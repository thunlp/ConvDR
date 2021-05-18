import argparse
import logging
import os
import torch
import random
from tensorboardX import SummaryWriter

from utils.util import pad_input_ids_with_mask, getattr_recursive

from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup
from torch import nn

from model.models import MSMarcoConfigDict
from utils.util import ConvSearchDataset, NUM_FOLD, set_seed, load_model
from utils.dpr_utils import CheckpointState, get_model_obj, get_optimizer

logger = logging.getLogger(__name__)


def _save_checkpoint(args,
                     model,
                     output_dir,
                     optimizer=None,
                     scheduler=None,
                     step=0) -> str:
    offset = step
    epoch = 0
    model_to_save = get_model_obj(model)
    cp = os.path.join(output_dir, 'checkpoint-' + str(offset))

    meta_params = {}
    state = CheckpointState(model_to_save.state_dict(), optimizer.state_dict(),
                            scheduler.state_dict(), offset, epoch, meta_params)
    torch.save(state._asdict(), cp)
    logger.info('Saved checkpoint at %s', cp)
    return cp


def train(args,
          train_dataset,
          model,
          teacher_model,
          loss_fn,
          logger,
          writer: SummaryWriter,
          cross_validate_id=-1,
          loss_fn_2=None,
          tokenizer=None):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=train_dataset.get_collate_fn(
                                      args, "train"))

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader
        ) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer = get_optimizer(args, model, weight_decay=args.weight_decay)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    tr_loss1, tr_loss2 = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(
        args)  # Added here for reproducibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            concat_ids, concat_id_mask, target_ids, target_id_mask = (ele.to(
                args.device) for ele in [
                    batch["concat_ids"], batch["concat_id_mask"],
                    batch["target_ids"], batch["target_id_mask"]
                ])
            model.train()
            teacher_model.eval()
            embs = model(concat_ids, concat_id_mask)
            with torch.no_grad():
                teacher_embs = teacher_model(target_ids,
                                             target_id_mask).detach()
            loss1 = None
            if not args.no_mse:
                loss1 = loss_fn(embs, teacher_embs)
            loss = loss1
            loss2 = None
            if args.ranking_task:
                bs = concat_ids.shape[0]  # real batch size
                pos_and_negs = batch["documents"]  #.numpy()  # (B, K)
                total_token_list = []
                for group in pos_and_negs:
                    sampled = random.sample(group[1:], args.num_negatives)
                    this_pos_neg = [group[0]] + sampled
                    for doc in this_pos_neg:
                        try:
                            title, text = doc.split("[SEP]")
                            doc_ids = tokenizer.encode(title,
                                                       text_pair=text,
                                                       add_special_tokens=True,
                                                       max_length=512)
                        except ValueError:
                            doc_ids = tokenizer.encode(doc,
                                                       add_special_tokens=True,
                                                       max_length=512)
                        doc_ids, doc_mask = pad_input_ids_with_mask(
                            doc_ids, 512)
                        total_token_list.append((doc_ids, doc_mask))
                doc_batch_size = 8
                pos_and_negs_embeddings = []
                for i in range(0, len(total_token_list), doc_batch_size):
                    # batchify
                    batch_ids = []
                    batch_mask = []
                    for j in range(
                            i, min(i + doc_batch_size, len(total_token_list))):
                        batch_ids.append(total_token_list[j][0])
                        batch_mask.append(total_token_list[j][1])
                    batch_ids = torch.tensor(batch_ids,
                                             dtype=torch.long).to(args.device)
                    batch_mask = torch.tensor(batch_mask,
                                              dtype=torch.long).to(args.device)
                    with torch.no_grad():
                        pos_and_negs_embeddings_tmp = teacher_model(
                            batch_ids, batch_mask, is_query=False).detach()
                        pos_and_negs_embeddings.append(
                            pos_and_negs_embeddings_tmp)
                pos_and_negs_embeddings = torch.cat(pos_and_negs_embeddings,
                                                    dim=0)  # (B * K, E)
                pos_and_negs_embeddings = pos_and_negs_embeddings.view(
                    bs, args.num_negatives + 1, -1)  # (B, K, E)
                embs_for_ranking = embs.unsqueeze(-1)  # (B, E, 1)
                embs_for_ranking = embs_for_ranking.expand(
                    bs, 768, args.num_negatives + 1)  # (B, E, K)
                embs_for_ranking = embs_for_ranking.transpose(1,
                                                              2)  # (B, K, E)
                logits = embs_for_ranking * pos_and_negs_embeddings
                logits = torch.sum(logits, dim=-1)  # (B, K)
                labels = torch.zeros(bs, dtype=torch.long).to(args.device)
                loss2 = loss_fn_2(logits, labels)
                loss = loss1 + loss2 if loss1 != None else loss2

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            if not args.no_mse:
                tr_loss1 += loss1.item()
            if args.ranking_task:
                tr_loss2 += loss2.item()
            del loss
            torch.cuda.empty_cache()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if global_step % args.log_steps == 0:
                    writer.add_scalar(
                        str(cross_validate_id) + "/loss",
                        tr_loss / args.log_steps, global_step)
                    if not args.no_mse:
                        writer.add_scalar(
                            str(cross_validate_id) + "/mse_loss",
                            tr_loss1 / args.log_steps, global_step)
                    if args.ranking_task:
                        writer.add_scalar(
                            str(cross_validate_id) + "/ranking_loss",
                            tr_loss2 / args.log_steps, global_step)
                    tr_loss = 0.0
                    tr_loss1 = 0.0
                    tr_loss2 = 0.0

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                    output_dir = args.output_dir + (
                        ('-' + str(cross_validate_id))
                        if cross_validate_id != -1 else "")
                    if args.model_type == "rdot_nll":
                        output_dir = os.path.join(
                            output_dir, '{}-{}'.format(checkpoint_prefix,
                                                       global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(
                            model, 'module') else model
                        model_to_save.save_pretrained(output_dir)
                        torch.save(
                            args, os.path.join(output_dir,
                                               'training_args.bin'))
                    else:
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        _save_checkpoint(args, model, output_dir, optimizer,
                                         scheduler, global_step)
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.model_type == "dpr":
        output_dir = args.output_dir + (
            ('-' + str(cross_validate_id)) if cross_validate_id != -1 else "")
        # output_dir = os.path.join(output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        _save_checkpoint(args, model, output_dir, optimizer, scheduler,
                         global_step)

    return global_step, tr_loss / global_step


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="The model checkpoint for weights initialization.")
    parser.add_argument(
        "--max_concat_length",
        default=256,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens)."
    )
    parser.add_argument(
        "--max_query_length", 
        default=64, 
        type=int,
        help="Max input query length after tokenization."
             "This option is for single query input."
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        required=True,
        help=
        "Path of training file. Do not add fold suffix when cross validate, i.e. use 'data/eval_topics.jsonl' instead of 'data/eval_topics.jsonl.0'"
    )
    parser.add_argument(
        "--cross_validate",
        action='store_true',
        help="Set when doing cross validation"
    )
    parser.add_argument(
        "--init_from_multiple_models",
        action='store_true',
        help=
        "Set when initialize from different models during cross validation (Model-based+CV)"
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MSMarcoConfigDict.keys()),
    )

    parser.add_argument(
        "--ranking_task", 
        action='store_true',
        help="Whether to use ranking loss."
    )
    parser.add_argument(
        "--no_mse", 
        action="store_true",
        help="Whether to disable KD loss."
    )
    parser.add_argument(
        "--num_negatives", 
        type=int, 
        default=9,
        help="Number of negative documents per query."
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=1.0,
        type=float,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps to perform. Override num_train_epochs."
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        '--save_steps',
        type=int,
        default=-1,
        help="Save checkpoint every X updates steps."
    )
    parser.add_argument(
        "--no_cuda",
        action='store_true',
        help="Avoid using CUDA when available"
    )
    parser.add_argument(
        '--overwrite_output_dir',
        action='store_true',
        help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="random seed for initialization"
    )
    parser.add_argument(
        "--log_dir", 
        type=str,
        help="Directory for tensorboard logging."
    )
    parser.add_argument(
        "--log_steps", 
        type=int, 
        default=1,
        help="Log loss every x steps."
    )
    parser.add_argument(
        "--cache_dir", 
        type=str
    )
    parser.add_argument(
        "--teacher_model", 
        type=str,
        help="The teacher model. If None, use `model_name_or_path` as teacher."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="no_res",
        choices=["no_res", "man_can", "auto_can", "target", "output", "raw"],
        help="Input query format."
    )
    args = parser.parse_args()

    tb_writer = SummaryWriter(log_dir=args.log_dir)

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            .format(args.output_dir))

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args)

    loss_fn = nn.MSELoss()
    loss_fn.to(args.device)
    loss_fn_2 = nn.CrossEntropyLoss()
    loss_fn_2.to(args.device)

    if args.teacher_model == None:
        args.teacher_model = args.model_name_or_path
    _, __, teacher_model = load_model(args, args.teacher_model)

    if not args.cross_validate:

        config, tokenizer, model = load_model(args, args.model_name_or_path)
        if args.query in ["man_can", "auto_can"]:
            tokenizer.add_tokens(["<response>"])
            model.resize_token_embeddings(len(tokenizer))
        if args.max_concat_length <= 0:
            args.max_concat_length = tokenizer.max_len_single_sentence
        args.max_concat_length = min(args.max_concat_length,
                                     tokenizer.max_len_single_sentence)

        # Training
        logger.info("Training/evaluation parameters %s", args)
        train_dataset = ConvSearchDataset([args.train_file],
                                          tokenizer,
                                          args,
                                          mode="train")
        global_step, tr_loss = train(args,
                                     train_dataset,
                                     model,
                                     teacher_model,
                                     loss_fn,
                                     logger,
                                     tb_writer,
                                     cross_validate_id=0,
                                     loss_fn_2=loss_fn_2,
                                     tokenizer=tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step,
                    tr_loss)

        # Saving
        # Create output directory if needed

        if args.model_type == "rdot_nll":
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            logger.info("Saving model checkpoint to %s", args.output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir,
                                          'training_args.bin'))

    else:
        # K-Fold Cross Validation
        for i in range(NUM_FOLD):
            logger.info("Training Fold #{}".format(i))
            suffix = ('-' + str(i)) if args.init_from_multiple_models else ''
            config, tokenizer, model = load_model(
                args, args.model_name_or_path + suffix)
            if args.query in ["man_can", "auto_can"]:
                tokenizer.add_tokens(["<response>"])
                model.resize_token_embeddings(len(tokenizer))

            if args.max_concat_length <= 0:
                args.max_concat_length = tokenizer.max_len_single_sentence
            args.max_concat_length = min(args.max_concat_length,
                                         tokenizer.max_len_single_sentence)

            logger.info("Training/evaluation parameters %s", args)
            train_files = [
                "%s.%d" % (args.train_file, j) for j in range(NUM_FOLD)
                if j != i
            ]
            logger.info("train_files: {}".format(train_files))
            train_dataset = ConvSearchDataset(train_files,
                                              tokenizer,
                                              args,
                                              mode="train")
            global_step, tr_loss = train(args,
                                         train_dataset,
                                         model,
                                         teacher_model,
                                         loss_fn,
                                         logger,
                                         tb_writer,
                                         cross_validate_id=i,
                                         loss_fn_2=loss_fn_2,
                                         tokenizer=tokenizer)
            logger.info(" global_step = %s, average loss = %s", global_step,
                        tr_loss)

            if args.model_type == "rdot_nll":
                output_dir = args.output_dir + '-' + str(i)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                logger.info("Saving model checkpoint to %s", output_dir)
                model_to_save = model.module if hasattr(model,
                                                        'module') else model
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))

            del model
            torch.cuda.empty_cache()

    tb_writer.close()


if __name__ == "__main__":
    main()
