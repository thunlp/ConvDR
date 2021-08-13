import sys

sys.path += ['../']
import torch
import os
from utils.util import (
    barrier_array_merge,
    StreamingDataset,
    EmbeddingCache,
)
from data.tokenizing import GetProcessingFn
from model.models import MSMarcoConfigDict
from torch import nn
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import argparse
import logging
from utils.dpr_utils import load_states_from_checkpoint, get_model_obj
import re

torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)


def load_model(args, checkpoint_path):
    label_list = ["0", "1"]
    num_labels = len(label_list)
    args.model_type = args.model_type.lower()
    configObj = MSMarcoConfigDict[args.model_type]
    args.model_name_or_path = checkpoint_path

    config, tokenizer, model = None, None, None
    if args.model_type != "dpr":
        config = configObj.config_class.from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task="MSMarco",
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        tokenizer = configObj.tokenizer_class.from_pretrained(
            args.model_name_or_path,
            do_lower_case=True,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model = configObj.model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    else:  # dpr
        model = configObj.model_class(args)
        saved_state = load_states_from_checkpoint(checkpoint_path)
        model_to_load = get_model_obj(model)
        logger.info('Loading saved model state ...')
        model_to_load.load_state_dict(saved_state.model_dict)

    model.to(args.device)
    logger.info("Inference parameters %s", args)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    return config, tokenizer, model


def InferenceEmbeddingFromStreamDataLoader(
    args,
    model,
    train_dataloader,
    is_query_inference=True,
):
    # expect dataset from ReconstructTrainingSet
    results = {}
    eval_batch_size = args.per_gpu_eval_batch_size

    # Inference!
    logger.info("***** Running ANN Embedding Inference *****")
    logger.info("  Batch size = %d", eval_batch_size)

    embedding = []
    embedding2id = []

    if args.local_rank != -1:
        dist.barrier()
    model.eval()

    for batch in tqdm(train_dataloader,
                      desc="Inferencing",
                      disable=args.local_rank not in [-1, 0],
                      position=0,
                      leave=True):

        idxs = batch[3].detach().numpy()  # [#B]

        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0].long(),
                "attention_mask": batch[1].long()
            }
            if is_query_inference:
                embs = model.module.query_emb(**inputs)
            else:
                embs = model.module.body_emb(**inputs)

        embs = embs.detach().cpu().numpy()

        # check for multi chunk output for long sequence
        if len(embs.shape) == 3:
            for chunk_no in range(embs.shape[1]):
                embedding2id.append(idxs)
                embedding.append(embs[:, chunk_no, :])
        else:
            embedding2id.append(idxs)
            embedding.append(embs)

    embedding = np.concatenate(embedding, axis=0)
    embedding2id = np.concatenate(embedding2id, axis=0)
    return embedding, embedding2id


# streaming inference
def StreamInferenceDoc(args,
                       model,
                       fn,
                       prefix,
                       f,
                       is_query_inference=True,
                       merge=True):
    inference_batch_size = args.per_gpu_eval_batch_size  # * max(1, args.n_gpu)
    inference_dataset = StreamingDataset(f, fn)
    inference_dataloader = DataLoader(inference_dataset,
                                      batch_size=inference_batch_size)

    if args.local_rank != -1:
        dist.barrier()  # directory created

    _embedding, _embedding2id = InferenceEmbeddingFromStreamDataLoader(
        args,
        model,
        inference_dataloader,
        is_query_inference=is_query_inference,
        )

    logger.info("merging embeddings")

    # preserve to memory
    full_embedding = barrier_array_merge(args,
                                         _embedding,
                                         prefix=prefix + "_emb_p_",
                                         load_cache=False,
                                         only_load_in_master=True,
                                         merge=merge)
    full_embedding2id = barrier_array_merge(args,
                                            _embedding2id,
                                            prefix=prefix + "_embid_p_",
                                            load_cache=False,
                                            only_load_in_master=True,
                                            merge=merge)

    return full_embedding, full_embedding2id


def generate_new_ann(
    args,
    checkpoint_path,
):

    _, __, model = load_model(args, checkpoint_path)
    merge = False

    logger.info("***** inference of passages *****")
    passage_collection_path = os.path.join(args.data_dir,
                                           "passages")
    passage_cache = EmbeddingCache(passage_collection_path)
    with passage_cache as emb:
        passage_embedding, passage_embedding2id = StreamInferenceDoc(
            args,
            model,
            GetProcessingFn(args, query=False),
            "passage_",
            emb,
            is_query_inference=False,
            merge=merge)
    logger.info("***** Done passage inference *****")


def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the tokenized passage ids.",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        required=True,
        help="Checkpoint of the ad hoc retriever",
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
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the training data will be written",
    )

    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        required=True,
        help="The directory where cached data will be written",
    )

    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--max_doc_character",
        default=10000,
        type=int,
        help="used before tokenizer to save tokenizer latency",
    )

    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=64,
        type=int,
        help="The starting output file number",
    )

    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Avoid using CUDA when available",
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    parser.add_argument(
        "--server_ip",
        type=str,
        default="",
        help="For distant debugging.",
    )

    parser.add_argument(
        "--server_port",
        type=str,
        default="",
        help="For distant debugging.",
    )

    args = parser.parse_args()

    return args


def set_env(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )


def ann_data_gen(args):

    logger.info("start generate ann data")
    generate_new_ann(
        args,
        args.checkpoint,
    )

    if args.local_rank != -1:
        dist.barrier()


def main():
    args = get_arguments()
    set_env(args)
    ann_data_gen(args)


if __name__ == "__main__":
    main()
