import argparse
import csv
import logging
import json
from model.models import MSMarcoConfigDict
import os
import pickle
import time

import copy
from utils.dpr_utils import get_model_obj, load_states_from_checkpoint
# from utils.msmarco_eval import compute_metrics
import faiss
import pytrec_eval
import torch
import random
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import SequentialSampler
import torch.nn.functional as F

from utils.util import convert_to_string_id, pad_input_ids, pad_input_ids_with_mask

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.util import ConvSearchDataset


global_responses = {}

ngpu = faiss.get_num_gpus()

gpu_resources = []
tempmem = -1

for i in range(ngpu):
    res = faiss.StandardGpuResources()
    if tempmem >= 0:
        res.setTempMemory(tempmem)
    gpu_resources.append(res)

def make_vres_vdev(i0=0, i1=-1):
    " return vectors of device ids and resources useful for gpu_multiple"
    vres = faiss.GpuResourcesVector()
    vdev = faiss.IntVector()
    if i1 == -1:
        i1 = ngpu
    for i in range(i0, i1):
        vdev.push_back(i)
        vres.push_back(gpu_resources[i])
    return vres, vdev

logger = logging.getLogger(__name__)
NUM_FOLD = 5

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def EvalDevQuery(query_embedding2id, merged_D, 
                 dev_query_positive_id, I_nearest_neighbor, 
                 topN, output_file, output_trec_file,
                 offset2pid, raw_data_dir, dev_query_name,
                 raw_sequences=None):
    prediction = {} #[qid][docid] = docscore, here we use -rank as score, so the higher the rank (1 > 2), the higher the score (-1 > -2)

    total = 0
    labeled = 0
    Atotal = 0
    Alabeled = 0
    qids_to_ranked_candidate_passages = {} 
    qids_to_ranked_candidate_passages_ori = {}
    qids_to_raw_sequences = {}
    for query_idx in range(len(I_nearest_neighbor)): 
        seen_pid = set()
        inputs = raw_sequences[query_idx]
        query_id = int(query_embedding2id[query_idx])
        prediction[query_id] = {}

        top_ann_pid = I_nearest_neighbor[query_idx].copy()
        top_ann_score = merged_D[query_idx].copy()
        selected_ann_idx = top_ann_pid[:topN]
        selected_ann_score = top_ann_score[:topN].tolist()
        rank = 0
        
        if query_id in qids_to_ranked_candidate_passages:
            pass    
        else:
            # By default, all PIDs in the list of 1000 are 0. Only override those that are given
            tmp = [(0, 0)] * 100
            tmp_ori = [0] * 100
            qids_to_ranked_candidate_passages[query_id] = tmp
            qids_to_ranked_candidate_passages_ori[query_id] = tmp_ori
        qids_to_raw_sequences[query_id] = inputs
                
        for idx, score in zip(selected_ann_idx, selected_ann_score):
            pred_pid = offset2pid[idx]
            
            if not pred_pid in seen_pid:
                # this check handles multiple vector per document
                qids_to_ranked_candidate_passages[query_id][rank]=(pred_pid, score)
                qids_to_ranked_candidate_passages_ori[query_id][rank] = pred_pid

                rank += 1
                prediction[query_id][pred_pid] = -rank
                seen_pid.add(pred_pid)

    # use out of the box evaluation script
    evaluator = pytrec_eval.RelevanceEvaluator(
        convert_to_string_id(dev_query_positive_id), {'map_cut', 'ndcg_cut.3', 'recip_rank','recall'})

    eval_query_cnt = 0
    result = evaluator.evaluate(convert_to_string_id(prediction))

    logger.info("Reading queries and passages...")
    queries = ["xxx"] * 200_0000
    with open(os.path.join(raw_data_dir, "queries." + dev_query_name + ".tsv"), "r") as f:
        for line in f:
            qid, query = line.strip().split("\t")
            qid = int(qid)
            queries[qid] = query
    collection = os.path.join(raw_data_dir, "collection.jsonl")
    all_passages = ["xx"] * 4000_0000
    with open(collection, "r") as f:
        for line in f:
            try:
                line = line.strip()
                line_arr = json.loads(line)
                pid = int(line_arr["id"])
                passage = line_arr["title"] + " " + line_arr["text"]
                all_passages[pid] = passage
            except IndexError:
                print(line)

    # write to file! qid \t pid \t q \t d \t
    with open(output_file, "w") as f, open(output_trec_file, "w") as g:
        for qid, passages in qids_to_ranked_candidate_passages.items():
            ori_qid = qid
            query_text = queries[ori_qid]
            sequences = qids_to_raw_sequences[ori_qid]
            for i in range(100):
                pid, score = passages[i]
                ori_pid = pid
                passage_text = all_passages[ori_pid]
                label = 0 if qid not in dev_query_positive_id else (dev_query_positive_id[qid][ori_pid] if ori_pid in dev_query_positive_id[qid] else 0)
                f.write(json.dumps({"query": query_text, 
                                    "doc": passage_text, 
                                    "label": label,
                                    "query_id": str(ori_qid), 
                                    "doc_id": str(ori_pid), 
                                    "retrieval_score": score,
                                    "input": sequences}) + "\n")
                g.write(str(ori_qid) + " Q0 " + str(ori_pid) + " " + str(i+1) + " " + str(-i-1+200) + " ance\n")
    
    # qids_to_relevant_passageids = {}
    # for qid in dev_query_positive_id:
    #     qid = int(qid)
    #     if qid in qids_to_relevant_passageids:
    #         pass
    #     else:
    #         qids_to_relevant_passageids[qid] = []
    #         for pid in dev_query_positive_id[qid]:
    #             if dev_query_positive_id[qid][pid]>0:
    #                 qids_to_relevant_passageids[qid].append(pid)
    
    # ms_mrr = compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages_ori)

    # ndcg = 0
    # Map = 0
    # mrr = 0
    # recall = 0
    # recall_1000 = 0

    # for k in result.keys():
    #     eval_query_cnt += 1
    #     ndcg += result[k]["ndcg_cut_3"]
    #     Map += result[k]["map_cut_10"]
    #     mrr += result[k]["recip_rank"]
    #     recall += result[k]["recall_"+str(topN)]

    # final_ndcg = ndcg / eval_query_cnt
    # final_Map = Map / eval_query_cnt
    # final_mrr = mrr / eval_query_cnt
    # final_recall = recall / eval_query_cnt
    # hole_rate = labeled/total
    # Ahole_rate = Alabeled/Atotal

    # return final_ndcg, eval_query_cnt, final_Map, final_mrr, final_recall, 0, ms_mrr, 0, result, prediction


def evaluate(args, eval_dataset, model, logger):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu) if not args.y2 else 1
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=eval_dataset.get_collate_fn(args, "inference"))

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_eval_batch_size)
    collection = os.path.join(args.raw_data_dir, "collection.tsv")

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    embedding = []
    embedding2id = []
    raw_sequences = []
    # epoch_iterator = tqdm(eval_dataloader, desc="Iteration")
    epoch_iterator = eval_dataloader
    for step, batch in enumerate(epoch_iterator):
        # topic_number, query_number = batch[:2]
        qids = batch["qid"]
        ids, id_mask = (ele.to(args.device) for ele in [batch["concat_ids"], batch["concat_id_mask"]]) 
        model.eval()
        with torch.no_grad():
            embs = model(ids, id_mask)
        embs = embs.detach().cpu().numpy()
        embedding.append(embs)
        for qid in qids:
            embedding2id.append(qid)
        
        sequences = batch["history_utterances"]
        raw_sequences.extend(sequences)

    embedding = np.concatenate(embedding, axis=0)
    return embedding, embedding2id, raw_sequences


def load_model(args, checkpoint_path):
    label_list = ["0", "1"]
    num_labels = len(label_list)
    args.model_type = args.model_type.lower()
    configObj = MSMarcoConfigDict[args.model_type]
    args.model_path = checkpoint_path

    config, tokenizer, model = None, None, None
    if args.model_type != "dpr":
        config = configObj.config_class.from_pretrained(
            args.model_path,
            num_labels=num_labels,
            finetuning_task="MSMarco",
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        tokenizer = configObj.tokenizer_class.from_pretrained(
            args.model_path,
            do_lower_case=True,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model = configObj.model_class.from_pretrained(
            args.model_path,
            from_tf=bool(".ckpt" in args.model_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    else:  # dpr
        model = configObj.model_class(args)
        saved_state = load_states_from_checkpoint(checkpoint_path)
        model_to_load = get_model_obj(model)
        logger.info('Loading saved model state ...')
        model_to_load.load_state_dict(saved_state.model_dict)
        tokenizer = configObj.tokenizer_class.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True,
            cache_dir=None,
        )

    model.to(args.device)
    logger.info("Inference parameters %s", args)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    return config, tokenizer, model


def search_one_by_one(ann_data_dir, gpu_index, query_embedding):
    merged_candidate_matrix = None
    for block_id in range(8):
        logger.info("Loading passage reps " + str(block_id))
        passage_embedding = None
        passage_embedding2id = None
        try:
            with open(os.path.join(ann_data_dir, "passage__emb_p__data_obj_"+str(block_id)+".pb"), 'rb') as handle:
                passage_embedding = pickle.load(handle)
            with open(os.path.join(ann_data_dir, "passage__embid_p__data_obj_"+str(block_id)+".pb"), 'rb') as handle:
                passage_embedding2id = pickle.load(handle)
        except:
            break
        print('passage embedding shape: ' + str(passage_embedding.shape))
        gpu_index.add(passage_embedding)
        ts = time.time()
        D, I = gpu_index.search(query_embedding, 100)
        te = time.time()
        elapsed_time = te - ts
        print({"total": elapsed_time, "data": query_embedding.shape[0], "per_query": elapsed_time / query_embedding.shape[0]})
        candidate_id_matrix = passage_embedding2id[I]  # passage_idx -> passage_id
        D = D.tolist()
        candidate_id_matrix = candidate_id_matrix.tolist()
        candidate_matrix = []
        for score_list, passage_list in zip(D, candidate_id_matrix):
            candidate_matrix.append([])
            for score, passage in zip(score_list, passage_list):
                candidate_matrix[-1].append((score, passage))
            assert len(candidate_matrix[-1]) == len(passage_list)
        assert len(candidate_matrix) == I.shape[0]

        gpu_index.reset()
        del passage_embedding
        del passage_embedding2id

        if merged_candidate_matrix == None:
            merged_candidate_matrix = candidate_matrix
            continue

        # merge
        merged_candidate_matrix_tmp = copy.deepcopy(merged_candidate_matrix)
        merged_candidate_matrix = []
        for merged_list, cur_list in zip(merged_candidate_matrix_tmp, candidate_matrix):
            p1, p2 = 0, 0
            merged_candidate_matrix.append([])
            while p1 < 100 and p2 < 100:
                if merged_list[p1][0] >= cur_list[p2][0]:
                    merged_candidate_matrix[-1].append(merged_list[p1])
                    p1 += 1
                else:
                    merged_candidate_matrix[-1].append(cur_list[p2])
                    p2 += 1
            while p1 < 100:
                merged_candidate_matrix[-1].append(merged_list[p1])
                p1 += 1
            while p2 < 100:
                merged_candidate_matrix[-1].append(cur_list[p2])
                p2 += 1
    
    merged_D, merged_I = [], []
    for merged_list in merged_candidate_matrix:
        merged_D.append([])
        merged_I.append([])
        for candidate in merged_list:
            merged_D[-1].append(candidate[0])
            merged_I[-1].append(candidate[1])
    merged_D, merged_I = np.array(merged_D), np.array(merged_I)

    print(merged_I)

    return merged_D, merged_I


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--eval_file", type=str)
    parser.add_argument("--block_size", default=256, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--max_query_length", default=64, type=int)
    parser.add_argument("--cross_validate", action='store_true',
                        help="Set when doing cross validation")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--ann_data_dir", type=str)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--qrels", type=str)
    parser.add_argument("--processed_data_dir", type=str)
    parser.add_argument("--raw_data_dir", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--output_trec_file", type=str)
    parser.add_argument("--query", type=str, default="concat")
    parser.add_argument("--y2", action="store_true")
    parser.add_argument("--dev_query_name", type=str)
    parser.add_argument("--mse", action='store_true', help="Whether to measure mse loss")
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--eval_on_train", action='store_true')
    parser.add_argument("--fold", type=int, default=-1)
    # parser.add_argument("--step", type=int)
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(
            MSMarcoConfigDict.keys()),
    )
    args = parser.parse_args()

    tb_writer = SummaryWriter(log_dir=args.log_dir) if args.log_dir else None

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args)

    with open(os.path.join(args.processed_data_dir, "offset2pid.pickle"), "rb") as f:
        offset2pid = pickle.load(f)

    logger.info("Building index")
    faiss.omp_set_num_threads(16)
    cpu_index = faiss.IndexFlatIP(768)
    index = None
    if args.use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True 
        gpu_vector_resources, gpu_devices_vector = make_vres_vdev(0, ngpu)
        gpu_index = faiss.index_cpu_to_gpu_multiple(gpu_vector_resources, gpu_devices_vector, cpu_index, co)
        index = gpu_index
    else:
        index = cpu_index

    dev_query_positive_id = {}
    with open(args.qrels, 'r', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, _, docid, rel] in tsvreader:
            topicid = int(topicid)
            docid = int(docid)
            rel = int(rel)
            if topicid not in dev_query_positive_id:
                if rel > 0:
                    dev_query_positive_id[topicid] = {}
                    dev_query_positive_id[topicid][docid] = rel
            else:
                dev_query_positive_id[topicid][docid] = rel
            
    total_embedding = []
    total_embedding2id = []
    total_raw_sequences = []

    if not args.cross_validate:

        config, tokenizer, model = load_model(args, args.model_path)
	
        if args.block_size <= 0:
            args.block_size = tokenizer.max_len_single_sentence
        args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

        # eval
        logger.info("Training/evaluation parameters %s", args)
        eval_dataset = ConvSearchDataset([args.eval_file], tokenizer, args, mode="inference")
        total_embedding, total_embedding2id, raw_sequences = evaluate(args, eval_dataset, model, logger)
        # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        total_raw_sequences.extend(raw_sequences)
        del model
        torch.cuda.empty_cache()

    else:
        # K-Fold Cross Validation

        for i in range(NUM_FOLD):
            if args.fold != -1 and i != args.fold:
                continue
            
            logger.info("Testing Fold #{}".format(i))
            suffix = ('-' + str(i))
            config, tokenizer, model = load_model(args, args.model_path + suffix)

            if args.block_size <= 0:
                args.block_size = tokenizer.max_len_single_sentence
            args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    
            logger.info("Training/evaluation parameters %s", args)
            eval_file = "%s.%d" % (args.eval_file, i)
            logger.info("eval_file: {}".format(eval_file))
            train_files = ["%s.%d" % (args.eval_file, j) for j in range(NUM_FOLD) if j != i]
            eval_dataset = ConvSearchDataset([eval_file], tokenizer, args, mode="inference") if not args.eval_on_train else ConvSearchDataset(train_files, tokenizer, args, mode="inference")
            embedding, embedding2id, raw_sequences = evaluate(args, eval_dataset, model, logger)
            total_embedding.append(embedding)
            total_embedding2id.extend(embedding2id)
            total_raw_sequences.extend(raw_sequences)

            del model
            torch.cuda.empty_cache() 
            
        total_embedding = np.concatenate(total_embedding, axis=0)

    merged_D, merged_I = search_one_by_one(args.ann_data_dir, index, total_embedding)
    logger.info("start EvalDevQuery...")
    EvalDevQuery(total_embedding2id, merged_D,
                    dev_query_positive_id=dev_query_positive_id,
                    I_nearest_neighbor=merged_I, topN=100,
                    output_file=args.output_file, output_trec_file=args.output_trec_file,
                    offset2pid=offset2pid, raw_data_dir=args.raw_data_dir,
                    dev_query_name=args.dev_query_name,
                    raw_sequences=total_raw_sequences)

    if args.log_dir:
        tb_writer.close()


if __name__ == "__main__":
    main()
