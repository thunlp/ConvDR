import sys
import os
import torch
sys.path += ['../']
import pickle
from utils.util import pad_input_ids, multi_file_process, numbered_byte_file_generator, EmbeddingCache
from model.models import MSMarcoConfigDict, ALL_MODELS
from torch.utils.data import TensorDataset
import numpy as np
import argparse
import json


def preprocess(args):

    pid2offset = {}
    offset2pid = []
    in_passage_path = args.collection

    out_passage_path = os.path.join(
        args.out_data_dir,
        "passages",
    )

    if os.path.exists(out_passage_path):
        print("preprocessed data already exist, exit preprocessing")
        return

    out_line_count = 0

    print('start passage file split processing')
    multi_file_process(
        args,
        32,
        in_passage_path,
        out_passage_path,
        PassagePreprocessingFn)

    print('start merging splits')
    with open(out_passage_path, 'wb') as f:
        for idx, record in enumerate(numbered_byte_file_generator(
                out_passage_path, 32, 8 + 4 + args.max_seq_length * 4)):
            p_id = int.from_bytes(record[:8], 'big')
            f.write(record[8:])
            pid2offset[p_id] = idx
            offset2pid.append(p_id)
            if idx < 3:
                print(str(idx) + " " + str(p_id))
            out_line_count += 1

    print("Total lines written: " + str(out_line_count))
    meta = {
        'type': 'int32',
        'total_number': out_line_count,
        'embedding_size': args.max_seq_length}
    with open(out_passage_path + "_meta", 'w') as f:
        json.dump(meta, f)
    embedding_cache = EmbeddingCache(out_passage_path)
    print("First line")
    with embedding_cache as emb:
        print(emb[0])

    pid2offset_path = os.path.join(
        args.out_data_dir,
        "pid2offset.pickle",
    )
    offset2pid_path = os.path.join(
        args.out_data_dir,
        "offset2pid.pickle",
    )
    with open(pid2offset_path, 'wb') as handle:
        pickle.dump(pid2offset, handle, protocol=4)
    with open(offset2pid_path, "wb") as handle:
        pickle.dump(offset2pid, handle, protocol=4)
    print("done saving pid2offset")


def PassagePreprocessingFn(args, line, tokenizer):
    line = line.strip()
    ext = args.collection[args.collection.rfind("."):]
    passage = None
    if ext == ".jsonl":
        obj = json.loads(line)
        p_id = int(obj["id"])
        p_text = obj["text"]
        p_title = obj["title"]

        full_text = p_text[:args.max_doc_character]

        passage = tokenizer.encode(
            p_title, 
            text_pair=full_text,
            add_special_tokens=True,
            max_length=args.max_seq_length,
        )
    elif ext == ".tsv":
        try:
            line_arr = line.split('\t')
            p_id = int(line_arr[0])
            p_text = line_arr[1].rstrip()
        except IndexError:  # split error
            raise ValueError  # empty passage
        else:
            full_text = p_text[:args.max_doc_character]
            passage = tokenizer.encode(
                full_text,
                add_special_tokens=True,
                max_length=args.max_seq_length,
            )
    else:
        raise TypeError("Unrecognized file type")

    passage_len = min(len(passage), args.max_seq_length)
    input_id_b = pad_input_ids(passage, args.max_seq_length)

    return p_id.to_bytes(8,'big') + passage_len.to_bytes(4,'big') + np.array(input_id_b,np.int32).tobytes()


def QueryPreprocessingFn(args, line, tokenizer):
    line_arr = line.split('\t')
    q_id = int(line_arr[0])

    passage = tokenizer.encode(
        line_arr[1].rstrip(),
        add_special_tokens=True,
        max_length=args.max_query_length)
    passage_len = min(len(passage), args.max_query_length)
    input_id_b = pad_input_ids(passage, args.max_query_length)

    return q_id.to_bytes(8,'big') + passage_len.to_bytes(4,'big') + np.array(input_id_b,np.int32).tobytes()


def GetProcessingFn(args, query=False):
    def fn(vals, i):
        passage_len, passage = vals
        max_len = args.max_query_length if query else args.max_seq_length

        pad_len = max(0, max_len - passage_len)
        token_type_ids = ([0] if query else [1]) * passage_len + [0] * pad_len
        attention_mask = [1] * passage_len + [0] * pad_len

        passage_collection = [(i, passage, attention_mask, token_type_ids)]

        query2id_tensor = torch.tensor(
            [f[0] for f in passage_collection], dtype=torch.long)
        all_input_ids_a = torch.tensor(
            [f[1] for f in passage_collection], dtype=torch.int)
        all_attention_mask_a = torch.tensor(
            [f[2] for f in passage_collection], dtype=torch.bool)
        all_token_type_ids_a = torch.tensor(
            [f[3] for f in passage_collection], dtype=torch.uint8)

        dataset = TensorDataset(
            all_input_ids_a,
            all_attention_mask_a,
            all_token_type_ids_a,
            query2id_tensor)

        return [ts for ts in dataset]

    return fn


def GetTrainingDataProcessingFn(args, query_cache, passage_cache):
    def fn(line, i):
        line_arr = line.split('\t')
        qid = int(line_arr[0])
        pos_pid = int(line_arr[1])
        neg_pids = line_arr[2].split(',')
        neg_pids = [int(neg_pid) for neg_pid in neg_pids]

        all_input_ids_a = []
        all_attention_mask_a = []

        query_data = GetProcessingFn(
            args, query=True)(
            query_cache[qid], qid)[0]
        pos_data = GetProcessingFn(
            args, query=False)(
            passage_cache[pos_pid], pos_pid)[0]

        pos_label = torch.tensor(1, dtype=torch.long)
        neg_label = torch.tensor(0, dtype=torch.long)

        for neg_pid in neg_pids:
            neg_data = GetProcessingFn(
                args, query=False)(
                passage_cache[neg_pid], neg_pid)[0]
            yield (query_data[0], query_data[1], query_data[2], pos_data[0], pos_data[1], pos_data[2], pos_label)
            yield (query_data[0], query_data[1], query_data[2], neg_data[0], neg_data[1], neg_data[2], neg_label)

    return fn


def GetTripletTrainingDataProcessingFn(args, query_cache, passage_cache):
    def fn(line, i):
        line_arr = line.split('\t')
        qid = int(line_arr[0])
        pos_pid = int(line_arr[1])
        neg_pids = line_arr[2].split(',')
        neg_pids = [int(neg_pid) for neg_pid in neg_pids]

        all_input_ids_a = []
        all_attention_mask_a = []

        query_data = GetProcessingFn(
            args, query=True)(
            query_cache[qid], qid)[0]
        pos_data = GetProcessingFn(
            args, query=False)(
            passage_cache[pos_pid], pos_pid)[0]

        for neg_pid in neg_pids:
            neg_data = GetProcessingFn(
                args, query=False)(
                passage_cache[neg_pid], neg_pid)[0]
            yield (query_data[0], query_data[1], query_data[2], pos_data[0], pos_data[1], pos_data[2],
                   neg_data[0], neg_data[1], neg_data[2])

    return fn


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--collection",
        default=None,
        type=str,
        required=True,
        help="The input data dir",
    )
    parser.add_argument(
        "--out_data_dir",
        default=None,
        type=str,
        required=True,
        help="The output data dir",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " +
        ", ".join(ALL_MODELS),
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
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_doc_character",
        default=10000,
        type=int,
        help="used before tokenizer to save tokenizer latency",
    )

    args = parser.parse_args()

    return args


def main():
    args = get_arguments()

    if not os.path.exists(args.out_data_dir):
        os.makedirs(args.out_data_dir)
    preprocess(args)


if __name__ == '__main__':
    main()
