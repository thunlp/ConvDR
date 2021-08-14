import sys

sys.path += ['../']
# import pandas as pd
# from sklearn.metrics import roc_curve, auc
import gzip
import copy
import torch
from torch import nn
import torch.distributed as dist
from tqdm import tqdm, trange
import os
from os import listdir
from os.path import isfile, join
import json
import logging
import random
import pytrec_eval
import pickle
import numpy as np
import torch

torch.multiprocessing.set_sharing_strategy('file_system')
from multiprocessing import Process
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
from utils.dpr_utils import get_model_obj, load_states_from_checkpoint
import re
from model.models import MSMarcoConfigDict, ALL_MODELS
from typing import List, Set, Dict, Tuple, Callable, Iterable, Any

logger = logging.getLogger(__name__)
NUM_FOLD = 5


class InputFeaturesPair(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """
    def __init__(self,
                 input_ids_a,
                 attention_mask_a=None,
                 token_type_ids_a=None,
                 input_ids_b=None,
                 attention_mask_b=None,
                 token_type_ids_b=None,
                 label=None):

        self.input_ids_a = input_ids_a
        self.attention_mask_a = attention_mask_a
        self.token_type_ids_a = token_type_ids_a

        self.input_ids_b = input_ids_b
        self.attention_mask_b = attention_mask_b
        self.token_type_ids_b = token_type_ids_b

        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def getattr_recursive(obj, name):
    for layer in name.split("."):
        if hasattr(obj, layer):
            obj = getattr(obj, layer)
        else:
            return None
    return obj


def barrier_array_merge(args,
                        data_array,
                        merge_axis=0,
                        prefix="",
                        load_cache=False,
                        only_load_in_master=False,
                        merge=True):
    # data array: [B, any dimension]
    # merge alone one axis

    if args.local_rank == -1:
        return data_array

    if not load_cache:
        rank = args.rank
        if is_first_worker():
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        dist.barrier()  # directory created
        pickle_path = os.path.join(
            args.output_dir, "{1}_data_obj_{0}.pb".format(str(rank), prefix))
        with open(pickle_path, 'wb') as handle:
            pickle.dump(data_array, handle, protocol=4)

        # make sure all processes wrote their data before first process
        # collects it
        dist.barrier()

    data_array = None

    data_list = []

    if not merge:
        return None

    # return empty data
    if only_load_in_master:
        if not is_first_worker():
            dist.barrier()
            return None

    for i in range(args.world_size
                   ):  # TODO: dynamically find the max instead of HardCode
        pickle_path = os.path.join(
            args.output_dir, "{1}_data_obj_{0}.pb".format(str(i), prefix))
        try:
            with open(pickle_path, 'rb') as handle:
                b = pickle.load(handle)
                data_list.append(b)
        except BaseException:
            continue

    data_array_agg = np.concatenate(data_list, axis=merge_axis)
    dist.barrier()
    return data_array_agg


def pad_input_ids(input_ids, max_length, pad_on_left=False, pad_token=0):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    # attention_mask = [1] * len(input_ids) + [0] * padding_length

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            input_ids = input_ids + padding_id

    return input_ids


def pad_input_ids_with_mask(input_ids,
                            max_length,
                            pad_on_left=False,
                            pad_token=0):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    attention_mask = []

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
        attention_mask = [1] * max_length
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            input_ids = input_ids + padding_id

    assert len(input_ids) == max_length
    assert len(attention_mask) == max_length

    return input_ids, attention_mask


def pad_ids(input_ids,
            attention_mask,
            token_type_ids,
            max_length,
            pad_on_left=False,
            pad_token=0,
            pad_token_segment_id=0,
            mask_padding_with_zero=True):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length
    padding_type = [pad_token_segment_id] * padding_length
    padding_attention = [0 if mask_padding_with_zero else 1] * padding_length

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        token_type_ids = token_type_ids[:max_length]
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
            attention_mask = padding_attention + attention_mask
            token_type_ids = padding_type + token_type_ids
        else:
            input_ids = input_ids + padding_id
            attention_mask = attention_mask + padding_attention
            token_type_ids = token_type_ids + padding_type

    return input_ids, attention_mask, token_type_ids


# to reuse pytrec_eval, id must be string
def convert_to_string_id(result_dict):
    string_id_dict = {}

    # format [string, dict[string, val]]
    for k, v in result_dict.items():
        _temp_v = {}
        for inner_k, inner_v in v.items():
            _temp_v[str(inner_k)] = inner_v

        string_id_dict[str(k)] = _temp_v

    return string_id_dict


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


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
    return config, tokenizer, model


def is_first_worker():
    return not dist.is_available() or not dist.is_initialized(
    ) or dist.get_rank() == 0


def concat_key(all_list, key, axis=0):
    return np.concatenate([ele[key] for ele in all_list], axis=axis)


def get_checkpoint_no(checkpoint_path):
    return int(re.findall(r'\d+', checkpoint_path)[-1])


def get_latest_ann_data(ann_data_path):
    ANN_PREFIX = "ann_ndcg_"
    if not os.path.exists(ann_data_path):
        return -1, None, None
    files = list(next(os.walk(ann_data_path))[2])
    num_start_pos = len(ANN_PREFIX)
    data_no_list = [
        int(s[num_start_pos:]) for s in files
        if s[:num_start_pos] == ANN_PREFIX
    ]
    if len(data_no_list) > 0:
        data_no = max(data_no_list)
        with open(os.path.join(ann_data_path, ANN_PREFIX + str(data_no)),
                  'r') as f:
            ndcg_json = json.load(f)
        return data_no, os.path.join(ann_data_path, "ann_training_data_" +
                                     str(data_no)), ndcg_json
    return -1, None, None


def numbered_byte_file_generator(base_path, file_no, record_size):
    for i in range(file_no):
        with open('{}_split{}'.format(base_path, i), 'rb') as f:
            while True:
                b = f.read(record_size)
                if not b:
                    # eof
                    break
                yield b


def load_collection(collection_file):
    all_passages = ["[INVALID DOC ID]"] * 5000_0000
    ext = collection_file[collection_file.rfind(".") + 1:]
    if ext not in ["jsonl", "tsv"]:
        raise TypeError("Unrecognized file type")
    with open(collection_file, "r") as f:
        if ext == "jsonl":
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                pid = int(obj["id"])
                passage = obj["title"] + "[SEP]" + obj["text"]
                all_passages[pid] = passage
        else:
            for line in f:
                line = line.strip()
                try:
                    line_arr = line.split("\t")
                    pid = int(line_arr[0])
                    passage = line_arr[1].rstrip()
                    all_passages[pid] = passage
                except IndexError:
                    print("bad passage")
                except ValueError:
                    print("bad pid")
    return all_passages


class EmbeddingCache:
    def __init__(self, base_path, seed=-1):
        self.base_path = base_path
        with open(base_path + '_meta', 'r') as f:
            meta = json.load(f)
            self.dtype = np.dtype(meta['type'])
            self.total_number = meta['total_number']
            self.record_size = int(
                meta['embedding_size']) * self.dtype.itemsize + 4
        if seed >= 0:
            self.ix_array = np.random.RandomState(seed).permutation(
                self.total_number)
        else:
            self.ix_array = np.arange(self.total_number)
        self.f = None

    def open(self):
        self.f = open(self.base_path, 'rb')

    def close(self):
        self.f.close()

    def read_single_record(self):
        record_bytes = self.f.read(self.record_size)
        passage_len = int.from_bytes(record_bytes[:4], 'big')
        passage = np.frombuffer(record_bytes[4:], dtype=self.dtype)
        return passage_len, passage

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __getitem__(self, key):
        if key < 0 or key > self.total_number:
            raise IndexError(
                "Index {} is out of bound for cached embeddings of size {}".
                format(key, self.total_number))
        self.f.seek(key * self.record_size)
        return self.read_single_record()

    def __iter__(self):
        self.f.seek(0)
        for i in range(self.total_number):
            new_ix = self.ix_array[i]
            yield self.__getitem__(new_ix)

    def __len__(self):
        return self.total_number


class StreamingDataset(IterableDataset):
    def __init__(self, elements, fn):
        super().__init__()
        self.elements = elements
        self.fn = fn
        self.num_replicas = -1

    def __iter__(self):
        if dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            print("Rank:", self.rank, "world:", self.num_replicas)
        else:
            print("Not running in distributed mode")
        for i, element in enumerate(self.elements):
            if self.num_replicas != -1 and i % self.num_replicas != self.rank:
                continue
            records = self.fn(element, i)
            for rec in records:
                # print("yielding record")
                # print(rec)
                yield rec


class ConvSearchExample:
    def __init__(self,
                 qid,
                 concat_ids,
                 concat_id_mask,
                 target_ids,
                 target_id_mask,
                 doc_pos=None,
                 doc_negs=None,
                 raw_sequences=None):
        self.qid = qid
        self.concat_ids = concat_ids
        self.concat_id_mask = concat_id_mask
        self.target_ids = target_ids
        self.target_id_mask = target_id_mask
        self.doc_pos = doc_pos
        self.doc_negs = doc_negs
        self.raw_sequences = raw_sequences


class ConvSearchDataset(Dataset):
    def __init__(self, filenames, tokenizer, args, mode="train"):
        self.examples = []
        for filename in filenames:
            with open(filename, encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    input_sents = record['input']
                    target_sent = record['target']
                    auto_sent = record.get('output', "no")
                    raw_sent = record["input"][-1]
                    responses = record[
                        "manual_response"] if args.query == "man_can" else (
                            record["automatic_response"]
                            if args.query == "auto_can" else [])
                    topic_number = record.get('topic_number', None)
                    query_number = record.get('query_number', None)
                    qid = str(topic_number) + "_" + str(
                        query_number) if topic_number != None else str(
                            record["qid"])
                    sequences = record['input']
                    concat_ids = []
                    concat_id_mask = []
                    target_ids = None
                    target_id_mask = None
                    doc_pos = None
                    doc_negs = None
                    if mode == "train" and args.ranking_task:
                        doc_pos = record["doc_pos"]
                        doc_negs = record["doc_negs"]

                    if mode == "train" or args.query in [
                            "no_res", "man_can", "auto_can"
                    ]:
                        if args.model_type == "dpr":
                            concat_ids.append(
                                tokenizer.cls_token_id
                            )  # dpr (on OR-QuAC) uses BERT-style sequence [CLS] q1 [SEP] q2 [SEP] ...
                        for sent in input_sents[:-1]:  # exlude last one
                            if args.model_type != "dpr":
                                concat_ids.append(
                                    tokenizer.cls_token_id
                                )  # RoBERTa-style sequence <s> q1 </s> <s> q2 </s> ...
                            concat_ids.extend(
                                tokenizer.convert_tokens_to_ids(
                                    tokenizer.tokenize(sent)))
                            concat_ids.append(tokenizer.sep_token_id)

                        if args.query in [
                                "man_can", "auto_can"
                        ] and len(responses) >= 2:  # add response
                            if args.model_type != "dpr":
                                concat_ids.append(tokenizer.cls_token_id)
                            concat_ids.extend(
                                tokenizer.convert_tokens_to_ids(["<response>"
                                                                 ]))
                            concat_ids.extend(
                                tokenizer.convert_tokens_to_ids(
                                    tokenizer.tokenize(responses[-2])))
                            concat_ids.append(tokenizer.sep_token_id)
                            sequences.insert(-1, responses[-2])

                        if args.model_type != "dpr":
                            concat_ids.append(tokenizer.cls_token_id)
                        concat_ids.extend(
                            tokenizer.convert_tokens_to_ids(
                                tokenizer.tokenize(input_sents[-1])))
                        concat_ids.append(tokenizer.sep_token_id)

                        # We do not use token type id for BERT (and for RoBERTa, of course)
                        concat_ids, concat_id_mask = pad_input_ids_with_mask(
                            concat_ids, args.max_concat_length)
                        assert len(concat_ids) == args.max_concat_length

                    elif args.query == "target":  # manual

                        concat_ids = tokenizer.encode(
                            target_sent,
                            add_special_tokens=True,
                            max_length=args.max_query_length)
                        concat_ids, concat_id_mask = pad_input_ids_with_mask(
                            concat_ids, args.max_query_length)
                        assert len(concat_ids) == args.max_query_length

                    elif args.query == "output":  # reserved for query rewriter output

                        concat_ids = tokenizer.encode(
                            auto_sent,
                            add_special_tokens=True,
                            max_length=args.max_query_length)
                        concat_ids, concat_id_mask = pad_input_ids_with_mask(
                            concat_ids, args.max_query_length)
                        assert len(concat_ids) == args.max_query_length

                    elif args.query == "raw":

                        concat_ids = tokenizer.encode(
                            raw_sent,
                            add_special_tokens=True,
                            max_length=args.max_query_length)
                        concat_ids, concat_id_mask = pad_input_ids_with_mask(
                            concat_ids, args.max_query_length)
                        assert len(concat_ids) == args.max_query_length

                    else:
                        raise KeyError("Unsupported query type")

                    if mode == "train":
                        target_ids = tokenizer.encode(
                            target_sent,
                            add_special_tokens=True,
                            max_length=args.max_query_length)
                        target_ids, target_id_mask = pad_input_ids_with_mask(
                            target_ids, args.max_query_length)
                        assert len(target_ids) == args.max_query_length

                    self.examples.append(
                        ConvSearchExample(qid, concat_ids, concat_id_mask,
                                          target_ids, target_id_mask, doc_pos,
                                          doc_negs, sequences))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args, mode):
        def collate_fn(batch_dataset: list):
            collated_dict = {
                "qid": [],
                "concat_ids": [],
                "concat_id_mask": [],
            }
            if mode == "train":
                collated_dict.update({"target_ids": [], "target_id_mask": []})
                if args.ranking_task:
                    collated_dict.update({"documents": []})
            else:
                collated_dict.update({"history_utterances": []})
            for example in batch_dataset:
                collated_dict["qid"].append(example.qid)
                collated_dict["concat_ids"].append(example.concat_ids)
                collated_dict["concat_id_mask"].append(example.concat_id_mask)
                if mode == "train":
                    collated_dict["target_ids"].append(example.target_ids)
                    collated_dict["target_id_mask"].append(
                        example.target_id_mask)
                    if args.ranking_task:
                        collated_dict["documents"].append([example.doc_pos] +
                                                          example.doc_negs)
                else:
                    collated_dict["history_utterances"].append(
                        example.raw_sequences)
            should_be_tensor = [
                "concat_ids", "concat_id_mask", "target_ids", "target_id_mask"
            ]
            for key in should_be_tensor:
                if key in collated_dict:
                    collated_dict[key] = torch.tensor(collated_dict[key],
                                                      dtype=torch.long)

            return collated_dict

        return collate_fn


def tokenize_to_file(args, i, num_process, in_path, out_path, line_fn):

    configObj = MSMarcoConfigDict[args.model_type]
    tokenizer = configObj.tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=True,
        cache_dir=None,
    )

    with open(in_path, 'r', encoding='utf-8') if in_path[-2:] != "gz" else gzip.open(in_path, 'rt', encoding='utf8') as in_f,\
            open('{}_split{}'.format(out_path, i), 'wb') as out_f:
        for idx, line in enumerate(in_f):
            if idx % num_process != i:
                continue
            try:
                res = line_fn(args, line, tokenizer)
            except ValueError:
                print("Bad passage.")
            else:
                out_f.write(res)


#                      args, 32,        , collection.tsv, passages,
def multi_file_process(args, num_process, in_path, out_path, line_fn):
    processes = []
    for i in range(num_process):
        p = Process(target=tokenize_to_file,
                    args=(
                        args,
                        i,
                        num_process,
                        in_path,
                        out_path,
                        line_fn,
                    ))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [data]

    world_size = dist.get_world_size()
    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size, )).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size, )).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list
