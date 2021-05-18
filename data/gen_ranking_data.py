import argparse
import copy
import json
import csv
import os
import pickle
import random
from utils.util import load_collection

# For CAsT cross-validation, I manually split the data into five folds to ensure the balance of judged queries in each fold.
qid_to_fold_test = {
    "31_1": 0,
    "31_2": 0,
    "31_3": 0,
    "31_4": 0,
    "31_5": 0,
    "31_6": 0,
    "31_7": 0,
    "31_8": 0,
    "31_9": 0,
    "32_1": 0,
    "32_2": 0,
    "32_3": 0,
    "32_4": 0,
    "32_5": 0,
    "32_6": 0,
    "32_7": 0,
    "32_8": 0,
    "32_9": 0,
    "32_10": 0,
    "32_11": 0,
    "33_1": 0,
    "33_2": 0,
    "33_3": 0,
    "33_4": 0,
    "33_5": 0,
    "33_6": 0,
    "33_7": 0,
    "33_8": 0,
    "33_9": 0,
    "33_10": 0,
    "34_1": 0,
    "34_2": 0,
    "34_3": 0,
    "34_4": 0,
    "34_5": 0,
    "34_6": 0,
    "34_7": 0,
    "34_8": 0,
    "34_9": 0,
    "35_1": 0,
    "35_2": 0,
    "35_3": 0,
    "35_4": 0,
    "35_5": 0,
    "35_6": 0,
    "35_7": 0,
    "35_8": 0,
    "35_9": 0,
    "36_1": 0,
    "36_2": 0,
    "36_3": 0,
    "36_4": 0,
    "36_5": 0,
    "36_6": 0,
    "36_7": 0,
    "36_8": 0,
    "36_9": 0,
    "36_10": 0,
    "36_11": 0,
    "37_1": 1,
    "37_2": 1,
    "37_3": 1,
    "37_4": 1,
    "37_5": 1,
    "37_6": 1,
    "37_7": 1,
    "37_8": 1,
    "37_9": 1,
    "37_10": 1,
    "37_11": 1,
    "37_12": 1,
    "38_1": 0,
    "38_2": 0,
    "38_3": 0,
    "38_4": 0,
    "38_5": 0,
    "38_6": 0,
    "38_7": 0,
    "38_8": 0,
    "39_1": 0,
    "39_2": 0,
    "39_3": 0,
    "39_4": 0,
    "39_5": 0,
    "39_6": 0,
    "39_7": 0,
    "39_8": 0,
    "39_9": 0,
    "40_1": 1,
    "40_2": 1,
    "40_3": 1,
    "40_4": 1,
    "40_5": 1,
    "40_6": 1,
    "40_7": 1,
    "40_8": 1,
    "40_9": 1,
    "40_10": 1,
    "41_1": 1,
    "41_2": 1,
    "41_3": 1,
    "41_4": 1,
    "41_5": 1,
    "41_6": 1,
    "41_7": 1,
    "41_8": 1,
    "41_9": 1,
    "42_1": 1,
    "42_2": 1,
    "42_3": 1,
    "42_4": 1,
    "42_5": 1,
    "42_6": 1,
    "42_7": 1,
    "42_8": 1,
    "43_1": 1,
    "43_2": 1,
    "43_3": 1,
    "43_4": 1,
    "43_5": 1,
    "43_6": 1,
    "43_7": 1,
    "43_8": 1,
    "44_1": 1,
    "44_2": 1,
    "44_3": 1,
    "44_4": 1,
    "44_5": 1,
    "44_6": 1,
    "44_7": 1,
    "44_8": 1,
    "45_1": 1,
    "45_2": 1,
    "45_3": 1,
    "45_4": 1,
    "45_5": 1,
    "45_6": 1,
    "45_7": 1,
    "45_8": 1,
    "46_1": 1,
    "46_2": 1,
    "46_3": 1,
    "46_4": 1,
    "46_5": 1,
    "46_6": 1,
    "46_7": 1,
    "46_8": 1,
    "46_9": 1,
    "46_10": 1,
    "47_1": 1,
    "47_2": 1,
    "47_3": 1,
    "47_4": 1,
    "47_5": 1,
    "47_6": 1,
    "47_7": 1,
    "48_1": 1,
    "48_2": 1,
    "48_3": 1,
    "48_4": 1,
    "48_5": 1,
    "48_6": 1,
    "48_7": 1,
    "48_8": 1,
    "48_9": 1,
    "49_1": 1,
    "49_2": 1,
    "49_3": 1,
    "49_4": 1,
    "49_5": 1,
    "49_6": 1,
    "49_7": 1,
    "49_8": 1,
    "49_9": 1,
    "49_10": 1,
    "50_1": 1,
    "50_2": 1,
    "50_3": 1,
    "50_4": 1,
    "50_5": 1,
    "50_6": 1,
    "50_7": 1,
    "50_8": 1,
    "50_9": 1,
    "50_10": 1,
    "51_1": 2,
    "51_2": 2,
    "51_3": 2,
    "51_4": 2,
    "51_5": 2,
    "51_6": 2,
    "51_7": 2,
    "51_8": 2,
    "51_9": 2,
    "51_10": 2,
    "52_1": 2,
    "52_2": 2,
    "52_3": 2,
    "52_4": 2,
    "52_5": 2,
    "52_6": 2,
    "52_7": 2,
    "52_8": 2,
    "52_9": 2,
    "52_10": 2,
    "53_1": 2,
    "53_2": 2,
    "53_3": 2,
    "53_4": 2,
    "53_5": 2,
    "53_6": 2,
    "53_7": 2,
    "53_8": 2,
    "53_9": 2,
    "54_1": 2,
    "54_2": 2,
    "54_3": 2,
    "54_4": 2,
    "54_5": 2,
    "54_6": 2,
    "54_7": 2,
    "54_8": 2,
    "54_9": 2,
    "55_1": 2,
    "55_2": 2,
    "55_3": 2,
    "55_4": 2,
    "55_5": 2,
    "55_6": 2,
    "55_7": 2,
    "55_8": 2,
    "55_9": 2,
    "55_10": 2,
    "56_1": 2,
    "56_2": 2,
    "56_3": 2,
    "56_4": 2,
    "56_5": 2,
    "56_6": 2,
    "56_7": 2,
    "56_8": 2,
    "57_1": 2,
    "57_2": 2,
    "57_3": 2,
    "57_4": 2,
    "57_5": 2,
    "57_6": 2,
    "57_7": 2,
    "57_8": 2,
    "57_9": 2,
    "57_10": 2,
    "58_1": 2,
    "58_2": 2,
    "58_3": 2,
    "58_4": 2,
    "58_5": 2,
    "58_6": 2,
    "58_7": 2,
    "58_8": 2,
    "59_1": 2,
    "59_2": 2,
    "59_3": 2,
    "59_4": 2,
    "59_5": 2,
    "59_6": 2,
    "59_7": 2,
    "59_8": 2,
    "60_1": 2,
    "60_2": 2,
    "60_3": 2,
    "60_4": 2,
    "60_5": 2,
    "60_6": 2,
    "60_7": 2,
    "61_1": 4,
    "61_2": 4,
    "61_3": 4,
    "61_4": 4,
    "61_5": 4,
    "61_6": 4,
    "61_7": 4,
    "61_8": 4,
    "61_9": 4,
    "62_1": 3,
    "62_2": 3,
    "62_3": 3,
    "62_4": 3,
    "62_5": 3,
    "62_6": 3,
    "62_7": 3,
    "62_8": 3,
    "62_9": 3,
    "62_10": 3,
    "62_11": 3,
    "63_1": 3,
    "63_2": 3,
    "63_3": 3,
    "63_4": 3,
    "63_5": 3,
    "63_6": 3,
    "63_7": 3,
    "63_8": 3,
    "63_9": 3,
    "63_10": 3,
    "64_1": 3,
    "64_2": 3,
    "64_3": 3,
    "64_4": 3,
    "64_5": 3,
    "64_6": 3,
    "64_7": 3,
    "64_8": 3,
    "64_9": 3,
    "64_10": 3,
    "64_11": 3,
    "65_1": 3,
    "65_2": 3,
    "65_3": 3,
    "65_4": 3,
    "65_5": 3,
    "65_6": 3,
    "65_7": 3,
    "65_8": 3,
    "65_9": 3,
    "65_10": 3,
    "66_1": 3,
    "66_2": 3,
    "66_3": 3,
    "66_4": 3,
    "66_5": 3,
    "66_6": 3,
    "66_7": 3,
    "66_8": 3,
    "66_9": 3,
    "67_1": 3,
    "67_2": 3,
    "67_3": 3,
    "67_4": 3,
    "67_5": 3,
    "67_6": 3,
    "67_7": 3,
    "67_8": 3,
    "67_9": 3,
    "67_10": 3,
    "67_11": 3,
    "68_1": 3,
    "68_2": 3,
    "68_3": 3,
    "68_4": 3,
    "68_5": 3,
    "68_6": 3,
    "68_7": 3,
    "68_8": 3,
    "68_9": 3,
    "68_10": 3,
    "68_11": 3,
    "69_1": 3,
    "69_2": 3,
    "69_3": 3,
    "69_4": 3,
    "69_5": 3,
    "69_6": 3,
    "69_7": 3,
    "69_8": 3,
    "69_9": 3,
    "69_10": 3,
    "70_1": 3,
    "70_2": 3,
    "70_3": 3,
    "70_4": 3,
    "70_5": 3,
    "70_6": 3,
    "70_7": 3,
    "70_8": 3,
    "70_9": 3,
    "70_10": 3,
    "71_1": 4,
    "71_2": 4,
    "71_3": 4,
    "71_4": 4,
    "71_5": 4,
    "71_6": 4,
    "71_7": 4,
    "71_8": 4,
    "71_9": 4,
    "71_10": 4,
    "71_11": 4,
    "71_12": 4,
    "72_1": 4,
    "72_2": 4,
    "72_3": 4,
    "72_4": 4,
    "72_5": 4,
    "72_6": 4,
    "72_7": 4,
    "72_8": 4,
    "72_9": 4,
    "72_10": 4,
    "73_1": 4,
    "73_2": 4,
    "73_3": 4,
    "73_4": 4,
    "73_5": 4,
    "73_6": 4,
    "73_7": 4,
    "73_8": 4,
    "73_9": 4,
    "73_10": 4,
    "74_1": 4,
    "74_2": 4,
    "74_3": 4,
    "74_4": 4,
    "74_5": 4,
    "74_6": 4,
    "74_7": 4,
    "74_8": 4,
    "74_9": 4,
    "74_10": 4,
    "74_11": 4,
    "74_12": 4,
    "75_1": 4,
    "75_2": 4,
    "75_3": 4,
    "75_4": 4,
    "75_5": 4,
    "75_6": 4,
    "75_7": 4,
    "75_8": 4,
    "75_9": 4,
    "75_10": 4,
    "76_1": 4,
    "76_2": 4,
    "76_3": 4,
    "76_4": 4,
    "76_5": 4,
    "76_6": 4,
    "76_7": 4,
    "76_8": 4,
    "76_9": 4,
    "76_10": 4,
    "77_1": 4,
    "77_2": 4,
    "77_3": 4,
    "77_4": 4,
    "77_5": 4,
    "77_6": 4,
    "77_7": 4,
    "77_8": 4,
    "77_9": 4,
    "77_10": 4,
    "78_1": 4,
    "78_2": 4,
    "78_3": 4,
    "78_4": 4,
    "78_5": 4,
    "78_6": 4,
    "78_7": 4,
    "78_8": 4,
    "78_9": 4,
    "78_10": 4,
    "79_1": 4,
    "79_2": 4,
    "79_3": 4,
    "79_4": 4,
    "79_5": 4,
    "79_6": 4,
    "79_7": 4,
    "79_8": 4,
    "79_9": 4,
    "80_1": 4,
    "80_2": 4,
    "80_3": 4,
    "80_4": 4,
    "80_5": 4,
    "80_6": 4,
    "80_7": 4,
    "80_8": 4,
    "80_9": 4,
    "80_10": 4
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str)
    parser.add_argument("--run", type=str)
    parser.add_argument("--qrels", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--collection", type=str)
    parser.add_argument(
        "--cast",
        action="store_true",
        help=
        "Set this flag if you are working on the TREC CAsT dataset to enable cross-validation"
    )
    parser.add_argument("--num_negs", type=int, default=9)
    args = parser.parse_args()

    print("Selecting negative documents...")
    query_positive_id = {}
    query_negative_id = {}
    with open(args.qrels, 'r', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, _, docid, rel] in tsvreader:
            docid = int(docid)
            rel = int(rel)
            if rel > 0:
                if topicid not in query_positive_id:
                    query_positive_id[topicid] = {}
                    query_positive_id[topicid][docid] = rel
                else:
                    query_positive_id[topicid][docid] = rel
            else:
                if topicid not in query_negative_id:
                    query_negative_id[topicid] = []
                    query_negative_id[topicid].append(docid)
                else:
                    query_negative_id[topicid].append(docid)

    with open(args.train, "r") as f:
        cqr = {}
        for line in f:
            obj = json.loads(line)
            qid = (
                obj["topic_number"] + "_" +
                obj["query_number"]) if "topic_number" in obj else obj["qid"]
            cqr[qid] = obj

    # find negatives documents first in the retrieved & annotated negatives
    negatives = {}
    with open(args.run, "r") as f:
        for line in f:
            qid, _, pid, _, _, _ = line.strip().split()
            pid = int(pid)
            positive_ids = query_positive_id[
                qid] if qid in query_positive_id else {}
            if positive_ids != {} and pid not in positive_ids:
                if qid in query_negative_id and pid in query_negative_id[qid]:
                    if qid not in negatives:
                        negatives[qid] = [pid]
                    else:
                        negatives[qid].append(pid)

    # not annotated negatives may be false negatives
    with open(args.run, "r") as f:
        for line in f:
            qid, _, pid, _, _, _ = line.strip().split()
            pid = int(pid)
            if qid in negatives and len(negatives[qid]) >= 20:
                continue
            positive_ids = query_positive_id[
                qid] if qid in query_positive_id else {}
            if positive_ids != {} and pid not in positive_ids:
                if qid not in negatives:
                    negatives[qid] = [pid]
                else:
                    negatives[qid].append(pid)
    print(len(negatives))

    print("Loading document collection...")
    all_passages = load_collection(args.collection)

    print("Writing to file...")
    items = copy.deepcopy(list(negatives.items()))
    random.shuffle(items)
    file_id = 0
    fs = None
    if args.cast:
        fs = [open(args.output + "." + str(x), "w") for x in range(5)]
    f = open(args.output, "w")
    for qid, negs in items:
        if qid not in query_positive_id:
            continue
        positives = query_positive_id[qid]
        max_positive = -1
        max_rel = -1
        for pos_id, rel in positives.items():
            if rel > max_rel:
                max_rel = rel
                max_positive = pos_id
        sampled_negs = random.sample(
            negs, args.num_negs) if len(negs) > args.num_negs else negs
        cqr_record = cqr[qid]
        target_obj = copy.deepcopy(cqr_record)
        target_obj.update({
            "doc_pos": all_passages[max_positive],
            "doc_pos_id": max_positive,
            "doc_negs": [all_passages[x] for x in sampled_negs],
            "doc_negs_id": [x for x in sampled_negs]
        })
        to_write = json.dumps(target_obj) + "\n"
        if args.cast:
            fs[qid_to_fold_test[qid]].write(to_write)
        f.write(to_write)

    if args.cast:
        for x in range(5):
            fs[x].close()
    f.close()