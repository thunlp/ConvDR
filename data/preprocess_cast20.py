import argparse
from trec_car import read_data
from tqdm import tqdm
import pickle
import os
import json
import copy
from utils.util import NUM_FOLD

topic_range = range(81, 106)
fold_dict = {x: (x - 81) // NUM_FOLD for x in topic_range}


def parse_sim_file(filename):
    """
    Reads the deduplicated documents file and stores the 
    duplicate passage ids into a dictionary
    """

    sim_dict = {}
    lines = open(filename).readlines()
    for line in lines:
        data = line.strip().split(':')
        if len(data[1]) > 0:
            sim_docs = data[-1].split(',')
            for docs in sim_docs:
                sim_dict[docs] = 1

    return sim_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--car_cbor", type=str)
    parser.add_argument("--msmarco_collection", type=str)
    parser.add_argument("--duplicate_file", type=str)
    parser.add_argument("--cast_dir", type=str)

    parser.add_argument("--out_data_dir", type=str)
    parser.add_argument("--out_collection_dir", type=str)
    args = parser.parse_args()

    # INPUT
    sim_file = args.duplicate_file
    cast_topics_automatic_file = os.path.join(
        args.cast_dir, "2020_automatic_evaluation_topics_v1.0.json")
    cast_topics_manual_file = os.path.join(
        args.cast_dir, "2020_manual_evaluation_topics_v1.0.json")
    cast_qrels_file = os.path.join(args.cast_dir, "2020qrels.txt")

    # OUTPUT
    out_topics_file = os.path.join(args.out_data_dir, "eval_topics.jsonl")
    out_raw_queries_file = os.path.join(args.out_data_dir, "queries.raw.tsv")
    out_manual_queries_file = os.path.join(args.out_data_dir,
                                           "queries.manual.tsv")
    out_qrels_file = os.path.join(args.out_data_dir, "qrels.tsv")
    car_id_to_idx_file = os.path.join(args.out_collection_dir,
                                      "car_id_to_idx.pickle")
    car_idx_to_id_file = os.path.join(args.out_collection_dir,
                                      "car_idx_to_id.pickle")
    out_collection_file = os.path.join(args.out_collection_dir,
                                       "collection.tsv")

    # 1. Combine TREC-CAR & MS MARCO, remove duplicate passages, assign new ids
    car_id_to_idx = {}
    car_idx_to_id = []
    collection = ["xx"] * 4000_0000
    if os.path.exists(out_collection_file) and os.path.exists(
            car_id_to_idx_file) and os.path.exists(car_idx_to_id_file):
        print("Preprocessed collection found. Loading car_id_to_idx...")
        with open(car_id_to_idx_file, "rb") as f:
            car_id_to_idx = pickle.load(f)
        print("Loading processed document collection...")
        with open(out_collection_file, "r") as f:
            for line in f:
                try:
                    line = line.strip()
                    obj = line.split("\t")
                    pid = obj[0]
                    text = obj[1]
                    pid = int(pid)
                    collection[pid] = text
                except IndexError:
                    print(line)
    else:
        sim_dict = parse_sim_file(sim_file)
        car_base_id = 10000000
        i = 0
        with open(car_idx_to_id_file, "w") as f:
            print("Processing TREC-CAR...")
            for para in tqdm(
                    read_data.iter_paragraphs(open(args.car_cbor, 'rb'))):
                car_id = "CAR_" + para.para_id
                text = para.get_text()
                text = text.replace("\t", " ").replace("\n",
                                                       " ").replace("\r", " ")
                idx = car_base_id + i
                car_id_to_idx[
                    car_id] = idx  # e.g. CAR_76a4a716d4b1b01995c6663ee16e94b4ca35fdd3 -> 10000044
                collection[idx] = text
                car_idx_to_id.append(car_id)
                f.write("{}\t{}\n".format(idx, text))
                i += 1
            print("Processing MS MARCO...")
            removed = 0
            with open(args.msmarco_collection, "r") as m:
                for line in tqdm(m):
                    marco_id, text = line.strip().split("\t")
                    if ("MARCO_" + marco_id) in sim_dict:
                        removed += 1
                        continue
                    collection[int(marco_id)] = text
                    f.write("{}\t{}\n".format(marco_id, text))
            print("Removed " + str(removed) + " passages")
        print("Dumping id mappings...")
        with open(car_id_to_idx_file, "wb") as f:
            pickle.dump(car_id_to_idx, f)
        with open(car_idx_to_id_file, "wb") as f:
            pickle.dump(car_idx_to_id, f)

    # 2. Process queries
    print("Processing CAsT utterances...")

    def get_text_by_raw_id(raw_id):
        new_id = None
        if raw_id.startswith("MARCO_"):
            new_id = int(raw_id[6:])
        elif raw_id.startswith("CAR_"):
            new_id = car_id_to_idx[raw_id]
        else:
            raise ValueError("Invalid document id")
        text = collection[new_id]
        if text == "xx":
            raise ValueError("Unknown document")
        return text

    with open(cast_topics_automatic_file, "r") as f:
        auto_raw = json.load(f)
    with open(cast_topics_manual_file, "r") as f:
        manual_raw = json.load(f)
    out_topics = open(out_topics_file, "w")
    out_topics_fold = open(out_topics_file + ".0", "w")
    out_raw_queries = open(out_raw_queries_file, "w")
    out_manual_queries = open(out_manual_queries_file, "w")
    cur_fold = 0
    for auto_topic, manual_topic in zip(auto_raw, manual_raw):
        topic_number = auto_topic["number"]
        assert topic_number == manual_topic["number"]
        auto_turns = auto_topic["turn"]
        manual_turns = manual_topic["turn"]
        assert len(auto_turns) == len(manual_turns)
        inputs = []
        manual_responses = []
        auto_responses = []
        manual_res_ids = []
        auto_res_ids = []
        for auto_turn, manual_turn in zip(auto_turns, manual_turns):
            query_number = auto_turn["number"]

            raw = auto_turn["raw_utterance"]
            inputs.append(raw)
            target = manual_turn["manual_rewritten_utterance"]

            manual_res_ids.append(manual_turn["manual_canonical_result_id"])
            response = get_text_by_raw_id(
                manual_turn["manual_canonical_result_id"])
            manual_responses.append(response)

            auto_res_ids.append(auto_turn["automatic_canonical_result_id"])
            response = get_text_by_raw_id(
                auto_turn["automatic_canonical_result_id"])
            auto_responses.append(response)

            output_dict = {
                "topic_number": topic_number,
                "query_number": query_number,
                "input": copy.deepcopy(inputs),
                "automatic_response_id": copy.deepcopy(auto_res_ids),
                "automatic_response": copy.deepcopy(auto_responses),
                "manual_response_id": copy.deepcopy(manual_res_ids),
                "manual_response": copy.deepcopy(manual_responses),
                "target": target
            }

            dumped_str = json.dumps(output_dict) + "\n"
            out_topics.write(dumped_str)
            if fold_dict[topic_number] != cur_fold:
                out_topics_fold.close()
                out_topics_fold = open(
                    out_topics_file + "." + str(fold_dict[topic_number]), "w")
                cur_fold = fold_dict[topic_number]
            out_topics_fold.write(dumped_str)

            out_raw_queries.write(
                str(topic_number) + "_" + str(query_number) + "\t" + raw +
                "\n")
            out_manual_queries.write(
                str(topic_number) + "_" + str(query_number) + "\t" + target +
                "\n")

    # 3. Process and convert qrels
    print("Processing qrels...")
    with open(cast_qrels_file, "r") as oq, open(out_qrels_file, "w") as nq:
        for line in oq:
            qid, _, pid, rel = line.strip().split()
            if pid.startswith("CAR_"):
                assert car_id_to_idx[pid] != -1
                pid = car_id_to_idx[pid]
            elif pid.startswith("MARCO_"):
                pid = int(pid[6:])
            else:
                continue
            nq.write(qid + "\t0\t" + str(pid) + "\t" + rel + "\n")

    print("End")