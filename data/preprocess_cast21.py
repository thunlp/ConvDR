import argparse
from trec_car import read_data
from tqdm import tqdm
import pickle
import os
import json
import copy
from utils.util import NUM_FOLD

topic_range = range(106, 132)
fold_dict = {x: (x - 106) // NUM_FOLD for x in topic_range}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kilt", type=str)
    parser.add_argument("--msmarco", type=str)
    parser.add_argument("--wapo", type=str)
    # parser.add_argument("--duplicate_file", type=str)
    parser.add_argument("--cast_dir", type=str)

    parser.add_argument("--out_data_dir", type=str)
    parser.add_argument("--out_collection_dir", type=str)
    args = parser.parse_args()

    # INPUT
    cast_topics_manual_file = os.path.join(
        args.cast_dir, "2021_manual_evaluation_topics_v1.0.json")

    # OUTPUT
    out_topics_file = os.path.join(args.out_data_dir, "eval_topics.jsonl")
    out_raw_queries_file = os.path.join(args.out_data_dir, "queries.raw.tsv")
    out_manual_queries_file = os.path.join(args.out_data_dir,
                                           "queries.manual.tsv")
    doc_id_to_idx_file = os.path.join(args.out_collection_dir,
                                      "doc_id_to_idx.pickle")
    doc_idx_to_id_file = os.path.join(args.out_collection_dir,
                                      "doc_idx_to_id.pickle")
    out_collection_file = os.path.join(args.out_collection_dir,
                                       "collection.tsv")
    out_psuedo_qrels_file = os.path.join(args.out_data_dir, "qrels.tsv")

    # 1. Combine KILT, MS MARCO * WaPo, remove duplicate passages, assign new ids
    doc_id_to_idx = {}
    doc_idx_to_id = []
    collection = ["xx"] * 8000_0000
    if os.path.exists(out_collection_file) and os.path.exists(
            doc_id_to_idx_file) and os.path.exists(doc_idx_to_id_file):
        print("Preprocessed collection found. Loading car_id_to_idx...")
        with open(doc_id_to_idx_file, "rb") as f:
            doc_id_to_idx = pickle.load(f)
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
                except ValueError:
                    print(line)
    else:
        with open(out_collection_file, "w") as f:
            print("Processing KILT...")
            with open(args.kilt, "r") as k:
                all_content = k.read()
            pidx = parse_documents(all_content, doc_id_to_idx, doc_idx_to_id, collection, f, pidx=0)
            print("Processing MS MARCO...")
            with open(args.msmarco, "r") as m:
                all_content = m.read()
            pidx = parse_documents(all_content, doc_id_to_idx, doc_idx_to_id, collection, f, pidx)
            print("Processing WaPo...")
            with open(args.wapo, "r") as w:
                all_content = w.read()
            pidx = parse_documents(all_content, doc_id_to_idx, doc_idx_to_id, collection, f, pidx)
            print("Total document num: {}".format(pidx))
        print("Dumping id mappings...")
        with open(doc_id_to_idx_file, "wb") as f:
            pickle.dump(doc_id_to_idx, f)
        with open(doc_idx_to_id_file, "wb") as f:
            pickle.dump(doc_idx_to_id, f)

    # 2. Process queries
    print("Processing CAsT utterances...")

    def get_text_by_raw_id(raw_id):
        new_id = doc_id_to_idx[raw_id]
        text = collection[new_id]
        if text == "xx":
            raise ValueError("Unknown document")
        return text, new_id

    with open(cast_topics_manual_file, "r") as f:
        manual_raw = json.load(f)
    out_topics = open(out_topics_file, "w")
    out_topics_fold = open(out_topics_file + ".0", "w")
    out_raw_queries = open(out_raw_queries_file, "w")
    out_manual_queries = open(out_manual_queries_file, "w")
    out_psuedo_qrels = open(out_psuedo_qrels_file, "w")
    cur_fold = 0
    for manual_topic in manual_raw:
        topic_number = manual_topic["number"]
        manual_turns = manual_topic["turn"]
        inputs = []
        manual_responses = []
        auto_responses = []
        manual_res_ids = []
        auto_res_ids = []
        for manual_turn in manual_turns:
            query_number = manual_turn["number"]

            raw = manual_turn["raw_utterance"]
            inputs.append(raw)
            target = manual_turn["manual_rewritten_utterance"]

            res_id = manual_turn["canonical_result_id"] + "-" + str(manual_turn["passage_id"])
            manual_res_ids.append(res_id)
            response, new_id = get_text_by_raw_id(
                res_id)
            manual_responses.append(response)

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

            out_psuedo_qrels.write(str(topic_number) + "_" + str(query_number)+ "\t0\t" + str(new_id) + "\t1\n")

            out_raw_queries.write(
                str(topic_number) + "_" + str(query_number) + "\t" + raw +
                "\n")
            out_manual_queries.write(
                str(topic_number) + "_" + str(query_number) + "\t" + target +
                "\n")

    print("End")

def parse_documents(all_content, doc_id_to_idx, doc_idx_to_id, collection, f, pidx=0):
    docid = None
    title = None
    passage = None
    pid = None
    char_id = 0
    last_char_id = 0
    with tqdm(total=len(all_content)) as pbar:
        while char_id < len(all_content):
            last_char_id = char_id
            if all_content[char_id] == "<":
                char_id += 1
                if all_content[char_id] not in ["D", "T", "p"]:
                    continue
                if all_content[char_id:char_id+len("DOCNO>")] == "DOCNO>":
                    char_id += len("DOCNO>")
                    end_pos = all_content.find("</DOCNO>", char_id)
                    assert end_pos != -1
                    docid = all_content[char_id:end_pos]
                    char_id = end_pos + len("</DOCNO>")
                elif all_content[char_id:char_id+len("TITLE>")] == "TITLE>":
                    char_id += len("TITLE>")
                    end_pos = all_content.find("</TITLE>", char_id)
                    assert end_pos != -1
                    title = all_content[char_id:end_pos]
                    char_id = end_pos + len("</TITLE>")
                elif all_content[char_id:char_id+len("passage id=")] == "passage id=":
                    char_id += len("passage id=")
                    end_pos = all_content.find(">", char_id)
                    assert end_pos != -1
                    pid = str(int(all_content[char_id:end_pos]))
                    char_id = end_pos + 1
                    end_pos = all_content.find("</passage>", char_id)
                    assert end_pos != -1
                    passage = all_content[char_id:end_pos].strip().replace("\n", " ").replace("\t", " ").strip()
                    text = title + " " + passage
                    char_id = end_pos + len("</passage>")
                    doc_id_to_idx[docid + "-" + pid] = pidx
                    doc_idx_to_id.append(docid + "-" + pid)
                    collection[pidx] = text
                    f.write("{}\t{}\n".format(pidx, text))
                    pidx += 1
            else:
                char_id += 1
            pbar.update(char_id - last_char_id)
    
    return pidx


if __name__ == "__main__":
    main()