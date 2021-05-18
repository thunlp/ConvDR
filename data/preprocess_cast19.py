import argparse
from trec_car import read_data
from tqdm import tqdm
import pickle
import os
import json
import copy
from utils.util import NUM_FOLD


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
    cast_topics_raw_file = os.path.join(args.cast_dir,
                                        "evaluation_topics_v1.0.json")
    cast_topics_manual_file = os.path.join(
        args.cast_dir, "evaluation_topics_annotated_resolved_v1.0.tsv")
    cast_qrels_file = os.path.join(args.cast_dir, "2019qrels.txt")

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
    if os.path.exists(out_collection_file) and os.path.exists(
            car_id_to_idx_file) and os.path.exists(car_idx_to_id_file):
        print("Preprocessed collection found. Loading car_id_to_idx...")
        with open(car_id_to_idx_file, "rb") as f:
            car_id_to_idx = pickle.load(f)
    else:
        sim_dict = parse_sim_file(sim_file)
        car_base_id = 10000000
        i = 0
        with open(out_collection_file, "w") as f:
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
                    f.write("{}\t{}\n".format(marco_id, text))
            print("Removed " + str(removed) + " passages")
        print("Dumping id mappings...")
        with open(car_id_to_idx_file, "wb") as f:
            pickle.dump(car_id_to_idx, f)
        with open(car_idx_to_id_file, "wb") as f:
            pickle.dump(car_idx_to_id, f)

    # 2. Process queries
    print("Processing CAsT utterances...")
    with open(cast_topics_raw_file, "r") as fin:
        raw_data = json.load(fin)

    with open(cast_topics_manual_file, "r") as fin:
        annonated_lines = fin.readlines()

    out_raw_queries = open(out_raw_queries_file, "w")
    out_manual_queries = open(out_manual_queries_file, "w")

    all_annonated = {}
    for line in annonated_lines:
        splitted = line.split('\t')
        out_manual_queries.write(line)
        topic_query = splitted[0]
        query = splitted[1].strip()
        topic_id = topic_query.split('_')[0]
        query_id = topic_query.split('_')[1]
        if topic_id not in all_annonated:
            all_annonated[topic_id] = {}
        all_annonated[topic_id][query_id] = query
    out_manual_queries.close()

    topic_number_dict = {}
    data = []
    for group in raw_data:
        topic_number, description, turn, title = str(
            group['number']), group.get('description',
                                        ''), group['turn'], group.get(
                                            'title', '')
        queries = []
        for query in turn:
            query_number, raw_utterance = str(
                query['number']), query['raw_utterance']
            queries.append(raw_utterance)
            record = {}
            record['topic_number'] = topic_number
            record['query_number'] = query_number
            record['description'] = description
            record['title'] = title
            record['input'] = copy.deepcopy(queries)
            record['target'] = all_annonated[topic_number][query_number]
            out_raw_queries.write("{}_{}\t{}\n".format(topic_number,
                                                       query_number,
                                                       raw_utterance))
            if not topic_number in topic_number_dict:
                topic_number_dict[topic_number] = len(topic_number_dict)
            data.append(record)
    out_raw_queries.close()

    with open(out_topics_file, 'w') as fout:
        for item in data:
            json_str = json.dumps(item)
            fout.write(json_str + '\n')

    # Split eval data into K-fold
    topic_per_fold = len(topic_number_dict) // NUM_FOLD
    for i in range(NUM_FOLD):
        with open(out_topics_file + "." + str(i), 'w') as fout:
            for item in data:
                idx = topic_number_dict[item['topic_number']]
                if idx // topic_per_fold == i:
                    json_str = json.dumps(item)
                    fout.write(json_str + '\n')

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