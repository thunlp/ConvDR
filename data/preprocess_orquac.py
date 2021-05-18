import argparse
import os
import json
from tqdm import tqdm
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orquac_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # all_blocks.txt -> collection.tsv
    print("Processing all_blocks.txt...")
    all_blocks = os.path.join(args.orquac_dir, "all_blocks.txt")
    collection = os.path.join(args.output_dir, "collection.jsonl")
    passage_id_to_idx = {}
    with open(all_blocks, "r") as f, open(collection, "w") as g:
        idx = 0
        for line in tqdm(f):
            obj = json.loads(line)
            passage = obj['text'].replace('\n', ' ').replace('\t', ' ')
            pid = obj['id']
            g.write(
                json.dumps({
                    "id": idx,
                    "title": obj["title"],
                    "text": passage
                }) + "\n")
            passage_id_to_idx[pid] = idx
            idx += 1

    # train/dev/test.txt -> queries.train/dev/test.manual/raw.tsv
    targets = ['train', 'dev', 'test']
    qids_set = {"train": set(), "dev": set(), "test": set()}
    idx = 0
    for target in targets:
        print(f"Processing {target}.txt")
        train = os.path.join(args.orquac_dir, "preprocessed", f"{target}.txt")
        queries_manual = os.path.join(args.output_dir,
                                      f"queries.{target}.manual.tsv")
        queries_raw = os.path.join(args.output_dir,
                                   f"queries.{target}.raw.tsv")
        cqr = os.path.join(args.output_dir, "{}.jsonl".format(target))
        with open(train, "r") as f, open(queries_manual, "w") as g, open(
                cqr, "w") as h, open(queries_raw, "w") as i:
            responses = []
            last_dialog_id = None
            for line in f:
                obj = json.loads(line)
                qid, query = obj['qid'], obj['rewrite']
                raw_query = obj["question"]
                dialog_id = qid[:qid.rfind('#')]
                if dialog_id != last_dialog_id:
                    last_dialog_id = dialog_id
                    responses.clear()
                cur_response = obj["answer"]["text"]
                responses.append(cur_response)
                input_sents = []
                for his in obj["history"]:
                    input_sents.append(his["question"])
                input_sents.append(obj["question"])
                h.write(
                    json.dumps({
                        "qid": qid,
                        "input": input_sents,
                        "target": query,
                        "manual_response": responses
                    }) + "\n")
                g.write(f"{qid}\t{query}\n")
                i.write(f"{qid}\t{raw_query}\n")
                qids_set[target].add(qid)
                idx += 1

    # qrels.txt -> qrels.train.tsv
    print("Processing qrels.txt...")
    qrels = os.path.join(args.orquac_dir, "qrels.txt")
    with open(qrels, "r") as f:
        qrels_dict = json.load(f)
    target_qrels_file = open(os.path.join(args.output_dir, "qrels.tsv"), "w")
    for qid, v in qrels_dict.items():
        for pid in v.keys():
            passage_idx = passage_id_to_idx[pid]
            target_qrels_file.write(f"{qid}\t0\t{passage_idx}\t1\n")
    target_qrels_file.close()

    print("End")