import argparse
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--convdr_trec", type=str)
    parser.add_argument("--doc_idx_to_id", type=str)
    parser.add_argument("--out_trec", type=str)
    args = parser.parse_args()

    print("Loading doc_idx_to_id...")
    with open(args.doc_idx_to_id, "rb") as f:
        doc_idx_to_id = pickle.load(f)

    with open(args.convdr_trec, "r") as f, open(args.out_trec, "w") as g:
        for line in f:
            qid, _, pid, rank, score, label = line.strip().split()
            g.write("{} Q0 {} {} {} {}\n".format(qid, doc_idx_to_id[int(pid)], rank, score, label))