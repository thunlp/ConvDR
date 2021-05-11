import argparse
import json

# TODO: Finish this file
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str)  # train.jsonl
    parser.add_argument("--run", type=str)
    parser.add_argument("--qrels", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    kd_train = []
    with open(args.train, "r") as f:
        for line in f:
            kd_train.append(json.loads(line))

    run_file = []