import json, random
import numpy as np
from collections import defaultdict

def read_jsonl(path):
    out = []
    with open(path, "r") as f:
        for line in f:
            out.append(json.loads(line))
    return out

def main():
    random.seed(42)

    ids = np.load("data/semantic_ids/semantic_ids_train2017.npy")
    meta = read_jsonl("data/clip_emb/meta_train2017.jsonl")

    q0 = ids[:,0].astype(int)
    buckets = defaultdict(list)
    for i, c in enumerate(q0):
        buckets[c].append(i)

    code = sorted(buckets.keys())[0]
    print("Choose Q0 code =", code, "bucket size =", len(buckets[code]))

    for i in random.sample(buckets[code], k=min(8, len(buckets[code]))):
        m = meta[i]
        print("-"*60)
        print("file:", m["file_name"])
        print("caption:", m.get("caption",""))

if __name__ == "__main__":
    main()
