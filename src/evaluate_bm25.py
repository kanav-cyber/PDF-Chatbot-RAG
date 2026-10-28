from src.bm25_retriever import load_bm25,retrieve_bm25
import json
from pathlib import Path

eval_path = Path("data/eval_queries.json")


if not eval_path.exists():
    raise ValueError(f"There in no valid path persent {eval_path}")


with open(eval_path,"r",encoding="utf-8") as f:
    eval_queries = json.load(f)

if not eval_queries:
    raise ValueError("There are no evaluation questions present")

bm25,chunk = load_bm25()

for i in eval_queries:
    results = retrieve_bm25(
        i["question"],
        bm25,
        chunk,
        top_k=5
    )

    retrieved_ids = [result["chunk_id"]for result in results]
    relevant_ids = i["relevant_chunk_ids"]

    print(f"\nQuestion: {i['question']}")
    print(f"Relevant IDs: {relevant_ids}")
    print(f"Retrieved IDs: {retrieved_ids}")
