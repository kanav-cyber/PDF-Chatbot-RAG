from pathlib import Path
import json
from rank_bm25 import BM25Okapi

chunk_path = Path("vector_store/chunks.json")


def load_bm25():

    if not chunk_path.exists():
        raise ValueError("The Chunks do not exist")

    with open(chunk_path,"r",encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        raise ValueError("There are no Chunks")

    tokenize_chunks = [
        chunk["text"].lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenize_chunks)

    return bm25,chunks


def retrieve_bm25(query :str , bm25, chunks, top_k:int=5):
    if not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k < 1:
        raise ValueError("Top K cannot be less than 3")

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = scores.argsort()[::-1][:min(top_k, len(chunks))]
    
    results = []

    for idx in top_indices:
        results.append({
            "chunk_id":int(idx),
            "text": chunks[idx]["text"],
            "source":chunks[idx]["source"],
            "page": chunks[idx]["page"],
            "score": float(scores[idx])
        })

    return results



if __name__ == "__main__":
    bm25,chunks = load_bm25()

    query = input("Ask the Question : ")
    results = retrieve_bm25(query,bm25,chunks)

    for result in results:
        print(f"\nChunk ID: {result['chunk_id']}")
        print(f"BM25 score: {result['score']:.4f}")
        print(f"Page: {result['page']}")
        print(result["text"])