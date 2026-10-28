import faiss
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

model_name = "sentence-transformers/all-MiniLM-L6-v2"

vec_dir = Path("vector_store")

def load_retrievers():
    embedding_path = vec_dir / "embeddings.npy"
    chunk_path = vec_dir / "chunks.json"

    if not embedding_path.exists() or not chunk_path.exists():
        raise FileNotFoundError(
            "Vector files not found. Run python -m src.embeddings first."
        )

    embeddings = np.load(embedding_path).astype("float32")

    with open(chunk_path ,"r" ,encoding="utf-8")as f:
        chunks = json.load(f)

    if(len(chunks)) != (len(embeddings)):
        raise ValueError("Number of chunks and embeddings do not match")

    if(len(embeddings)) == 0:
        raise ValueError("No Embedding Found")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    model = SentenceTransformer(model_name)

    return index,chunks,model


    


def retrieve(query : str,index,chunks,model,top_k : int = 3):

    if not query.strip():
        raise ValueError("Query not found")

    if top_k < 1:
        raise ValueError("The top k should me more than 1")

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")


    k = min(top_k,len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append({
            "chunk_id": int(idx),
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "page": chunks[idx]["page"],
            "score": float(score),
        })
    return results

if __name__ == "__main__":
    index, chunks, model = load_retrievers()

    query = input("Ask a question about your PDF: ")

    results = retrieve(query, index, chunks, model, top_k=3)

    print("\nRetrieved results:")

    for i, result in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(f"Similarity score: {result['score']:.4f}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(result["text"])
        