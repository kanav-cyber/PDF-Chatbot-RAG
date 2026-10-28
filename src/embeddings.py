from sentence_transformers import SentenceTransformer
from src.ingestion import load_pdf
from src.chunking import chunk_doc
from pathlib import Path
import numpy as np
import json

model_name = "sentence-transformers/all-MiniLM-L6-v2"


def generate_embeddings(chunks:list[dict]):

    if not chunks:
        raise ValueError("No chunking text's Available")

    model = SentenceTransformer(model_name)

    texts = [chunk["text"]for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embeddings




if __name__ == "__main__":

    pages = load_pdf("data/deep_lr.pdf")

    chunks = chunk_doc(pages)

    embeddings = generate_embeddings(chunks)

    output_dir = Path("vector_store")

    output_dir.mkdir(parents=True,exist_ok=True)

    np.save(output_dir / "embeddings.npy",embeddings)

    with open(output_dir / "chunks.json","w",encoding="utf-8") as f:
        json.dump(chunks,f,ensure_ascii=False,indent=2)

        print("Embeddings generated successfully!")
        print("Number of chunks:", len(chunks))
        print("Embedding shape:", embeddings.shape)
        print("Saved to:", output_dir.resolve())
