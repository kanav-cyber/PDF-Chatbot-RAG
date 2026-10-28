
import json
from pathlib import Path

from src.retriever import load_retrievers, retrieve


EVAL_PATH = Path("data/eval_queries.json")
TOP_K = 5


def calculate_precision_recall(retrieved_ids, relevant_ids, k=5):
    retrieved_ids = retrieved_ids[:k]
    relevant_ids = set(relevant_ids)

    if not relevant_ids:
        raise ValueError("Each question must have at least one relevant chunk ID.")

    hits = len(set(retrieved_ids) & relevant_ids)

    precision = hits / len(retrieved_ids) if retrieved_ids else 0.0
    recall = hits / len(relevant_ids)

    return precision, recall, hits


def main():
    if not EVAL_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {EVAL_PATH}"
        )

    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)

    if not eval_queries:
        raise ValueError("No evaluation questions found.")

    index, chunks, model = load_retrievers()

    all_precision = []
    all_recall = []

    for item in eval_queries:
        question = item["question"]
        relevant_ids = item["relevant_chunk_ids"]

        # Validate the manually assigned chunk IDs.
        invalid_ids = [
            chunk_id for chunk_id in relevant_ids
            if not isinstance(chunk_id, int)
            or chunk_id < 0
            or chunk_id >= len(chunks)
        ]

        if invalid_ids:
            raise ValueError(
                f"Invalid relevant chunk IDs for '{question}': {invalid_ids}"
            )

        results = retrieve(
            question,
            index,
            chunks,
            model,
            top_k=TOP_K,
        )

        retrieved_ids = [result["chunk_id"] for result in results]

        precision, recall, hits = calculate_precision_recall(
            retrieved_ids,
            relevant_ids,
            TOP_K,
        )

        all_precision.append(precision)
        all_recall.append(recall)

        print(f"\nQuestion: {question}")
        print(f"Relevant chunk IDs: {relevant_ids}")
        print(f"Retrieved chunk IDs: {retrieved_ids}")
        print(f"Relevant chunks retrieved: {hits}")
        print(f"Precision@{TOP_K}: {precision:.2%}")
        print(f"Recall@{TOP_K}: {recall:.2%}")

    print("\n" + "=" * 45)
    print(f"Evaluated questions: {len(eval_queries)}")
    print(f"Mean Precision@{TOP_K}: {sum(all_precision) / len(all_precision):.2%}")
    print(f"Mean Recall@{TOP_K}: {sum(all_recall) / len(all_recall):.2%}")


if __name__ == "__main__":
    main()

