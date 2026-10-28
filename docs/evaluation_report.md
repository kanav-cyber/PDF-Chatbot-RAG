# PDF-Rag Chatbot — Evaluation Report

## Project overview

A Retrieval-Augmented Generation (RAG) system that extracts PDF content, splits it into chunks, retrieves relevant context, and uses Groq to generate answers.

## Retrieval configuration

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector search: FAISS
- Keyword retrieval: BM25
- Evaluation cutoff: Top 5 chunks
- Previously reported evaluation set: 10 queries

## Previously reported baseline results

| Method | Precision@5 | Recall@5 |
|---|---:|---:|
| FAISS semantic retrieval | 36% | 90% |
| BM25 keyword retrieval | 20% | 50% |

## Interpretation

FAISS achieved higher reported Precision@5 and Recall@5 than BM25 on the existing evaluation set.

These results are preliminary until reproduced using the evaluation scripts and relevance labels.

## Next steps

1. Rerun the evaluation and verify the metrics.
2. Record retrieval latency and end-to-end response latency.
3. Preserve the test questions and relevance labels.
4. Document the best verified configuration.
5. Commit the evaluation scripts and results to Git.
