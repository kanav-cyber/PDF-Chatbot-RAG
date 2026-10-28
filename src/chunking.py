from langchain_text_splitters import RecursiveCharacterTextSplitter
from  src.ingestion import load_pdf


def chunk_doc(pages = list[dict] ,chunk_size :int  = 800,chunk_overlap : int = 150) ->list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    chunks = []

    for page in pages:
        page_chunk = splitter.create_documents(
            texts=[page["text"]],
            metadatas=[{
                "source" : page["source"],
                "page":page["page"]
            }],
        )

        for chunk in page_chunk:
            chunks.append({
                "text":chunk.page_content,
                "source" : chunk.metadata["source"],
                "page":chunk.metadata["page"],
                "start_index":chunk.metadata["start_index"]
            })



    return chunks



if __name__ == "__main__":
    # Extracted pages
    pages = load_pdf("data/deep_lr.pdf")
    # Done chunking
    chunks = chunk_doc(pages)
    print(f"Total number of chunks : {len(chunks)}")


    for i,chunk in enumerate(chunks[:3]):
        print(f"\n----Chunk{i+1}-----")
        print(f"Source :{chunk['source']}")
        print(f"Page : {chunk['page']}")
        print(f"Start Index : {chunk['start_index']}")
        print(chunk["text"])



