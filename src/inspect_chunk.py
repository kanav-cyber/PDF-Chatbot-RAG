import json 
with open("vector_store/chunks.json" , "r",encoding="utf-8") as f:
    chunks = json.load(f)

print(F"Total Chunks : {len(chunks)}")


for i ,chunk in enumerate(chunks):
    print(f"\n{'=' * 60}:")
    print(f"chunk id :{i}")
    print(f"Source : {chunk['source']}")
    print(f"Page {chunk['page']}")
    print(chunk["text"])