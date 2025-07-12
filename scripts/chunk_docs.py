from verbatim_rag.loader import DocumentLoader
from verbatim_rag.text_splitter import TextSplitter

# 1) Load all your docs
docs = DocumentLoader.load_directory("examples/example_docs", recursive=True)
print(f"🔍 Found {len(docs)} source documents.")

# 2) Configure your splitter
splitter = TextSplitter(chunk_size=200, chunk_overlap=50)

# 3) Chunk them
chunks = splitter.split_documents(docs)
print(f"🗂️  Produced {len(chunks)} chunks from those documents.")

# (Optional) inspect first few chunks
for i, doc in enumerate(chunks[:3]):
    print(f"{i+1}. {doc.metadata['source']} [chunk {doc.metadata['chunk']+1}/{doc.metadata['chunk_of']}] → {doc.content[:60]!r}")
