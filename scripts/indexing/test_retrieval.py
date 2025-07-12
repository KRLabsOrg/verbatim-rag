#!/usr/bin/env python3
from verbatim_rag.index import VerbatimIndex

def main():
    # 1) load the index you just built
    idx = VerbatimIndex.load("models/index")

    # 2) try a few queries
    for query in [
        "How long is the Golden Gate Bridge?",
        "Who designed the Golden Gate Bridge?",
        "When was it completed?"
    ]:
        print(f"\n🔎 Query: {query}")
        hits = idx.search(query, k=3)
        for i, doc in enumerate(hits, 1):
            src = doc.metadata.get("source", "<no-source>")
            snippet = doc.content.replace("\n"," ")[:120]
            print(f"  {i}. ({src}) → {snippet}…")

if __name__ == "__main__":
    main()
