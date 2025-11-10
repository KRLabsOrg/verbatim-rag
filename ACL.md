
## Prerequisites

### Getting PDFs
PDFs are currently scraped via the Makefile of the acl-anthology repo

### Preprocessing

Sample command to preprocess all PDFs to MDs under a top-level directory:

```bash
python scripts/preprocess_acl.py --input-dir ../acl-anthology/build/anthology-files/pdf --output-dir acl_md --metadata-file papers.json --doc-batch-size 512 --page-batch-size 1024 &> acl_logs/20251103/202511030845.log
```

You can find everyhing we have on both `neptun` and `datalab` in my home dirs, under
`projects/verbatim-rag/acl_md`

### Indexing

Sample command to chunk and index all md files in a given directory (using a GPU):

```bash
time python scripts/index_acl.py --input-dir acl_md/acl --index-file acl.db --metadata-file papers.json --collection-name acl --device cuda &> acl_log/20251103_index_acl.log
```

### Querying

Sample command for loading an index and trying some queries

```bash
python scripts/query_index.py --index-file acl.db --device cuda  --collection-name acl
```




## NOTES

### 2025. 11. 10
Finished processing all anthology papers, stats:
|   |   |
|---|---|
|total PDFs | 111 640 |
|total MDs created | 107 559 |
|skipped because not listed in metadata file | 4 037 |
|fails to convert | 63 |
|empty MDs (??) | 154 |

Out of 60 PDFs that fail to convert, 47 will fail fast with an exception, the other 16 are listed
as `TO_SKIP` because they either cause segfaults (7 papers) or take a long time to fail (9 papers)

