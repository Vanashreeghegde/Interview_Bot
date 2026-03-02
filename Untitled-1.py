from retriver import HybridRetriever  # make sure filename matches exactly

retriever = HybridRetriever()
docs = retriever.retrieve("What is overfitting?")

for i, d in enumerate(docs, 1):
    print(f"\nResult {i}:\n")
    print(d.page_content[:300])