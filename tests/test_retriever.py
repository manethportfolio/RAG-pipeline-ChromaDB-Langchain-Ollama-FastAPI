from classes.chromadb_retriever import ChromaDBRetriever
import time

def test_retriever_returns_results():
    retriever = ChromaDBRetriever(
        vectordb_dir="data/vectordb",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        collection_name="collections"
    )
    start_time = time.perf_counter()
    try:
        results = retriever.query("test question")
        elapsed = time.perf_counter() - start_time
        print(f"Retriever query completed in {elapsed:.2f} seconds.")
        assert isinstance(results, list)
    except Exception as e:
        assert False, f"Retriever failed: {e}"