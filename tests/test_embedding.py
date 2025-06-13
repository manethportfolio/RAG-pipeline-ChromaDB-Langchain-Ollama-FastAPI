from classes.embedding_preparer import EmbeddingPreparer
import time

def test_embedding_generation_runs():
    preparer = EmbeddingPreparer(
        file_list=["example.txt"],
        input_dir="data/cleaned_text",
        output_dir="data/embeddings",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    start_time = time.perf_counter()
    try:
        preparer.process_files()
        elapsed = time.perf_counter() - start_time
        print(f"Embedding generation completed in {elapsed:.2f} seconds.")
        assert True
    except Exception as e:
        assert False, f"Embedding generation failed: {e}"