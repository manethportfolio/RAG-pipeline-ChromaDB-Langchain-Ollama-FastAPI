from classes.document_ingestor import DocumentIngestor
import time

def test_document_ingestor_runs_without_error():
    ingestor = DocumentIngestor(
        file_list=["example.txt"],
        input_dir="data/raw_input",
        output_dir="data/cleaned_text",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    start_time = time.perf_counter()
    try:
        ingestor.process_files()
        elapsed = time.perf_counter() - start_time
        print(f"Document ingestion completed in {elapsed:.2f} seconds.")
        assert True
    except Exception as e:
        assert False, f"Document ingestion failed: {e}"

