import time
from main import run_pipeline

def test_full_pipeline_runs():
    question = "How can RAG assist in diagnosis?"

    start = time.perf_counter()
    response = run_pipeline(
        step="step05_generate_response",
        query_args=question,
        use_rag=True
    )
    duration = time.perf_counter() - start

    print(f"\nFull pipeline response time: {duration:.2f} seconds")
    
    assert isinstance(response, str)
    assert len(response) > 20