import time
from main import run_pipeline

def test_latency_measurement():
    query = "How does RAG assist in diagnosis?"

    start_rag = time.time()
    rag_response = run_pipeline(
        step="step05_generate_response",
        query_args=query,
        use_rag=True
    )
    rag_duration = time.time() - start_rag

    start_baseline = time.time()
    baseline_response = run_pipeline(
        step="step05_generate_response",
        query_args=query,
        use_rag=False
    )
    baseline_duration = time.time() - start_baseline

    print(f"\nRAG Response Time:      {rag_duration:.2f} seconds")
    print(f"Baseline Response Time: {baseline_duration:.2f} seconds")
    print(f"RAG Response Length:    {len(rag_response)}")
    print(f"Baseline Response Length: {len(baseline_response)}")

    # You can add asserts if needed:
    assert isinstance(rag_response, str) and len(rag_response) > 10
    assert isinstance(baseline_response, str) and len(baseline_response) > 10
