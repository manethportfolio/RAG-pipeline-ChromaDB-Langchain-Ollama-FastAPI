def test_rag_vs_baseline_length():
    from main import run_pipeline

    query = "How does RAG help with diagnostics?"

    rag_response = run_pipeline(
        step="step05_generate_response",
        query_args=query,
        use_rag=True
    )

    baseline_response = run_pipeline(
        step="step05_generate_response",
        query_args=query,
        use_rag=False
    )

    print("\n--- RAG Response ---")
    print(f"Length: {len(rag_response)}")
    print(f"Preview: {rag_response[:300]}")

    print("\n--- Baseline Response ---")
    print(f"Length: {len(baseline_response)}")
    print(f"Preview: {baseline_response[:300]}")

    assert len(rag_response) > len(baseline_response), \
        "RAG response should generally be more informative than baseline."


