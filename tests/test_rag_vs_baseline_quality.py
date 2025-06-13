from main import run_pipeline
from rouge_score import rouge_scorer

def test_rag_vs_baseline_quality():
    query = "How does RAG help with diagnostics?"
    reference_answer = (
        "RAG helps with diagnostics by retrieving relevant clinical guidelines, previous cases, "
        "and contextual information from medical literature to assist decision-making."
    )

    # Get RAG and baseline responses
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

    # Compute ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rag_score = scorer.score(reference_answer, rag_response)['rougeL'].fmeasure
    baseline_score = scorer.score(reference_answer, baseline_response)['rougeL'].fmeasure

    print("\n--- ROUGE-L F1 Scores ---")
    print(f"RAG: {rag_score:.4f}")
    print(f"Baseline: {baseline_score:.4f}")

    # Simple assertion
    assert rag_score >= baseline_score, "Expected RAG response to match reference better than baseline."
