from main import run_pipeline
import time

def test_full_rag_pipeline():
    queries = [
        "How does RAG assist in diagnostics?",
        "What are the advantages of using a vector database?",
        "Explain the role of embeddings in retrieval systems.",
        "How can RAG be applied in hospital management?",
        "What is the function of prompt engineering in LLMs?"
    ]

    for question in queries:
        start_time = time.perf_counter()
        response = run_pipeline(
            step="step05_generate_response",
            query_args=question,
            use_rag=True
        )
        duration = time.perf_counter() - start_time
        print(f"Query: '{question}' took {duration:.2f} seconds")
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 20, "Response too short"
