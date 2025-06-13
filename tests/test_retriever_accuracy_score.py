from classes.chromadb_retriever import ChromaDBRetriever

def test_retriever_accuracy_score():
    retriever = ChromaDBRetriever(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        collection_name="collections",
        vectordb_dir="data/vectordb",
        score_threshold=0.0
    )

    queries = [
        ("What role does RAG play in ensuring accurate medication lists?", "rag_for_medication_reconciliation_cleaned.txt"),
        ("How can RAG systems support real-time clinical decision-making?", "real-time_clinical_query_answering_with_rag_cleaned.txt"),
        ("In what ways can RAG improve personalized treatment planning?", "personalized_treatment_generation_using_rag_cleaned.txt"),
        ("What methods are used by RAG to summarize clinical patient notes?", "clinical_notes_summarization_with_rag_cleaned.txt"),
        ("How is RAG applied to aid diagnostic decisions in healthcare?", "rag_for_diagnostic_decision_support_cleaned.txt"),
    ]

    correct = 0
    for query, expected_id in queries:
        results = retriever.query(query, top_k=3)
        retrieved_ids = [r.get("id") for r in results]

        print(f"\nQuery: {query}")
        print(f"Expected: {expected_id}")
        print(f"Retrieved: {retrieved_ids}")

        if expected_id in retrieved_ids:
            correct += 1

    accuracy = correct / len(queries)
    print(f"\n Retrieval Accuracy: {accuracy:.2%}")

    # Add a threshold-based assertion to make it a real test
    assert accuracy >= 0.2, "Retrieval accuracy is too low"

