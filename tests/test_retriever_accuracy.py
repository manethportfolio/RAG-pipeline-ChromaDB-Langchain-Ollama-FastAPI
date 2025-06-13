def test_known_query_returns_expected_doc():
    
    from classes.chromadb_retriever import ChromaDBRetriever

    retriever = ChromaDBRetriever(
        vectordb_dir="data/vectordb",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        collection_name="collections"
    )

    query = "Explain case-based retrieval for diagnostics"
    expected_doc_id = "rag_for_diagnostic_decision_support_cleaned.txt"

    results = retriever.query(query, top_k=3)
    returned_ids = [doc.get("id", "") for doc in results]

    assert expected_doc_id in returned_ids, f"Expected {expected_doc_id} not found in top_k results."
