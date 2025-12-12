"""
Example usage of the Travel RAG System
"""

from rag_system import TravelRAGSystem


def main():
    # Initialize the RAG system
    print("Initializing Travel RAG System...")
    rag = TravelRAGSystem(
        documents_dir="documents",
        vector_store_path="vector_store",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        ollama_base_url="http://localhost:11434",
        llm_model="gemma2"
    )
    
    print("\n" + "="*60)
    print("Travel RAG System Ready!")
    print("="*60 + "\n")
    
    # Interactive query loop
    while True:
        question = input("Enter your question (or 'quit' to exit): ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        # Query the system
        result = rag.query(question, use_enrichment=True)
        
        # Display results
        print("\n" + "-"*60)
        print(f"Answer: {result['answer']}")
        print("-"*60)
        
        if result['sources']:
            print(f"\nRetrieved from {len(result['sources'])} source(s):")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. {source['file']} ({source['file_type']})")
        
        if result['source_based']:
            print("\n✓ Answer based on your documents")
        elif result['enriched']:
            print("\n✓ Answer enriched with additional knowledge")
        
        print("\n")


if __name__ == "__main__":
    main()


