# compare_chunking.py
"""
Script to compare different chunking parameters for the RAG system.
Tests various chunk_size and chunk_overlap combinations and compares results.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from rag_system import TravelRAGSystem


class ChunkingComparison:
    """Compare different chunking configurations."""
    
    def __init__(
        self,
        documents_dir: str = "documents",
        test_queries: List[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        ollama_base_url: str = "http://localhost:11434",
        llm_model: str = "gemma2"
    ):
        """
        Initialize the comparison tool.
        
        Args:
            documents_dir: Directory containing documents
            test_queries: List of test queries to evaluate
            embedding_model: Embedding model to use
            ollama_base_url: Ollama API URL
            llm_model: Ollama model name
        """
        self.documents_dir = documents_dir
        self.embedding_model = embedding_model
        self.ollama_base_url = ollama_base_url
        self.llm_model = llm_model
        
        # Default test queries if none provided
        self.test_queries = test_queries or [
            "When is my flight from TLV?",
            "What is my insurance policy number?",
            "What's the hotel name?",
            "Who is operating the flight?",
            "What is the hotel address?",
        ]
    
    def test_configuration(
        self,
        chunk_size: int,
        chunk_overlap: int,
        k: int = 5
    ) -> Dict:
        """
        Test a specific chunking configuration.
        
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            k: Number of documents to retrieve
        
        Returns:
            Dictionary with test results
        """
        config_name = f"chunk{chunk_size}_overlap{chunk_overlap}"
        vector_store_path = f"vector_store_{config_name}"
        
        print(f"\n{'='*70}")
        print(f"Testing Configuration: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        print(f"{'='*70}")
        
        # Initialize RAG system with this configuration
        start_time = time.time()
        rag = TravelRAGSystem(
            documents_dir=self.documents_dir,
            vector_store_path=vector_store_path,
            embedding_model=self.embedding_model,
            ollama_base_url=self.ollama_base_url,
            llm_model=self.llm_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        init_time = time.time() - start_time
        
        # Get statistics about the vector store
        stats = rag.get_statistics()
        
        # Test each query
        query_results = []
        total_query_time = 0
        
        for query in self.test_queries:
            query_start = time.time()
            result = rag.query(query, use_enrichment=True, k=k)
            query_time = time.time() - query_start
            total_query_time += query_time
            
            query_results.append({
                "query": query,
                "answer": result['answer'],
                "answer_length": len(result['answer']),
                "sources_count": len(result['sources']),
                "source_based": result['source_based'],
                "enriched": result['enriched'],
                "query_time": query_time,
                "source_files": [s['file'] for s in result['sources']]
            })
        
        avg_query_time = total_query_time / len(self.test_queries)
        
        # Calculate metrics
        source_based_ratio = sum(1 for r in query_results if r['source_based']) / len(query_results)
        avg_answer_length = sum(r['answer_length'] for r in query_results) / len(query_results)
        avg_sources_count = sum(r['sources_count'] for r in query_results) / len(query_results)
        
        return {
            "configuration": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "overlap_percentage": (chunk_overlap / chunk_size) * 100 if chunk_size > 0 else 0
            },
            "vector_store_stats": {
                "total_chunks": stats.get('total_chunks', 0),
                "total_files": stats.get('total_files', 0),
                "avg_chunks_per_file": stats.get('avg_chunks_per_file', 0)
            },
            "performance": {
                "init_time": init_time,
                "avg_query_time": avg_query_time,
                "total_query_time": total_query_time
            },
            "quality_metrics": {
                "source_based_ratio": source_based_ratio,
                "avg_answer_length": avg_answer_length,
                "avg_sources_count": avg_sources_count
            },
            "query_results": query_results
        }
    
    def compare_configurations(
        self,
        configurations: List[Tuple[int, int]],
        k: int = 5
    ) -> Dict:
        """
        Compare multiple chunking configurations.
        
        Args:
            configurations: List of (chunk_size, chunk_overlap) tuples
            k: Number of documents to retrieve
        
        Returns:
            Dictionary with comparison results
        """
        print("\n" + "="*70)
        print("CHUNKING PARAMETER COMPARISON")
        print("="*70)
        print(f"Testing {len(configurations)} configurations with {len(self.test_queries)} queries each")
        print("="*70)
        
        results = []
        
        for chunk_size, chunk_overlap in configurations:
            result = self.test_configuration(chunk_size, chunk_overlap, k=k)
            results.append(result)
        
        return {
            "test_queries": self.test_queries,
            "configurations": results,
            "summary": self._generate_summary(results)
        }
    
    def _generate_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics for comparison."""
        summary = {
            "best_source_based": None,
            "best_performance": None,
            "most_chunks": None,
            "least_chunks": None,
            "configurations": []
        }
        
        best_source_ratio = -1
        best_perf_time = float('inf')
        max_chunks = -1
        min_chunks = float('inf')
        
        for result in results:
            config = result['configuration']
            quality = result['quality_metrics']
            perf = result['performance']
            stats = result['vector_store_stats']
            
            config_summary = {
                "chunk_size": config['chunk_size'],
                "chunk_overlap": config['chunk_overlap'],
                "overlap_percentage": config['overlap_percentage'],
                "total_chunks": stats['total_chunks'],
                "source_based_ratio": quality['source_based_ratio'],
                "avg_query_time": perf['avg_query_time'],
                "avg_answer_length": quality['avg_answer_length']
            }
            summary["configurations"].append(config_summary)
            
            # Track best/worst
            if quality['source_based_ratio'] > best_source_ratio:
                best_source_ratio = quality['source_based_ratio']
                summary["best_source_based"] = config_summary
            
            if perf['avg_query_time'] < best_perf_time:
                best_perf_time = perf['avg_query_time']
                summary["best_performance"] = config_summary
            
            if stats['total_chunks'] > max_chunks:
                max_chunks = stats['total_chunks']
                summary["most_chunks"] = config_summary
            
            if stats['total_chunks'] < min_chunks:
                min_chunks = stats['total_chunks']
                summary["least_chunks"] = config_summary
        
        return summary
    
    def print_comparison_table(self, comparison_results: Dict):
        """Print a formatted comparison table."""
        print("\n" + "="*100)
        print("COMPARISON RESULTS")
        print("="*100)
        
        # Table header
        print(f"\n{'Config':<20} {'Chunks':<10} {'Source%':<10} {'Query Time':<12} {'Answer Len':<12} {'Overlap%':<10}")
        print("-" * 100)
        
        # Table rows
        for config_data in comparison_results['summary']['configurations']:
            config_str = f"{config_data['chunk_size']}/{config_data['chunk_overlap']}"
            chunks = config_data['total_chunks']
            source_pct = config_data['source_based_ratio'] * 100
            query_time = config_data['avg_query_time']
            answer_len = config_data['avg_answer_length']
            overlap_pct = config_data['overlap_percentage']
            
            print(f"{config_str:<20} {chunks:<10} {source_pct:<10.1f} {query_time:<12.2f}s {answer_len:<12.0f} {overlap_pct:<10.1f}")
        
        # Summary
        print("\n" + "="*100)
        print("SUMMARY")
        print("="*100)
        
        if comparison_results['summary']['best_source_based']:
            best = comparison_results['summary']['best_source_based']
            print(f"\n✓ Best Source-Based Ratio: {best['chunk_size']}/{best['chunk_overlap']} "
                  f"({best['source_based_ratio']*100:.1f}%)")
        
        if comparison_results['summary']['best_performance']:
            best = comparison_results['summary']['best_performance']
            print(f"✓ Fastest Queries: {best['chunk_size']}/{best['chunk_overlap']} "
                  f"({best['avg_query_time']:.2f}s avg)")
        
        if comparison_results['summary']['most_chunks']:
            most = comparison_results['summary']['most_chunks']
            print(f"✓ Most Chunks: {most['chunk_size']}/{most['chunk_overlap']} ({most['total_chunks']} chunks)")
        
        if comparison_results['summary']['least_chunks']:
            least = comparison_results['summary']['least_chunks']
            print(f"✓ Least Chunks: {least['chunk_size']}/{least['chunk_overlap']} ({least['total_chunks']} chunks)")
    
    def save_results(self, comparison_results: Dict, filename: str = "chunking_comparison_results.json"):
        """Save comparison results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        print(f"\n✓ Results saved to {filename}")


def main():
    """Main function to run the comparison."""
    
    # Define configurations to test
    # Format: (chunk_size, chunk_overlap)
    configurations = [
        (800, 150),   # Small chunks, ~19% overlap
        (1000, 200),  # Current default, 20% overlap
        (1200, 240),  # Medium chunks, 20% overlap
        (1500, 300),  # Large chunks, 20% overlap
        (1000, 100),  # Current size, less overlap (10%)
        (1000, 300),  # Current size, more overlap (30%)
    ]
    
    # Optional: Customize test queries
    test_queries = [
        "When is my flight from TLV?",
        "What is my insurance policy number?",
        "What's the hotel name?",
        "Who is operating the flights?",
        "What is the hotel address?",
    ]
    
    # Create comparison tool
    comparator = ChunkingComparison(
        documents_dir="documents",
        test_queries=test_queries,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        ollama_base_url="http://localhost:11434",
        llm_model="gemma2"
    )
    
    # Run comparison
    print("\nStarting chunking parameter comparison...")
    print("This may take several minutes depending on your documents and LLM speed.\n")
    
    comparison_results = comparator.compare_configurations(configurations, k=5)
    
    # Print results
    comparator.print_comparison_table(comparison_results)
    
    # Save results
    comparator.save_results(comparison_results, filename="comparison_results.json")
    
    # Print detailed results for each query
    print("\n" + "="*100)
    print("DETAILED QUERY RESULTS")
    print("="*100)
    
    for config_result in comparison_results['configurations']:
        config = config_result['configuration']
        print(f"\n{'='*70}")
        print(f"Configuration: {config['chunk_size']}/{config['chunk_overlap']} "
              f"({config['overlap_percentage']:.1f}% overlap)")
        print(f"{'='*70}")
        
        for i, query_result in enumerate(config_result['query_results'], 1):
            print(f"\nQuery {i}: {query_result['query']}")
            print(f"  Answer: {query_result['answer'][:150]}...")
            print(f"  Source-based: {query_result['source_based']}")
            print(f"  Sources: {query_result['sources_count']} files")
            print(f"  Time: {query_result['query_time']:.2f}s")


if __name__ == "__main__":
    main()