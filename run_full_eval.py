"""
Full comprehensive evaluation of all 4 plagiarism detection systems.

Tests all systems on the complete 35-case test dataset:
- System 1: Pure Embedding Search
- System 2: Direct LLM (no retrieval)
- System 3: Standard RAG
- System 4: Hybrid RAG
"""

import json
import pickle
import time
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from src.config import DEFAULT_SIMILARITY_THRESHOLD, DEFAULT_TOP_K, DEFAULT_ALPHA
from src.chunking import CodeChunk
from src.embeddings import EmbeddingGenerator
from src.llm import GeminiLLM
from src.retrieval import DenseRetriever, BM25Retriever, HybridRetriever
from src.evaluation import EvaluationMetrics, ErrorAnalyzer
from src.visualization import plot_comparison_chart


def load_test_dataset(path: str = "data/test_dataset.json") -> List[Dict]:
    """Load test dataset."""
    with open(path, 'r') as f:
        return json.load(f)


def load_retrievers(indexes_dir: str = "indexes/"):
    """Load pre-built retrievers."""
    print("\n[Loading Retrievers]")

    # Load dense retriever using class method
    dense_retriever = DenseRetriever.load(f"{indexes_dir}/dense_retriever.pkl")
    print(f"✓ Dense retriever loaded ({len(dense_retriever.chunks)} functions)")

    # Load BM25 retriever using class method
    bm25_retriever = BM25Retriever.load(f"{indexes_dir}/bm25_retriever.pkl")
    print(f"✓ BM25 retriever loaded ({len(bm25_retriever.chunks)} functions)")

    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(dense_retriever, bm25_retriever)
    print(f"✓ Hybrid retriever created\n")

    return dense_retriever, bm25_retriever, hybrid_retriever


def evaluate_system1_embedding(test_cases: List[Dict], dense_retriever: DenseRetriever) -> List[bool]:
    """Evaluate System 1: Pure Embedding Search."""
    print(f"\n{'='*70}")
    print("SYSTEM 1: PURE EMBEDDING SEARCH")
    print(f"{'='*70}")
    print(f"Threshold: {DEFAULT_SIMILARITY_THRESHOLD}")

    predictions = []

    for test_case in tqdm(test_cases, desc="Embedding"):
        query_code = test_case['code']

        # Retrieve similar functions
        results = dense_retriever.retrieve(query_code, top_k=DEFAULT_TOP_K)

        # Check if any result exceeds threshold
        is_plagiarism = any(score >= DEFAULT_SIMILARITY_THRESHOLD for _, score in results)
        predictions.append(is_plagiarism)

        time.sleep(5)  # Rate limiting (15 requests/min limit)

    return predictions


def evaluate_system2_direct_llm(test_cases: List[Dict], llm: GeminiLLM) -> List[bool]:
    """Evaluate System 2: Direct LLM (no retrieval)."""
    print(f"\n{'='*70}")
    print("SYSTEM 2: DIRECT LLM (No Retrieval)")
    print(f"{'='*70}")

    predictions = []

    for test_case in tqdm(test_cases, desc="Direct LLM"):
        query_code = test_case['code']

        # Analyze without context (direct LLM)
        result = llm.analyze_plagiarism_direct(query_code)
        predictions.append(result['is_plagiarism'])

        time.sleep(5)  # Rate limiting (15 requests/min limit)

    return predictions


def evaluate_system3_standard_rag(test_cases: List[Dict], dense_retriever: DenseRetriever, llm: GeminiLLM) -> List[bool]:
    """Evaluate System 3: Standard RAG."""
    print(f"\n{'='*70}")
    print("SYSTEM 3: STANDARD RAG")
    print(f"{'='*70}")

    predictions = []

    for test_case in tqdm(test_cases, desc="Standard RAG"):
        query_code = test_case['code']

        # Retrieve similar functions
        results = dense_retriever.retrieve(query_code, top_k=DEFAULT_TOP_K)
        context_functions = [chunk for chunk, score in results]

        # Analyze with LLM
        result = llm.analyze_plagiarism_with_context(query_code, context_functions)
        predictions.append(result['is_plagiarism'])

        time.sleep(5)  # Rate limiting (15 requests/min limit)

    return predictions


def evaluate_system4_hybrid_rag(test_cases: List[Dict], hybrid_retriever: HybridRetriever, llm: GeminiLLM) -> List[bool]:
    """Evaluate System 4: Hybrid RAG."""
    print(f"\n{'='*70}")
    print("SYSTEM 4: HYBRID RAG")
    print(f"{'='*70}")
    print(f"Alpha: {DEFAULT_ALPHA} (0=BM25, 1=Dense)")

    predictions = []

    for test_case in tqdm(test_cases, desc="Hybrid RAG"):
        query_code = test_case['code']

        # Retrieve using hybrid approach
        results = hybrid_retriever.retrieve(query_code, top_k=DEFAULT_TOP_K, alpha=DEFAULT_ALPHA)
        context_functions = [chunk for chunk, score in results]

        # Analyze with LLM
        result = llm.analyze_plagiarism_with_context(query_code, context_functions)
        predictions.append(result['is_plagiarism'])

        time.sleep(5)  # Rate limiting (15 requests/min limit)

    return predictions


def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION - All 4 Systems on 35 Test Cases")
    print("="*70)

    # Load test dataset
    print("\n[1/6] Loading test dataset...")
    test_cases = load_test_dataset()
    y_true = [tc.get('is_plagiarism', False) for tc in test_cases]

    print(f"✓ Loaded {len(test_cases)} test cases")
    print(f"  - Positive (plagiarism): {sum(y_true)}")
    print(f"  - Negative (original): {len(y_true) - sum(y_true)}")

    # Load retrievers
    print("\n[2/6] Loading pre-built indexes...")
    dense_retriever, bm25_retriever, hybrid_retriever = load_retrievers()

    # Initialize LLM
    llm = GeminiLLM()

    # Evaluate all systems
    all_results = {}

    # System 1: Pure Embedding
    print("\n[3/6] Evaluating System 1: Pure Embedding Search...")
    y_pred_embedding = evaluate_system1_embedding(test_cases, dense_retriever)
    metrics_embedding = EvaluationMetrics.calculate_metrics(y_true, y_pred_embedding)
    all_results["Pure Embedding"] = metrics_embedding
    print(f"✓ Embedding: F1={metrics_embedding['f1_score']:.3f}, Acc={metrics_embedding['accuracy']:.3f}")

    # System 2: Direct LLM
    print("\n[4/6] Evaluating System 2: Direct LLM...")
    y_pred_direct = evaluate_system2_direct_llm(test_cases, llm)
    metrics_direct = EvaluationMetrics.calculate_metrics(y_true, y_pred_direct)
    all_results["Direct LLM"] = metrics_direct
    print(f"✓ Direct LLM: F1={metrics_direct['f1_score']:.3f}, Acc={metrics_direct['accuracy']:.3f}")

    # System 3: Standard RAG
    print("\n[5/6] Evaluating System 3: Standard RAG...")
    y_pred_rag = evaluate_system3_standard_rag(test_cases, dense_retriever, llm)
    metrics_rag = EvaluationMetrics.calculate_metrics(y_true, y_pred_rag)
    all_results["Standard RAG"] = metrics_rag
    print(f"✓ Standard RAG: F1={metrics_rag['f1_score']:.3f}, Acc={metrics_rag['accuracy']:.3f}")

    # System 4: Hybrid RAG
    print("\n[6/6] Evaluating System 4: Hybrid RAG...")
    y_pred_hybrid = evaluate_system4_hybrid_rag(test_cases, hybrid_retriever, llm)
    metrics_hybrid = EvaluationMetrics.calculate_metrics(y_true, y_pred_hybrid)
    all_results["Hybrid RAG"] = metrics_hybrid
    print(f"✓ Hybrid RAG: F1={metrics_hybrid['f1_score']:.3f}, Acc={metrics_hybrid['accuracy']:.3f}")

    # Print summary
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*70)

    for system_name, metrics in all_results.items():
        print(f"\n{system_name}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "full_evaluation_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to results/full_evaluation_results.json")

    # Generate comparison chart
    print("\n" + "="*70)
    print("GENERATING COMPARISON CHART")
    print("="*70)

    chart_path = results_dir / "full_comparison_chart.png"
    plot_comparison_chart(all_results, str(chart_path))
    print(f"Comparison chart saved to: {chart_path}")

    # Error analysis for each system
    print("\n" + "="*70)
    print("ERROR ANALYSIS")
    print("="*70)

    for system_name, y_pred in [
        ("Pure Embedding", y_pred_embedding),
        ("Direct LLM", y_pred_direct),
        ("Standard RAG", y_pred_rag),
        ("Hybrid RAG", y_pred_hybrid)
    ]:
        errors = ErrorAnalyzer.analyze_errors(test_cases, y_pred)
        print(f"\n{system_name}:")
        print(f"  False Positives: {len(errors['false_positives'])}")
        print(f"  False Negatives: {len(errors['false_negatives'])}")

        if errors['false_negatives']:
            print("  Missed cases:")
            for fn in errors['false_negatives'][:3]:  # Show first 3
                tc = fn['test_case']
                print(f"    - {tc.get('description', 'No description')}")
                if 'transformation_type' in tc:
                    print(f"      Type: {tc['transformation_type']}")

    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  - View results/full_comparison_chart.png for visual comparison")
    print("  - Review results/full_evaluation_results.json for detailed metrics")
    print("  - Run ablation studies to analyze hyperparameter sensitivity")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
