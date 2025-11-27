#!/usr/bin/env python3
"""
one-click test runner for the plagiarism detection system.
tests all modules, retrieval systems, and detection functions.

usage: python run_all_tests.py
"""

import sys
import os
import json
import time

sys.path.insert(0, '.')

def print_header(text):
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)

def print_status(name, passed):
    status = "PASS" if passed else "FAIL"
    symbol = "✓" if passed else "✗"
    print(f"  {symbol} {name}: {status}")
    return passed

def main():
    start_time = time.time()
    all_passed = True

    print_header("plagiarism detection system - full test suite")
    print(f"started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # test 1: imports
    print_header("1. testing imports")
    try:
        from src.config import GEMINI_API_KEY, GEMINI_MODEL, EMBEDDING_MODEL
        from src.chunking import CodeChunk, PythonFunctionExtractor
        from src.embeddings import EmbeddingGenerator, cosine_similarity_batch
        from src.retrieval import DenseRetriever, BM25Retriever, HybridRetriever
        from src.llm import GeminiLLM
        from src.evaluation import EvaluationMetrics, ErrorAnalyzer
        from src.visualization import plot_comparison_chart, plot_confusion_matrices
        all_passed &= print_status("all module imports", True)
    except Exception as e:
        all_passed &= print_status(f"imports failed: {e}", False)
        return 1

    # test 2: load indexes
    print_header("2. testing index loading")
    try:
        dense = DenseRetriever.load("indexes/dense_retriever.pkl")
        all_passed &= print_status(f"dense retriever ({len(dense.chunks)} chunks)", True)

        bm25 = BM25Retriever.load("indexes/bm25_retriever.pkl")
        all_passed &= print_status(f"bm25 retriever ({len(bm25.chunks)} chunks)", True)

        hybrid = HybridRetriever(dense, bm25)
        all_passed &= print_status("hybrid retriever", True)
    except Exception as e:
        all_passed &= print_status(f"index loading failed: {e}", False)
        return 1

    # test 3: retrieval
    print_header("3. testing retrieval systems")
    test_code = '''
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
'''

    try:
        dense_results = dense.retrieve(test_code, top_k=5)
        all_passed &= print_status(f"dense retrieval ({len(dense_results)} results)", len(dense_results) == 5)

        bm25_results = bm25.retrieve(test_code, top_k=5)
        all_passed &= print_status(f"bm25 retrieval ({len(bm25_results)} results)", len(bm25_results) == 5)

        hybrid_results = hybrid.retrieve(test_code, top_k=5, alpha=0.5)
        all_passed &= print_status(f"hybrid retrieval ({len(hybrid_results)} results)", len(hybrid_results) == 5)
    except Exception as e:
        all_passed &= print_status(f"retrieval failed: {e}", False)

    # test 4: llm
    print_header("4. testing llm connection")
    try:
        llm = GeminiLLM()
        all_passed &= print_status(f"gemini llm initialized ({GEMINI_MODEL})", True)
    except Exception as e:
        all_passed &= print_status(f"llm init failed: {e}", False)
        return 1

    # test 5: detection systems
    print_header("5. testing detection systems (with api calls)")

    # system 1: pure embedding
    try:
        results = dense.retrieve(test_code, top_k=1)
        similarity = results[0][1] if results else 0
        all_passed &= print_status(f"pure embedding (similarity: {similarity:.4f})", True)
    except Exception as e:
        all_passed &= print_status(f"pure embedding failed: {e}", False)

    # system 2: direct llm
    try:
        result = llm.analyze_plagiarism_direct(test_code)
        is_plag = result.get('is_plagiarism', 'error')
        all_passed &= print_status(f"direct llm (is_plagiarism: {is_plag})", 'is_plagiarism' in result)
    except Exception as e:
        all_passed &= print_status(f"direct llm failed: {e}", False)

    # system 3: standard rag
    try:
        context = [chunk for chunk, _ in dense.retrieve(test_code, top_k=10)]
        result = llm.analyze_plagiarism_with_context(test_code, context)
        is_plag = result.get('is_plagiarism', 'error')
        all_passed &= print_status(f"standard rag (is_plagiarism: {is_plag})", 'is_plagiarism' in result)
    except Exception as e:
        all_passed &= print_status(f"standard rag failed: {e}", False)

    # system 4: hybrid rag
    try:
        context = [chunk for chunk, _ in hybrid.retrieve(test_code, top_k=10, alpha=0.5)]
        result = llm.analyze_plagiarism_with_context(test_code, context)
        is_plag = result.get('is_plagiarism', 'error')
        all_passed &= print_status(f"hybrid rag (is_plagiarism: {is_plag})", 'is_plagiarism' in result)
    except Exception as e:
        all_passed &= print_status(f"hybrid rag failed: {e}", False)

    # test 6: evaluation
    print_header("6. testing evaluation module")
    try:
        y_true = [True, True, False, False, True]
        y_pred = [True, False, False, True, True]
        metrics = EvaluationMetrics.calculate_metrics(y_true, y_pred)

        all_passed &= print_status(f"precision: {metrics['precision']:.3f}", True)
        all_passed &= print_status(f"recall: {metrics['recall']:.3f}", True)
        all_passed &= print_status(f"f1_score: {metrics['f1_score']:.3f}", True)
    except Exception as e:
        all_passed &= print_status(f"evaluation failed: {e}", False)

    # test 7: saved results
    print_header("7. checking saved results")
    result_files = [
        "results/evaluation_results.json",
        "results/comparison_chart.png",
        "results/confusion_matrices.png",
        "results/ablation_k_values.png",
        "results/ablation_alpha_values.png",
        "results/cost_vs_performance.png"
    ]

    for f in result_files:
        exists = os.path.exists(f)
        size = os.path.getsize(f) / 1024 if exists else 0
        all_passed &= print_status(f"{os.path.basename(f)} ({size:.1f} KB)", exists and size > 0)

    # test 8: test dataset
    print_header("8. checking test dataset")
    try:
        with open("data/test_dataset.json", "r") as f:
            dataset = json.load(f)
        num_cases = len(dataset)
        num_plagiarism = sum(1 for case in dataset if case.get('is_plagiarism', False))
        all_passed &= print_status(f"test cases: {num_cases} ({num_plagiarism} plagiarism, {num_cases - num_plagiarism} original)", num_cases > 0)
    except Exception as e:
        all_passed &= print_status(f"test dataset failed: {e}", False)

    # summary
    elapsed = time.time() - start_time
    print_header("test summary")

    if all_passed:
        print(f"\n  ALL TESTS PASSED!")
    else:
        print(f"\n  SOME TESTS FAILED - check output above")

    print(f"\n  total time: {elapsed:.2f} seconds")
    print(f"  finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
