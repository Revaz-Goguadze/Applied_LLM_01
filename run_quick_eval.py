"""
Quick evaluation of plagiarism detection systems on test dataset.
"""

import sys
import os
import json
import time
from tqdm import tqdm

sys.path.append('.')

from src.retrieval import DenseRetriever, BM25Retriever, HybridRetriever
from src.llm import GeminiLLM
from src.evaluation import EvaluationMetrics
from src.visualization import plot_comparison_chart

print("\n" + "="*70)
print("QUICK EVALUATION - Testing All 4 Systems on 35 Test Cases")
print("="*70)

# Load test dataset
print("\n[1/5] Loading test dataset...")
with open('data/test_dataset.json', 'r') as f:
    test_dataset = json.load(f)

print(f"✓ Loaded {len(test_dataset)} test cases")
print(f"  - Positive (plagiarism): {sum(1 for tc in test_dataset if tc['is_plagiarism'])}")
print(f"  - Negative (original): {sum(1 for tc in test_dataset if not tc['is_plagiarism'])}")

# Load indexes
print("\n[2/5] Loading pre-built indexes...")
dense_retriever = DenseRetriever.load("indexes/dense_retriever.pkl")
bm25_retriever = BM25Retriever.load("indexes/bm25_retriever.pkl")
hybrid_retriever = HybridRetriever(dense_retriever, bm25_retriever)
llm = GeminiLLM()
print(f"✓ All retrievers loaded ({len(dense_retriever.chunks)} functions indexed)")

# Define detection functions
def detect_embedding(query_code, threshold=0.85):
    results = dense_retriever.retrieve(query_code, top_k=5)
    if not results:
        return {'is_plagiarism': False}
    _, max_similarity = results[0]
    return {'is_plagiarism': max_similarity >= threshold}

def detect_rag(query_code, top_k=10):
    retrieved = dense_retriever.retrieve(query_code, top_k=top_k)
    candidate_chunks = [chunk for chunk, score in retrieved]
    result = llm.analyze_plagiarism_with_context(query_code, candidate_chunks)
    return result

def detect_hybrid_rag(query_code, top_k=10, alpha=0.5):
    retrieved = hybrid_retriever.retrieve(query_code, top_k=top_k, alpha=alpha)
    candidate_chunks = [chunk for chunk, score in retrieved]
    result = llm.analyze_plagiarism_with_context(query_code, candidate_chunks)
    return result

# Evaluate System 1: Pure Embedding
print("\n[3/5] Evaluating System 1: Pure Embedding Search...")
y_true_emb = []
y_pred_emb = []

for tc in tqdm(test_dataset, desc="Embedding"):
    y_true_emb.append(tc['is_plagiarism'])
    try:
        result = detect_embedding(tc['code'])
        y_pred_emb.append(result['is_plagiarism'])
    except:
        y_pred_emb.append(False)

metrics_embedding = EvaluationMetrics.calculate_metrics(y_true_emb, y_pred_emb)
print(f"✓ Embedding: F1={metrics_embedding['f1_score']:.3f}, Acc={metrics_embedding['accuracy']:.3f}")

# Evaluate System 2: Standard RAG (on subset to save time/cost)
print("\n[4/5] Evaluating System 2: Standard RAG (subset of 15 cases)...")
subset = test_dataset[:15]  # Use subset for speed
y_true_rag = []
y_pred_rag = []

for tc in tqdm(subset, desc="RAG"):
    y_true_rag.append(tc['is_plagiarism'])
    try:
        result = detect_rag(tc['code'])
        y_pred_rag.append(result['is_plagiarism'])
        time.sleep(0.5)  # Rate limiting
    except Exception as e:
        print(f"Error: {e}")
        y_pred_rag.append(False)

metrics_rag = EvaluationMetrics.calculate_metrics(y_true_rag, y_pred_rag)
print(f"✓ RAG: F1={metrics_rag['f1_score']:.3f}, Acc={metrics_rag['accuracy']:.3f}")

# Evaluate System 3: Hybrid RAG (on same subset)
print("\n[5/5] Evaluating System 3: Hybrid RAG (subset of 15 cases)...")
y_pred_hybrid = []

for tc in tqdm(subset, desc="Hybrid"):
    try:
        result = detect_hybrid_rag(tc['code'])
        y_pred_hybrid.append(result['is_plagiarism'])
        time.sleep(0.5)  # Rate limiting
    except Exception as e:
        print(f"Error: {e}")
        y_pred_hybrid.append(False)

metrics_hybrid = EvaluationMetrics.calculate_metrics(y_true_rag, y_pred_hybrid)
print(f"✓ Hybrid: F1={metrics_hybrid['f1_score']:.3f}, Acc={metrics_hybrid['accuracy']:.3f}")

# Create results summary
results = {
    'Pure Embedding (full 35)': metrics_embedding,
    'Standard RAG (15)': metrics_rag,
    'Hybrid RAG (15)': metrics_hybrid
}

# Print summary
print("\n" + "="*70)
print("EVALUATION RESULTS SUMMARY")
print("="*70)

for system, metrics in results.items():
    print(f"\n{system}:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1 Score:  {metrics['f1_score']:.3f}")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")

# Generate comparison chart
print("\n" + "="*70)
print("GENERATING COMPARISON CHART")
print("="*70)

plot_comparison_chart(results, save_path="results/comparison_chart.png")

# Save results
os.makedirs('results', exist_ok=True)
with open('results/evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to results/evaluation_results.json")
print("\n" + "="*70)
print("QUICK EVALUATION COMPLETE!")
print("="*70)
print("\nNext steps:")
print("  - View results/comparison_chart.png for visual comparison")
print("  - Run full 03_evaluation.ipynb for comprehensive analysis")
print("  - All indexes are ready for interactive testing")
print("="*70)
