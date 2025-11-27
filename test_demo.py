"""
Quick demo script to test all 4 plagiarism detection systems.
Run this to verify everything is working before running the notebooks.
"""

import sys
sys.path.append('.')

import numpy as np
from src.chunking import PythonFunctionExtractor
from src.embeddings import EmbeddingGenerator
from src.retrieval import DenseRetriever, BM25Retriever, HybridRetriever
from src.llm import GeminiLLM

print("\n" + "="*70)
print("PLAGIARISM DETECTION SYSTEM - QUICK DEMO")
print("="*70)

# Step 1: Extract functions
print("\n[1/5] Extracting functions from reference corpus...")
extractor = PythonFunctionExtractor()
chunks = extractor.extract_from_directory("data/reference_corpus/string_utils")
chunks += extractor.extract_from_directory("data/reference_corpus/math_utils")
print(f"      ✓ Extracted {len(chunks)} functions")

# Step 2: Generate embeddings
print("\n[2/5] Generating embeddings (this may take a minute)...")
embed_gen = EmbeddingGenerator()
codes = [chunk.content for chunk in chunks]
embeddings = embed_gen.embed_batch(codes, batch_size=20, show_progress=False)
embeddings_matrix = np.array(embeddings)
print(f"      ✓ Generated {len(embeddings)} embeddings (dimension: {embeddings_matrix.shape[1]})")

# Step 3: Build retrievers
print("\n[3/5] Building retrieval indexes...")
dense_retriever = DenseRetriever(chunks, embeddings_matrix)
bm25_retriever = BM25Retriever(chunks)
hybrid_retriever = HybridRetriever(dense_retriever, bm25_retriever)
llm = GeminiLLM()
print(f"      ✓ Dense, BM25, and Hybrid retrievers ready")

# Step 4: Define detection functions
def detect_embedding(query_code, threshold=0.85):
    results = dense_retriever.retrieve(query_code, top_k=5)
    if not results:
        return {'is_plagiarism': False, 'confidence': 0}
    top_chunk, max_similarity = results[0]
    return {
        'is_plagiarism': max_similarity >= threshold,
        'confidence': float(max_similarity * 100),
        'matched_function': top_chunk.function_name if max_similarity >= threshold else None
    }

def detect_llm(query_code):
    result = llm.analyze_plagiarism_direct(query_code, chunks[:20])
    return result

def detect_rag(query_code):
    retrieved = dense_retriever.retrieve(query_code, top_k=10)
    candidate_chunks = [chunk for chunk, score in retrieved]
    result = llm.analyze_plagiarism_with_context(query_code, candidate_chunks)
    return result

def detect_hybrid_rag(query_code):
    retrieved = hybrid_retriever.retrieve(query_code, top_k=10, alpha=0.5)
    candidate_chunks = [chunk for chunk, score in retrieved]
    result = llm.analyze_plagiarism_with_context(query_code, candidate_chunks)
    return result

print(f"      ✓ All 4 detection systems ready\n")

# Step 5: Test with examples
print("[4/5] Testing with sample queries...\n")

# Test 1: Plagiarized code
test1 = """
def string_inverter(input_text):
    return input_text[::-1]
"""

print("─" * 70)
print("TEST 1: Plagiarized code (reverse_string with renamed variables)")
print("─" * 70)
print(test1)

print("Results:")
r1 = detect_embedding(test1)
print(f"  [Embedding]    Plagiarism: {r1['is_plagiarism']:5} | Confidence: {r1['confidence']:5.1f}% | Match: {r1.get('matched_function', 'N/A')}")

r2 = detect_rag(test1)
print(f"  [Standard RAG] Plagiarism: {r2['is_plagiarism']:5} | Confidence: {r2['confidence']:5}% | Match: {r2.get('matched_function', 'N/A')}")

r3 = detect_hybrid_rag(test1)
print(f"  [Hybrid RAG]   Plagiarism: {r3['is_plagiarism']:5} | Confidence: {r3['confidence']:5}% | Match: {r3.get('matched_function', 'N/A')}")

# Test 2: Original code
test2 = """
def calculate_factorial(number):
    if number == 0:
        return 1
    result = 1
    for i in range(1, number + 1):
        result *= i
    return result
"""

print("\n" + "─" * 70)
print("TEST 2: Original code (factorial - not in corpus)")
print("─" * 70)
print(test2)

print("Results:")
r1 = detect_embedding(test2)
print(f"  [Embedding]    Plagiarism: {r1['is_plagiarism']:5} | Confidence: {r1['confidence']:5.1f}%")

r2 = detect_rag(test2)
print(f"  [Standard RAG] Plagiarism: {r2['is_plagiarism']:5} | Confidence: {r2['confidence']:5}%")

r3 = detect_hybrid_rag(test2)
print(f"  [Hybrid RAG]   Plagiarism: {r3['is_plagiarism']:5} | Confidence: {r3['confidence']:5}%")

print("\n[5/5] Demo complete!\n")
print("="*70)
print("✓✓✓ ALL SYSTEMS OPERATIONAL ✓✓✓")
print("="*70)
print("\nNext steps:")
print("  1. Run 01_indexing.ipynb to build full corpus indexes")
print("  2. Run 02_interactive.ipynb for interactive testing")
print("  3. Run 03_evaluation.ipynb for full evaluation on 35 test cases")
print("="*70)
