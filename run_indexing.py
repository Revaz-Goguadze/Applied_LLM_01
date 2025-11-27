"""
Execute 01_indexing.ipynb functionality as a Python script.
This builds all indexes for the plagiarism detection system.
"""

import sys
import os
import numpy as np
from tqdm import tqdm
import pickle
import json

# Add src to path
sys.path.append(os.path.abspath('.'))

from src.chunking import PythonFunctionExtractor, CodeChunk
from src.embeddings import EmbeddingGenerator
from src.retrieval import DenseRetriever, BM25Retriever
from src.config import RANDOM_SEED

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

print("\n" + "="*70)
print("01_INDEXING - Building All Indexes")
print("="*70)

# Step 1: Extract Functions from Reference Corpus
print("\n[Step 1/5] Extracting functions from reference corpus...")
print("-" * 70)

extractor = PythonFunctionExtractor(min_lines=3, max_lines=500)
reference_corpus_dir = "data/reference_corpus"

print(f"Scanning directory: {reference_corpus_dir}")
all_chunks = extractor.extract_from_directory(reference_corpus_dir)

print(f"\n✓ Extracted {len(all_chunks)} functions from reference corpus")
print(f"\nSample functions:")
for i, chunk in enumerate(all_chunks[:5]):
    print(f"  {i+1}. {chunk.function_name} from {chunk.file_path}")

# Step 2: Generate Embeddings for All Functions
print("\n[Step 2/5] Generating embeddings for all functions...")
print("-" * 70)

embedding_gen = EmbeddingGenerator()
code_texts = [chunk.content for chunk in all_chunks]

print(f"Generating embeddings for {len(code_texts)} functions...")
print("This may take several minutes...\n")

# Generate embeddings with progress bar
embeddings = embedding_gen.embed_batch(code_texts, batch_size=50, show_progress=True)

# Convert to numpy array
embeddings_matrix = np.array(embeddings)

print(f"\n✓ Generated embeddings with shape: {embeddings_matrix.shape}")

# Step 3: Build Dense Retriever Index
print("\n[Step 3/5] Building dense retriever index...")
print("-" * 70)

dense_retriever = DenseRetriever(all_chunks, embeddings_matrix)

# Save to disk
os.makedirs("indexes", exist_ok=True)
dense_retriever.save("indexes/dense_retriever.pkl")

print("✓ Dense retriever saved to indexes/dense_retriever.pkl")

# Step 4: Build BM25 Index
print("\n[Step 4/5] Building BM25 index...")
print("-" * 70)

print("Building BM25 index...")
bm25_retriever = BM25Retriever(all_chunks)

# Save to disk
bm25_retriever.save("indexes/bm25_retriever.pkl")

print("✓ BM25 retriever saved to indexes/bm25_retriever.pkl")

# Step 5: Save Metadata
print("\n[Step 5/5] Saving metadata...")
print("-" * 70)

metadata = {
    "num_functions": len(all_chunks),
    "embedding_dimension": embeddings_matrix.shape[1],
    "reference_corpus_dir": reference_corpus_dir,
    "repositories": [
        "algorithms (TheAlgorithms/Python)",
        "string_utils (text processing)",
        "data_structures (linked list)",
        "math_utils (statistics)",
        "file_utils (file operations)",
        "sorting_algos (sorting algorithms)"
    ]
}

with open("indexes/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("✓ Metadata saved to indexes/metadata.json")

# Step 6: Test Index Loading (Verification)
print("\n[Step 6/6] Testing index loading (verification)...")
print("-" * 70)

print("Loading indexes to verify...\n")

dense_test = DenseRetriever.load("indexes/dense_retriever.pkl")
print(f"✓ Dense retriever loaded: {len(dense_test.chunks)} chunks")

bm25_test = BM25Retriever.load("indexes/bm25_retriever.pkl")
print(f"✓ BM25 retriever loaded: {len(bm25_test.chunks)} chunks")

# Test retrieval with a sample query
test_query = "def reverse_string(s): return s[::-1]"
print(f"\nTest query: {test_query}")

dense_results = dense_test.retrieve(test_query, top_k=3)
print(f"\nTop-3 Dense Retrieval Results:")
for i, (chunk, score) in enumerate(dense_results):
    print(f"  {i+1}. {chunk.function_name} (similarity: {score:.4f})")

bm25_results = bm25_test.retrieve(test_query, top_k=3)
print(f"\nTop-3 BM25 Retrieval Results:")
for i, (chunk, score) in enumerate(bm25_results):
    print(f"  {i+1}. {chunk.function_name} (score: {score:.4f})")

print("\n" + "="*70)
print("INDEXING COMPLETE!")
print("="*70)
print("\nMetadata Summary:")
print(json.dumps(metadata, indent=2))
print("\n" + "="*70)
print("All indexes built and verified successfully!")
print("Ready to use in 02_interactive.ipynb")
print("="*70)
