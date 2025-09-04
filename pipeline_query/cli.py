#!/usr/bin/env python3
"""
cli.py

Command-line entry point for running natural-language clinical queries
against the system. This script wires up the embedding model,
initializes the QueryCompiler, and delegates execution to `run_query`.

Usage
-----
	./cli.py "How did creatinine trend from admission to discharge?"

The script will print the model's final answer to stdout (or a fallback
message if none was produced).
"""

import argparse, uuid
from main import run_query
from config import settings
from abstraction_layer import QueryCompiler

def load_compiler() -> QueryCompiler:
	from sentence_transformers import SentenceTransformer, models
	word_emb = models.Transformer(settings.params.embedder_model)
	pooling = models.Pooling(word_emb.get_word_embedding_dimension())
	model = SentenceTransformer(modules=[word_emb, pooling], device="cpu")
	embed_fn = lambda s: model.encode([s], normalize_embeddings=True, convert_to_numpy=True)[0].astype("float32")
	return QueryCompiler(settings.params.duckdb_path, settings.params.faiss_path, settings.params.idmap_path, embed_fn)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run a query")
	parser.add_argument("query", help="The question to ask (wrap in quotes)")
	args = parser.parse_args()

	qc = load_compiler()
	result = run_query(user_query=args.query, session_id=str(uuid.uuid4()), query_compiler=qc)
	print(result.get("final_answer") or "No answer generated.")