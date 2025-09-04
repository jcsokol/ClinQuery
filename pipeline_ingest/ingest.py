#!/usr/bin/env python3
"""
ingest.py — Batch pipeline for converting raw EHR notes into structured outputs.

This script processes a JSONL file of raw notes through four stages:
1. Named Entity Recognition (NER)
2. Relation Extraction (REL)
3. Normalization (NORM)
4. Database/CSV export

Usage:
	python ingest.py JSONL_IN OUT_DIR \
		--mrconso.rrf PATH --mrrel.rrf PATH --mrsty.rrf PATH \
		[--to_csv] [--keep] [--ner-model DIR] [--rel-model DIR]

Arguments:
	JSONL_IN   Input JSONL file of raw notes.
	OUT_DIR    Output directory for intermediates and final artifacts.

Options:
	--mrconso.rrf PATH   Path to UMLS mrconso.rrf file (required).
	--mrrel.rrf PATH     Path to UMLS mrrel.rrf file (required).
	--mrsty.rrf PATH     Path to UMLS mrsty.rrf file (required).
	--to_csv             Export the normalized table to CSV (default: SQL database + alias embeddings for query engine).
	--keep               Preserve intermediate files.
	--ner-model DIR      Directory containing the NER model (default: ./db/ner_model).
	--rel-model DIR      Directory containing the relation extraction model (default: ./db/rel_model).

Input Format:
	Each line in JSONL_IN must be a JSON object with:
		- 'uid'       Unique identifier for the note (required).
		- 'raw_text'  Free-text clinical note (required).
		- 'name'      Patient or case name (required).
		- 'filename'  Source filename (optional).

Outputs:
	- Intermediate JSONL files with NER and REL annotations.
	- Term statistics (CSV, optional).
	- SQL database + embedded aliases for query engine / csv database if --keep is invoked.
	
Action items for the near future:
	- Enable users to plug in their own structured extractions (e.g. for users who have used LLMs for structured entity+timestamp+negation extractions). 
"""

import argparse, sys
from pathlib import Path
import ner
import relations
import normalize
import sql_writer

def parse_args():
	p = argparse.ArgumentParser(description="Batch ingest EHR notes JSONL → SQL DB/TABLE")
	p.add_argument("jsonl_path", type=Path, help="Input raw notes JSONL")
	p.add_argument("out_path", type=Path, help="Output workspace directory (intermediates + DB/CSV)")
	p.add_argument("--to_csv", action="store_true", help="Write normalized table to CSV instead of SQL DB (OK for small data)")
	p.add_argument("--keep", action="store_true", help="Keep intermediate files")
	p.add_argument("--ner-model", dest="ner_model_path", type=Path, default=Path("./db/ner_model"), help="NER model directory (default: ./db/ner_model)")
	p.add_argument("--rel-model", dest="rel_model_path", type=Path, default=Path("./db/rel_model"), help="REL model directory (default: ./db/rel_model)")
	p.add_argument("--mrconso_rrf", dest="mrconso_rrf", type=Path, help="mrconso.rrf file directory", required=True)
	p.add_argument("--mrrel_rrf", dest="mrrel_rrf", type=Path, help="mrrel.rrf file directory", required=True)
	p.add_argument("--mrsty_rrf", dest="mrsty_rrf", type=Path, help="mrsty.rrf file directory", required=True)
	return p.parse_args()

def main() -> int:
	args = parse_args()
	if not args.jsonl_path.exists():
		print(f"[ingest] Input not found: {args.jsonl_path}", file=sys.stderr)
		return 1
	if not args.ner_model_path.exists():
		print(f"[ingest] NER model path does not exist: {args.ner_model_path}", file=sys.stderr)
		return 1
	if not args.rel_model_path.exists():
		print(f"[ingest] REL model path does not exist: {args.rel_model_path}", file=sys.stderr)
		return 1 
	if not args.mrconso_rrf.exists():
		print(f"[ingest] mrconso.rrf path does not exist: {args.mrconso_rrf}", file=sys.stderr)
		return 1         
	if not args.mrrel_rrf.exists():
		print(f"[ingest] mrrel.rrf path does not exist: {args.mrrel_rrf}", file=sys.stderr)
		return 1          
	if not args.mrsty_rrf.exists():
		print(f"[ingest] mrsty.rrf path does not exist: {args.mrsty_rrf}", file=sys.stderr)
		return 1  
								
	workdir = args.out_path
	workdir.mkdir(parents=True, exist_ok=True)
	
	ner_rel_out = workdir / "ner_rel_predictions.jsonl"
	rel_tmp_out = workdir / "ner_rel_predictions.tmp.jsonl"
	term_stats_out = workdir / "term_stats.csv"
	csv_out = workdir / "db.csv"  
	
	print("[ingest] Stage 1: NER …")
	ner.run_file(str(args.jsonl_path), str(ner_rel_out), str(args.ner_model_path)) # also conducts file checks on args.jsonl_path to ensure it has all required fields and that all uids are unique

	print("[ingest] Stage 2: RELATIONS …")
	relations.run_file(str(ner_rel_out), str(rel_tmp_out), str(args.rel_model_path))
	rel_tmp_out.replace(ner_rel_out)

	print("[ingest] Stage 3: NORMALIZE …")
	norm_out = normalize.run_file(str(ner_rel_out), str(term_stats_out), str(args.mrconso_rrf), str(args.mrrel_rrf), str(args.mrsty_rrf), args.keep) # args.keep==False will not produce term_stats_out

	print("[ingest] Stage 4: DB WRITE …")
	sql_writer.to_db(norm_out, str(csv_out), str(workdir), args.to_csv) # will write to csv if args.to_csv==True else will write to sql db
	
	# now remove intermediary files if user does not need them
	if not args.keep:
		try:
			if ner_rel_out.exists():
				ner_rel_out.unlink()
			if rel_tmp_out.exists():
				rel_tmp_out.unlink()
		except:
			pass
			
	print(f"[ingest] Done")
			
	return 0
	
if __name__ == "__main__":
	main()