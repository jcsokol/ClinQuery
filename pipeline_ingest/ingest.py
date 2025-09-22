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
    --keep               Produce stats showing mapped+unmapped terms and their frequencies; use this to get a sense of the quality of the resolved ontology mappings.
    --ont_corr PATH      Path to ontology correction yml file (default: ./pipeline_ingest/db/ontology_corrections.yml).    
    --ner-model DIR      Directory containing the NER model (default: ./pipeline_ingest/db/ner_model).
    --rel-model DIR      Directory containing the relation extraction model (default: ./pipeline_ingest/db/rel_model).
    --no_pruning         Flag to disable vocab pruning. Resulting database will be large (>1GB) but more expressive.

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
    - no network calls; all outputs are local
    
Action items for the near future:
    - Enable users to plug in their own structured extractions (e.g. for users who have used LLMs for structured entity+timestamp+negation extractions). 
    - The admission date is currently extracted from the records, but I want to let users optionally provide it in the JSON input to override the record-derived value.
    
Additional notes:
    - If you want good term recall you will need to manually tune the ontology alias->term mappings within db/ontology_corrections.yml. Tune this for your own data using the stats you can get with --keep. 

"""

import argparse
import logging
import sys
from pathlib import Path

from logging_setup import setup_logging
from pipeline_ingest import ner, relations
from pipeline_ingest.normalize_and_write import Normalizer

log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Batch ingest EHR notes JSONL → SQL DB/TABLE")
    p.add_argument("jsonl_path", type=Path, help="Input raw notes JSONL")
    p.add_argument("out_path", type=Path, help="Output workspace directory (intermediates + DB/CSV)")
    p.add_argument("--to_csv", action="store_true", help="Write normalized table to CSV instead of SQL DB (OK for small data)")
    p.add_argument("--keep", action="store_true", help="Keep intermediate files")
    p.add_argument(
        "--ner-model",
        dest="ner_model_path",
        type=Path,
        default=Path("./pipeline_ingest/db/ner_model"),
        help="NER model directory (default: ./pipeline_ingest/db/ner_model)",
    )
    p.add_argument(
        "--rel-model",
        dest="rel_model_path",
        type=Path,
        default=Path("./pipeline_ingest/db/rel_model"),
        help="REL model directory (default: ./pipeline_ingest/db/rel_model)",
    )
    p.add_argument(
        "--ont_corr",
        dest="ont_corr",
        type=Path,
        help="ontology correction yml file directory",
        default=Path("./pipeline_ingest/db/ontology_corrections.yml"),
    )
    p.add_argument("--mrconso_rrf", dest="mrconso_rrf", type=Path, help="mrconso.rrf file directory", required=True)
    p.add_argument("--mrrel_rrf", dest="mrrel_rrf", type=Path, help="mrrel.rrf file directory", required=True)
    p.add_argument("--mrsty_rrf", dest="mrsty_rrf", type=Path, help="mrsty.rrf file directory", required=True)
    p.add_argument(
        "--no_pruning",
        action="store_true",
        help="by default the ontology is pruned to only conain terms in dataset; call --no_pruning to disable this pruning",
    )
    return p.parse_args()


def main() -> int:
    setup_logging("INFO")
    args = parse_args()
    if not args.jsonl_path.exists():
        log.error(f"Input not found: {args.jsonl_path}")
        raise SystemExit(1)
    if not args.ner_model_path.exists():
        log.error(f"NER model path does not exist: {args.ner_model_path}")
        raise SystemExit(1)
    if not args.rel_model_path.exists():
        log.error(f"REL model path does not exist: {args.rel_model_path}")
        raise SystemExit(1)
    if not args.mrconso_rrf.exists():
        log.error(f"mrconso.rrf path does not exist: {args.mrconso_rrf}")
        raise SystemExit(1)
    if not args.mrrel_rrf.exists():
        log.error(f"[ingest] mrrel.rrf path does not exist: {args.mrrel_rrf}")
        raise SystemExit(1)
    if not args.mrsty_rrf.exists():
        log.error(f"mrsty.rrf path does not exist: {args.mrsty_rrf}")
        raise SystemExit(1)
    if not args.ont_corr.exists():
        log.error(f"ontology correction file does not exist: {args.ont_corr}")
        raise SystemExit(1)

    workdir = args.out_path
    workdir.mkdir(parents=True, exist_ok=True)

    ner_rel_out = workdir / "ner_rel_predictions.jsonl"
    rel_tmp_out = workdir / "ner_rel_predictions.tmp.jsonl"
    term_stats_out = workdir / "term_stats_resolved.csv", workdir / "term_stats_unresolved.csv"
    csv_out = workdir / "db.csv"

    log.info("Stage 1: NER …")
    ner.run_file(
        str(args.jsonl_path), str(ner_rel_out), str(args.ner_model_path)
    )  # also conducts file checks on args.jsonl_path to ensure it has all required fields and that all uids are unique

    log.info("Stage 2: RELATIONS …")
    relations.run_file(str(ner_rel_out), str(rel_tmp_out), str(args.rel_model_path))
    rel_tmp_out.replace(ner_rel_out)

    log.info("Stage 3: NORMALIZE …")
    norm = Normalizer(
        str(args.mrconso_rrf),
        str(args.mrrel_rrf),
        str(args.mrsty_rrf),
        str(args.ont_corr),
        keep=args.keep,
        no_pruning=args.no_pruning,
    )
    norm.normalize(
        in_jsonl=str(ner_rel_out), term_stats_csv=str(term_stats_out)
    )  # when args.keep==False will not produce term_stats_out

    log.info("Stage 4: DB WRITE …")
    norm.write_db(str(csv_out), str(workdir), args.to_csv)

    # now remove intermediary files if user does not need them
    if not args.keep:
        try:
            if ner_rel_out.exists():
                ner_rel_out.unlink()
            if rel_tmp_out.exists():
                rel_tmp_out.unlink()
        except Exception:
            pass

    log.info("Done")

    return 0


if __name__ == "__main__":
    sys.exit(main())
