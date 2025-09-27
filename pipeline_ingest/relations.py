"""
relations.py — Relation extraction inference for annotated clinical notes.

This module loads a fine-tuned sequence classifier and predicts relations between pre-extracted entity spans (e.g., TIME_RELATION, NEGATION_RELATION).
Pipeline:
  1) Read JSONL with fields: uid, name, text, tables, **spans**.
  2) Chunk text for long-sequence handling.
  3) Generate span pairs that match allowed (source, target, relation) patterns.
  4) Score each pair; keep high-confidence relations.
  5) Write updated JSONL with a 'relations' list per entry.

Debug XML-tagged strings can be collected for inspection if needed.
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import logging
from itertools import product
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

log = logging.getLogger(__name__)


# ---------------- Public API ----------------


def run_file(in_jsonl: str, out_jsonl: str, model_dir: str, prob_threshold: float = 0.9, max_token_distance: int = 100) -> None:
    """
    Main entrypoint for running relations inference.

    Loads a trained model and tokenizer, parses input JSONL notes with predicted spans, predicts relations, and writes annotated out_jsonl to disk.
    """

    # ensure input jsonl and model paths exist
    in_path = Path(in_jsonl)
    out_path = Path(out_jsonl)
    model_path = Path(model_dir)
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {in_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model dir not found: {model_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # load model and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"gpu: {torch.cuda.is_available()}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model = model.to(device)

    # generate relations predictions
    id2labels = {0: "NEGATION_RELATION", 1: "NO_RELATION", 2: "TIME_RELATION"}
    label2id = {v: k for k, v in id2labels.items()}
    allowed_relations = [
        ("C_ENT", "TIME", "TIME_RELATION"),
        ("TABLE", "TIME", "TIME_RELATION"),
        ("C_ENT", "NEGATION", "NEGATION_RELATION"),
    ]
    with open(in_path) as f:
        entries = [json.loads(line) for line in f]
    with open(out_path, "w") as out_f:
        total = len(entries)
        checkpoints = {int(total * 0.15): "15%", int(total * 0.50): "50%", int(total * 0.75): "75%"}
        for i, entry in enumerate(entries, start=1):
            if i in checkpoints:
                log.info(f"Progress: {checkpoints[i]} ({i}/{total} entries)")
            if not {"uid", "name", "text", "tables", "spans"}.issubset(entry.keys()):
                raise ValueError("uid,name,text,tables,spans fields all need to be present in every jsonl line")
            relations, xml_tagged_texts, xml_tagged_texts_no_relations = predict_relations_with_chunking(
                entry["text"],
                entry["spans"],
                model,
                tokenizer,
                label2id,
                id2labels,
                allowed_relations,
                device,
                prob_threshold=0.9,
                max_token_distance=max_token_distance,
            )
            relations_filtered = filter_relations_for_predictions(relations)
            entry["relations"] = relations_filtered
            out_f.write(json.dumps(entry) + "\n")


# ---------------- Internal helpers ----------------


def chunk_text_for_predictions(text: str, tokenizer, max_length=512, overlap=0.50, pred_window=0.80):
    """
    Split long text into overlapping token windows for inference.

    Produces chunk metadata with token offsets and a 'middle_range' index window
    to reduce edge effects when merging chunk predictions.

    Args:
        text: Raw input text.
        tokenizer: Hugging Face tokenizer.
        max_length: Max tokens per chunk (including specials).
        overlap: Fractional overlap between consecutive chunks (0..1).
        pred_window: Fraction of each chunk considered the 'middle' for scoring.

    Returns:
        List of dicts with keys: 'input_ids', 'offsets', 'start_token', 'middle_range'.
    """
    tokens = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    input_ids = tokens["input_ids"]
    offsets = tokens["offset_mapping"]
    stride = int(max_length * (1 - overlap))
    chunks = []

    for start in range(0, len(input_ids), stride):
        end = min(start + max_length, len(input_ids))
        chunk_ids = input_ids[start:end]
        chunk_offsets = offsets[start:end]

        # Ensure middle range is based on current chunk size (not always max_length)
        chunk_length = end - start
        margin = int(chunk_length * (1 - pred_window) / 2)
        middle_start = margin
        middle_end = chunk_length - margin

        chunks.append({"input_ids": chunk_ids, "offsets": chunk_offsets, "start_token": start, "middle_range": (middle_start, middle_end)})

        if end == len(input_ids):
            break

    return chunks


def spans_in_middle_range_for_predictions(spans: list[dict], offsets, middle_range: tuple[int, int], is_start, is_end) -> list[dict]:
    """
    Filter spans to those whose token positions land inside the given middle_range,
    relaxing the lower/upper bound for the first/last chunks.

    Args:
        spans (list[dict]): Candidate spans with "start" and "end" character offsets (global text).
        offsets (list[tuple[int,int]]): Token offsets (char start,end) for the chunk.
        middle_range (tuple[int,int]): (start_idx, end_idx) token indices within chunk to keep.
        is_start (bool): If True, allow tokens from the chunk start.
        is_end (bool): If True, allow tokens through the chunk end.

    Returns:
        list[dict]: Spans that fall within the trusted region of the chunk.
    """
    selected = []
    start_idx, end_idx = middle_range
    if is_start:
        start_idx = 0
    if is_end:
        end_idx = float("inf")

    for span in spans:
        # Find the first token whose offset overlaps with span start
        for i, (start, end) in enumerate(offsets):
            if start is None or end is None:
                continue
            if start <= span["start"] < end or start < span["end"] <= end or (span["start"] <= start and end <= span["end"]):
                if start_idx <= i <= end_idx:
                    selected.append(span)
                break
    return selected


def generate_allowed_span_pairs_for_predictions(spans: list[dict], allowed_relations: list[tuple[str, str, str]]) -> list[tuple[dict, dict, str]]:
    """
    Generate ordered span pairs that match allowed type patterns.

    Args:
        spans (list[dict]): Candidate spans (each with "id" and "label").
        allowed_relations (list[tuple[str,str,str]]): Triples of (source_type, target_type, rel_label).

    Returns:
        list[tuple[dict, dict, str]]: (head_span, child_span, rel_label) triples for scoring.
    """
    span_pairs = []
    for a, b in product(spans, spans):
        if a["id"] == b["id"]:
            continue
        for source_type, target_type, rel_label in allowed_relations:
            if a["label"] == source_type and b["label"] == target_type:
                span_pairs.append((a, b, rel_label))
    return span_pairs


def predict_relations_with_chunking(
    text: str,
    spans: list[dict],
    model,
    tokenizer,
    label2id: dict[str, int],
    id2label: dict[int, str],
    allowed_relations: list[tuple[str, str, str]],
    device,
    prob_threshold=0.6,
    max_token_distance=100,
):
    """
    Score candidate (HEAD, CHILD) pairs inside overlapping chunks, enforcing a maximum
    token distance and trusting only the central region per chunk to reduce edge effects.

    Args:
        text (str): Original document text.
        spans (list[dict]): Detected entity spans with global char offsets and "id"/"label".
        model: Trained relation classifier (returns logits).
        tokenizer: Matching tokenizer.
        label2id (dict[str,int]): Mapping label -> class index (unused here but kept for symmetry).
        id2label (dict[int,str]): Mapping class index -> label string.
        allowed_relations (list[tuple[str,str,str]]): Allowed (source_type, target_type, label) patterns.
        device (str): "cpu" or "cuda".
        prob_threshold (float): Minimum softmax probability to accept a prediction.
        max_token_distance (int): Discard candidate pairs farther than this distance.

    Returns:
        tuple:
            - all_rels (list[dict]): Accepted relations with {"head","child","label","confidence"}.
            - xml_texts (list[str]): Example tagged strings for accepted relations.
            - xml_texts_no_relations (list[str]): Example tagged strings predicted as NO_RELATION (confident).
    """
    model.to(device)
    all_rels = []
    xml_texts = []
    xml_texts_no_relations = []

    chunks = chunk_text_for_predictions(text, tokenizer)

    for chunk in chunks:
        chunk_offsets = chunk["offsets"]
        middle_range = chunk["middle_range"]
        chunk_start = chunk_offsets[0][0]
        chunk_end = chunk_offsets[-1][1]
        if chunks.index(chunk) == len(chunks) - 1:
            chunk_end = len(text)

        is_start = chunk_start == 0
        is_end = chunk_offsets[-1][1] >= len(text) - 5 or chunks.index(chunk) == len(chunks) - 1

        spans_in_chunk = [span for span in spans if chunk_start <= span["start"] < chunk_end]
        selected_spans = spans_in_middle_range_for_predictions(spans_in_chunk, chunk_offsets, middle_range, is_start, is_end)
        pairs = generate_allowed_span_pairs_for_predictions(selected_spans, allowed_relations)

        chunk_text = text[chunk_start:chunk_end]

        for head, child, _ in pairs:
            # Convert global character offsets to chunk-local positions
            head_start = head["start"] - chunk_start
            head_end = head["end"] - chunk_start
            child_start = child["start"] - chunk_start
            child_end = child["end"] - chunk_start

            if head_start < 0 or head_end > len(chunk_text) or child_start < 0 or child_end > len(chunk_text):
                continue

            if head_start < child_start:
                tagged = (
                    chunk_text[:head_start]
                    + "<HEAD>"
                    + chunk_text[head_start:head_end]
                    + "</HEAD>"
                    + chunk_text[head_end:child_start]
                    + "<CHILD>"
                    + chunk_text[child_start:child_end]
                    + "</CHILD>"
                    + chunk_text[child_end:]
                )
            else:
                tagged = (
                    chunk_text[:child_start]
                    + "<CHILD>"
                    + chunk_text[child_start:child_end]
                    + "</CHILD>"
                    + chunk_text[child_end:head_start]
                    + "<HEAD>"
                    + chunk_text[head_start:head_end]
                    + "</HEAD>"
                    + chunk_text[head_end:]
                )

            tok = tokenizer(tagged, return_offsets_mapping=True, add_special_tokens=True)
            offsets = tok["offset_mapping"]

            def char_to_token_index(pos, offsets=offsets):
                for idx, (start, end) in enumerate(offsets):
                    if start <= pos < end:
                        return idx
                return None

            idx1 = char_to_token_index(head_start)
            idx2 = char_to_token_index(child_start)
            if idx1 is None or idx2 is None:
                continue
            if abs(idx1 - idx2) > max_token_distance:
                continue

            enc = tokenizer(tagged, return_tensors="pt", truncation=True, padding=True, max_length=512, add_special_tokens=True).to(device)

            with torch.no_grad():
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1)
                pred = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred].item()

            label = id2label[pred]
            if label == "TIME_RELATION" and confidence > prob_threshold:
                if head["label"] in {"C_ENT", "TABLE"} and child["label"] == "TIME":
                    xml_texts.append(tagged)
                    all_rels.append({"head": head["id"], "child": child["id"], "label": label, "confidence": round(confidence, 3)})
            elif label == "NEGATION_RELATION" and confidence > prob_threshold:
                if head["label"] == "C_ENT" and child["label"] == "NEGATION":
                    xml_texts.append(tagged)
                    all_rels.append({"head": head["id"], "child": child["id"], "label": label, "confidence": round(confidence, 3)})
            elif label == "NO_RELATION" and confidence > prob_threshold:
                xml_texts_no_relations.append(tagged)

    return all_rels, xml_texts, xml_texts_no_relations


def filter_relations_for_predictions(relations: list[dict]) -> list[dict]:
    """
    Reduce duplicate/competing relation predictions by:
      1) keeping the highest-confidence record for each (head, child, label),
      2) then keeping only the single best label for each (head, child).

    Args:
        relations (list[dict]): Items with "head", "child", "label", and "confidence".

    Returns:
        list[dict]: Deduplicated, best-per-pair relation predictions.
    """
    # Step 1: Keep highest-confidence for each (head, child, label)
    relation_map: dict[tuple[str, str, str], dict] = {}
    for rel in relations:
        key = (rel["head"], rel["child"], rel["label"])
        if key not in relation_map or rel["confidence"] > relation_map[key]["confidence"]:
            relation_map[key] = rel

    # Step 2: For each (head, child), keep only the label with max confidence
    best_by_pair: dict[tuple[str, str], dict] = {}
    for (head, child, _), rel in relation_map.items():
        pair_key = (head, child)
        if pair_key not in best_by_pair or rel["confidence"] > best_by_pair[pair_key]["confidence"]:
            best_by_pair[pair_key] = rel

    return list(best_by_pair.values())


# ---------------- CLI to run this script on its own ----------------


def _parse_args_cli():
    """
    Parse command-line arguments for running relations model as a standalone script.
    """
    import argparse

    p = argparse.ArgumentParser(description="Run relations model over a JSONL file and write relations predictions.")
    p.add_argument("jsonl_in", type=Path)
    p.add_argument("jsonl_out", type=Path)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--prob", type=float, default=0.9)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args_cli()
    from logging_setup import setup_logging

    setup_logging("INFO")
    run_file(str(args.jsonl_in), str(args.jsonl_out), str(args.model), prob_threshold=args.prob)
