"""
ner.py — Named Entity Recognition (NER) inference pipeline for clinical notes.

This script loads a trained token classification model and applies it to EHR notes stored in JSONL format. It performs the following steps:

1. Preprocess input notes by extracting pipe-delimited tables and replacing them with a '[TABLE]' marker in the free text.
2. Run NER inference with a fine-tuned transformer model to identify entities.
3. Post-process predictions to merge overlapping spans and filter out noise.
4. Write the enriched notes (with entity spans and parsed tables) back to JSONL.

Intended to be called by ingest.py, but can also be run as a standalone function.
"""

import json
import logging
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

log = logging.getLogger(__name__)


# ---------------- Public API ----------------


def run_file(in_jsonl: str, out_jsonl: str, model_dir: str, prob_threshold: float = 0.6) -> None:
    """
    Main entrypoint for running NER inference.

    Loads a trained model and tokenizer, parses input JSONL notes (with raw ehr texts containing pipe-delimited tables),
    predicts entity spans, and writes annotated out_jsonl to disk.
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

    # separate tables from texts (assumes all table are pipe-delimited), and ensure input json is valid
    _extract_tables(in_path, out_path)

    # load model and set device
    id2label, label_list = _load_model_labels(model_dir)
    model = WeightedTokenClassificationModel("UFNLP/gatortron-base", label_list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"gpu: {torch.cuda.is_available()}")

    model.load_state_dict(torch.load(model_path / "model_weights.pt"))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = model.to(device)

    # generate NER predictions
    with open(out_path) as f:
        lines = [json.loads(line) for line in f]
    with open(out_path, "w") as out:
        for example in lines:
            spans = predict_entities(example["text"], model, tokenizer, id2label, device=device, prob_threshold=prob_threshold)
            spans = filter_overlapping_spans_for_predictions(spans=spans, min_char_length=1)
            example["spans"] = spans
            out.write(json.dumps(example) + "\n")


# ---------------- Internal helpers ----------------


def _load_model_labels(model_dir):
    """
    Load label mappings saved during training.

    Expects labels.json with 'id2label' (dict of index→label) and 'label_list' (ordered list).
    Returns (id2label, label_list).
    """
    with open(Path(model_dir) / "labels.json", encoding="utf-8") as f:
        data = json.load(f)
    id2label = {int(k): v for k, v in data["id2label"].items()}
    label_list = data["label_list"]
    return id2label, label_list


def _extract_tables(in_path, out_path):
    """
    Parse EHR notes to separate pipe-delimited tables from free text.

    Stores a new jsonl file in out_path that contains a 'text' and 'tables' entry for each line. The 'text' entry contains the raw text with all tables replaced with a '[TABLE]' marker. The 'tables' entry contains a list of parsed tables in a json format.
    """

    def parse_ehr_string(input_EHR_string):
        json_dicts_output_list = []
        output_EHR_string = ""
        _skip_index = False
        table_str = ""
        for input_i in range(len(input_EHR_string)):
            ch = input_EHR_string[input_i]
            if ch != "|" and len(table_str) == 0:
                output_EHR_string += ch
            else:
                table_str += ch
                # need two chars ahead: positions input_i+1 and input_i+2
                if input_i + 2 < len(input_EHR_string) and ch == "|" and input_EHR_string[input_i + 1] == "\n" and input_EHR_string[input_i + 2] != "|":
                    json_dicts_output_list.append(parse_txt_table_to_json(table_str))
                    table_str = ""
                    output_EHR_string += "[TABLE]"
        return (output_EHR_string, json_dicts_output_list)

    def parse_txt_table_to_json(input_table_string):
        # convert table string to list of lines
        lines = [line.strip() for line in input_table_string.strip().split("\n") if line.strip()]
        # filter out lines that are just separator rows filled with {'|','-',' '} characters
        lines = [line for line in lines if not set(line.strip()) <= {"|", "-", " "}]
        # split headers
        header = [h.strip() for h in lines[0].split("|") if h.strip()]
        # finally create json dict
        json_output = []
        malformed_row_counter = 0
        for row in lines[1:]:
            values = [col.strip() for col in row.split("|")][1:-1]
            if len(values) == len(header):
                entry = dict(zip(header, values, strict=False))
                json_output.append(entry)
            else:
                malformed_row_counter += 1
        # print('# of malformed rows: '+str(malformed_row_counter))

        if malformed_row_counter == 0:
            return json_output
        else:
            return None

    with open(in_path) as f:
        lines = [json.loads(line) for line in f]
    uids_set = set()
    with open(out_path, "w") as out:
        for example in lines:
            if not {"uid", "name", "raw_text"}.issubset(example.keys()):
                raise ValueError("uid, name, or raw_text key not present in at least one example in input jsonl")
            example["filename"] = example.get("filename")  # sets to None if filename not present
            uids_set.add(example["uid"])
            example["text"], example["tables"] = parse_ehr_string(example["raw_text"])
            out.write(json.dumps(example) + "\n")
    if len(uids_set) != len(lines):
        raise ValueError("uids must be unique")


class WeightedTokenClassificationModel(nn.Module):
    """
    Lightweight token classification head on top of a base Transformer.

    Supports per-class weighting to emphasize rare labels during training.
    """

    def __init__(self, base_model_name, label_list, weight_boosts=None):
        super().__init__()
        self.num_labels = len(label_list)
        self.label_list = label_list
        self.weight_boosts = weight_boosts or {}

        self.config = AutoConfig.from_pretrained(base_model_name, num_labels=self.num_labels)
        self.base_model = AutoModel.from_pretrained(base_model_name, config=self.config)

        hidden_size = self.config.hidden_size
        self.classifier = nn.Linear(hidden_size, self.num_labels)

        # Set up loss
        self.loss_fct = nn.CrossEntropyLoss()  # weights added later in forward()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        if labels is not None:
            # Compute per-token class weights
            if self.weight_boosts:
                class_weights = torch.ones(self.num_labels, device=logits.device)
                for i, label in enumerate(self.label_list):
                    if label in self.weight_boosts:
                        class_weights[i] = self.weight_boosts[label]
                loss_fct = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}


# chunk texts for predictions
def chunk_text_for_predictions(text, tokenizer, max_length=512, overlap_ratio=0.25):
    """
    Split a long text into overlapping sub-sequences for NER inference.

    Returns chunks with input_ids, character offsets, and chunk text.
    """
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False, return_attention_mask=False)

    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    stride = int(max_length * (1 - overlap_ratio))
    chunks = []

    for start_idx in range(0, len(input_ids), stride):
        end_idx = start_idx + max_length
        chunk_input_ids = input_ids[start_idx:end_idx]
        chunk_offsets = offsets[start_idx:end_idx]

        if not chunk_input_ids:
            continue

        # Extract chunk text from first to last character span
        char_start = chunk_offsets[0][0]
        char_end = chunk_offsets[-1][1]
        chunk_text = text[char_start:char_end]

        chunks.append(
            {
                "input_ids": chunk_input_ids,
                "offsets": chunk_offsets,  # still relative to full text
                "text": chunk_text,  # needed by predict_entities
            }
        )

        if end_idx >= len(input_ids):
            break

    return chunks


# predict entities
def predict_entities(text, model, tokenizer, id2label: dict[int, str], device="cpu", prob_threshold=0.6, max_length=512):
    """
    Run NER model inference on text and return extracted entity spans.

    Uses sliding-window chunking to handle long inputs and merges predictions
    into a list of span dicts (start, end, label, text, confidence, id).
    """
    chunks = chunk_text_for_predictions(text, tokenizer, max_length=max_length)
    spans = []

    for chunk in chunks:
        chunk_text = chunk["text"]
        input_ids = chunk["input_ids"]
        offset_mapping = chunk["offsets"]  # use full-text-relative offsets

        # Determine chunk prediction window
        length = len(input_ids)
        lower = int(length * 0.15)
        upper = int(length * 0.85)

        # Tokenize the **chunk text only**
        inputs = tokenizer(
            chunk_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
            is_split_into_words=False,
            return_offsets_mapping=True,
        )

        chunk_relative_offsets = inputs.pop("offset_mapping").squeeze().tolist()
        chunk_start_char = offset_mapping[0][0]  # where this chunk starts in full text

        offset_mapping = [(s + chunk_start_char, e + chunk_start_char) if s is not None and e is not None else (0, 0) for (s, e) in chunk_relative_offsets]

        is_first_chunk = offset_mapping[0][0] == 0
        is_last_chunk = offset_mapping[-1][1] == len(text)
        middle_start = 0 if is_first_chunk else lower
        middle_end = length if is_last_chunk else upper

        inputs = {k: v.to(device) for k, v in inputs.items()}

        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs["logits"].squeeze(0)
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1).tolist()

        current_span = None
        for i in range(middle_start, min(middle_end, len(predictions))):
            pred_label_idx = predictions[i]
            label = id2label[pred_label_idx]
            confidence = probs[i][pred_label_idx].item()

            # Skip special tokens or padding
            if offset_mapping[i] == [0, 0]:
                continue

            # Get char offsets for current token (within chunk), map to full text
            chunk_token_start, chunk_token_end = offset_mapping[i]
            full_text_start = offset_mapping[i][0]
            full_text_end = offset_mapping[i][1]

            if label.startswith("B-") and confidence >= prob_threshold:
                if current_span:
                    spans.append(current_span)
                entity_type = label[2:]
                current_span = {"start": full_text_start, "end": full_text_end, "label": entity_type, "confidence": confidence}
            elif label.startswith("I-") and current_span and label[2:] == current_span["label"] and confidence >= prob_threshold:
                current_span["end"] = full_text_end
                current_span["confidence"] = max(current_span["confidence"], confidence)
            else:
                if current_span:
                    spans.append(current_span)
                    current_span = None

        if current_span:
            spans.append(current_span)

    # Handle literal '[TABLE]' patterns
    spans = [span for span in spans if span["label"] != "TABLE"]
    for match in re.finditer(r"\[TABLE\]", text):
        spans.append({"start": match.start(), "end": match.end(), "label": "TABLE", "text": "[TABLE]", "confidence": 1.0})

    # Finalize span info and assign IDs
    spans = sorted(spans, key=lambda s: (s["start"], s["end"]))  # Ensure consistent order
    for idx, span in enumerate(spans):
        span["text"] = text[span["start"] : span["end"]]
        span["confidence"] = round(span["confidence"], 3)
        span["id"] = f"e{idx+1}"

    return spans


def filter_overlapping_spans_for_predictions(spans: list[dict], min_char_length: int = 1) -> list[dict]:
    """
    Post-process predicted spans by removing overlaps and very short spans.

    Keeps the highest-confidence spans when conflicts occur.
    """
    # Sort by confidence (highest first)
    spans = sorted(spans, key=lambda x: x.get("confidence", 1.0), reverse=True)

    selected = []
    for span in spans:
        # Skip spans that are too short
        if (span["end"] - span["start"]) < min_char_length:
            continue
        # Check for any overlap with already selected spans
        overlap = False
        for sel in selected:
            if not (span["end"] <= sel["start"] or span["start"] >= sel["end"]):
                overlap = True
                break
        if not overlap:
            selected.append(span)

    return selected


# ---------------- CLI to run this script on its own ----------------


def _parse_args_cli():
    """
    Parse command-line arguments for running NER as a standalone script.
    """
    import argparse

    p = argparse.ArgumentParser(description="Run NER over a JSONL file and write span predictions.")
    p.add_argument("jsonl_in", type=Path)
    p.add_argument("jsonl_out", type=Path)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--prob", type=float, default=0.6)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args_cli()
    from logging_setup import setup_logging

    setup_logging("INFO")
    run_file(str(args.jsonl_in), str(args.jsonl_out), str(args.model), prob_threshold=args.prob)
