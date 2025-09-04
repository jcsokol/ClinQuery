import numpy as np
import random
import torch
import torch.nn.functional as F
from torch import nn
import json
import copy
import re
from pathlib import Path
from typing import List, Dict
from seqeval.metrics import (classification_report,accuracy_score,f1_score,precision_score,recall_score,)
from transformers import AutoConfig, AutoModel

def load_train_and_devsets_from_jsonl(path,initial_shuffle=False,dev_examples_n=5):
    """
    Load a Prodigy-style JSONL with fields like {"text": ..., "spans": [...]} and split into train/dev.

    Args:
        path (str|Path): Path to JSONL file.
        initial_shuffle (bool): If True, shuffle once before splitting.
        dev_examples_n (int): Number of examples to put into the dev split (taken from the start after optional shuffle).

    Returns:
        (trainset_output, devset_output): Two lists of dict examples. (Deep-copied to avoid accidental mutation.)
    """
    with open(path, 'r') as f:
        all_examples_list = [json.loads(line) for line in f]
        if initial_shuffle: random.shuffle(all_examples_list) 
        devset_output = all_examples_list[:dev_examples_n]
        trainset_output = all_examples_list[dev_examples_n:]
        devset_output = copy.deepcopy(devset_output)
        trainset_output = copy.deepcopy(trainset_output)
        random.shuffle(devset_output)
        random.shuffle(trainset_output)
        return trainset_output,devset_output

def chunk_text(example, tokenizer, max_length=512, overlap=0.25):
    """
    Chunk a single labeled example into overlapping token windows using tokenizer offsets.

    Notes:
        - Uses add_special_tokens=False so offsets map cleanly to the original text.
        - Intersects character spans with the chunk [char_start, char_end) and remaps to chunk-local coordinates.

    Args:
        example (dict): {"text": str, "spans": [{"start": int, "end": int, "label": str}, ...]}
        tokenizer: HF tokenizer with return_offsets_mapping support.
        max_length (int): Max tokens per chunk.
        overlap (float): Fractional overlap between chunks (0.0–0.99).

    Returns:
        List[dict]: Each item has {"text": chunk_text, "spans": [adjusted_spans]}.
    """
    text = example['text']
    spans = example.get('spans', [])
    tokens = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = tokens['input_ids']
    offsets = tokens['offset_mapping']

    stride = int(max_length * (1 - overlap))
    chunks = []

    for start in range(0, len(input_ids), stride):
        end = min(start + max_length, len(input_ids))
        chunk_ids = input_ids[start:end]
        chunk_offsets = offsets[start:end]

        if not chunk_offsets:
            continue

        # Use character boundaries from original text
        char_start = chunk_offsets[0][0]
        char_end = chunk_offsets[-1][1]
        chunk_text = text[char_start:char_end]  # Extract text slice directly

        # Find and adjust spans that intersect this chunk
        chunk_spans = []
        for span in spans:
            if span['end'] > char_start and span['start'] < char_end:
                adjusted = {
                    'start': max(span['start'], char_start) - char_start,
                    'end': min(span['end'], char_end) - char_start,
                    'label': span['label']
                }
                chunk_spans.append(adjusted)

        chunks.append({
            'text': chunk_text,
            'spans': chunk_spans
        })

        if end == len(input_ids):
            break

    return chunks
    
def convert_spans_to_bio_labels(example, tokenizer, label2id, max_length=512, expand_entities=False):
    """
    Convert character-level spans into token-level BIO labels aligned to tokenizer offsets.

    Behavior:
        - Special tokens/padding (offset (0,0)) get label -100 (ignored by loss).
        - If expand_entities=True, softly expand entity boundaries by 1 token left/right where safe.

    Args:
        example (dict): {"text": str, "spans": [{"start": int, "end": int, "label": str}, ...]}
        tokenizer: HF tokenizer.
        label2id (dict): e.g., {"O":0, "B-C_ENT":1, "I-C_ENT":2, ...}
        max_length (int): Max tokens (uses padding='max_length', add_special_tokens=False).
        expand_entities (bool): Optional one-token expansion for recall.

    Returns:
        dict: tokenized inputs plus "labels" (list[int]) where non-entity/padding positions are "O"/-100 respectively.
    """
    encoding = tokenizer(
        example['text'],
        return_offsets_mapping=True,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        add_special_tokens=False
    )

    labels = ['O'] * len(encoding['input_ids'])
    offsets = encoding['offset_mapping']
    spans = example.get('spans', [])

    # Define priority
    PRIORITY = {
        'PLAN': 3,
        'TIME': 2,
        'C_ENT': 1
    }

    # Build a character-to-label map with priority handling
    char_label_map = {}
    for span in spans:
        label = span['label']
        start = span['start']
        end = span['end']
        priority = PRIORITY.get(label.upper(), 0)
        for i in range(start, end):
            if i not in char_label_map or priority > char_label_map[i][1]:
                char_label_map[i] = (label, priority)

    # Assign BIO labels
    for i, (start, end) in enumerate(offsets):
        if start == end or (start == 0 and end == 0):  # special token
            continue
        span_chars = list(range(start, end))
        token_labels = [char_label_map[c][0] for c in span_chars if c in char_label_map]
        if not token_labels:
            continue
        label = token_labels[0]
        if i == 0 or labels[i - 1][2:] != label:
            labels[i] = f'B-{label}'
        else:
            labels[i] = f'I-{label}'

    # Expand BIO spans by 1 token in both directions (if safe)
    if expand_entities:
        new_labels = labels[:]
        for i, label in enumerate(labels):
            if label.startswith("B-") or label.startswith("I-"):
                ent_type = label[2:]

                # Try expanding left
                if i > 0 and new_labels[i - 1] == 'O':
                    new_labels[i - 1] = f'B-{ent_type}'
                    if new_labels[i].startswith("B-"):
                        new_labels[i] = f'I-{ent_type}'  # demote to I- after expanding left

                # Try expanding right
                if i < len(labels) - 1 and new_labels[i + 1] == 'O':
                    new_labels[i + 1] = f'I-{ent_type}'

        labels = new_labels

    # Convert to IDs and apply masking
    encoding['labels'] = [
        label2id.get(l, 0) if o != (0, 0) else -100
        for l, o in zip(labels, offsets)
    ]
    encoding.pop('offset_mapping')

    return encoding

def training_metrics_func(label_list):
    """
    Build a compute_metrics function for Hugging Face Trainer (token classification).

    Prints (seqeval) a full classification_report each eval step and returns accuracy/precision/recall/F1.
    """
    def compute_metrics(p):
        predictions = np.argmax(p.predictions, axis=2)
        labels = p.label_ids

        # Convert to label strings and ignore -100 (padding)
        true_predictions = [
            [label_list[p_i] for p_i, l_i in zip(pred, lab) if l_i != -100]
            for pred, lab in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l_i] for p_i, l_i in zip(pred, lab) if l_i != -100]
            for pred, lab in zip(predictions, labels)
        ]

        # print(predictions[20,:])
        # print(labels[20,:])
        print(classification_report(true_labels, true_predictions))

        return {
            "accuracy": accuracy_score(true_labels, true_predictions),
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }
    return compute_metrics

def render_labeled_text_with_xml_sanity_check(text: str, tokens: List[int], labels: List[str], tokenizer) -> str:
    """
    Render BIO labels back into inline XML-like tags for a quick sanity check.

    Args:
        text (str): Original text.
        tokens (List[int]): Not used directly (kept for compatibility).
        labels (List[str]): BIO tags, one per token position.
        tokenizer: Tokenizer used for offsets (must match training).

    Returns:
        str: Text with tags like <C_ENT>... </C_ENT> (tag names derived from BIO labels).
    """
    encoding = tokenizer(text, return_offsets_mapping=True, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=False, max_length=512)
    offsets = encoding["offset_mapping"][0].tolist()  # shape: (seq_len, 2)

    assert len(labels) == len(offsets), f"Expected {len(offsets)} labels, but got {len(labels)}."

    output = ""
    last_end = 0
    open_tag = None

    for (start, end), label in zip(offsets, labels):
        if start == end or (start == 0 and end == 0):  # likely special tokens or padding
            continue

        # Add the raw text between last token and this one
        output += text[last_end:start]

        token_text = text[start:end]

        if label.startswith("B-"):
            if open_tag:
                output += f"</{open_tag}>"
            open_tag = label[2:]
            output += f"<{open_tag}>{token_text}"
        elif label.startswith("I-") and open_tag == label[2:]:
            output += token_text
        else:
            if open_tag:
                output += f"</{open_tag}>"
                open_tag = None
            output += token_text

        last_end = end

    if open_tag:
        output += f"</{open_tag}>"
    output += text[last_end:]

    return output

class WeightedTokenClassificationModel(nn.Module):
    """
    Lightweight wrapper for token classification with optional per-class weight boosts.

    Args:
        base_model_name (str): HF model id or path.
        label_list (List[str]): Label vocabulary in order (maps index->label).
        weight_boosts (dict|None): Optional boosts by label name (e.g., {"B-TIME": 1.5, "I-TIME": 1.5}).
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

def set_dropout(self, dropout_prob):
    """
    Update dropout probabilities in the transformer base model after instantiation.
    """
    if hasattr(self.base_model, 'dropout'):
        self.base_model.dropout.p = dropout_prob

    if hasattr(self.base_model, 'encoder') and hasattr(self.base_model.encoder, 'layer'):
        for layer in self.base_model.encoder.layer:
            if hasattr(layer.attention, 'self') and hasattr(layer.attention.self, 'dropout'):
                layer.attention.self.dropout.p = dropout_prob
            if hasattr(layer.attention, 'output') and hasattr(layer.attention.output, 'dropout'):
                layer.attention.output.dropout.p = dropout_prob
            if hasattr(layer, 'output') and hasattr(layer.output, 'dropout'):
                layer.output.dropout.p = dropout_prob

# chunk texts for predictions
def chunk_text_for_predictions(text, tokenizer, max_length=512, overlap_ratio=0.25):
    """
    Split raw text into overlapping token windows for inference.

    Returns:
        List[dict]: Each with {"input_ids", "offsets", "text"} where offsets are full-text-relative.
    """
    encoding = tokenizer(text,return_offsets_mapping=True,add_special_tokens=False,return_attention_mask=False)

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

        chunks.append({
            "input_ids": chunk_input_ids,
            "offsets": chunk_offsets,      # still relative to full text
            "text": chunk_text,            # needed by predict_entities
        })

        if end_idx >= len(input_ids):
            break

    return chunks

# predict entities
def predict_entities(text, model, tokenizer, id2label: Dict[int, str], device="cpu", prob_threshold=0.6, max_length=512):
    """
    Run sliding-window token classification with overlap trimming and merge contiguous B/I segments.

    Args:
        text (str): Input note text (can be long).
        model: Trained token classifier (returns {"logits"}).
        tokenizer: Matching tokenizer.
        id2label (dict): Maps index->BIO string label (e.g., 1->"B-C_ENT").
        device (str): "cpu" or "cuda".
        prob_threshold (float): Minimum softmax probability to accept a token label.
        max_length (int): Window size (tokens).

    Returns:
        List[dict]: Spans with {"start","end","label","confidence","text","id"} sorted by (start,end).
    """
    chunks = chunk_text_for_predictions(text, tokenizer, max_length=max_length)
    spans = []

    for chunk in chunks:
        chunk_text = chunk["text"]
        input_ids = chunk["input_ids"]
        offset_mapping = chunk["offsets"]    # use full-text-relative offsets

        # Determine chunk prediction window
        length = len(input_ids)
        lower = int(length * 0.15)
        upper = int(length * 0.85)

        # Tokenize the **chunk text only**
        inputs = tokenizer(chunk_text, return_tensors="pt", truncation=True, padding="max_length",
                           max_length=max_length, is_split_into_words=False, return_offsets_mapping=True)

        chunk_relative_offsets = inputs.pop("offset_mapping").squeeze().tolist()
        chunk_start_char = offset_mapping[0][0]  # where this chunk starts in full text

        offset_mapping = [(s + chunk_start_char, e + chunk_start_char) if s is not None and e is not None else (0, 0)
        for (s, e) in chunk_relative_offsets]  

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
                current_span = {
                    "start": full_text_start,
                    "end": full_text_end,
                    "label": entity_type,
                    "confidence": confidence
                }
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
        spans.append({
            "start": match.start(),
            "end": match.end(),
            "label": "TABLE",
            "text": "[TABLE]",
            "confidence": 1.0
        })

    # Finalize span info and assign IDs
    spans = sorted(spans, key=lambda s: (s["start"], s["end"]))  # Ensure consistent order
    for idx, span in enumerate(spans):
        span["text"] = text[span["start"]:span["end"]]
        span["confidence"] = round(span["confidence"], 3)
        span["id"] = f"e{idx+1}"

    return spans

def filter_overlapping_spans_for_predictions(spans: List[Dict], min_char_length: int=1) -> List[Dict]:
    """
    Greedy filter to drop lower-confidence spans that overlap higher-confidence ones.

    Args:
        spans (List[dict]): Each with {"start","end","label","confidence",...}
        min_char_length (int): Ignore spans shorter than this many characters.

    Returns:
        List[dict]: Non-overlapping spans, sorted by initial greedy selection order.
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

def convert_prodigy_jsonl_to_tagged_text_and_relations(input_jsonl_path):
    """
    Convert Prodigy JSONL to (raw_text, xml_tagged_text, relations) tuples.

    - Inserts inline entity tags like <c_ent id="e1"> ... </c_ent> (label lowercased).
    - Preserves any provided 'id' on spans; otherwise assigns e1, e2, ...
    - Replaces long label 'condition_symptom_measurement_device_procedure_med' with 'c_ent' in the output text.

    Returns:
        List[Tuple[str, str, List[dict]]]
    """
    input_path = Path(input_jsonl_path)
    output = []

    with input_path.open("r", encoding="utf-8") as infile:
        for line in infile:
            data = json.loads(line)
            raw_text = data["text"]
            text = data["text"]
            spans = data.get("spans", [])
            relations = data.get("relations", [])

            # Sort spans by start index to apply tags properly
            spans = sorted(spans, key=lambda x: x["start"])

            # Assign unique IDs if not already present
            id_map = {}
            for i, span in enumerate(spans):
                ent_id = span.get("id", f"e{i+1}")
                span["id"] = ent_id
                id_map[(span["start"], span["end"])] = ent_id

            # Apply inline XML-style tags from back to front
            for span in reversed(spans):
                start, end = span["start"], span["end"]
                label = span["label"].lower()
                ent_id = span["id"]
                open_tag = f'<{label} id="{ent_id}">'
                close_tag = f'</{label}>'
                text = text[:start] + open_tag + text[start:end] + close_tag + text[end:]

            # Format relation list
            example_relations = []
            for rel in relations:
                head_id = rel["head"]
                child_id = rel["child"]
                label = rel["label"]
                example_relations.append({
                    "head": head_id,
                    "child": child_id,
                    "label": label
                })

            output.append((raw_text, text.replace('condition_symptom_measurement_device_procedure_med', 'c_ent'), example_relations))

    return output