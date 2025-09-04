import numpy as np 
import random 
import torch 
from torch import nn 
import json 
import copy 
from itertools import product
from datasets import Dataset 
from typing import List, Dict, Tuple 
from sklearn.metrics import confusion_matrix, classification_report
from transformers import Trainer

def load_train_and_devsets_from_jsonl(path,initial_shuffle=False,dev_examples_n=5):
    """
    Load Prodigy-style JSONL with {"text", "spans": [...]} and split into train/dev.

    Returns:
        (train_list, dev_list)
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
    Chunk text for relation training, preserving only spans/relations that intersect the chunk.

    Differences vs NER chunking:
        - Uses add_special_tokens=True to mirror classification model inputs.
        - Drops special tokens from the char-offset window used to slice text.

    Returns:
        List[dict]: {"text", "spans", "relations"} for each chunk.
    """
    text = example['text']
    spans = example.get('spans', [])
    relations = example.get('relations', [])

    # Use add_special_tokens=True to match model input behavior
    tokens = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    input_ids = tokens['input_ids']
    offsets = tokens['offset_mapping']

    stride = int(max_length * (1 - overlap))
    chunks = []

    for start in range(0, len(input_ids), stride):
        end = min(start + max_length, len(input_ids))
        chunk_ids = input_ids[start:end]
        chunk_offsets = offsets[start:end]

        # Remove special tokens from character offset consideration
        chunk_offsets_clean = [o for o in chunk_offsets if o[1] > o[0]]
        if not chunk_offsets_clean:
            continue

        char_start = chunk_offsets_clean[0][0]
        char_end = chunk_offsets_clean[-1][1]
        chunk_text = text[char_start:char_end]

        # Adjust spans for this chunk
        chunk_spans = []
        id_map = {}
        for span in spans:
            if span['end'] > char_start and span['start'] < char_end:
                adjusted = {
                    'id': span['id'],
                    'start': max(span['start'], char_start) - char_start,
                    'end': min(span['end'], char_end) - char_start,
                    'label': span['label'],
                    'text': text[max(span['start'], char_start):min(span['end'], char_end)]
                }
                chunk_spans.append(adjusted)
                id_map[span['id']] = True

        chunk_relations = [
            rel for rel in relations if rel['head'] in id_map and rel['child'] in id_map
        ]

        if len(chunk_spans) > 0 or len(chunk_relations) > 0:
            chunks.append({
            'text': chunk_text,
            'spans': chunk_spans,
            'relations': chunk_relations
        })

        if end == len(input_ids):
            break

    return chunks

def prepare_relation_examples_with_chunking(data, tokenizer, label_encoder, max_length=512, overlap=0.25, max_control_token_distance=30, max_control_token_distance_lr=150, max_positives_token_distance=150, lr_controls_prop=0.05):
    """
    Build a relation classification dataset by inserting <HEAD>...</HEAD> and <CHILD>...</CHILD> markers.

    Positives:
        - TIME_RELATION: (C_ENT or TABLE) ↔ TIME
        - NEGATION_RELATION: C_ENT ↔ NEGATION
        - Discards overly long token separations.

    Controls (NO_RELATION):
        - Sampled pairs matching valid type patterns but not annotated as a relation.
        - Mix of short-range controls and a small fraction (lr_controls_prop) of long-range controls.

    Returns:
        datasets.Dataset with fields: input_ids/attention_mask (from tokenizer), labels, debug, distance.
    """
    examples = []
    long_range_controls = []

    for entry in data:
        chunks = chunk_text(entry, tokenizer, max_length=max_length, overlap=overlap)

        for chunk in chunks:
            text = chunk["text"]
            spans = {s["id"]: s for s in chunk["spans"]}
            relations = chunk.get("relations", [])

            def insert_xml_tags(text, head, child):
                first, second = (head, child) if head["start"] <= child["start"] else (child, head)
                before = text[:first["start"]]
                between = text[first["end"]:second["start"]]
                after = text[second["end"]:]

                middle1 = text[first["start"]:first["end"]]
                middle2 = text[second["start"]:second["end"]]

                first_tagged = (
                    f"<HEAD>{middle1}</HEAD>" if first == head else f"<CHILD>{middle1}</CHILD>"
                )
                second_tagged = (
                    f"<CHILD>{middle2}</CHILD>" if second == child else f"<HEAD>{middle2}</HEAD>"
                )

                return before + first_tagged + between + second_tagged + after

            def char_to_token_index(offsets, pos):
                for idx, (start, end) in enumerate(offsets):
                    if start <= pos < end:
                        return idx
                return None

            # Positive relation examples
            for rel in relations:
                head = spans.get(rel["head"])
                child = spans.get(rel["child"])
                if not head or not child:
                    continue

                if rel["label"] == "TIME_RELATION":
                    if head["label"] not in {"C_ENT", "TABLE"} or child["label"] != "TIME":
                        continue
                elif rel["label"] == "NEGATION_RELATION":
                    if head["label"] != "C_ENT" or child["label"] != "NEGATION":
                        continue
                else:
                    continue

                tagged = insert_xml_tags(text, head, child)
                tok = tokenizer(tagged, return_offsets_mapping=True, add_special_tokens=True)
                offsets = tok["offset_mapping"]

                idx1 = char_to_token_index(offsets, head["start"])
                idx2 = char_to_token_index(offsets, child["start"])
                if idx1 is None or idx2 is None:
                    continue
                distance = abs(idx1 - idx2)

                if distance > max_positives_token_distance:
                    continue  # discard long-distance positives

                examples.append({
                    "text": tagged,
                    "label": rel["label"],
                    "debug": tagged,
                    "distance": distance
                })

            # Negative examples (NO_RELATION)
            span_list = list(spans.values())
            for i in range(len(span_list)):
                for j in range(len(span_list)):
                    if i == j:
                        continue

                    s1, s2 = span_list[i], span_list[j]

                    if s1["start"] == s2["start"]:
                        continue

                    if any((r["head"] == s1["id"] and r["child"] == s2["id"]) for r in relations):
                        continue

                    valid = (
                        (s1["label"] in {"C_ENT", "TABLE"} and s2["label"] == "TIME") or
                        (s1["label"] == "C_ENT" and s2["label"] == "NEGATION")
                    )
                    if not valid:
                        continue

                    tagged = insert_xml_tags(text, s1, s2)
                    tok = tokenizer(tagged, return_offsets_mapping=True, add_special_tokens=True)
                    offsets = tok["offset_mapping"]

                    idx1 = char_to_token_index(offsets, s1["start"])
                    idx2 = char_to_token_index(offsets, s2["start"])
                    if idx1 is None or idx2 is None:
                        continue
                    distance = abs(idx1 - idx2)

                    ex = {
                        "text": tagged,
                        "label": "NO_RELATION",
                        "debug": tagged,
                        "distance": distance
                    }

                    if distance <= max_control_token_distance:
                        examples.append(ex)
                    elif max_control_token_distance < distance <= max_control_token_distance_lr:
                        long_range_controls.append(ex)

    # Add ~10% of NO_RELATION examples from long-range band
    max_extra = int(lr_controls_prop*len([e for e in examples if e["label"] == "NO_RELATION"]))
    sampled = random.sample(long_range_controls, min(len(long_range_controls), max_extra))
    examples.extend(sampled)

    # Encode
    labels = label_encoder.transform([ex["label"] for ex in examples])
    encodings = tokenizer(
        [ex["text"] for ex in examples],
        truncation=True,
        padding=True,
        max_length=max_length,
        add_special_tokens=True
    )
    encodings["labels"] = labels.tolist()
    encodings["debug"] = [ex["debug"] for ex in examples]
    encodings["distance"] = [ex["distance"] for ex in examples]
    return Dataset.from_dict(encodings)

    
def compute_metrics_with_confusion_matrix(eval_pred):
    """
    compute_metrics for HF Trainer (relation classification).

    Prints a scikit-learn classification_report and returns:
        - "accuracy"
        - Flattened confusion-matrix entries as keys: confusion_matrix_i_j
    """
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    # Optional: print full report
    print(classification_report(labels, preds, digits=4))

    # Confusion matrix as a dictionary for compatibility
    cm = confusion_matrix(labels, preds)
    cm_dict = {
        f"confusion_matrix_{i}_{j}": cm[i][j]
        for i in range(cm.shape[0])
        for j in range(cm.shape[1])
    }

    # You can also add accuracy, f1, etc.
    return {
        "accuracy": (preds == labels).mean(),
        **cm_dict
    }
    

class WeightedTrainer(Trainer):
    """
    Hugging Face Trainer with optional class weights for relation classification.

    Args:
        class_weights (torch.Tensor|None): Tensor of per-class weights (size C), or None.
    """
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def chunk_text_for_predictions(text: str, tokenizer, max_length=512, overlap=0.50, pred_window=0.80):
    """
    Split raw text into overlapping token windows for relation **inference** (not training)
    using add_special_tokens=True. Each chunk includes a "middle_range" (token index window)
    where predictions should be trusted to reduce edge effects.

    Args:
        text (str): Input document text.
        tokenizer: Matching HF tokenizer with return_offsets_mapping=True.
        max_length (int): Maximum chunk length in tokens.
        overlap (float): Overlap fraction between consecutive chunks in [0,1).
        pred_window (float): Proportion of the chunk to treat as the central, trusted region.

    Returns:
        list[dict]: Each item has:
            - "input_ids": list[int]
            - "offsets": list[tuple[int,int]] (character offsets in original text)
            - "start_token": int (starting token index in the full tokenized sequence)
            - "middle_range": tuple[int,int] (start,end token indices within the chunk)
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

        chunks.append({
            "input_ids": chunk_ids,
            "offsets": chunk_offsets,
            "start_token": start,
            "middle_range": (middle_start, middle_end)
        })

        if end == len(input_ids):
            break

    return chunks

def spans_in_middle_range_for_predictions(spans: List[Dict], offsets, middle_range: Tuple[int, int], is_start, is_end) -> List[Dict]:
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
    if is_start: start_idx = 0
    if is_end: end_idx = float('inf')

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

def generate_allowed_span_pairs_for_predictions(spans: List[Dict], allowed_relations: List[Tuple[str, str, str]]) -> List[Tuple[Dict, Dict, str]]:
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

def predict_relations_with_chunking(text: str, spans: List[Dict], model, tokenizer, label2id: Dict[str, int], id2label: Dict[int, str], allowed_relations: List[Tuple[str, str, str]], device, prob_threshold=0.6, max_token_distance=100):
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
                    chunk_text[:head_start] +
                    "<HEAD>" + chunk_text[head_start:head_end] + "</HEAD>" +
                    chunk_text[head_end:child_start] +
                    "<CHILD>" + chunk_text[child_start:child_end] + "</CHILD>" +
                    chunk_text[child_end:]
                )
            else:
                tagged = (
                    chunk_text[:child_start] +
                    "<CHILD>" + chunk_text[child_start:child_end] + "</CHILD>" +
                    chunk_text[child_end:head_start] +
                    "<HEAD>" + chunk_text[head_start:head_end] + "</HEAD>" +
                    chunk_text[head_end:]
                )

            tok = tokenizer(tagged, return_offsets_mapping=True, add_special_tokens=True)
            offsets = tok["offset_mapping"]

            def char_to_token_index(pos):
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

            enc = tokenizer(tagged, return_tensors="pt", truncation=True,
                            padding=True, max_length=512, add_special_tokens=True).to(device)

            with torch.no_grad():
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1)
                pred = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred].item()

            label = id2label[pred]
            if label == "TIME_RELATION" and confidence > prob_threshold:
                if head['label'] in {"C_ENT", "TABLE"} and child['label'] == "TIME":
                    xml_texts.append(tagged)
                    all_rels.append({
                        "head": head["id"],
                        "child": child["id"],
                        "label": label,
                        "confidence": round(confidence, 3)
                    })
            elif label == "NEGATION_RELATION" and confidence > prob_threshold:
                if head['label'] == "C_ENT" and child['label'] == "NEGATION":
                    xml_texts.append(tagged)
                    all_rels.append({
                        "head": head["id"],
                        "child": child["id"],
                        "label": label,
                        "confidence": round(confidence, 3)
                    })
            elif label == "NO_RELATION" and confidence > prob_threshold:
                xml_texts_no_relations.append(tagged)

    return all_rels, xml_texts, xml_texts_no_relations

def filter_relations_for_predictions(relations: List[Dict]) -> List[Dict]:
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
    relation_map: Dict[Tuple[str, str, str], Dict] = {}
    for rel in relations:
        key = (rel["head"], rel["child"], rel["label"])
        if key not in relation_map or rel["confidence"] > relation_map[key]["confidence"]:
            relation_map[key] = rel

    # Step 2: For each (head, child), keep only the label with max confidence
    best_by_pair: Dict[Tuple[str, str], Dict] = {}
    for (head, child, _), rel in relation_map.items():
        pair_key = (head, child)
        if pair_key not in best_by_pair or rel["confidence"] > best_by_pair[pair_key]["confidence"]:
            best_by_pair[pair_key] = rel

    return list(best_by_pair.values())


def render_text_with_xml_tags(text: str, spans: list[dict], span_type: str = None) -> str:
    """
    Insert inline XML tags for each span, e.g., <c_ent id="e1">...</c_ent>.

    Args:
        text (str): Original text.
        spans (list[dict]): Spans with "start", "end", "label", and "id".
        span_type (str | None): If provided, only render spans of this label.

    Returns:
        str: Text with spans wrapped in XML-like tags. Spans are inserted from the end to preserve offsets.
    """
    # Filter spans if span_type is set
    if span_type is not None:
        spans = [s for s in spans if s["label"] == span_type]

    # Sort spans by start (descending to preserve character positions when inserting)
    sorted_spans = sorted(spans, key=lambda x: x["start"], reverse=True)

    tagged_text = text
    for span in sorted_spans:
        start = span["start"]
        end = span["end"]
        label = span["label"].lower()
        span_id = span["id"]

        inner_text = tagged_text[start:end]
        tagged = f'<{label} id="{span_id}">{inner_text}</{label}>'
        tagged_text = tagged_text[:start] + tagged + tagged_text[end:]

    return tagged_text
        
