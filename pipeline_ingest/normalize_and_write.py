"""
normalize_and_write.py — clinical entity normalization pipeline using UMLS (MRCONSO/MRREL/MRSTY) and SapBERT.

What it does
------------
- Ingests span/relation JSONL (NER + relations) and table extractions.
- Resolves mentions to canonical terms; builds alias↔canonical maps.
- Adds UMLS-derived concept classes, timestamps (relative to admission), negation, and values/modifiers.
- Writes normalized artifacts (DuckDB/FAISS) for downstream querying.

Inputs
------
- UMLS files: MRCONSO.RRF, MRREL.RRF, MRSTY.RRF (English rows only are used).
- `in_jsonl`: JSONL with keys `spans`, `relations`, `tables` (see `_create_master_list`).

Outputs
-------
- DuckDB at `{workdir}/master_db.duckdb` when `keep=False`
- FAISS index `{workdir}/alias_vectors.faiss` and `{workdir}/alias_vector_ids.npy` when `keep=False`
- term-stat CSVs `(resolved.csv, unresolved.csv)` when `keep=True`

Performance & curation
----------------------
If you want to optimize performance/accuracy you will have to carefully review the unmapped and mapped terms that this script gives you and then manually tune **term→alias** and **class→term-set** mappings in ontology_corrections.yml.

Assumptions & caveats
---------------------
- Dates like `xx/xx` and `xx/xx/xxxx` are interpreted as **mm/dd** and **mm/dd/yyyy** (US order); `yyyy-mm-dd` is also supported.
- There are a few constants/tunables that I still need to move out.
- I need to add an additional sanity check that will give the user likely term sets that were merged erroneously or that should be merged. This will help them tune ontology_corrections.yml better.
"""

import itertools
import json
import logging
import os
import re
from collections import Counter, OrderedDict, defaultdict, deque
from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
import yaml
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer, models

from pipeline_ingest.write_db import to_db

log = logging.getLogger(__name__)


class Normalizer:
    """Build and apply UMLS-based normalization; resolves spans/tables and writes outputs."""

    def __init__(
        self,
        mrconso_rrf: str,
        mrrel_rrf: str,
        mrsty_rrf: str,
        ont_corr: str,
        keep: bool = False,
        no_pruning: bool = False,
        cli_entry: bool = False,
    ):
        """Initialize paths, defaults, caches; validate UMLS inputs."""
        self.mrconso_rrf = Path(mrconso_rrf)
        self.mrrel_rrf = Path(mrrel_rrf)
        self.mrsty_rrf = Path(mrsty_rrf)
        self.ont_corr = Path(ont_corr)
        self.keep = keep
        self.no_pruning = no_pruning
        self.embedder_model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        self.min_alias_len = 6  # shorter terms are not considered by the embedder
        self.max_alias_len = 35  # ignore longer terms within umls
        self.resolved_cent_concept_cache = {}
        self.embedding_cache = {}
        self.cli_entry = cli_entry

        # validate umls inputs
        for p in (self.mrconso_rrf, self.mrrel_rrf, self.mrsty_rrf):
            if not p.exists():
                raise FileNotFoundError(f"UMLS file not found: {p}")

    # ---- Phase 1: normalize ----

    def normalize(self, in_jsonl: str, term_stats_csv: tuple[str, str]) -> None:
        """Run end-to-end normalization: load UMLS, resolve spans/tables, post-process, prune/collapse."""

        # read in master_list (and ensure that all patients have an admission date)
        self._create_master_list(in_jsonl)

        # create vocabulary from UMLS and load embedder
        log.info("loading ontology mappings …")
        self._create_umls_vocab()
        self._load_embedder()

        # resolve spans and table entries and combine resolved nontabular and tabular entities
        log.info("resolving spans …")
        self._resolve_spans()
        if not self.cli_entry:
            log.info("resolving tabular data …")
            self._resolve_table_entries()
        for master_list_i in range(len(self.master_list)):
            self.master_list[master_list_i][7] = self.master_list[master_list_i][7] + self.master_list[master_list_i][8]
            del self.master_list[master_list_i][8]

        # produce term_stats
        if self.keep:
            self._produce_term_stats_output(term_stats_csv)

        # post-process (remove None entities, expand SBP/DBPs into separate BP+DP entities, etc..)
        self._post_process_entities()

        # prune out vocab that does not appear in data, and then merge similar term sets
        log.info("post-processing vocab …")
        self._prune_and_collapse_vocab()

    # ---- Phase 2: write ----

    def write_db(self, csv_out: str, workdir: str, to_csv: bool = False) -> None:
        """Write DuckDB/FAISS artifacts OR only csv output when to_csv=True."""

        log.info("writing db …")
        to_db(
            self,
            str(Path(workdir) / "master_db.duckdb"),
            str(Path(workdir) / "alias_vectors.faiss"),
            str(Path(workdir) / "alias_vector_ids.npy"),
            csv_out,
            to_csv,
        )

    # ---------------- Internal helpers ----------------

    def filter_vocab_by_usage(self, c_to_a_dict, entity_list):
        """Keep only canonicals whose terms/aliases appear in data."""
        # build set of all terms and concept classes in entity_list
        seen = set(entity_list)
        seen.discard(None)

        # now remove any term->aliases sets where not a single term is seen in the dataset
        filtered_vocab = {}
        for canonical, aliases in c_to_a_dict.items():
            # keep only if the canonical term or one of its aliases appears
            if canonical.lower().strip() in seen or any(a.lower().strip() in seen for a in aliases):
                filtered_vocab[canonical] = aliases

        # return result
        return filtered_vocab

    def merge_canonical_alias_sets_by_embeddings(
        self,
        c_to_a_dict,
        a_to_c_resolver=None,
        k_neighbors=16,
        centroid_sim_threshold=50,
        fuzzy_threshold=95,
        emb_threshold=90,
        match_ratio_threshold=0.30,
    ) -> dict[str, list[str]]:

        # ---- collect + length-filter ----
        canonicals = list(c_to_a_dict.keys())
        members, uniq = [], set()
        for c in canonicals:
            m = [c] + (c_to_a_dict.get(c, []) or [])
            m = list(dict.fromkeys(s for s in m if s and len(s) >= self.min_alias_len))
            members.append(m)
            uniq.update(m)
        if not uniq:
            return {c: [c] for c in canonicals}

        # ---- embed aliases (cache-aware) ----
        to_embed = [s for s in uniq if s not in self.embedding_cache]
        if to_embed:
            E = (
                self.emb_model.encode(to_embed, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
                if self.device == "cuda"
                else self.emb_model.encode(to_embed, normalize_embeddings=True, show_progress_bar=False)
            )
            for s, e in zip(to_embed, E, strict=False):
                self.embedding_cache[s] = e
        D = next(iter(self.embedding_cache.values())).shape[0]

        # ---- set centroids (plain mean; L2-normalized) ----
        C = np.zeros((len(canonicals), D), dtype=np.float32)
        valid = np.zeros(len(canonicals), dtype=bool)
        for i, m in enumerate(members):
            embs = [self.embedding_cache[t] for t in m if t in self.embedding_cache]
            if not embs:
                continue
            v = np.stack(embs, axis=0).mean(axis=0)
            v /= max(np.linalg.norm(v), 1e-9)
            C[i] = v
            valid[i] = True

        vidx = np.where(valid)[0]
        if len(vidx) <= 1:
            return {c: list(set([c] + members[i])) for i, c in enumerate(canonicals)}

        # ---- ANN index (HNSW, L2 metric) ----
        index = faiss.IndexHNSWFlat(D, 32)
        index.hnsw.efSearch = 128
        index.hnsw.efConstruction = 200
        index.add(C[vidx])

        pos2idx = {pos: idx for pos, idx in enumerate(vidx)}

        # ---- union-find ----
        parents = list(range(len(canonicals)))

        def find(x):
            while parents[x] != x:
                parents[x] = parents[parents[x]]
                x = parents[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parents[rb] = ra

        # ---- thresholds as decimals ----
        cos_pref = float(centroid_sim_threshold) / 100.0
        emb_thr = float(emb_threshold) / 100.0

        # ---- verifier (your rule) ----
        def verify_pair(i: int, j: int) -> bool:
            A, B = members[i], members[j]
            if not A or not B:
                return False
            # fuzzy
            ab = sum(any(fuzz.ratio(a, b) >= fuzzy_threshold for b in B) for a in A)
            ba = sum(any(fuzz.ratio(b, a) >= fuzzy_threshold for a in A) for b in B)
            U = len(set(A).union(B))
            if (ab + ba) / (2 * max(U, 1)) >= match_ratio_threshold:
                return True
            # embeddings (cosine on unit vectors)
            Ei = [self.embedding_cache[s] for s in A if s in self.embedding_cache]
            Ej = [self.embedding_cache[s] for s in B if s in self.embedding_cache]
            if not Ei or not Ej:
                return False
            Ei = np.stack(Ei)
            Ej = np.stack(Ej)
            Ei /= np.linalg.norm(Ei, axis=1, keepdims=True) + 1e-9
            Ej /= np.linalg.norm(Ej, axis=1, keepdims=True) + 1e-9
            sims = Ei @ Ej.T
            ab += int((sims.max(axis=1) >= emb_thr).sum())
            ba += int((sims.max(axis=0) >= emb_thr).sum())
            return (ab + ba) / (2 * max(U, 1)) >= match_ratio_threshold

        # ---- retrieve candidates (convert L2→cosine), then verify ----
        for i in vidx:
            v = C[i][None, :].astype(np.float32)
            dists, idxs = index.search(v, min(k_neighbors + 1, len(vidx)))  # includes self
            # unit vectors: L2^2 = 2 - 2*cos  =>  cos = 1 - L2^2/2
            for p, d2 in zip(idxs[0], dists[0], strict=False):
                j = pos2idx.get(p)
                if j is None or j == i:
                    continue
                cos_sim = 1.0 - float(d2) / 2.0
                if cos_sim < cos_pref:
                    continue
                if verify_pair(i, j):
                    union(i, j)

        # ---- build merged result ----
        groups = {}
        for i in range(len(canonicals)):
            r = find(i)
            groups.setdefault(r, []).append(i)

        def pick_rep(idxs):
            if a_to_c_resolver:
                return min(idxs, key=lambda ix: (len(a_to_c_resolver.get(canonicals[ix], "§" * 64)), len(canonicals[ix]), ix))
            return min(idxs)

        merged = {}
        for idxs in groups.values():
            rep_idx = pick_rep(idxs)
            rep_key = canonicals[rep_idx]
            bag = set()
            for ix in idxs:
                bag.update(members[ix])
                bag.add(canonicals[ix])
            merged[rep_key] = list(set(bag | {rep_key}))

        merged = {key: [item for item in values if item is not None] for key, values in merged.items() if key is not None}  # remove any None entries
        merged = {key: list(set(values + [key])) for key, values in merged.items() if key is not None}  # remove duplicates

        return merged

    def deduplicate_concept_classes(self, concept_classes: Iterable[str], *, fuzzy_threshold: int = 90, embed_threshold: int = 90) -> dict[str, list[str]]:
        """Collapse near-duplicate class labels with light normalize + fuzzy + optional embeddings."""

        # define phrases to remove for normalization -- entire phrases can also be put here
        _DROP_PHRASES = [
            "finding of",
            "finding by site",
            "other",
            "unspecified",
            "general observation of patient",
            "clinical history/examination observable",
            "finding",
            "measurements",
            "measurement",
            "measure",
            "by site",
            "by intent",
            "by method",
            "action",
            "observation",
            "finding of",
            "procedure by site",
            "procedure on body region",
            "procedure on body part",
            "structure of solid organ transplantation",
            "mediastinum implantation",
            "trunk implantation",
            "mediastinal finding",
            "related procedure",
            "of trunk structure",
            "introduction of",
            "biomedical equipment procedure",
            "introduction",
            "thorax implantation",
            "procedure on organ",
            "thoracic surgical procedure",
            "lung &/or mediastinum operations",
            "body wall and cavity procedures",
            "insertion procedure",
            "general finding of soft tissue",
            "procedure on body system",
            "blood vessel feature",
            "finding of region of thorax",
            "insertion of tube",
            "disorder by body site",
            "disorder of body system",
            "preventive intent",
            "interpretation of findings",
            "specific test feature",
            "markers",
            "markers of",
            "study",
            "measurement of",
            "attribute",
            "observation regimes",
            "personal and environmental management regime",
            "diagnostic procedure by site",
            "introduction procedure",
            "fluid observable",
            "examination by method",
            "assessment regimes",
            "anesthetic observable",
            "evacuation procedure",
            "procedure related observable",
            "procedure by focus",
            "navigational concept",
            "panel",
            "function",
            "panel",
            "markers",
            "marker",
            "status",
            "measurement",
            "study",
            "test",
            "monitoring",
            "symptom",
            "feature",
            "investigation",
            "tests",
            "function tests",
            "function test",
        ]

        # compile with flexible whitespace and reorder
        _DROP_PHRASES = sorted(_DROP_PHRASES, key=len, reverse=True)
        _DROP_PHRASES = [re.compile(r"\b" + r"\s+".join(map(re.escape, phrase.split())) + r"\b") for phrase in _DROP_PHRASES]

        # exact concept classes that should be removed after normalization
        CLASS_DENY_EXACT = {
            "appliance",
            "equipment",
            "finding",
            "substance",
            "action",
            "symptoms",
            "chronic",
            "disease",
            "feature",
            "function",
            "testing",
            "functional",
            "anatomic",
            "techniques",
            "evaluation",
        }

        def normalize_class_label(s: str) -> str:
            """Light, conservative normalization for concept class labels."""
            s = s.lower().strip()
            s = re.sub(r"[(),:_]+", " ", s)
            s = re.sub(r"[-–—]", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            # remove common scaffolding phrases
            for pat in _DROP_PHRASES:
                s = pat.sub(" ", s)
            # collapse whitespace again
            s = re.sub(r"\s+", " ", s).strip()
            return s

        def drop_unhelpful_class(s: str | None) -> bool:
            """Return True if a class label is too generic/noisy and should be dropped."""
            if not s:
                return True
            if s in CLASS_DENY_EXACT:
                return True
            # Extremely short labels are rarely helpful as classes
            if len(s) <= 1:
                return True
            return False

        def _precluster_strings(strings: list[str], threshold: int = 90) -> list[list[str]]:
            """
            Group near-duplicate strings using token_set_ratio into connected components.
            Uses simple length bucketing to keep O(n^2) manageable for a few thousand strings.
            """
            strings = list(strings)
            n = len(strings)
            if n == 0:
                return []

            # buckets by length // 5 to avoid comparing very different lengths
            buckets = defaultdict(list)
            for i, s in enumerate(strings):
                buckets[len(s) // 5].append(i)

            adj = [[] for _ in range(n)]
            for _, idxs in buckets.items():
                m = len(idxs)
                for ii in range(m):
                    for jj in range(ii + 1, m):
                        a, b = idxs[ii], idxs[jj]
                        if len(strings[a]) <= 5 and len(strings[b]) <= 5:  # short terms require more stringent fuzzy threshold
                            threshold_upd = min(threshold + 10, 100)
                        else:
                            threshold_upd = threshold
                        if fuzz.ratio(strings[a], strings[b]) >= threshold_upd:
                            adj[a].append(b)
                            adj[b].append(a)

            # connected components
            visited, groups = set(), []
            for i in range(n):
                if i in visited:
                    continue
                stack, comp = [i], []
                while stack:
                    k = stack.pop()
                    if k in visited:
                        continue
                    visited.add(k)
                    comp.append(k)
                    stack.extend(adj[k])
                groups.append([strings[k] for k in comp])

            return groups

        def _semantic_merge_groups(groups: list[list[str]], embed_threshold: float = 90) -> list[list[str]]:
            """
            Merge groups whose representatives are semantically equivalent using embeddings.
            Only requires embeddings for one representative per group.
            """
            if self.emb_model is None or len(groups) <= 1:
                return groups

            # 1) Collect eligible terms per group
            eligible_terms_per_group = []
            eligible_group_idx = []
            for gi, g in enumerate(groups):
                terms = [s for s in g if len(s) >= self.min_alias_len]
                if terms:
                    eligible_terms_per_group.append(terms)
                    eligible_group_idx.append(gi)

            # If fewer than 2 groups have eligible terms, nothing to semantically merge
            if len(eligible_group_idx) < 2:
                return groups

            # 2) Embed all uncached eligible terms (dedup across groups)
            all_terms = list({t for terms in eligible_terms_per_group for t in terms})
            uncached_terms = [t for t in all_terms if t not in self.embedding_cache]
            if uncached_terms:
                new_embs = self.emb_model.encode(uncached_terms, normalize_embeddings=True, show_progress_bar=False)
                for t, emb in zip(uncached_terms, new_embs, strict=False):
                    self.embedding_cache[t] = emb

            # 3) Compute mean embedding per eligible group, then L2-normalize
            group_means = []
            for terms in eligible_terms_per_group:
                M = np.vstack([self.embedding_cache[t] for t in terms])
                mean = M.mean(axis=0)
                # normalize the mean to unit length (avoid division by zero)
                norm = np.linalg.norm(mean)
                if norm > 0:
                    mean = mean / norm
                group_means.append(mean)

            # 4) Cosine similarity between group means (dot product since unit-length)
            means_mat = np.vstack(group_means)  # shape: (E, D) for E eligible groups
            sims = means_mat @ means_mat.T

            # Union-find over all groups; only union among eligible ones
            parent = list(range(len(groups)))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            thr = embed_threshold / 100.0
            E = len(eligible_group_idx)
            for i in range(E):
                for j in range(i + 1, E):
                    if sims[i, j] >= thr:
                        gi, gj = eligible_group_idx[i], eligible_group_idx[j]
                        union(gi, gj)

            # Collate merged groups
            merged = defaultdict(list)
            for idx, g in enumerate(groups):
                merged[find(idx)].extend(g)

            return list(merged.values())

        # 1) Normalize + collect originals per normalized key, drop junk
        norm_to_originals = defaultdict(set)
        for cls in concept_classes:
            raw = (cls or "").strip()
            if not raw:
                continue
            norm = normalize_class_label(raw)
            if drop_unhelpful_class(norm):
                continue
            norm_to_originals[norm].add(raw)

        # Early exit
        if not norm_to_originals:
            return {}

        # 2) Fuzzy pre-cluster over normalized keys
        norm_keys = list(norm_to_originals.keys())
        pre_groups = _precluster_strings(norm_keys, threshold=fuzzy_threshold)
        # pre_groups = [[k] for k in norm_keys] # disable pregrouping entirely

        # 3) Optional semantic merge (embeddings) with min_alias_len filtering
        pre_groups = _semantic_merge_groups(pre_groups, embed_threshold=embed_threshold)

        # 4) For each (possibly merged) group of normalized keys, choose canonical and assemble aliases
        output_dict: dict[str, list[str]] = {}

        for group_norm_keys in pre_groups:
            # Gather all originals for every normalized member in the group
            originals = set()
            for nk in group_norm_keys:
                originals.update(norm_to_originals[nk])

            # Canonical = shortest normalized label
            canonical_norm = sorted(group_norm_keys, key=len)[0]

            # Prefer an original whose normalized form equals canonical_norm, else fallback to normalized text
            candidates = [o for o in originals if normalize_class_label(o) == canonical_norm]
            canonical = sorted(candidates, key=len)[0] if candidates else canonical_norm

            aliases = sorted(o.lower().strip() for o in originals)
            output_dict[canonical.lower().strip()] = aliases

            aliases = sorted(o.lower().strip() for o in originals)
            output_dict[canonical.lower().strip()] = aliases

        def expand_aliases_remove_terms(alias_dict, removable_terms):
            expanded = {}
            for canonical, aliases in alias_dict.items():
                new_aliases = set(aliases)
                for alias in aliases:
                    words = alias.split()
                    indices_to_remove = [i for i, word in enumerate(words) if word in removable_terms]
                    for r in range(1, len(indices_to_remove) + 1):
                        for subset in itertools.combinations(indices_to_remove, r):
                            reduced_words = [word for i, word in enumerate(words) if i not in subset]
                            if reduced_words:
                                new_aliases.add(" ".join(reduced_words))
                expanded[canonical] = sorted(new_aliases)
            return expanded

        output_dict = expand_aliases_remove_terms(output_dict, ["function", "panel", "markers", "marker", "status", "measurement", "study", "test"])

        output_dict = {key: [value for value in values if value is not None] for key, values in output_dict.items() if key is not None}  # remove None entries
        output_dict = {key: list(set(values + [key])) for key, values in output_dict.items() if key is not None}  # remove duplicates

        return output_dict

    def expand_terms_list_with_aliases(self, input_list: list[str], term_dict: dict[str, list[str]]) -> list[str]:
        """Expand a term list by its full alias groups."""

        # Step 1: Build reverse map: term → all canonical+alias terms it's ever associated with
        term_to_group = defaultdict(set)
        for canonical, aliases in term_dict.items():
            full_set = set([canonical] + aliases)
            for term in full_set:
                term_to_group[term].update(full_set)

        # Step 2: Expand input list using reverse lookup
        expanded_set = set(input_list)
        for term in input_list:
            if term in term_to_group:
                expanded_set.update(term_to_group[term])

        return list(expanded_set)

    def expand_concept_class_aliases_from_term_dict(
        self,
        concept_class_dict: dict[str, list[str]],
        term_dict: dict[str, list[str]],
        fuzzy_threshold: int = 90,
        emb_threshold: int = 90,
        match_ratio_threshold: float = 0.30,
    ) -> dict[str, list[str]]:
        """Augment class aliases using the best-matching term alias set (fuzzy+embedding)."""

        # Precompute all concept class sets and term sets
        concept_sets = {k: set([k] + (v or [])) for k, v in concept_class_dict.items()}
        term_sets = {k: set([k] + (v or [])) for k, v in term_dict.items()}

        # Step 1: Embed all terms needed (skip if already cached)
        all_terms = set()
        for s in concept_sets.values():
            all_terms.update(t for t in s if len(t) >= self.min_alias_len)
        for s in term_sets.values():
            all_terms.update(t for t in s if len(t) >= self.min_alias_len)

        to_embed = [t for t in all_terms if t not in self.embedding_cache]
        if to_embed:
            embs = self.emb_model.encode(to_embed, normalize_embeddings=True, show_progress_bar=False)
            for t, emb in zip(to_embed, embs, strict=False):
                self.embedding_cache[t] = emb

        # Step 2: Match and expand
        expanded_dict = {}

        for c_key, c_set in concept_sets.items():
            best_match = None
            best_ratio = 0.0

            for _t_key, t_set in term_sets.items():
                union_size = len(c_set.union(t_set))
                if union_size == 0:
                    continue

                # Count C → T matches
                match_count_ct = 0
                for a in c_set:
                    matched = False
                    for b in t_set:
                        if fuzz.ratio(a, b) >= fuzzy_threshold:
                            matched = True
                            break
                    if not matched and len(a) >= self.min_alias_len:
                        if a in self.embedding_cache:
                            emb_b_list = [self.embedding_cache[b] for b in t_set if b in self.embedding_cache and len(b) >= self.min_alias_len]
                            if not emb_b_list:
                                continue
                            emb_b = np.vstack(emb_b_list)
                            sims = np.dot(emb_b, self.embedding_cache[a])
                            if np.any(sims >= emb_threshold / 100):
                                matched = True
                    if matched:
                        match_count_ct += 1

                # Count T → C matches
                match_count_tc = 0
                for b in t_set:
                    matched = False
                    for a in c_set:
                        if fuzz.ratio(b, a) >= fuzzy_threshold:
                            matched = True
                            break
                    if not matched and len(b) >= self.min_alias_len:
                        if b in self.embedding_cache:
                            emb_c_list = [self.embedding_cache[a] for a in c_set if a in self.embedding_cache and len(a) >= self.min_alias_len]
                            if not emb_c_list:
                                continue
                            emb_c = np.vstack(emb_c_list)
                            sims = np.dot(emb_c, self.embedding_cache[b])
                            if np.any(sims >= emb_threshold / 100):
                                matched = True
                    if matched:
                        match_count_tc += 1

                # Compute average match ratio
                match_ratio = (match_count_ct + match_count_tc) / (2 * union_size)

                if match_ratio >= match_ratio_threshold and match_ratio > best_ratio:
                    best_ratio = match_ratio
                    best_match = t_set

            # Expand with best-matching term set
            expanded_aliases = set(concept_class_dict.get(c_key, []))  # exclude canonical key
            if best_match:
                expanded_aliases.update(best_match)
                expanded_aliases.discard(c_key)

            expanded_dict[c_key] = sorted(expanded_aliases)

        expanded_dict = {key: [value for value in values if value is not None] for key, values in expanded_dict.items() if key is not None}  # remove None entries
        expanded_dict = {key: list(set(values + [key])) for key, values in expanded_dict.items() if key is not None}  # remove duplicates

        return expanded_dict

    def _prune_and_collapse_vocab(self):
        """Prune to observed terms, merge equivalents, build alias→canonical maps, and rewrite data."""

        # get all entities and concept classes that actually appear in data
        entities = [ent for patient in self.master_list for ent in patient[7]]

        def normalize(s: str) -> str:
            return re.sub(r"\s+", " ", s).lower().strip()

        all_terms_list = list({normalize(ent["normalized_term"]) for ent in entities})
        concept_class_list = [c for c in {normalize(cls) for ent in entities for cls in ent["concept_classes"]} if len(c) >= self.min_alias_len]

        self.c_to_a_dict = {normalize(key): [normalize(v) for v in values] for key, values in self.c_to_a_dict.items()}

        # create term dict
        c_to_a_dict_filtered = self.filter_vocab_by_usage(self.c_to_a_dict, list(set(all_terms_list + concept_class_list)))
        self.c_to_a_dict = self.merge_canonical_alias_sets_by_embeddings(
            self.c_to_a_dict,
            a_to_c_resolver=self.reverse_can_vocab_to_aliases_dict(c_to_a_dict_filtered),
            fuzzy_threshold=95,
            emb_threshold=90,
            match_ratio_threshold=0.30,
        )

        # create class aliases dict
        self.c_to_a_dict_concept_classes = self.deduplicate_concept_classes(concept_class_list, fuzzy_threshold=95, embed_threshold=85)
        self.c_to_a_dict_concept_classes = self.merge_canonical_alias_sets_by_embeddings(
            self.c_to_a_dict_concept_classes,
            a_to_c_resolver=self.reverse_can_vocab_to_aliases_dict(self.c_to_a_dict_concept_classes),
            fuzzy_threshold=95,
            emb_threshold=85,
            match_ratio_threshold=0.20,
        )
        self.c_to_a_dict_concept_classes = self.expand_concept_class_aliases_from_term_dict(
            self.c_to_a_dict_concept_classes,
            c_to_a_dict_filtered,
            fuzzy_threshold=95,
            emb_threshold=85,
            match_ratio_threshold=0.20,
        )
        self.c_to_a_dict_concept_classes = self.merge_canonical_alias_sets_by_embeddings(
            self.c_to_a_dict_concept_classes,
            a_to_c_resolver=self.reverse_can_vocab_to_aliases_dict(self.c_to_a_dict_concept_classes),
            fuzzy_threshold=95,
            emb_threshold=85,
            match_ratio_threshold=0.20,
        )

        # apply final ontology corrections
        self.c_to_a_dict, self.concept_class_map, self.c_to_a_dict_concept_classes = self.apply_ontology_corrections(
            self.c_to_a_dict,
            self.concept_class_map,
            self.c_to_a_dict_concept_classes,
            a_to_c_resolver=self.reverse_can_vocab_to_aliases_dict(c_to_a_dict_filtered),
            post_edits=True,
        )
        self.a_to_c_dict = self.reverse_can_vocab_to_aliases_dict(self.c_to_a_dict)
        self.a_to_c_dict_concept_classes = self.reverse_can_vocab_to_aliases_dict(self.c_to_a_dict_concept_classes)

        # now choose whether or not to prune vocab
        if not self.no_pruning:
            self.c_to_a_dict = self.filter_vocab_by_usage(self.c_to_a_dict, list(set(all_terms_list + concept_class_list)))
            self.a_to_c_dict = self.reverse_can_vocab_to_aliases_dict(self.c_to_a_dict)

        # log vocab stats
        log.info("vocab size: " + str(len(self.c_to_a_dict) + len(self.c_to_a_dict_concept_classes)))

        # cycle through the dataset and replace everything with its canonical term
        unmapped_term_list, unmapped_class_list = [], []
        for master_list_i in range(len(self.master_list)):
            drop_indices = []
            for c_ent_i in range(len(self.master_list[master_list_i][7])):
                ent = self.master_list[master_list_i][7][c_ent_i]

                # replace term
                normalized_term = normalize(ent["normalized_term"])
                if normalized_term in self.a_to_c_dict:
                    ent["normalized_term"] = self.a_to_c_dict[normalized_term]
                else:
                    drop_indices.append(c_ent_i)
                    unmapped_term_list.append(normalized_term)

                # replace classes
                unmapped_class_list += [normalize(item) for item in ent["concept_classes"] if normalize(item) not in self.a_to_c_dict_concept_classes]
                ent["concept_classes"] = list(set(self.a_to_c_dict_concept_classes[normalize(item)] for item in ent["concept_classes"] if normalize(item) in self.a_to_c_dict_concept_classes))

            # drop c_ent entries that were not mapped
            for c_ent_i in sorted(drop_indices, reverse=True):
                del self.master_list[master_list_i][7][c_ent_i]

    def _post_process_entities(self):
        """Drop None, split BP 120/80 into SBP/DBP, strip stray slashed values for non-BP."""

        # also removes any values that contain a '/' if they are not blood pressure entities
        for master_list_i in range(len(self.master_list)):
            new_cent_list = []
            for c_ent_i in range(len(self.master_list[master_list_i][7])):
                if self.master_list[master_list_i][7][c_ent_i]["normalized_term"] is not None:
                    if (
                        {"blood pressure", "systolic blood pressure", "diastolic blood pressure"} <= self.a_to_c_dict.keys()
                        and self.master_list[master_list_i][7][c_ent_i]["normalized_term"] == self.a_to_c_dict["blood pressure"]
                        and len(self.master_list[master_list_i][7][c_ent_i]["values"]) > 0
                    ):
                        for value_i in range(len(self.master_list[master_list_i][7][c_ent_i]["values"])):
                            if type(self.master_list[master_list_i][7][c_ent_i]["values"][value_i]["value"]) is str and "/" in self.master_list[master_list_i][7][c_ent_i]["values"][value_i]["value"]:
                                new_sbp_cent_entry = deepcopy(self.master_list[master_list_i][7][c_ent_i])
                                new_dbp_cent_entry = deepcopy(self.master_list[master_list_i][7][c_ent_i])
                                new_sbp_cent_entry["normalized_term"] = self.a_to_c_dict["systolic blood pressure"]
                                new_dbp_cent_entry["normalized_term"] = self.a_to_c_dict["diastolic blood pressure"]
                                new_sbp_cent_entry["concept_classes"] = self.concept_class_map[new_sbp_cent_entry["normalized_term"]]
                                new_dbp_cent_entry["concept_classes"] = self.concept_class_map[new_dbp_cent_entry["normalized_term"]]
                                new_sbp_value_entry = deepcopy(new_sbp_cent_entry["values"][value_i])
                                new_dbp_value_entry = deepcopy(new_dbp_cent_entry["values"][value_i])
                                new_sbp_value_entry["value"] = int(new_sbp_value_entry["value"].split("/")[0])
                                new_dbp_value_entry["value"] = int(new_dbp_value_entry["value"].split("/")[1])
                                new_sbp_cent_entry["values"] = [new_sbp_value_entry]
                                new_dbp_cent_entry["values"] = [new_dbp_value_entry]
                                new_cent_list.append(new_sbp_cent_entry)
                                new_cent_list.append(new_dbp_cent_entry)
                    elif len(self.master_list[master_list_i][7][c_ent_i]["values"]) > 0:
                        self.master_list[master_list_i][7][c_ent_i]["values"] = [
                            item for item in self.master_list[master_list_i][7][c_ent_i]["values"] if not (type(item["value"]) is str and "/" in item["value"])
                        ]
                        new_cent_list.append(self.master_list[master_list_i][7][c_ent_i])
                    else:  # neither None entry nor bp entry nor has any values
                        new_cent_list.append(self.master_list[master_list_i][7][c_ent_i])
            self.master_list[master_list_i][7] = new_cent_list

    def _produce_term_stats_output(self, term_stats_csv: tuple[str, str]):
        """Write CSVs of resolved vs. unresolved term strings with counts."""

        resolved_entities_list = [(key, value[0][0][0], value[1]) for key, value in self.resolved_cent_concept_cache.items() if value[0] is not None]
        unresolved_entities_list = [(key, None, value[1]) for key, value in self.resolved_cent_concept_cache.items() if value[0] is None]
        resolved_entities_list = sorted(resolved_entities_list, key=lambda elem: elem[2], reverse=True)
        unresolved_entities_list = sorted(unresolved_entities_list, key=lambda elem: elem[2], reverse=True)
        pd.DataFrame(resolved_entities_list, columns=["raw_string", "resolved_term", "count"]).to_csv(term_stats_csv[0], index=False)
        pd.DataFrame(unresolved_entities_list, columns=["raw_string", "resolved_term", "count"]).to_csv(term_stats_csv[1], index=False)

    def transpose_time_series_table(self, table: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Transpose row-per-time tables to a canonical time-keyed form."""

        TIME_LIKE_PATTERNS = [r"\bday\b", r"\bd\s*(?:\s?#\s*)?\d+\b", r"\bhd\s*(?:\s?#\s*)?\d+\b", r"\bd\d+\b", r"\bhd\d+\b"]
        DATE_PATTERNS = [r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$", r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}$", r"^\d{1,2}[-/]\d{1,2}$"]

        def is_time_like(s: str) -> bool:
            s_lower = s.lower().strip()
            for pattern in TIME_LIKE_PATTERNS:
                if re.search(pattern, s_lower):
                    return True
            for pattern in DATE_PATTERNS:
                if re.fullmatch(pattern, s):
                    return True
            return False

        def should_transpose(table: list[dict[str, Any]]) -> bool:
            for row in table:
                keys = list(row.keys())
                time_keys = [k for k in keys if is_time_like(k)]
                if len(keys) - len(time_keys) != 1:
                    return False
            return True

        if not table or not should_transpose(table):
            return table

        # Identify label key (e.g., 'Measure' or 'Parameter')
        first_row = table[0]
        label_key = next(k for k in first_row if not is_time_like(k))

        # Collect time keys and row labels
        time_keys = [k for k in first_row if k != label_key]

        transposed = []
        for time_key in time_keys:
            new_row = {"time": time_key}
            for row in table:
                label = row[label_key]
                new_row[label] = row.get(time_key)
            transposed.append(new_row)

        return transposed

    def add_standardized_entities(self, standardized_entities_list, cent_text, time_entries_list, negation_entries_list, adm_timestamp, debug_tuple):
        """Resolve one chunk to canonical term, values, modifier, timestamp, and negation."""

        cent_text, time_entries_list, negation_entries_list = (
            re.sub(r"\s+", " ", cent_text.lower().strip()),
            [re.sub(r"\s+", " ", item.lower().strip()) for item in time_entries_list],
            [re.sub(r"\s+", " ", item.lower().strip()) for item in negation_entries_list],
        )

        time_entries_list_copy = deepcopy(time_entries_list)
        negation_entries_list_copy = deepcopy(negation_entries_list)
        resolved_negation, cent_text, time_entries_list, negation_entries_list = self.resolve_cent_negation(cent_text, time_entries_list, negation_entries_list)
        resolved_timestamp, cent_text, time_entries_list, negation_entries_list = self.resolve_cent_timestamp(cent_text, time_entries_list, negation_entries_list, adm_timestamp)
        resolved_modifier, cent_text, time_entries_list, negation_entries_list = self.resolve_cent_modifier(cent_text, time_entries_list, negation_entries_list)
        resolved_values, cent_text = self.resolve_cent_value(cent_text)
        cent_text, time_entries_list, negation_entries_list = (
            re.sub(r"[-–—]", " ", cent_text),
            [re.sub(r"[-–—]", " ", item) for item in time_entries_list],
            [re.sub(r"[-–—]", " ", item) for item in negation_entries_list],
        )  # replace dashes with a space because in the functions that follow dashes do not have any additive information

        # now handle the idenified values
        if len(resolved_values) > 0:
            resolved_values_filtered = []
            for resolved_value in resolved_values:
                if resolved_value["unit"] is not None:
                    # check if it is a normalized concept -- if yes create entirely new entry
                    resolved_concept = self.resolve_cent_concept(
                        resolved_value["unit"],
                        self.a_to_c_dict,
                        list(self.a_to_c_dict.keys()),
                        fuzzy_threshold=85,
                        embedding_threshold=88,
                        embedding_matching_prefilter=75,
                        top_n=7,
                    )
                    if resolved_concept is not None:
                        # append to standardized entities list
                        resolved_concept = resolved_concept[0][0]
                        resolved_concept_classes = self.concept_class_map[resolved_concept]
                        resolved_value["unit"] = None
                        standardized_entities_list.append(
                            {
                                "normalized_term": resolved_concept,
                                "concept_classes": resolved_concept_classes,
                                "values": [resolved_value],
                                "modifier": resolved_modifier,
                                "timestamp": resolved_timestamp,
                                "negated": resolved_negation,
                                "timestamp_texts": time_entries_list_copy,
                                "negation_texts": negation_entries_list_copy,
                                "source": debug_tuple,
                            }
                        )
                    else:
                        resolved_values_filtered.append(resolved_value)
                else:
                    resolved_values_filtered.append(resolved_value)
        else:
            resolved_values_filtered = resolved_values
        if len(resolved_values_filtered) > 0 or len(resolved_values) == 0:
            resolved_concept = self.resolve_cent_concept(
                cent_text,
                self.a_to_c_dict,
                list(self.a_to_c_dict.keys()),
                fuzzy_threshold=85,
                embedding_threshold=88,
                embedding_matching_prefilter=75,
                top_n=7,
            )
            if resolved_concept is not None:
                resolved_concept = resolved_concept[0][0]
                resolved_concept_classes = self.concept_class_map[resolved_concept]
            else:
                resolved_concept_classes = []
            standardized_entities_list.append(
                {
                    "normalized_term": resolved_concept,
                    "concept_classes": resolved_concept_classes,
                    "values": resolved_values_filtered,
                    "modifier": resolved_modifier,
                    "timestamp": resolved_timestamp,
                    "negated": resolved_negation,
                    "timestamp_texts": time_entries_list_copy,
                    "negation_texts": negation_entries_list_copy,
                    "source": debug_tuple,
                }
            )

        return standardized_entities_list

    def add_standardized_entities_wrapper(self, standardized_entities_list, cent_text, time_entries_list, negation_entries_list, adm_timestamp, debug_tuple):
        """Resolve a raw entity+timestamp+negation stamp."""

        # ensure that inputs are strings (defensive)
        cent_text, time_entries_list, negation_entries_list = str(cent_text), [str(item) for item in time_entries_list], [str(item) for item in negation_entries_list]

        standardized_entities_list_new = deepcopy(standardized_entities_list)
        standardized_entities_list_new = self.add_standardized_entities(standardized_entities_list_new, cent_text, time_entries_list, negation_entries_list, adm_timestamp, debug_tuple)

        if len(standardized_entities_list_new) > len(standardized_entities_list) and any(
            [standardized_entities_list_new[i]["normalized_term"] is not None for i in range(len(standardized_entities_list), len(standardized_entities_list_new))]
        ):
            return standardized_entities_list_new
        else:
            # try to split cent_text by commas/colons/parentheses and see if this resolves any entities
            standardized_entities_list_new = deepcopy(standardized_entities_list)
            cent_text_parts = re.split(r"[:,()]", cent_text)
            for cent_text_part in cent_text_parts:
                standardized_entities_list_new = self.add_standardized_entities(
                    standardized_entities_list_new,
                    cent_text_part.strip(),
                    time_entries_list,
                    negation_entries_list,
                    adm_timestamp,
                    debug_tuple,
                )
            if len(standardized_entities_list_new) > len(standardized_entities_list) and any(
                [standardized_entities_list_new[i]["normalized_term"] is not None for i in range(len(standardized_entities_list), len(standardized_entities_list_new))]
            ):
                return standardized_entities_list_new
            else:
                # try to split cent by ' for '
                standardized_entities_list_new = deepcopy(standardized_entities_list)
                cent_text_parts = cent_text.split(" for ")
                for cent_text_part in cent_text_parts:
                    standardized_entities_list_new = self.add_standardized_entities(
                        standardized_entities_list_new,
                        cent_text_part.strip(),
                        time_entries_list,
                        negation_entries_list,
                        adm_timestamp,
                        debug_tuple,
                    )
                if len(standardized_entities_list_new) > len(standardized_entities_list) and any(
                    [standardized_entities_list_new[i]["normalized_term"] is not None for i in range(len(standardized_entities_list), len(standardized_entities_list_new))]
                ):
                    return standardized_entities_list_new
                else:
                    return standardized_entities_list

    def clean_str_for_table_parsing(self, input_str):
        """Light cleanup for table cells to avoid parser collisions."""

        input_str = re.sub(r"[()]", "", input_str).lower().strip()  # remove parentheses (individual or combined) as these frequently contain units
        input_str = re.sub(r"\(k\)", "", input_str, flags=re.IGNORECASE)  # remove '(k)' patterns
        input_str = input_str.replace("alk phos", "alk-phos")
        input_str = input_str.replace("t. bili", "tbili")
        return input_str

    def connect_substrings_with_underscore(self, text: str) -> str:
        """Replace spaces with underscores outside parentheses."""

        # converts entries like 'substr1 substr2 (substr3)' into 'substr1_substr2 (substr3)'
        segments = []
        pattern = re.compile(r"\([^\)]*\)")  # match non-nested parentheses
        last_end = 0

        for match in pattern.finditer(text):
            # Text before parentheses — transform
            before = text[last_end : match.start()]
            transformed = re.sub(r"(?<=\S) (?=\S)", "_", before)
            segments.append(transformed)

            # Parentheses content — preserve as-is
            segments.append(match.group(0))
            last_end = match.end()

        # Process remaining text after last parenthesis group
        after = text[last_end:]
        transformed = re.sub(r"(?<=\S) (?=\S)", "_", after)
        segments.append(transformed)

        return "".join(segments)

    def resolve_cent_concept(
        self,
        text,
        alias_to_canonical,
        all_aliases_list,
        fuzzy_threshold=90,
        embedding_threshold=90,
        embedding_matching_prefilter=70,
        top_n=5,
    ):
        """Resolve text to top canonical candidates via fuzzy prefilter + embedding tie-break."""

        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        cleaned = re.sub(r"[.,:;<>()~]+", "", text)  # removes parentheses, commas, full stops, colons, semicolons, <>~ chars
        cleaned = re.sub(r"[-–—_]", " ", cleaned)  # replace underscores and dashes with a whitespace
        cleaned = cleaned.lower().strip()
        cleaned = re.sub(r"\s+", " ", cleaned)  # replace multiple contiguous spaces with a single space

        if len(cleaned) <= 4:
            fuzzy_threshold = min(fuzzy_threshold + 5, 95)  # short strings need a higher threshold

        if cleaned in self.resolved_cent_concept_cache.keys():
            self.resolved_cent_concept_cache[cleaned][1] += 1
            return self.resolved_cent_concept_cache[cleaned][0]

        # Step 1: Fuzzy match
        initial_fast_matches = process.extract(cleaned, all_aliases_list, scorer=fuzz.ratio, score_cutoff=embedding_matching_prefilter, limit=top_n * 100)
        reranked_matches = sorted(
            [(alias, fuzz.token_sort_ratio(cleaned, alias)) for alias, _, _ in initial_fast_matches],
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

        # Apply short-alias safety filter
        fuzzy_filtered = []
        for alias, score in reranked_matches:
            if len(alias) <= 3:
                if alias in cleaned and len(cleaned) <= len(alias) + 1:
                    fuzzy_filtered.append((alias, alias_to_canonical[alias], score))
            else:
                fuzzy_filtered.append((alias, alias_to_canonical[alias], score))

        if not fuzzy_filtered:
            self.resolved_cent_concept_cache[cleaned] = [None, 1]
            return None

        # Group fuzzy matches by canonical term (deduplicated)
        fuzzy_by_canon = {}
        for alias, canon, score in fuzzy_filtered:
            if canon not in fuzzy_by_canon or score > fuzzy_by_canon[canon][1]:
                fuzzy_by_canon[canon] = (alias, score)

        fuzzy_sorted = sorted(fuzzy_by_canon.items(), key=lambda x: x[1][1], reverse=True)

        # Case 1: Top match is high confidence
        if fuzzy_sorted[0][1][1] >= fuzzy_threshold:
            top_score = fuzzy_sorted[0][1][1]
            close_matches = [item for item in fuzzy_sorted if abs(item[1][1] - top_score) < 1]

            # If no tie, return top fuzzy match(es)
            if len(close_matches) == 1:
                self.resolved_cent_concept_cache[cleaned] = [[(canon, score) for canon, (_, score) in close_matches], 1]
                return self.resolved_cent_concept_cache[cleaned][0]

            # Tie exists — use embedding similarity among tied aliases
            if cleaned in self.embedding_cache.keys():
                query_emb = self.embedding_cache[cleaned]
            else:
                query_emb = self.emb_model.encode(cleaned, normalize_embeddings=True, show_progress_bar=False)
                self.embedding_cache[cleaned] = query_emb

            emb_by_canon = {}

            for canon, (alias, _) in close_matches:
                if len(alias) < self.min_alias_len:
                    continue
                if alias in self.embedding_cache.keys():
                    emb = self.embedding_cache[alias]
                else:
                    emb = self.emb_model.encode(alias, normalize_embeddings=True, show_progress_bar=False)
                    self.embedding_cache[alias] = emb
                if emb is None:
                    continue
                emb_score = cosine_sim(query_emb, emb)
                if emb_score >= embedding_threshold / 100:
                    score_percent = round(emb_score * 100, 2)
                    if canon not in emb_by_canon or score_percent > emb_by_canon[canon]:
                        emb_by_canon[canon] = score_percent

            if emb_by_canon:
                self.resolved_cent_concept_cache[cleaned] = [sorted(emb_by_canon.items(), key=lambda x: x[1], reverse=True), 1]
                return self.resolved_cent_concept_cache[cleaned][0]

            # If embedding didn't help, return fuzzy tie set
            self.resolved_cent_concept_cache[cleaned] = [[(canon, score) for canon, (_, score) in close_matches], 1]
            return self.resolved_cent_concept_cache[cleaned][0]

        # Case 2: Fuzzy matches exist but are all below fuzzy_threshold
        # Use all fuzzy_filtered (>= prefilter threshold) in embedding matching
        if cleaned in self.embedding_cache.keys():
            query_emb = self.embedding_cache[cleaned]
        else:
            query_emb = self.emb_model.encode(cleaned, normalize_embeddings=True, show_progress_bar=False)
            self.embedding_cache[cleaned] = query_emb

        emb_by_canon = {}

        for alias, canon, _ in fuzzy_filtered:
            if len(alias) < self.min_alias_len:
                continue
            if alias in self.embedding_cache.keys():
                emb = self.embedding_cache[alias]
            else:
                emb = self.emb_model.encode(alias, normalize_embeddings=True, show_progress_bar=False)
                self.embedding_cache[alias] = emb
            if emb is None:
                continue
            emb_score = cosine_sim(query_emb, emb)
            if emb_score >= embedding_threshold / 100:
                score_percent = round(emb_score * 100, 2)
                if canon not in emb_by_canon or score_percent > emb_by_canon[canon]:
                    emb_by_canon[canon] = score_percent

        if emb_by_canon:
            self.resolved_cent_concept_cache[cleaned] = [sorted(emb_by_canon.items(), key=lambda x: x[1], reverse=True), 1]
            return self.resolved_cent_concept_cache[cleaned][0]

        # No matches
        self.resolved_cent_concept_cache[cleaned] = [None, 1]
        return None

    def resolve_cent_negation(self, cent_text, time_texts, negation_texts):
        """Detect/strip negation across cent/time/neg spans; return flag and cleaned texts."""

        NEGATION_TERMS = {
            "no",
            "not",
            "none",
            "denies",
            "denied",
            "without",
            "negative for",
            "never",
            "absence of",
            "no known family history of",
            "no evidence of",
            "free of",
            "rules out",
            "rule out",
            "ruled out",
            "no hx of",
            "no known",
            "no new",
            "no signs of",
            "off",
            "negative",
        }

        # Phrases where "off" should not trigger negation
        OFF_EXCEPTIONS = [r"weaned off", r"tapered off", r"came off", r"taken off"]

        def strip_and_detect_negation(text):
            found = False
            cleaned = text.lower()

            # Pre-check: If an exception phrase is present, remove "off" from NEGATION_TERMS for this instance
            exception_detected = any(re.search(r"\b" + phrase + r"\b", cleaned) for phrase in OFF_EXCEPTIONS)

            terms_to_use = NEGATION_TERMS - {"off"} if exception_detected else NEGATION_TERMS

            for term in sorted(terms_to_use, key=len, reverse=True):
                pattern = r"\b" + re.escape(term) + r"\b"
                if re.search(pattern, cleaned):
                    found = True
                    cleaned = re.sub(pattern, "", cleaned)

            return found, cleaned.strip()

        negated = False
        cleaned_time_texts = []
        cleaned_negation_texts = []

        # 1. Check and strip NEGATION entities
        for txt in negation_texts:
            found, cleaned = strip_and_detect_negation(txt)
            if found:
                negated = True
            cleaned_negation_texts.append(cleaned)

        # 2. Check and strip TIME entities
        for txt in time_texts:
            found, cleaned = strip_and_detect_negation(txt)
            if found:
                negated = True
            cleaned_time_texts.append(cleaned)

        # 3. Check and strip C_ENT entity
        found_cent, cleaned_cent_text = strip_and_detect_negation(cent_text)
        if found_cent:
            negated = True

        return negated, cleaned_cent_text, cleaned_time_texts, cleaned_negation_texts

    def resolve_cent_timestamp(self, cent_text: str, time_texts: list[str], negation_texts: list[str], admission_timestamp: tuple[str, str | None]):
        """Extract the most specific coherent date/range/history relative to admission."""

        TimestampType = tuple[tuple[str, None], None, None] | tuple[tuple[str, None], tuple[str, None], None] | tuple[str, None, None]

        def normalize_year(year: int) -> int:
            return 2000 + year if year < 100 else year

        def normalize_day_month_year(day: int, month: int, year: int) -> str:
            if not (1 <= month <= 12 and 1 <= day <= 31):
                log.warning("warning: timestamp appears to be in a DD/MM format but this pipeline assumes a MM/DD format")
            return f"{year}-{month:02d}-{day:02d}"

        def parse_relative_day(offset: int, admission_date: datetime) -> str:
            date = admission_date + timedelta(days=offset)
            return date.strftime("%Y-%m-%d")

        def parse_relative_month(offset: int, admission_date: datetime) -> str:
            date = admission_date - timedelta(days=30 * offset)
            return date.strftime("%Y-%m-%d")

        # Parse individual timestamps
        def parse_timestamp_by_label(label: str, match: re.Match, admission_date: datetime) -> TimestampType | None:
            try:
                if label in {"a", "b"}:
                    offset = int(match.group(1))
                    date = parse_relative_day(offset, admission_date)
                    return ((date, None), None, None)

                elif label == "c":
                    month, _, day, year = map(int, match.groups())
                    year = normalize_year(int(year))
                    return ((normalize_day_month_year(day, month, year), None), None, None)

                elif label == "r":
                    year, _, month, day = map(int, match.groups())
                    year = normalize_year(int(year))
                    return ((normalize_day_month_year(day, month, year), None), None, None)

                elif label == "d":
                    month, day = map(int, match.groups())
                    return ((normalize_day_month_year(day, month, admission_date.year), None), None, None)

                elif label in {"e", "f"}:
                    if "yesterday" in match.group(0):
                        date = admission_date - timedelta(days=1)
                    else:
                        date = admission_date
                    return ((date.strftime("%Y-%m-%d"), None), None, None)

                elif label == "g":
                    offset = int(match.group(1)) if match.group(1).isdigit() else ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"].index(match.group(1).lower()) + 1
                    date = admission_date - timedelta(days=offset)
                    return ((date.strftime("%Y-%m-%d"), None), None, None)

                elif label == "h":
                    offset = int(match.group(1)) if match.group(1).isdigit() else ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"].index(match.group(1).lower()) + 1
                    date = parse_relative_month(offset, admission_date)
                    return ((date, None), None, None)

                elif label == "i":
                    m1, d1, y1, m2, d2, y2 = map(int, match.groups())
                    y1 = normalize_year(int(y1))
                    y2 = normalize_year(int(y2))
                    start = normalize_day_month_year(d1, m1, y1)
                    end = normalize_day_month_year(d2, m2, y2)
                    return ((start, None), (end, None), None)

                elif label == "j":
                    m1, d1, m2, d2 = map(int, match.groups())
                    start = normalize_day_month_year(d1, m1, admission_date.year)
                    end = normalize_day_month_year(d2, m2, admission_date.year)
                    return ((start, None), (end, None), None)

                elif label == "k":
                    start_day, end_day = map(int, match.groups())
                    start = parse_relative_day(start_day, admission_date)
                    end = parse_relative_day(end_day, admission_date)
                    return ((start, None), (end, None), None)

                elif label == "l":
                    match_days = re.search(
                        r"(?:(\d+)|(?:" + "|".join(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]) + "))",
                        match.group(0).lower(),
                    )
                    if match_days:
                        offset = int(match_days.group(1)) if match_days.group(1) else ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"].index(match_days.group(0)) + 1
                        end = admission_date
                        start = admission_date - timedelta(days=offset)
                        return ((start.strftime("%Y-%m-%d"), None), (end.strftime("%Y-%m-%d"), None), None)

                elif label == "p":
                    num_text = re.search(r"(?P<number>\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)", match.group(0).lower())
                    if num_text:
                        word = num_text.group("number")
                        offset = int(word) if word.isdigit() else ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"].index(word) + 1
                        end = admission_date
                        start = end - timedelta(days=7 * offset)
                        return ((start.strftime("%Y-%m-%d"), None), (end.strftime("%Y-%m-%d"), None), None)

                elif label == "s":
                    y1, m1, d1, y2, m2, d2 = map(int, match.groups())
                    start = normalize_day_month_year(d1, m1, y1)
                    end = normalize_day_month_year(d2, m2, y2)
                    return ((start, None), (end, None), None)

                elif label == "m":
                    return ("history", None, None)

                elif label == "n":
                    return ("baseline", None, None)

            except Exception:
                return None

        # Check if two timestamps conflict
        def are_timestamps_conflicting(t1: TimestampType, t2: TimestampType) -> bool:
            if "baseline" in str(t1) or "history" in str(t1) or "baseline" in str(t2) or "history" in str(t2):
                return False

            def to_day_tuple(t):
                if isinstance(t, tuple) and isinstance(t[0], tuple):
                    return t[0][0], t[1][0] if t[1] else None
                return t[0] if isinstance(t[0], str) else None, None

            s1, e1 = to_day_tuple(t1)
            s2, e2 = to_day_tuple(t2)

            if e1 and e2:
                # Compare two ranges
                return s1 > e2 or s2 > e1
            elif e1:
                return not (s1 <= s2 <= e1)
            elif e2:
                return not (s2 <= s1 <= e2)
            else:
                return s1 != s2

        # Step 1: Prepare texts
        admission_date = datetime.strptime(admission_timestamp[0], "%Y-%m-%d")
        original_texts = {"cent": [cent_text], "time": time_texts, "neg": negation_texts}
        clean_texts = {k: [t.lower().strip() for t in v] for k, v in deepcopy(original_texts).items()}

        # Step 2: Define patterns
        pattern_priority = {
            **dict.fromkeys(["i", "j", "k", "l", "s"], 0),  # Highest priority
            **dict.fromkeys(["a", "b", "c", "d", "e", "f", "g", "h", "m", "n", "p", "r"], 1),
            **dict.fromkeys(["o", "t"], 1),  # Lowest priority
        }
        patterns = {
            "a": r"\b(?:on\s+)?(?:icu\s*|hospital\s*)?(?:day|d)\s*([0-9]{1,2})\b",
            "b": r"\b(?:on\s+)?(?:hd|d|day)\s*#?\s*([1-9][0-9]?)\b",
            "c": r"\b(?:on\s+)?(0?[1-9]|1[0-2])([\/\-])(0?[1-9]|[12][0-9]|3[01])\2(\d{2}|\d{4})\b",  # date
            "d": r"\b(?:on\s+)?(0?[1-9]|1[0-2])/(0?[1-9]|[12][0-9]|3[01])\b",  # date (does not allow for dash-based dates)
            "e": r"\btoday\b|\byesterday\b|\bcurrently\b|\bnow\b|\bon admission\b|\bon arrival|\bin progress\b|\bcurrent\b",
            "f": r"\b(presenting to the ed|presented to the ed)( with)?\b|\bpresenting with\b|\bpresented with|\bin the ed\b|\bactive\b",
            "g": r"\b(((?:[1-9][0-9]?)|(?:one|two|three|four|five|six|seven|eight|nine|ten)))\s+days?\s+(before|prior|of|earlier)\b",
            "h": r"\b((?:[1-9][0-9]?)|(?:one|two|three|four|five|six|seven|eight|nine|ten))\s+months?\s+ago\b",
            "i": r"\b(?:on\s+)?(0?[1-9]|1[0-2])[\/\-](0?[1-9]|[12][0-9]|3[01])[\/\-](\d{2,4})\s*(?:\-|to|until)\s*(0?[1-9]|1[0-2])[\/\-](0?[1-9]|[12][0-9]|3[01])[\/\-](\d{2,4})\b",  # range
            "j": r"\b(?:on\s+)?(0?[1-9]|1[0-2])[\/\-](0?[1-9]|[12][0-9]|3[01])\s*(?:\-|to|until)\s*(0?[1-9]|1[0-2])[\/\-](0?[1-9]|[12][0-9]|3[01])\b",  # range
            "k": r"\b(?:on\s+)?(?:icu\s*)?(?:days|day|hd|d)\s*(\d{1,2})\s*(?:\-|to)\s*(\d{1,2})\b",  # range
            "l": r"\b(?:(?:presenting|presented)\s+to\s+(?:the\s+)?(?:ed|emergency department|clinic)\s+(?:with|for)\s+)?(((?:[1-9]|1[0-2])\s+days?)|(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+days?)\s+of\b",  # range
            "m": r"\bpmhx\b|\bpmh\b|\bpast surgical history\b|\bpast medical history significant for\b|\bpast medical history\b|\bpast medical hx\b|\bprior to this admission\b|\bprior to admission\b|\bhistory of\b|\bremote\b|\bprior\b|\brecent\b|\bh/x\b|\bh/o\b|\bhx\b|\bat home\b",  # history
            "n": r"\b(?:at\s+)?XXXXXXXXXXXXXX\b",  # used to be baseline, now removed
            "o": r"\b(0?[1-9]|1[0-2]):[0-5][0-9]\s*(am|pm)\b",  # time -- is stripped but ignored during time parsing
            "p": r"\b(?:x\s*)?(?P<number>\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s+weeks?\s+(?:of|ago)\b",  # range
            "r": r"\b(?:on\s+)?(\d{4})([\/\-])(0?[1-9]|1[0-2])\2(0?[1-9]|[12][0-9]|3[01])\b",  # same as c but uses YYYYMMDD order
            "s": r"\b(?:on\s+)?(\d{4})[\/\-](0?[1-9]|1[0-2])[\/\-](0?[1-9]|[12][0-9]|3[01])\s*(?:\-|to|until)\s*(\d{4})[\/\-](0?[1-9]|1[0-2])[\/\-](0?[1-9]|[12][0-9]|3[01])\b",  # range, same as i but using YYYYMMDD format
            "t": r"\b(?:in\s+)?(?:am|pm)\b",  # am/pm -- is stripped but ignored during time parsing
        }

        # Step 3: Greedy timestamp extraction
        labels_to_ignore_in_parsing = {"o"}
        all_timestamps = []
        for pass_labels in [["i", "j", "k", "l", "s"], [k for k in patterns if k not in {"i", "j", "k", "l", "s"}]]:
            for source, texts in clean_texts.items():
                for i, t in enumerate(texts):
                    found = []
                    for label in pass_labels:
                        for match in re.finditer(patterns[label], t):
                            found.append((match.group(0), match, label))
                    found.sort(key=lambda x: (pattern_priority[x[2]], -len(x[0])))
                    seen = set()
                    for full_match, match_obj, label in found:
                        if full_match in seen:
                            continue
                        seen.add(full_match)
                        if label not in labels_to_ignore_in_parsing:
                            all_timestamps.append((label, match_obj))
                        texts[i] = re.sub(re.escape(full_match), "", texts[i], flags=re.IGNORECASE).strip()
                clean_texts[source] = texts

        # Step 4: Parse timestamps
        parsed_timestamps = []
        for label, match_obj in all_timestamps:
            parsed = parse_timestamp_by_label(label, match_obj, admission_date)
            if parsed:
                parsed_timestamps.append(parsed)

        # Step 5: Resolve logic
        if not parsed_timestamps:
            return (None, None, "no timestamps"), clean_texts["cent"][0], clean_texts["time"], clean_texts["neg"]

        for i in range(len(parsed_timestamps)):
            for j in range(i + 1, len(parsed_timestamps)):
                if are_timestamps_conflicting(parsed_timestamps[i], parsed_timestamps[j]):
                    return (
                        (None, None, "logically incompatible timestamps"),
                        clean_texts["cent"][0],
                        clean_texts["time"],
                        clean_texts["neg"],
                    )

        def timestamp_priority(ts: TimestampType):
            if isinstance(ts, tuple) and isinstance(ts[0], tuple) and ts[1] is None:  # specific date
                return 0
            elif isinstance(ts, tuple) and isinstance(ts[0], tuple):  # date range
                return 1
            elif ts[0] == "baseline":
                return 2
            elif ts[0] == "history":
                return 3
            return 4

        parsed_timestamps.sort(key=timestamp_priority)
        return parsed_timestamps[0], clean_texts["cent"][0], clean_texts["time"], clean_texts["neg"]

    def resolve_cent_modifier(self, current_cent_text, current_time_texts, current_negation_texts):
        """Extract qualitative modifiers and remove their surface forms from texts."""

        MODIFIER_MAP = {
            "decreased": [
                "decreased to",
                "decreased",
                "reduced to",
                "declined",
                "trending down",
                "trended down",
                "downtrending",
                "falling",
                "lowered",
                "drop in",
                "fall in",
                "down titrated",
                "decreasing",
                "down",
            ],
            "increased": [
                "increased to",
                "increased",
                "uptrending",
                "trending up",
                "trended up",
                "rising",
                "heightened",
                "rise in",
                "surge in",
                "spiked",
                "increasing",
                "up",
            ],
            "unchanged": [
                "unchanged",
                "persistent",
                "continued",
                "ongoing",
                "no change",
                "no improvement",
                "not improving",
                "remains the same",
            ],
            "stable": ["stable", "controlled", "at baseline", "baseline"],
            "started": [
                "started",
                "started on",
                "initiated",
                "initiated on",
                "placed",
                "placement",
                "transitioned to",
                "new onset",
                "commenced",
                "new",
                "starting",
                "developed",
            ],
            "stopped": [
                "stopped",
                "removed",
                "removal",
                "discontinued",
                "dcd",
                "weaned",
                "explanted",
                "weaned and explanted",
                "held",
                "taken off",
                "weaned off",
                "came off",
                "tapered off",
                "weaning",
                "dc'd",
            ],
            "improved": [
                "improved to",
                "improved",
                "improvement",
                "improving",
                "responding",
                "resolving",
                "recovered",
                "recovery of",
                "improved from admission",
            ],
            "worsened": [
                "worsened",
                "worsening",
                "deteriorating",
                "progressive",
                "exacerbated",
                "refractory",
                "decompensated",
                "unstable",
                "instability",
            ],
            "severe": ["high grade", "severe", "fulminant", "marked", "pronounced", "critical", "severely", "highly"],
            "mild/moderate": ["low grade", "mild", "moderate", "mod", "mildly"],
            "present": ["present", "noted", "evident", "visible", "positive for"],
            "resolved": ["resolved", "gone", "absent", "no longer present", "cleared", "subsided", "reversed"],
            "normal": ["normal", "within normal limits", "wnl"],
            "low": ["low dose", "low", "minimal dose", "small dose", "reduced", "below reference range"],
            "high": ["high dose", "high", "maximal dose", "large dose", "elevated", "above reference range"],
            "medium": ["medium dose", "medium", "normal dose", "standard dose", "usual dose"],
            "abnormal": ["abnormal", "not normal", "out of range"],
            "approximate": ["approximately", "approximate", "roughly", "generally"],
            "in 24 hours": ["over 24h", "over 24 hours", "within 24 hours"],
            "rapid": [
                "rapid",
                "rapidly",
                "fast",
                "quick",
                "quickly",
                "swift",
                "swiftly",
                "speedy",
                "expeditious",
                "brisk",
                "immediate",
                "immediately",
                "prompt",
                "promptly",
                "sudden",
                "acute onset",
            ],
            "slow": ["slow", "gradual", "incremental", "slowly", "gradually", "incrementally"],
            "recurrent": ["recurrent", "recurring", "cyclical", "periodic", "regular", "perennial", "repeat"],
            "weak": ["weak", "frail", "feeble"],
            "widespread": ["widespread", "diffuse", "extensive", "pervasive"],
            "bilateral": ["bilateral"],
            "unilateral": ["unilateral", "one sided", "left sided", "right sided", "left", "right"],
            "intermittent": ["intermittent", "occasional"],
        }
        matched_modifiers = OrderedDict()  # will hold {standard_modifier: [matched_phrases]}
        all_alias_to_std = []

        # Prepare a list of (standard, alias) pairs sorted by alias length (longest first)
        for std, aliases in MODIFIER_MAP.items():
            for alias in aliases:
                all_alias_to_std.append((std, alias))
                if " " in alias:
                    for dash in ["-", "–", "—"]:  # hyphen, en dash, em dash
                        all_alias_to_std.append((std, alias.replace(" ", dash)))
        all_alias_to_std.sort(key=lambda x: -len(x[1]))

        def match_and_clean(text):
            matched_phrases = []
            for std, alias in all_alias_to_std:
                pattern = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
                if pattern.search(text):
                    if std not in matched_modifiers:
                        matched_modifiers[std] = []
                    matched_modifiers[std].append(alias)
                    matched_phrases.append(alias)
            for phrase in matched_phrases:
                text = re.sub(rf"\b{re.escape(phrase)}\b", "", text, flags=re.IGNORECASE)
            return text.strip()

        cleaned_cent = match_and_clean(current_cent_text)
        cleaned_time = [match_and_clean(t) for t in current_time_texts]
        cleaned_neg = [match_and_clean(t) for t in current_negation_texts]

        return list(matched_modifiers.keys()), cleaned_cent, cleaned_time, cleaned_neg

    def resolve_cent_value(self, text):
        """Extract numeric values with optional unit/route/frequency; return values + cleaned text."""

        # Patterns
        unit_pattern = re.compile(
            r"(l/min/m2|l/min/m²|[a-z]{1,4}/[a-z]{1,4}/[a-z]{1,4}|ml/min|l/min|mg/dl|/min|mmol/l|k/ul|g/dl|10\^3/ul|u/l|ng/ml|pg/ml|meq/l|[a-z]{1,4}/[a-z]{1,4}|m2|°[cf]|%|\b[a-zA-Z]{1,6}\b)",
            re.IGNORECASE,
        )
        route_pattern = re.compile(r"\b(iv infusion|iv|po|im|subq|sc|inhaled)\b", re.IGNORECASE)
        freq_pattern = re.compile(
            r"\b(once daily|daily|bid|tid|qhs|q6h|q8h|q12h|q24h|per protocol|prn as needed|as needed|schedule)(?:\s+prn)?\b|\bprn\b",
            re.IGNORECASE,
        )
        value_regex = re.compile(r"(?:([<>~])\s*)?\b((?:\d{1,3}/\d{2,3})|(?:[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?))(s?\b)?", re.IGNORECASE)

        cleaned_text = text.strip().lower()

        results = []

        # Step 1: Look for value
        no_further_values = False
        while not no_further_values:
            match_value = value_regex.search(cleaned_text)
            if not match_value:
                no_further_values = True
                continue

            inequality = match_value.group(1)
            val_str = match_value.group(2)
            trailing_s = match_value.group(3)
            start_val, end_val = match_value.span()  # indices of the value

            # Abort if followed or preceded by a dash
            disallowed_set = {"-", "–", "—"}
            if cleaned_text[start_val - 1 : start_val] in disallowed_set or cleaned_text[end_val : end_val + 1] in disallowed_set:
                cleaned_text = cleaned_text[:start_val] + cleaned_text[end_val:]
                continue

            if "/" not in val_str:
                try:  # see if it is actually a number
                    value = float(val_str.replace(",", ""))
                except Exception:
                    cleaned_text = cleaned_text[:start_val] + cleaned_text[end_val:]
                    continue
            else:
                value = val_str

            modifier = None
            if inequality == "<":
                modifier = "less than"
            elif inequality == ">":
                modifier = "more than"
            if trailing_s == "s":
                modifier = "approximate"
                if "/" not in val_str:
                    if 10 <= value <= 90:
                        value = value + 5
                    elif 100 <= value <= 900:
                        value = value + 50
            elif inequality == "~":
                modifier = "approximate"

            unit = None
            route = None
            freq = None

            # Step 2: Look for unit (right first)
            unit_found = False
            right_text = cleaned_text[end_val : min(end_val + 10, len(cleaned_text))]
            left_text = cleaned_text[max(0, start_val - 10) : start_val]
            if unit_pattern.match(right_text):
                match_unit = unit_pattern.match(right_text)
                unit = match_unit.group(0)
                unit_found = True
                start_unit, end_unit = match_unit.span()
                start_unit, end_unit = start_unit + end_val, end_unit + end_val
            elif right_text.startswith(" ") and unit_pattern.match(right_text[1:]):
                match_unit = unit_pattern.match(right_text[1:])
                unit = match_unit.group(0)
                unit_found = True
                start_unit, end_unit = match_unit.span()
                start_unit, end_unit = start_unit + end_val + 1, end_unit + end_val + 1

            # if not found, look left
            matches_unit = list(unit_pattern.finditer(left_text))
            if not unit_found and matches_unit:
                match_unit = matches_unit[-1]
                start_unit, end_unit = match_unit.span()
                start_unit, end_unit = start_unit + max(0, start_val - 10), end_unit + max(0, start_val - 10)
                if end_unit == start_val:
                    unit = match_unit.group(0)
                    unit_found = True
                elif end_unit == start_val - 1 and left_text.endswith(" "):
                    unit = match_unit.group(0)
                    unit_found = True

            # Step 3 & 4: Look for route and frequency only if unit found
            if unit_found:
                if end_val > end_unit:
                    start_index = end_val
                elif end_unit >= end_val:
                    start_index = end_unit
                right_text = cleaned_text[start_index : min(start_index + 20, len(cleaned_text))]

                if right_text and right_text[0] == " ":  # only continue with route+freq parsing if the next character is a space

                    matches_route = list(route_pattern.finditer(right_text))
                    matches_freq = list(freq_pattern.finditer(right_text))
                    if matches_route:
                        match_route = matches_route[0]
                        start_route, end_route = match_route.span()
                    if matches_freq:
                        match_freq = matches_freq[0]
                        start_freq, end_freq = match_freq.span()

                    if matches_route and matches_freq:
                        if start_route < start_freq:  # route is first
                            route = match_route.group(0)
                            if right_text[end_route] == " " and start_freq == end_route + 1:  # there must be a space separating route and freq
                                freq = match_freq.group(0)
                        elif start_route >= start_freq:  # freq is first
                            freq = match_freq.group(0)
                            if right_text[end_freq] == " " and start_route == end_freq + 1:  # there must be a space separating freq and route
                                route = match_route.group(0)
                    elif matches_route and start_route == 1:  # has to start after a space
                        route = match_route.group(0)
                    elif matches_freq and start_freq == 1:  # has to start after a space
                        freq = match_freq.group(0)

                    if matches_route:
                        start_route, end_route = start_route + start_index, end_route + start_index
                    if matches_freq:
                        start_freq, end_freq = start_freq + start_index, end_freq + start_index

            # now remove the identified val, unit, route, freq
            spans_to_remove = [(start_val, end_val)]
            if unit is not None:
                spans_to_remove.append((start_unit, end_unit))
            if freq is not None:
                spans_to_remove.append((start_freq, end_freq))
            if route is not None:
                spans_to_remove.append((start_route, end_route))
            spans_to_remove.sort(reverse=True)
            for start, end in spans_to_remove:
                cleaned_text = cleaned_text[:start] + cleaned_text[end:]
            cleaned_text = cleaned_text.strip()

            # add entry to output list
            results.append({"value": value, "unit": unit, "route": route, "frequency": freq, "modifier": modifier})

        return results, cleaned_text

    def _resolve_spans(self):
        """Standardize span-level entities per patient."""

        for master_list_i in range(len(self.master_list)):
            standardized_entities_list = []
            for entity in self.master_list[master_list_i][7]:
                standardized_entities_list = self.add_standardized_entities_wrapper(
                    standardized_entities_list,
                    entity[0]["text"],
                    [item["text"] for item in entity[1]],
                    [item["text"] for item in entity[2]],
                    self.master_list[master_list_i][5],
                    ("spans", entity[0].get("id", ""), entity[0]["text"]),
                )
            self.master_list[master_list_i][7] = standardized_entities_list

    def _resolve_table_entries(self):
        """Standardize table-derived entities; handles generic, medication, and key/value rows."""

        # transpose tables where each row is a time series
        for master_list_i in range(len(self.master_list)):
            for table_i in range(len(self.master_list[master_list_i][8])):
                self.master_list[master_list_i][8][table_i] = (
                    self.transpose_time_series_table(self.master_list[master_list_i][8][table_i][0]),
                    self.master_list[master_list_i][8][table_i][1],
                )

        # now resolve each table entity
        for master_list_i in range(len(self.master_list)):

            ### create standardized_entities_list from the table data ###
            standardized_entities_list = []

            for table in self.master_list[master_list_i][8]:
                for table_row in table[0]:
                    # get time entries
                    time_entries = []
                    for key in table_row.keys():
                        if any(sub in key.lower() for sub in ["time", "day", "date"]):  # time entry
                            time_entries += [table_row[key]]
                    if len(time_entries) == 0:
                        time_entries += [time_span["text"] for time_span in table[1]]

                    # handle entries where individual entries must be separated and the keys must be included (only keep this for modular code as this is by far the most common entry)
                    if not any(key in ("Medication", "Dose", "Value") for key in table_row.keys()):
                        for key in table_row.keys():
                            if not any(sub in key.lower() for sub in ["time", "day", "date"]):
                                key_clean = self.connect_substrings_with_underscore(key) if len(key) < 18 else key  # ensures that the key doesn't throw the value parser off
                                cent_text = key_clean + " " + table_row[key]
                                cent_text = self.clean_str_for_table_parsing(cent_text)
                                standardized_entities_list = self.add_standardized_entities_wrapper(
                                    standardized_entities_list,
                                    cent_text,
                                    time_entries,
                                    [],
                                    self.master_list[master_list_i][5],
                                    ("table", cent_text),
                                )

                    # handle entries where the whole row needs to be combined (any key='Medication'/'Dose')
                    # do not include keys here
                    # only parse dose, route, frequency entries
                    elif any(key in ("Medication", "Dose") for key in table_row.keys()):
                        cent_text = ""
                        for key in table_row.keys():
                            if any(sub in key.lower() for sub in ["medication", "dose", "route", "frequency", "status", "name"]):
                                cent_text += table_row[key] + " "
                        cent_text = self.clean_str_for_table_parsing(cent_text)
                        standardized_entities_list = self.add_standardized_entities_wrapper(
                            standardized_entities_list,
                            cent_text,
                            time_entries,
                            [],
                            self.master_list[master_list_i][5],
                            ("table", cent_text),
                        )

                    # handle rows where there is one key and one value field
                    # do not include keys here
                    elif any(key in ("Value") for key in table_row.keys()):
                        cent_text = ""
                        for key in table_row.keys():
                            if not any(sub in key.lower() for sub in ["time", "day", "date"]):
                                if key != "Value" and len(table_row[key]) < 18:
                                    cent_text += self.connect_substrings_with_underscore(table_row[key]) + " "
                                else:
                                    cent_text += table_row[key] + " "
                        cent_text = self.clean_str_for_table_parsing(cent_text)
                        standardized_entities_list = self.add_standardized_entities_wrapper(
                            standardized_entities_list,
                            cent_text,
                            time_entries,
                            [],
                            self.master_list[master_list_i][5],
                            ("table", cent_text),
                        )

            self.master_list[master_list_i][8] = standardized_entities_list

    def parse_mrconso_with_classes(
        self,
        mrconso_path,
        mrrel_path,
        mrsty_path=None,
        sources=None,
        allowed_ttys=None,
        parent_sabs=None,
        allowed_parent_semtypes=None,
        max_depth=3,
        min_classes=1,
    ):
        """Build {canonical→aliases} and {canonical→class labels} from UMLS with ancestor labels."""

        if sources is None:
            sources = {"SNOMEDCT_US", "LOINC", "RXNORM"}
        if allowed_ttys is None:
            allowed_ttys = {"PT", "SY", "AB"}  # , "FN", "ET"}
        if parent_sabs is None:
            parent_sabs = {"SNOMEDCT_US", "LOINC", "RXNORM"}

        # Step 1: Load canonical terms and aliases from MRCONSO
        cui_to_terms = defaultdict(list)
        full_cui_to_terms = defaultdict(list)
        with open(mrconso_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                cui, lat, sab, tty, term = parts[0], parts[1], parts[11], parts[12], parts[14].lower().strip()
                if lat == "ENG" and sab in sources and tty in allowed_ttys:
                    cui_to_terms[cui].append((term, tty))
                if lat == "ENG":
                    full_cui_to_terms[cui].append((term, tty))

        canonical_vocab = {}
        term_to_cui = {}
        cui_to_canonical = {}
        for cui, term_list in cui_to_terms.items():
            pt_terms = [t for t, tty in term_list if tty == "PT"]
            canonical = pt_terms[0] if pt_terms else None  # sorted([t for t, _ in term_list], key=len)[0]
            aliases = sorted({t for t, _ in term_list if t != canonical})
            if canonical is not None:
                canonical_vocab[canonical] = aliases
            term_to_cui[canonical] = cui
            cui_to_canonical[cui] = canonical

        # Step 2: Load parent relationships from MRREL
        cui_to_parents = defaultdict(set)
        with open(mrrel_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if parts[3] == "PAR" and parts[10] in parent_sabs:
                    cui_to_parents[parts[0]].add(parts[4])

        # Step 3 (optional): Load semantic types from MRSTY
        cui_to_semtypes = defaultdict(set)
        if mrsty_path and os.path.exists(mrsty_path):
            with open(mrsty_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("|")
                    cui_to_semtypes[parts[0]].add(parts[3].strip().lower())

        def parent_semtype_ok(parent_cui):
            if allowed_parent_semtypes is None:
                return True
            return any(sty.lower() in allowed_parent_semtypes for sty in cui_to_semtypes.get(parent_cui, []))

        # Step 4: For each concept, collect all valid parent class labels up to max_depth
        def get_class_labels(cui, max_depth=3):
            visited = set()
            q = deque([(cui, 0)])
            results = set()

            while q:
                current, depth = q.popleft()
                if depth >= max_depth:
                    continue
                for pcui in cui_to_parents.get(current, set()):
                    if pcui in visited:
                        continue
                    visited.add(pcui)

                    terms = full_cui_to_terms.get(pcui, [])
                    if not terms:
                        continue  # no label to assign

                    pt_terms = [t for t, tty in terms if tty == "PT"]
                    label = pt_terms[0] if pt_terms else (terms[0][0] if terms else None)

                    # Apply semantic type filtering if desired
                    if not parent_semtype_ok(pcui):
                        q.append((pcui, depth + 1))
                        continue

                    if label:
                        results.add(label)

                    q.append((pcui, depth + 1))

            return sorted(results)

        concept_class_map = {}
        for canonical, cui in term_to_cui.items():
            labels = set()

            # Add class based on the concept's own semantic type
            if allowed_parent_semtypes is None or any(sty.lower() in allowed_parent_semtypes for sty in cui_to_semtypes.get(cui, [])):
                terms = full_cui_to_terms.get(cui, [])
                pt_terms = [t for t, tty in terms if tty == "PT"]
                label = pt_terms[0] if pt_terms else (terms[0][0] if terms else None)
                if label:
                    labels.add(label)

            # Add class labels from ancestors
            parent_labels = get_class_labels(cui, max_depth=max_depth)
            labels.update(parent_labels)

            concept_class_map[canonical] = sorted(labels) if len(labels) >= min_classes else []
            concept_class_map[canonical] = [item for item in concept_class_map[canonical] if item != canonical]

        canonical_vocab = {
            re.sub(r"[-–—_]", " ", key): [re.sub(r"[-–—_]", " ", val) for val in values if val is not None] for key, values in canonical_vocab.items() if key is not None
        }  # replace dashes with spaces
        concept_class_map = {
            re.sub(r"[-–—_]", " ", key): [re.sub(r"[-–—_]", " ", val) for val in values if val is not None] for key, values in concept_class_map.items() if key is not None
        }  # replace dashes with spaces

        return canonical_vocab, concept_class_map

    def reverse_can_vocab_to_aliases_dict(self, input_dict, duplicate_alias_warnings=False):
        """Invert {canonical:[aliases]} → {alias_or_canonical: canonical} (warn on duplicates)."""

        output_dict = {}
        for key, value in input_dict.items():
            for v in value:
                if duplicate_alias_warnings and v in output_dict.keys():
                    log.debug("warning: " + str(v) + " alias occurs multiple times")
                output_dict[v] = key
            output_dict[key] = key
        return output_dict

    def reverse_aliases_to_can_vocab_dict(self, input_dict):
        """Convert {alias: canonical} → {canonical: [aliases]}."""
        canonical_to_aliases = defaultdict(list)
        for alias, canonical in input_dict.items():
            canonical_to_aliases[canonical].append(alias)
        return dict(canonical_to_aliases)

    def filter_canonical_vocab_to_aliases_dict(self, c_to_a_dict, a_to_c_dict, concept_class_map, substrings_remove):
        """Filter noisy/too-long/unit-like aliases; pick a new canonical if the old is dropped.

        Removes the following:
        - any terms longer than max_alias_len
        - any terms containing any substring within substrings_remove
        - any terms containing units (e.g. mmhg) at least four characters long (does not need to be surrounded by word boundaries)
        - any terms exactly matching a unit less than four characters long (this patterns needs to be surrounded by word boundaries)
        - any terms that are less than two characters long
        - if the canonical term is removed, a new canonical term is chosen
        """
        unit_tokens = [
            r"mmhg",
            r"cmh2o",
            r"microgram",
            r"milligram",
            r"microg",
            r"osmol",
            r"gram",
            r"mg",
            r"ug",
            r"ng",
            r"pg",
            r"g",
            r"mcg",
            r"kg",
            r"litre",
            r"liter",
            r"l",
            r"ml",
            r"dl",
            r"mmol",
            r"mol",
            r"micromol",
            r"micromole",
            r"millimole",
            r"millimol",
            r"mole",
            r"umol",
            r"mcmol",
            r"cm",
            r"mm",
            r"mu",
            r"m",
            r"°c",
            r"°f",
            r"min",
            r"second",
            r"sec",
            r"s",
            r"m2",
            r"%",
            r"units",
            r"u",
        ]

        symbol_units = ["%", "°c", "°f"]
        word_units = [u for u in unit_tokens if u not in symbol_units]

        symbol_unit_pattern = "(" + "|".join(re.escape(u) for u in symbol_units) + ")"
        word_unit_pattern_no_boundaries = "(" + "|".join(re.escape(u) + r"(?![a-zA-Z])" for u in word_units) + ")"
        word_unit_pattern_with_boundaries = r"\b(" + "|".join(re.escape(u) for u in word_units) + r")\b"

        compound_unit_pattern_no_boundaries = rf"""
			(
				{word_unit_pattern_no_boundaries}
				(
					\s*(/|per)\s*
					{word_unit_pattern_no_boundaries}
				){{1,3}}
			)
			|
			{word_unit_pattern_no_boundaries}
			|
			{symbol_unit_pattern}
		"""

        compound_unit_pattern_with_boundaries = rf"""
			(
				{word_unit_pattern_with_boundaries}
				(
					\s*(/|per)\s*
					{word_unit_pattern_with_boundaries}
				){{1,3}}
			)
			|
			{word_unit_pattern_with_boundaries}
			|
			{symbol_unit_pattern}
		"""

        unit_regex_no_boundaries = re.compile(compound_unit_pattern_no_boundaries, re.IGNORECASE | re.VERBOSE)
        unit_regex_with_boundaries = re.compile(compound_unit_pattern_with_boundaries, re.IGNORECASE | re.VERBOSE)

        def is_valid(term):
            term = term.lower().strip()
            if len(term) > self.max_alias_len:
                return False
            if any(sub.lower() in term for sub in substrings_remove):
                return False
            unit_match = unit_regex_no_boundaries.search(term)
            if unit_match and len(unit_match.group(0)) >= 4:
                return False
            if unit_regex_with_boundaries.fullmatch(term) and len(term) < 4:
                return False
            if len(term) <= 3:
                return False
            return True

        filtered_dict = {}
        updated_concept_class_map = {}

        for canonical, aliases in c_to_a_dict.items():
            all_aliases = set(aliases)

            # Filter valid aliases
            valid_aliases = [alias for alias in all_aliases if is_valid(alias)]

            if is_valid(canonical):
                # Canonical remains the same
                new_canonical = canonical
                if canonical not in valid_aliases:
                    valid_aliases.append(canonical)
            else:
                if not valid_aliases:
                    continue  # Skip this entry entirely

                # select the best new canonical alias
                def sort_key(term):
                    mapped = a_to_c_dict.get(term, "____________________")
                    return (len(mapped), len(term))

                new_canonical = sorted(valid_aliases, key=sort_key)[0]
                if a_to_c_dict[new_canonical] in valid_aliases:
                    new_canonical = a_to_c_dict[new_canonical]

            # Build final alias list
            alias_list = [alias for alias in valid_aliases if alias != new_canonical]
            alias_list.append(new_canonical)

            filtered_dict[new_canonical] = alias_list

            if canonical in concept_class_map:
                updated_concept_class_map[new_canonical] = concept_class_map[canonical]

        return filtered_dict, updated_concept_class_map

    def apply_ontology_corrections(self, term_dict, class_dict, class_aliases=None, a_to_c_resolver=None, post_edits=False):

        rules = yaml.safe_load(open(self.ont_corr)) or {}
        # Work with SETS internally (aliases/classes); convert to lists at the end
        terms = {k: set(v) | {k} for k, v in deepcopy(term_dict).items()}
        classes = {k: set(v) for k, v in deepcopy(class_dict).items()}
        cls_alias = None if class_aliases is None else {k: set(v) | {k} for k, v in deepcopy(class_aliases).items()}

        # ----- util -----
        def normalize_term(s: str) -> str:
            s = s.lower()
            s = re.sub(r"\b(?:measurement|determination|level|test|value|taking|finding|monitoring)\b", "", s)
            s = re.sub(r"[,\s]+", "", s)
            return s.strip()

        def rep_pick(items, resolver):
            # Prefer item whose resolver mapping has the shortest string; then shortest item
            # (Matches your earlier heuristic; resolver may not contain all items)
            items = list(items)
            return sorted(items, key=lambda t: (len(resolver.get(t, "§" * 64)), len(t)))[0]

        def components_from_token_sets(key2toks: dict[str, set[str]]):
            if not key2toks:
                return []
            # Union-find over keys that share at least one token
            token2keys, parent = {}, {k: k for k in key2toks}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            for k, toks in key2toks.items():
                for t in toks:
                    token2keys.setdefault(t, []).append(k)
            for ks in token2keys.values():
                base = ks[0]
                for k in ks[1:]:
                    union(base, k)
            comp = {}
            for k in key2toks:
                comp.setdefault(find(k), []).append(k)
            return list(comp.values())

        def build_live_resolver(groups: dict[str, set[str]]) -> dict[str, str]:
            return {alias: canon for canon, als in groups.items() for alias in als}

        def add_only_expand_groups(groups: dict[str, set[str]], expansions: list[list[str]]):
            if not groups or not expansions:
                return
            exp_sets = [set(L) for L in expansions if L]
            for c in list(groups):
                crowd = groups[c]
                for E in exp_sets:
                    if crowd & E:
                        groups[c] |= E

        def coverage_on_spec(group_set: set[str], A: set[str], B: set[str]) -> float:
            U = A | B
            return 0.0 if not U else len(U & group_set) / len(U)

        # refresh live resolvers
        live_term = build_live_resolver(terms)
        live_cls = build_live_resolver(cls_alias) if cls_alias is not None else {}

        # ----- 1) PREPROCESS -----
        pre = rules.get("preprocess", {}) or {}
        minlen = int(pre.get("min_alias_length", 1))
        rm_alias = set(pre.get("aliases_to_remove", []) or [])

        def preprocess_groups(groups: dict[str, set[str]], resolver: dict, sync_classes=False, pref_resolver=None):
            if not groups:
                return
            for c in list(groups):
                filt = {a for a in groups[c] if len(a) >= minlen and a not in rm_alias}
                if not filt:
                    groups.pop(c, None)
                    if sync_classes:
                        classes.pop(c, None)
                    continue
                # choose representative and re-key if needed
                pref = pref_resolver or {}
                pref_rep = next((pref[t] for t in filt if t in pref), None)
                rep = pref_rep if pref_rep else rep_pick(filt, resolver)
                if rep != c:
                    groups.pop(c, None)
                    groups[rep] = filt | {rep}
                    if sync_classes and c in classes:
                        classes[rep] = classes.get(rep, set()) | classes.pop(c)
                else:
                    groups[c] = filt | {c}

        preprocess_groups(terms, live_term, sync_classes=True, pref_resolver=a_to_c_resolver)
        if cls_alias is not None:
            preprocess_groups(cls_alias, live_cls or {}, sync_classes=False, pref_resolver=a_to_c_resolver)

        # ----- 2) REMOVE WHOLE SETS that contain any banned token -----
        banned = set(pre.get("term_sets_to_remove", []) or [])

        def remove_sets_with_banned(groups: dict[str, set[str]], sync_classes=False):
            if not groups or not banned:
                return
            for c in list(groups):
                if groups[c] & banned:
                    groups.pop(c, None)
                    if sync_classes:
                        classes.pop(c, None)

        remove_sets_with_banned(terms, sync_classes=True)
        if cls_alias is not None:
            remove_sets_with_banned(cls_alias, sync_classes=False)

        # ----- 3) EQUIVALENT TERM SETS (add-only; no merges) -----
        add_only_expand_groups(terms, rules.get("equivalent_term_sets", []) or [])
        if cls_alias is not None:
            add_only_expand_groups(cls_alias, rules.get("equivalent_term_sets", []) or [])

        # refresh live resolvers
        live_term = build_live_resolver(terms)
        live_cls = build_live_resolver(cls_alias) if cls_alias is not None else {}

        # ----- 4) CLASS MAPPINGS (resolver-aware add-only) -----
        for k, cls_list in (rules.get("class_mappings", {}) or {}).items():
            tgt = live_term.get(k)
            if tgt in terms:
                classes.setdefault(tgt, set()).update(cls_list)

        # ----- 5) NEW CLASSES (resolver-aware add-only) -----
        for cls_name, members in (rules.get("new_classes", {}) or {}).items():
            for t in members or []:
                tgt = live_term.get(t)
                if tgt in terms:
                    classes.setdefault(tgt, set()).add(cls_name)

        # ----- 6) POST EDITS -----
        if post_edits:
            pe = rules.get("post_edits", {}) or {}

            # 6a) term_merges — add-only
            add_only_expand_groups(terms, pe.get("term_merges", []) or [])
            live_term = build_live_resolver(terms)

            # 6b) class_merges — add-only on class_aliases
            if cls_alias is not None:
                add_only_expand_groups(cls_alias, pe.get("class_merges", []) or [])
                live_cls = build_live_resolver(cls_alias)

            # 6c) term_removals (resolver-aware); repick canonical if removed
            for m in pe.get("term_removals", []) or []:
                ((name, aliases),) = m.items()  # <- parse this format only
                tgt = live_term.get(name, name)
                if tgt not in terms:
                    continue
                rem = set(aliases)
                kept = set(a for a in terms[tgt] if a not in rem)

                if not kept:  # empty -> drop group + its classes
                    terms.pop(tgt, None)
                    classes.pop(tgt, None)
                    continue

                # if canonical was removed, pick a new rep and move classes
                if tgt not in kept:
                    pref_map = a_to_c_resolver or {}
                    rep = min(kept, key=lambda t: (len(pref_map.get(t, "§" * 64)), len(t)))
                    terms.pop(tgt, None)
                    terms[rep] = kept | {rep}
                    if tgt in classes:
                        classes[rep] = classes.get(rep, set()) | classes.pop(tgt)
                else:
                    terms[tgt] = kept | {tgt}

            live_term = build_live_resolver(terms)

            # 6d) class_removals (resolver-aware) on class_aliases
            if cls_alias is not None:
                for m in pe.get("class_removals", []) or []:
                    ((cname, aliases),) = m.items()  # <- parse this format only
                    ccanon = live_cls.get(cname, cname)
                    if ccanon not in cls_alias:
                        continue
                    rem = set(aliases)
                    cls_alias[ccanon] = set(x for x in cls_alias[ccanon] if x not in rem)

                live_cls = build_live_resolver(cls_alias)

            # 6e) term_set_splits — coverage; ALWAYS drop source; copy classes to both
            split_cov = float(pe.get("split_coverage", 0.6))
            for spec in pe.get("term_set_splits", []) or []:
                if not (isinstance(spec, list) and len(spec) == 2):
                    continue
                A, B = set(spec[0]), set(spec[1])
                for c in list(terms):
                    G = set(terms[c])
                    if coverage_on_spec(G, A, B) >= split_cov:
                        cA, cB = spec[0][0], spec[1][0]
                        terms.setdefault(cA, set()).update(A | {cA})
                        terms.setdefault(cB, set()).update(B | {cB})
                        classes.setdefault(cA, set()).update(classes.get(c, set()))
                        classes.setdefault(cB, set()).update(classes.get(c, set()))
                        terms.pop(c, None)
                        classes.pop(c, None)

            live_term = build_live_resolver(terms)

        def pick_rep_with_resolver(keys, raw_sets, resolver, fallback_items, fallback_resolver):
            if resolver:
                # preserve merge order: first canonical whose set has any resolver-mapped term wins
                for k in keys:
                    for t in raw_sets[k]:
                        if t in resolver:
                            return resolver[t]
            # fallback to your existing heuristic
            return rep_pick(fallback_items, fallback_resolver)

        # ----- FINAL MERGE (normalized) and process terms -----
        raw_sets = {k: set(terms[k]) for k in terms}
        norm_sets = {k: {normalize_term(t) for t in v if normalize_term(t)} for k, v in raw_sets.items()}
        comps = components_from_token_sets(norm_sets)
        for keys in comps:
            if len(keys) <= 1:
                continue
            all_terms = set().union(*[raw_sets[k] for k in keys])
            all_classes = set().union(*[classes.get(k, set()) for k in keys])
            rep = pick_rep_with_resolver(keys, raw_sets, a_to_c_resolver, all_terms, live_term)
            terms[rep] = all_terms | {rep}
            classes[rep] = all_classes
            for k in keys:
                if k != rep:
                    terms.pop(k, None)
                    classes.pop(k, None)

        # ----- classes -----
        if cls_alias:
            c_raw = {k: set(cls_alias[k]) for k in cls_alias}
            c_norm = {k: {normalize_term(t) for t in v if normalize_term(t)} for k, v in c_raw.items()}
            c_comps = components_from_token_sets(c_norm)
            for ks in c_comps:
                if len(ks) <= 1:
                    continue
                all_alias = set().union(*[c_raw[k] for k in ks])
                repc = pick_rep_with_resolver(ks, c_raw, a_to_c_resolver, all_alias, live_cls or {})
                cls_alias[repc] = all_alias | {repc}
                for k in ks:
                    if k != repc:
                        cls_alias.pop(k, None)

        # ----- final checks -----
        for k in terms:
            classes.setdefault(k, set())  # to force every canonical term to be in the class dict
        terms = {k: list(set(v) | {k}) for k, v in terms.items()}
        classes = {k: list(set(v)) for k, v in classes.items()}
        cls_alias = None if cls_alias is None else {k: list(set(v) | {k}) for k, v in cls_alias.items()}

        return terms, classes, cls_alias

    def _create_umls_vocab(self):
        """Parse UMLS, collapse equivalents, filter noise, and build alias maps."""

        allowed_semtypes = {
            "clinical attribute",
            "finding",
            "sign or symptom",
            "disease or syndrome",
            "pathologic function",
            "laboratory or test result",
            "quantitative concept",
            "temporal concept",
            "therapeutic or preventive procedure",
            "diagnostic procedure",
            "laboratory procedure",
            "medical device",
            "functional concept",
            "pharmacologic substance",
            "antibiotic",
            "clinical drug",
            "patient or disabled group",
            "health care related organization",
            "health care activity",
            "mental or behavioral dysfunction",
            "biologically active substance",
        }
        self.c_to_a_dict, self.concept_class_map = self.parse_mrconso_with_classes(self.mrconso_rrf, self.mrrel_rrf, self.mrsty_rrf, allowed_parent_semtypes=allowed_semtypes, max_depth=3)

        # clean up canonical_vocab_to_aliases_dict
        substrings_remove = [
            "-one",
            "-ol",
            "-ine",
            "-ane",
            "-ene",
            "-azole",
            "-hydroxy",
        ]
        # clean up canonical_vocab_to_aliases_dict
        self.c_to_a_dict, self.concept_class_map = self.filter_canonical_vocab_to_aliases_dict(
            self.c_to_a_dict, self.reverse_can_vocab_to_aliases_dict(self.c_to_a_dict), self.concept_class_map, substrings_remove
        )
        self.c_to_a_dict, self.concept_class_map, _ = self.apply_ontology_corrections(self.c_to_a_dict, self.concept_class_map)

        # now create a reversed dict
        self.a_to_c_dict = self.reverse_can_vocab_to_aliases_dict(self.c_to_a_dict)

    def _load_embedder(self):
        """Load SapBERT encoder + pooling on CPU."""

        word_emb = models.Transformer(self.embedder_model_name)
        pooling = models.Pooling(word_emb.get_word_embedding_dimension())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"gpu: {torch.cuda.is_available()}")
        self.emb_model = SentenceTransformer(modules=[word_emb, pooling], device=self.device)

    def _create_master_list(self, json_input):
        """Build master list from JSONL: identity/demographics/admission, spans, and tables. If self.cli_entry then build from a JSONL containing name,uid,filename,age,sex,adm_date,entities keys."""

        # read in raw tables and ner+relations predictions
        with open(json_input) as f:
            json_lines_list = [json.loads(line) for line in f]
            if not self.cli_entry:
                table_extractions_list = [item["tables"] for item in json_lines_list]

        # map entities to timestamps + negations
        if not self.cli_entry:
            extracted_entities = self.extract_cent_time_negation_by_patient(json_lines_list)
        else:
            extracted_entities = self.extract_cent_time_negation_by_patient_from_prestructured(json_lines_list)

        # map tables to timestamps + negations
        if not self.cli_entry:
            extracted_tables = self.extract_table_time_by_patient(json_lines_list, table_extractions_list)

        # now create master_list
        self.master_list = []
        for i in range(len(json_lines_list)):
            resolved_age = self.resolve_age_function(extracted_entities[i][1])
            resolved_sex = self.resolve_sex_function(extracted_entities[i][2])
            resolved_adm_time = self.resolve_adm_time_function(extracted_entities[i][3])
            self.master_list.append(
                [
                    json_lines_list[i]["uid"],
                    json_lines_list[i].get("name", ""),
                    json_lines_list[i].get("filename", ""),
                    resolved_age,
                    resolved_sex,
                    resolved_adm_time,
                    extracted_entities[i][0] if not self.cli_entry else "",
                    extracted_entities[i][4],
                    extracted_tables[i] if not self.cli_entry else [],
                ]
            )

        # normalize uid, name, sex entries
        for master_list_i in range(len(self.master_list)):
            self.master_list[master_list_i][0] = re.sub(r"\s+", " ", re.sub(r"[-_]", " ", str(self.master_list[master_list_i][0]))).strip().lower()
            if self.master_list[master_list_i][1] is not None:
                self.master_list[master_list_i][1] = re.sub(r"\s+", " ", re.sub(r"[-_]", " ", str(self.master_list[master_list_i][1]))).strip().lower()
            if self.master_list[master_list_i][4] is not None:
                self.master_list[master_list_i][4] = re.sub(r"\s+", " ", re.sub(r"[-_]", " ", str(self.master_list[master_list_i][4]))).strip().lower()

        # ensure all uids are unique and remove any patients with no valid admission timestamps
        master_list_filtered = []
        uids_set = set()
        for pt_entry in self.master_list:
            if pt_entry[5][0] is not None:
                master_list_filtered.append(pt_entry)
                uids_set.add(pt_entry[0])
        if len(self.master_list) != len(master_list_filtered):
            log.info("warning: " + str(len(self.master_list) - len(master_list_filtered)) + " patients removed due to invalid admission timestamps (a critical datapoint)")
        if len(uids_set) != len(master_list_filtered):
            raise ValueError("all uids must be unique")
        self.master_list = master_list_filtered

    def extract_cent_time_negation_by_patient_from_prestructured(self, json_inputs: list[dict]):
        """
        Return ('', {'label':'AGE','text':X},{'label':'SEX','text':X},{'label':'ADM_TIME','text':X}, C_ENT triples) per patient.
        The admission date entry is set as the earliest timestamp if no admission date is supplied; if no adm_time nor timestamp is present adm_time is set to 01/01/2025.
        """

        DEFAULT_ADM_FALLBACK_ISO = "2025-01-01"

        out = []
        adm_iso = None

        for json_line in json_inputs:
            cent_triples = [({"text": t[0]}, [{"text": s} for s in t[1]], [{"text": s} for s in t[2]]) for t in json_line["entities"]]

            # AGE entity
            age_text = json_line.get("age")
            age_ent = {"label": "AGE", "text": "" if age_text is None else str(age_text)}

            # SEX entity
            sex_text = json_line.get("sex")
            sex_ent = {"label": "SEX", "text": "" if sex_text is None else sex_text}

            # Preferred field for admission time/date
            adm_date_raw = json_line.get("adm_date")

            # Try to resolve adm_time from provided field
            if adm_date_raw not in (None, ""):
                parsed = self.resolve_adm_time_function([{"text": str(adm_date_raw)}])
                if parsed is not None:
                    adm_iso, _ = parsed  # (YYYY-MM-DD, time)

            # If not provided or unparsable, scan timestamps in C_ENT triples for earliest date
            if adm_iso is None:
                earliest_iso: str | None = None

                for _cent, ts_list, _ in cent_triples:
                    for ts in ts_list or []:
                        ts_text = ts.get("text") if isinstance(ts, dict) else None
                        if not ts_text:
                            continue
                        parsed = self.resolve_adm_time_function([{"text": str(ts_text)}])
                        if parsed is None:
                            continue
                        ts_iso, _ = parsed
                        if ts_iso is None:
                            continue
                        if earliest_iso is None or ts_iso < earliest_iso:
                            earliest_iso = ts_iso

                if earliest_iso is not None:
                    adm_iso = earliest_iso

            # Final fallback if still unknown
            if adm_iso is None:
                adm_iso = DEFAULT_ADM_FALLBACK_ISO

            adm_ent = {"label": "ADM_TIME", "text": adm_iso}

            out.append(("", [age_ent], [sex_ent], [adm_ent], cent_triples))

        return out

    def extract_cent_time_negation_by_patient(self, samples: list[dict]) -> list[tuple[str, list[tuple[dict, list[dict], list[dict]]]]]:
        """Return (text, AGE spans, SEX spans, ADM_TIME spans, C_ENT triples) per patient."""

        all_results = []

        for sample in samples:
            text = sample["text"]
            span_dict = {span["id"]: span for span in sample["spans"]}
            relations = sample["relations"]

            # Collect AGE, SEX, and ADM_TIME spans
            age_list = [span for span in sample["spans"] if span["label"] == "AGE"]
            sex_list = [span for span in sample["spans"] if span["label"] == "SEX"]
            adm_time_list = [span for span in sample["spans"] if span["label"] == "ADM_TIME"]

            # Initialize relation mappings
            cent_to_times = {}
            cent_to_negations = {}

            for rel in relations:
                head_id = rel["head"]
                child_id = rel["child"]
                rel_label = rel["label"]

                head_span = span_dict.get(head_id)
                child_span = span_dict.get(child_id)

                if not head_span or not child_span:
                    continue

                if head_span["label"] == "C_ENT":
                    if rel_label == "TIME_RELATION" and child_span["label"] == "TIME":
                        cent_to_times.setdefault(head_id, []).append(child_span)
                    elif rel_label == "NEGATION_RELATION" and child_span["label"] == "NEGATION":
                        cent_to_negations.setdefault(head_id, []).append(child_span)

            # Collect triples of (C_ENT, associated TIMES, associated NEGATIONS)
            cent_triples = []
            for cent_id in set(cent_to_times.keys()).union(cent_to_negations.keys()):
                cent_span = span_dict[cent_id]
                time_spans = cent_to_times.get(cent_id, [])
                negation_spans = cent_to_negations.get(cent_id, [])
                cent_triples.append((cent_span, time_spans, negation_spans))

            all_results.append((text, age_list, sex_list, adm_time_list, cent_triples))

        return all_results

    def extract_table_time_by_patient(self, patients, table_extractions_list):
        """Attach TIME spans to TABLEs in source order; return per-patient (table, times)."""

        result = []

        for patient_idx, patient in enumerate(patients):
            spans = patient["spans"]
            relations = patient["relations"]
            patient_tables = table_extractions_list[patient_idx]

            # Build span ID to span and index lookup
            id_to_span = {span["id"]: span for span in spans}
            id_to_index = {span["id"]: idx for idx, span in enumerate(spans)}

            # Identify all TABLE and TIME spans
            table_spans = [span for span in spans if span["label"] == "TABLE"]
            time_ids = {span["id"] for span in spans if span["label"] == "TIME"}

            # Sort TABLE spans by their ID (lexicographically, e.g., e3 < e6)
            sorted_table_spans = sorted(table_spans, key=lambda s: int(s["id"][1:]))  # Assumes 'e###' format

            # Map table span index (in sorted list) to list of related TIME spans
            table_idx_to_time_spans = {}

            for rel in relations:
                if rel["label"] != "TIME_RELATION":
                    continue

                head_id = rel["head"]
                child_id = rel["child"]

                if head_id in time_ids and child_id in id_to_index:
                    table_id, time_id = child_id, head_id
                elif child_id in time_ids and head_id in id_to_index:
                    table_id, time_id = head_id, child_id
                else:
                    continue

                # Only continue if the table_id is in our sorted_table_spans list
                if table_id not in id_to_index:
                    continue

                # Map table_id to its index in the sorted table list
                table_span = id_to_span[table_id]
                try:
                    table_order_idx = sorted_table_spans.index(table_span)
                except ValueError:
                    continue  # Defensive, though shouldn't happen

                time_span = id_to_span[time_id]
                if table_order_idx not in table_idx_to_time_spans:
                    table_idx_to_time_spans[table_order_idx] = []
                table_idx_to_time_spans[table_order_idx].append(time_span)

            # Construct output for this patient
            patient_result = []
            for i, table in enumerate(patient_tables):
                time_spans = sorted(table_idx_to_time_spans.get(i, []), key=lambda s: s["start"])
                patient_result.append((table, time_spans))

            result.append(patient_result)

        return result

    def resolve_age_function(self, age_spans: list[dict]) -> int | None:
        """Choose most frequent plausible age (1–119) from AGE spans."""

        candidate_ages = []

        for span in age_spans:
            text = span.get("text", "")

            # Skip span if it contains 2 or more forward slashes (likely a date)
            if text.count("/") >= 2:
                continue

            # Find all 1–3 digit numbers surrounded by word boundaries
            numbers = re.findall(r"\d{1,3}(?=[^\d]|\b)", text)
            for num in numbers:
                age = int(num)
                if 0 < age < 120:
                    candidate_ages.append(age)

        if not candidate_ages:
            return None

        # Choose most frequent age; if tie, return the smallest
        age_counts = Counter(candidate_ages)
        most_common = age_counts.most_common()
        top_freq = most_common[0][1]
        top_candidates = [age for age, freq in most_common if freq == top_freq]

        return min(top_candidates)

    def resolve_sex_function(self, sex_spans: list[dict]) -> str | None:
        """Resolve sex as 'male'/'female' if clear; else None."""

        male_keywords = {"m", "male", "man"}
        female_keywords = {"f", "female", "woman"}

        candidates = []

        for span in sex_spans:
            text = span.get("text", "").lower().strip()

            if text in male_keywords:
                candidates.append("male")
            elif text in female_keywords:
                candidates.append("female")
            else:
                candidates.append(None)

        if not candidates:
            return None

        counts = Counter(candidates)
        most_common = counts.most_common()

        if not most_common:
            return None

        top_count = most_common[0][1]
        top_values = [value for value, count in most_common if count == top_count]

        # Prefer 'male' or 'female' if tied with None
        for preferred in ["male", "female"]:
            if preferred in top_values:
                return preferred

        # Otherwise return the most common (could be None)
        return most_common[0][0]

    def resolve_adm_time_function(self, adm_time_spans: list[dict]) -> tuple[str, str | None] | None:
        """Resolve admission date (YYYY-MM-DD) and optional HH:MM; else (None, None)."""

        def extract_date_time(text: str) -> tuple[str, str | None] | None:
            text = text.strip()

            # Match single/double-digit MM/DD/YYYY or YYYY-MM-DD
            date_match = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})", text)
            if not date_match:
                return None

            date_str = date_match.group(1)

            date_str_norm = date_str.replace("/", "-")

            # Try both formats
            for fmt in ("%Y-%m-%d", "%m-%d-%Y"):
                try:
                    date_obj = datetime.strptime(date_str_norm, fmt)
                    break
                except ValueError:
                    continue
            else:
                return None

            # Optional time: HH:MM or H:MM
            time_match = re.search(r"\b(\d{1,2}:\d{2})\b", text)
            time_str = time_match.group(1) if time_match else None

            return date_obj.date().isoformat(), time_str

        # Sort spans by descending confidence
        sorted_spans = sorted(adm_time_spans, key=lambda x: x.get("confidence", 0), reverse=True)

        for span in sorted_spans:
            raw_text = span.get("text", "").strip()
            result = extract_date_time(raw_text)
            if result:
                return result

        return (None, None)


# ---------------- CLI to run this script on its own using pre-structured EHR extractions ----------------


def _parse_args_cli():
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Run normalization script over pre-structured extractions (JSONL → SQL DB/TABLE)")

    # Keep same interface as parse_args()
    p.add_argument(
        "jsonl_path",
        type=Path,
        help="Input structured jsonl ('name', 'uid', 'filename', 'age', 'sex', 'adm_date', 'entities' for each patient; name,filename,age,sex,adm_date are optional)",
    )
    p.add_argument("out_path", type=Path, help="Output workspace directory (intermediates + DB/CSV)")
    p.add_argument("--to_csv", action="store_true", help="Write normalized table to CSV instead of SQL DB (OK for small data)")
    p.add_argument("--keep", action="store_true", help="Keep intermediate files")
    p.add_argument(
        "--ont_corr",
        dest="ont_corr",
        type=Path,
        help="ontology correction yml file",
        default=Path("./pipeline_ingest/db/ontology_corrections.yml"),
    )
    p.add_argument("--mrconso_rrf", dest="mrconso_rrf", type=Path, help="mrconso.rrf file directory", required=True)
    p.add_argument("--mrrel_rrf", dest="mrrel_rrf", type=Path, help="mrrel.rrf file directory", required=True)
    p.add_argument("--mrsty_rrf", dest="mrsty_rrf", type=Path, help="mrsty.rrf file directory", required=True)
    p.add_argument(
        "--no_pruning",
        action="store_true",
        help="By default the ontology is pruned to only contain terms in dataset; call --no_pruning to disable",
    )

    args = p.parse_args()

    # -------- Path validation -------- #
    if not args.jsonl_path.exists():
        log.error(f"Input not found: {args.jsonl_path}")
        sys.exit(1)
    if not args.mrconso_rrf.exists():
        log.error(f"mrconso.rrf path does not exist: {args.mrconso_rrf}")
        sys.exit(1)
    if not args.mrrel_rrf.exists():
        log.error(f"mrrel.rrf path does not exist: {args.mrrel_rrf}")
        sys.exit(1)
    if not args.mrsty_rrf.exists():
        log.error(f"mrsty.rrf path does not exist: {args.mrsty_rrf}")
        sys.exit(1)
    if not args.ont_corr.exists():
        log.error(f"Ontology correction file does not exist: {args.ont_corr}")
        sys.exit(1)

    # Ensure output directory exists
    args.out_path.mkdir(parents=True, exist_ok=True)

    return args


if __name__ == "__main__":
    """
    CLI for normalization over pre-structured extractions (e.g. from LLM output).

    Same arguments as `parse_args()`, except excludes NER/REL model paths.
    Input JSONL must contain:
      - uid
      - name, filename, age, sex, adm_date (optional)
      - entities: list of (term, timestamp list, negation list), where 'timestamp list' and 'negation list' may be empty or None.
    """

    from logging_setup import setup_logging

    setup_logging("INFO")

    # get input/output paths
    args = _parse_args_cli()
    workdir = args.out_path
    term_stats_out = workdir / "term_stats_resolved.csv", workdir / "term_stats_unresolved.csv"
    csv_out = workdir / "db.csv"

    # ensure that input jsonl file contains uid & entities keys for each patient
    with open(str(args.jsonl_path)) as f:
        for line in f:
            json_line = json.loads(line)
            if ("uid" not in json_line) or ("entities" not in json_line):
                raise ValueError("input JSONL is invalid as it must contain 'uid' & 'entities' entries for EVERY patient")
            if not isinstance(json_line["entities"], list):
                raise ValueError("input JSONL is invalid as every 'entities' entry must be a list")

    # now run normalization pipeline
    log.info("Stage 1: NORMALIZE …")
    norm = Normalizer(
        str(args.mrconso_rrf),
        str(args.mrrel_rrf),
        str(args.mrsty_rrf),
        str(args.ont_corr),
        keep=args.keep,
        no_pruning=args.no_pruning,
        cli_entry=True,
    )
    norm.normalize(in_jsonl=str(args.jsonl_path), term_stats_csv=(str(term_stats_out[0]), str(term_stats_out[1])))

    # now write outputs
    log.info("Stage 2: DB WRITE …")
    norm.write_db(str(csv_out), str(workdir), args.to_csv)

    log.info("Done")
