# write_db.py
"""
Utilities to persist normalized clinical extractions into analytical tables (DuckDB) and precompute alias embeddings (FAISS).

Overview
--------
Given an object that exposes a processed `master_list` of patients and concepts, this module builds:

1) event_df
   - One row per (patient, normalized_term, value/modifier, timestamp window)
2) patient_df
   - Patient-level demographic / file metadata
3) baselines_df
   - Rolling/anchored baselines (admission_start, latest, admission_end, etc.)
4) trends_df
   - Cross-anchor trend signals (uptrended / downtrended / unchanged) and deltas
5) alias_to_canonical_df
   - Map of string aliases to canonical terms / concept classes
6) term_concept_class_map_df
   - Map of terms to their concept classes

It can either:
- Write only the events CSV (when `to_csv=True`) and skip DB tables, or
- Materialize all tables into DuckDB and create indices, then build a FAISS vector index of alias embeddings and save vector IDs to a .npy file.

Extra notes
-----------
- Pump-related baselines and trends are NOT stored to ensure generalizability
- Tables are (re)created idempotently (DROP IF EXISTS + CREATE IF NOT EXISTS).
"""

import json
import logging
from collections import Counter
from datetime import datetime

import duckdb
import faiss
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def to_db(norm_obj, sql_out_path, faiss_out_path, vector_ids_out_path, csv_out, to_csv) -> None:
    """
    Build analytics tables from a normalized extraction object and optionally precompute alias embeddings.

    This function consumes a normalization object (`norm_obj`) that exposes patient-level extractions and dictionaries for alias canonicalization. It materializes only an events csv when to_csv flag is used, or a DuckDB database and alias embeddings used to run the query engine.

    Outputs
    -------
    When `to_csv == True`:
      - Writes `event_df` CSV to `csv_out`.

    When `to_csv == False`:
      - Persists the following DuckDB tables at `sql_out_path` (recreated idempotently):
          * event_df
          * patient_df
          * baselines_df
          * trends_df
          * alias_to_canonical_df
          * term_concept_class_map_df
        Appropriate indices are created to accelerate lookups.
      - Computes alias embeddings from `norm_obj.a_to_c_dict_*` (filtered by `min_alias_len`),
        writes a FAISS inner-product index to `faiss_out_path`, and saves aligned alias IDs
        to `vector_ids_out_path` as a .npy array.
    """

    # initialize database
    if not to_csv:
        con = duckdb.connect(sql_out_path)
        con.execute("DROP TABLE IF EXISTS event_df")
        con.execute("DROP TABLE IF EXISTS patient_df")
        con.execute("DROP TABLE IF EXISTS baselines_df")
        con.execute("DROP TABLE IF EXISTS trends_df")
        con.execute("DROP TABLE IF EXISTS alias_to_canonical_df")
        con.execute("DROP TABLE IF EXISTS term_concept_class_map_df")

    # create main events dataframe
    column_names = [
        "patient_id",
        "normalized_term",
        "value",
        "unit",
        "modifiers",
        "is_active",
        "is_started",
        "is_ended",
        "is_worsening",
        "level_entry",
        "day_offset_start",
        "day_offset_end",
        "history",
        "negated",
        "timestamp_start",
        "timestamp_end",
        "source_entity",
        "source_negations",
        "source_timestamps",
    ]
    rows_list = []
    for master_list_i in range(len(norm_obj.master_list)):
        adm_date = (
            datetime.strptime(norm_obj.master_list[master_list_i][5][0], "%Y-%m-%d").date()
            if norm_obj.master_list[master_list_i][5] is not None and norm_obj.master_list[master_list_i][5][0] is not None
            else None
        )
        for c_ent in norm_obj.master_list[master_list_i][7]:
            date_start = (
                datetime.strptime(c_ent["timestamp"][0][0], "%Y-%m-%d").date()
                if isinstance(c_ent["timestamp"][0], tuple) and c_ent["timestamp"][0][0] is not None
                else None
            )
            date_end = (
                datetime.strptime(c_ent["timestamp"][1][0], "%Y-%m-%d").date()
                if isinstance(c_ent["timestamp"][1], tuple) and c_ent["timestamp"][1][0] is not None
                else None
            )
            day_offset_start = int((date_start - adm_date).days) if adm_date is not None and date_start is not None else np.nan
            day_offset_end = (
                int((date_end - adm_date).days) if adm_date is not None and date_end is not None else day_offset_start
            )
            history_entry = isinstance(c_ent["timestamp"][0], str) and c_ent["timestamp"][0].lower() == "history"
            for value in c_ent["values"] or [{"value": None, "unit": None}]:
                rows_list.append(
                    [
                        norm_obj.master_list[master_list_i][0],
                        c_ent["normalized_term"],
                        (
                            float(value["value"])
                            if isinstance(value["value"], int | float) and not pd.isna(value["value"])
                            else np.nan
                        ),
                        str(value["unit"]) if not pd.isna(value["unit"]) else None,
                        c_ent["modifier"],
                        (
                            False
                            if (c_ent["negated"] or any(mod in c_ent["modifier"] for mod in {"resolved", "stopped"}))
                            else True
                        ),  # is_active entry
                        (
                            True if any(mod in c_ent["modifier"] for mod in {"started"}) and not c_ent["negated"] else False
                        ),  # is_started entry
                        (
                            True
                            if any(mod in c_ent["modifier"] for mod in {"resolved", "stopped"}) and not c_ent["negated"]
                            else False
                        ),  # is_ended entry
                        (
                            True if any(mod in c_ent["modifier"] for mod in {"worsened", "increased"}) else False
                        ),  # is_worsening entry
                        next((mod for mod in ["high", "low", "medium"] if mod in c_ent["modifier"]), None),  # level_entry entry
                        day_offset_start,
                        day_offset_end,
                        history_entry,
                        c_ent["negated"],
                        c_ent["timestamp"][0],
                        c_ent["timestamp"][1],
                        c_ent["source"],
                        c_ent["negation_texts"],
                        c_ent["timestamp_texts"],
                    ]
                )

    event_df = pd.DataFrame.from_records(rows_list, columns=column_names)

    event_df["modifiers"] = event_df["modifiers"].apply(json.dumps)  # to make duckdb friendly
    event_df["source_entity"] = event_df["source_entity"].apply(json.dumps)  # to make duckdb friendly
    event_df["source_timestamps"] = event_df["source_timestamps"].apply(json.dumps)  # to make duckdb friendly
    event_df["source_negations"] = event_df["source_negations"].apply(json.dumps)  # to make duckdb friendly

    if to_csv:
        event_df.to_csv(csv_out, index=False)
        return  # skip rest of function call if user only wants events csv

    # create second patient dataframe containing each unique patient, and their unique ids, and their demographic identifiers
    # columns: patient_id, patient_name, raw_ehr_filename, age, sex, adm_time
    column_names = ["patient_id", "patient_name", "raw_ehr_filename", "age", "sex", "adm_time"]
    rows_list = []
    for master_list_i in range(len(norm_obj.master_list)):
        rows_list.append(
            [
                norm_obj.master_list[master_list_i][0],
                (
                    norm_obj.master_list[master_list_i][1].strip().lower()
                    if isinstance(norm_obj.master_list[master_list_i][1], str)
                    else None
                ),
                norm_obj.master_list[master_list_i][2],
                (
                    int(norm_obj.master_list[master_list_i][3])
                    if isinstance(norm_obj.master_list[master_list_i][3], int | float)
                    and not pd.isna(norm_obj.master_list[master_list_i][3])
                    else np.nan
                ),
                (
                    norm_obj.master_list[master_list_i][4].strip().lower()
                    if isinstance(norm_obj.master_list[master_list_i][4], str)
                    else None
                ),
                norm_obj.master_list[master_list_i][5],
            ],
        )

    patient_df = pd.DataFrame.from_records(rows_list, columns=column_names)

    if to_csv:
        con.register("temp_patient_df", patient_df)
        con.execute("CREATE TABLE IF NOT EXISTS patient_df AS SELECT * FROM temp_patient_df")
        con.execute("CREATE INDEX IF NOT EXISTS idx_patient_df_patient_id ON patient_df(patient_id);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_patient_df_patient_name ON patient_df(patient_name);")

    # create third baselines dataframe
    # columns: patient_id, normalized_term, anchor (adm baseline, adm end, latest, pre-pump baseline, during-pump baseline, post-pump baseline), value (for numeric data), unit (for numeric data), modifier (for qualitative data), start_day, end_day, n_quant_measurements, n_qual_measurements, type (quant,qual or mixed), method ('median'), source_entities, source_negations, source_timestamps
    # parameters
    day_buffer = 1  # used for 'admission_start', and 'latest' to set how many days beyond timepoint to construct window over
    day_buffer_adm_end = 2  # used for 'admission_end'
    _blanking_days = 1  # used for pump-related baselines

    # first only get entries with timestamps
    event_df = event_df[event_df["day_offset_start"].notna()]

    # now generate the baselines dataframe using the events dataframe
    column_names = [
        "patient_id",
        "normalized_term",
        "anchor",
        "value",
        "unit",
        "level_entry",
        "start_day",
        "end_day",
        "n_quant_measurements",
        "n_qual_measurements",
        "type",
        "method",
        "source_entities",
        "source_negations",
        "source_timestamps",
    ]
    rows_list = []

    # 1) add admission_start baselines
    event_df_sub = event_df[
        (event_df["day_offset_end"] <= day_buffer)
        & (event_df["level_entry"].isin(["high", "low", "medium"]) | event_df["value"].notna())
    ]
    for pt_id in set(event_df_sub["patient_id"]):
        event_df_sub_pt = event_df_sub[event_df_sub["patient_id"] == pt_id]
        for term in set(event_df_sub_pt["normalized_term"]):
            event_df_sub_pt_term = event_df_sub_pt[event_df_sub_pt["normalized_term"] == term]
            values_list = list(event_df_sub_pt_term[event_df_sub_pt_term["value"].notna()]["value"])
            units_list = list(event_df_sub_pt_term[event_df_sub_pt_term["value"].notna()]["unit"])
            modifiers_list = list(
                event_df_sub_pt_term[event_df_sub_pt_term["level_entry"].isin(["high", "low", "medium"])]["level_entry"]
            )
            rows_list.append(
                [
                    pt_id,
                    term,
                    "admission_start",
                    np.median(values_list) if len(values_list) > 0 else np.nan,
                    Counter(units_list).most_common(1)[0][0] if len(units_list) > 0 else None,
                    Counter(modifiers_list).most_common(1)[0][0] if len(modifiers_list) > 0 else None,
                    0,
                    day_buffer,
                    len(values_list),
                    len(modifiers_list),
                    "quant" if len(values_list) > 0 else "qual",
                    "median" if len(values_list) > 0 else "mode",
                    list(event_df_sub_pt_term["source_entity"]),
                    list(event_df_sub_pt_term["source_negations"]),
                    list(event_df_sub_pt_term["source_timestamps"]),
                ]
            )

    # 2) add latest baselines
    event_df_sub = event_df[(event_df["level_entry"].isin(["high", "low", "medium"]) | event_df["value"].notna())]
    for pt_id in set(event_df_sub["patient_id"]):
        event_df_sub_pt = event_df_sub[event_df_sub["patient_id"] == pt_id]
        for term in set(event_df_sub_pt["normalized_term"]):
            event_df_sub_pt_term = event_df_sub_pt[event_df_sub_pt["normalized_term"] == term]
            start_day, end_day = (
                np.nanmax(event_df_sub_pt_term["day_offset_end"]) - day_buffer
                if np.nanmax(event_df_sub_pt_term["day_offset_end"]) != 0
                else 0
            ), np.nanmax(event_df_sub_pt_term["day_offset_end"])
            event_df_sub_pt_term = event_df_sub_pt_term[event_df_sub_pt_term["day_offset_start"] >= start_day]
            if len(event_df_sub_pt_term) > 0:
                values_list = list(event_df_sub_pt_term[event_df_sub_pt_term["value"].notna()]["value"])
                units_list = list(event_df_sub_pt_term[event_df_sub_pt_term["value"].notna()]["unit"])
                modifiers_list = list(
                    event_df_sub_pt_term[event_df_sub_pt_term["level_entry"].isin(["high", "low", "medium"])]["level_entry"]
                )
                rows_list.append(
                    [
                        pt_id,
                        term,
                        "latest",
                        np.median(values_list) if len(values_list) > 0 else np.nan,
                        Counter(units_list).most_common(1)[0][0] if len(units_list) > 0 else None,
                        Counter(modifiers_list).most_common(1)[0][0] if len(modifiers_list) > 0 else None,
                        start_day,
                        end_day,
                        len(values_list),
                        len(modifiers_list),
                        "quant" if len(values_list) > 0 else "qual",
                        "median" if len(values_list) > 0 else "mode",
                        list(event_df_sub_pt_term["source_entity"]),
                        list(event_df_sub_pt_term["source_negations"]),
                        list(event_df_sub_pt_term["source_timestamps"]),
                    ]
                )

    # 3) add admission_end baselines
    event_df_sub = event_df[(event_df["level_entry"].isin(["high", "low", "medium"]) | event_df["value"].notna())]
    for pt_id in set(event_df_sub["patient_id"]):
        event_df_sub_pt = event_df_sub[event_df_sub["patient_id"] == pt_id]
        start_day, end_day = (
            np.nanmax(event_df_sub_pt["day_offset_end"]) - day_buffer_adm_end
            if np.nanmax(event_df_sub_pt["day_offset_end"]) != 0
            else 0
        ), np.nanmax(event_df_sub_pt["day_offset_end"])
        event_df_sub_pt = event_df_sub_pt[
            (event_df_sub_pt["day_offset_start"] >= start_day)
            & (event_df_sub_pt["level_entry"].isin(["high", "low", "medium"]) | event_df_sub_pt["value"].notna())
        ]
        for term in set(event_df_sub_pt["normalized_term"]):
            event_df_sub_pt_term = event_df_sub_pt[event_df_sub_pt["normalized_term"] == term]
            values_list = list(event_df_sub_pt_term[event_df_sub_pt_term["value"].notna()]["value"])
            units_list = list(event_df_sub_pt_term[event_df_sub_pt_term["value"].notna()]["unit"])
            modifiers_list = list(
                event_df_sub_pt_term[event_df_sub_pt_term["level_entry"].isin(["high", "low", "medium"])]["level_entry"]
            )
            rows_list.append(
                [
                    pt_id,
                    term,
                    "admission_end",
                    np.median(values_list) if len(values_list) > 0 else np.nan,
                    Counter(units_list).most_common(1)[0][0] if len(units_list) > 0 else None,
                    Counter(modifiers_list).most_common(1)[0][0] if len(modifiers_list) > 0 else None,
                    start_day,
                    end_day,
                    len(values_list),
                    len(modifiers_list),
                    "quant" if len(values_list) > 0 else "qual",
                    "median" if len(values_list) > 0 else "mode",
                    list(event_df_sub_pt_term["source_entity"]),
                    list(event_df_sub_pt_term["source_negations"]),
                    list(event_df_sub_pt_term["source_timestamps"]),
                ]
            )

    baselines_df = pd.DataFrame.from_records(rows_list, columns=column_names)
    baselines_df["source_entities"] = baselines_df["source_entities"].apply(json.dumps)  # to make duckdb friendly
    baselines_df["source_negations"] = baselines_df["source_negations"].apply(json.dumps)  # to make duckdb friendly
    baselines_df["source_timestamps"] = baselines_df["source_timestamps"].apply(json.dumps)  # to make duckdb friendly

    con.register("temp_baselines_df", baselines_df)
    con.execute("CREATE TABLE IF NOT EXISTS baselines_df AS SELECT * FROM temp_baselines_df")
    con.execute("CREATE INDEX IF NOT EXISTS idx_baselines_df_patient_id ON baselines_df(patient_id);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_baselines_df_normalized_term ON baselines_df(normalized_term);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_baselines_df_anchor ON baselines_df(anchor);")

    # create fourth dataframe with trends
    # columns: patient_id, normalized_term, change (qualitative trend descriptor), delta (if from numeric data), unit (if from numeric data), time_delta, val_start (if from numeric data), val_end (if from numeric data), anchor1, anchor2, anchor1_start, anchor1_end, anchor2_start, anchor2_end, anchor1_n_quant_measurements, anchor1_n_qual_measurements, anchor2_n_quant_measurements, anchor2_n_qual_measurements, type ('delta from medians', 'modifier comparison')
    # parameters
    epsilon = 0.2

    def compute_change(quant_entries1, quant_entries2, qual_entries1, qual_entries2, epsilon):
        if len(quant_entries1) > 0 and len(quant_entries2) > 0:
            entry1, entry2 = np.median(quant_entries1), np.median(quant_entries2)
            if abs(entry1) < 0.001 and abs(entry2) < 0.001:
                return "unchanged"  # protect against division by zero
            if abs(abs(entry2 - entry1) / ((entry1 + entry2) / 2)) > epsilon:
                if entry2 - entry1 > 0:
                    return "uptrended"
                elif entry2 - entry1 < 0:
                    return "downtrended"
            else:
                return "unchanged"
        elif len(qual_entries1) > 0 and len(qual_entries2) > 0:
            entry1, entry2 = Counter(qual_entries1).most_common(1)[0][0], Counter(qual_entries2).most_common(1)[0][0]
            if entry2 == "high" and (entry1 == "medium" or entry1 == "low"):
                return "uptrended"
            elif entry2 == "medium" and entry1 == "low":
                return "uptrended"
            elif entry1 == "high" and (entry2 == "medium" or entry2 == "low"):
                return "downtrended"
            elif entry1 == "medium" and entry2 == "low":
                return "downtrended"
            elif entry1 == entry2:
                return "unchanged"

    def gather_trends_data(anchor_str, baselines_df_pt_term):
        baselines_df_pt_term_sub = baselines_df_pt_term[baselines_df_pt_term["anchor"] == anchor_str]
        anchor_quant_entries, anchor_qual_entries = list(
            baselines_df_pt_term_sub[baselines_df_pt_term_sub["value"].notna()]["value"]
        ), list(baselines_df_pt_term_sub[baselines_df_pt_term_sub["level_entry"].isin(["high", "low", "medium"])]["level_entry"])
        anchor_time, anchor_start_time, anchor_end_time = (
            (
                (np.nanmin(baselines_df_pt_term_sub["start_day"]) + np.nanmax(baselines_df_pt_term_sub["end_day"])) / 2
                if len(baselines_df_pt_term_sub) > 0
                else np.nan
            ),
            np.nanmin(baselines_df_pt_term_sub["start_day"]) if len(baselines_df_pt_term_sub) > 0 else np.nan,
            np.nanmax(baselines_df_pt_term_sub["end_day"]) if len(baselines_df_pt_term_sub) > 0 else np.nan,
        )
        anchor_unit = (
            Counter(baselines_df_pt_term_sub["unit"]).most_common(1)[0][0] if len(baselines_df_pt_term_sub) > 0 else None
        )
        anchor_n_quant_entries, anchor_n_qual_entries = (
            np.median(baselines_df_pt_term_sub["n_quant_measurements"]) if len(baselines_df_pt_term_sub) > 0 else np.nan
        ), (np.median(baselines_df_pt_term_sub["n_qual_measurements"]) if len(baselines_df_pt_term_sub) > 0 else np.nan)
        return (
            anchor_quant_entries,
            anchor_qual_entries,
            anchor_time,
            anchor_start_time,
            anchor_end_time,
            anchor_unit,
            anchor_n_quant_entries,
            anchor_n_qual_entries,
        )

    def compute_trends_row(
        anchor1_str,
        anchor2_str,
        pt_id,
        term,
        anchor1_quant_entries,
        anchor1_qual_entries,
        anchor1_time,
        anchor1_start_time,
        anchor1_end_time,
        anchor1_unit,
        anchor1_n_quant_entries,
        anchor1_n_qual_entries,
        anchor2_quant_entries,
        anchor2_qual_entries,
        anchor2_time,
        anchor2_start_time,
        anchor2_end_time,
        anchor2_unit,
        anchor2_n_quant_entries,
        anchor2_n_qual_entries,
    ):
        # ensure that there is no temporal overlap between these two anchors
        if anchor1_end_time < anchor2_start_time or anchor2_end_time < anchor1_start_time:
            return [
                pt_id,
                term,
                compute_change(anchor1_quant_entries, anchor2_quant_entries, anchor1_qual_entries, anchor2_qual_entries, epsilon),
                (
                    np.median(anchor2_quant_entries) - np.median(anchor1_quant_entries)
                    if (len(anchor1_quant_entries) > 0 and len(anchor2_quant_entries) > 0)
                    else np.nan
                ),
                anchor1_unit if anchor1_unit == anchor2_unit else None,
                anchor2_time - anchor1_time,
                (
                    np.median(anchor1_quant_entries)
                    if (len(anchor1_quant_entries) > 0 and len(anchor2_quant_entries) > 0)
                    else np.nan
                ),
                (
                    np.median(anchor2_quant_entries)
                    if (len(anchor1_quant_entries) > 0 and len(anchor2_quant_entries) > 0)
                    else np.nan
                ),
                anchor1_str,
                anchor2_str,
                anchor1_start_time,
                anchor1_end_time,
                anchor2_start_time,
                anchor2_end_time,
                anchor1_n_quant_entries,
                anchor1_n_qual_entries,
                anchor2_n_quant_entries,
                anchor2_n_qual_entries,
                "medians delta" if (len(anchor1_quant_entries) > 0 and len(anchor2_quant_entries) > 0) else "modifier comparison",
            ]
        else:
            return None

    column_names = [
        "patient_id",
        "normalized_term",
        "change",
        "delta",
        "unit",
        "time_delta",
        "val_start",
        "val_end",
        "anchor1",
        "anchor2",
        "anchor1_start",
        "anchor1_end",
        "anchor2_start",
        "anchor2_end",
        "anchor1_quant_measurements",
        "anchor1_qual_measurements",
        "anchor2_quant_measurements",
        "anchor2_qual_measurements",
        "type",
    ]
    rows_list = []
    for pt_id in set(baselines_df["patient_id"]):
        baselines_df_pt = baselines_df[baselines_df["patient_id"] == pt_id]
        for term in set(baselines_df_pt["normalized_term"]):
            baselines_df_pt_term = baselines_df_pt[baselines_df_pt["normalized_term"] == term]

            # gather data
            (
                quant_entries_dict,
                qual_entries_dict,
                time_dict,
                start_time_dict,
                end_time_dict,
                unit_dict,
                n_quant_entries_dict,
                n_qual_entries_dict,
            ) = ({}, {}, {}, {}, {}, {}, {}, {})
            for anchor in [
                "admission_start",
                "admission_end",
                "latest",
                "prepump",
                "pump",
                "postpump",
                "prepump_earliest",
                "pump_earliest",
                "postpump_earliest",
                "prepump_latest",
                "pump_latest",
                "postpump_latest",
            ]:
                (
                    quant_entries_dict[anchor],
                    qual_entries_dict[anchor],
                    time_dict[anchor],
                    start_time_dict[anchor],
                    end_time_dict[anchor],
                    unit_dict[anchor],
                    n_quant_entries_dict[anchor],
                    n_qual_entries_dict[anchor],
                ) = gather_trends_data(anchor, baselines_df_pt_term)

            # compute cross-anchor changes
            trend_anchors_list = ["admission_start", "admission_end", "latest"]
            for anchor1 in trend_anchors_list:
                for anchor2 in trend_anchors_list:
                    if anchor1 != anchor2:
                        if (len(quant_entries_dict[anchor1]) > 0 and len(quant_entries_dict[anchor2]) > 0) or (
                            len(qual_entries_dict[anchor1]) > 0 and len(qual_entries_dict[anchor2]) > 0
                        ):
                            new_row = compute_trends_row(
                                anchor1,
                                anchor2,
                                pt_id,
                                term,
                                quant_entries_dict[anchor1],
                                qual_entries_dict[anchor1],
                                time_dict[anchor1],
                                start_time_dict[anchor1],
                                end_time_dict[anchor1],
                                unit_dict[anchor1],
                                n_quant_entries_dict[anchor1],
                                n_qual_entries_dict[anchor1],
                                quant_entries_dict[anchor2],
                                qual_entries_dict[anchor2],
                                time_dict[anchor2],
                                start_time_dict[anchor2],
                                end_time_dict[anchor2],
                                unit_dict[anchor2],
                                n_quant_entries_dict[anchor2],
                                n_qual_entries_dict[anchor2],
                            )
                            if new_row is not None:
                                rows_list.append(new_row)

    trends_df = pd.DataFrame.from_records(rows_list, columns=column_names)

    con.register("temp_trends_df", trends_df)
    con.execute("CREATE TABLE IF NOT EXISTS trends_df AS SELECT * FROM temp_trends_df")
    con.execute("CREATE INDEX IF NOT EXISTS idx_trends_df_patient_id ON trends_df(patient_id);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_trends_df_normalized_term ON trends_df(normalized_term);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_trends_df_anchor1 ON trends_df(anchor1);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_trends_df_anchor2 ON trends_df(anchor2);")

    # create fifth dataframe mapping from a string to its canonical term/concept class
    # columns: alias_id, alias, target_kind ('term' or 'concept_class'), canonical, status (always 'active'), priority (always 100)
    column_names = ["alias_id", "alias", "target_kind", "canonical", "status", "priority"]
    rows_list = []
    for alias, canonical in norm_obj.a_to_c_dict.items():
        rows_list.append([len(rows_list) + 1, alias, "term", canonical, "active", 100])
    for alias, canonical in norm_obj.a_to_c_dict_concept_classes.items():
        rows_list.append([len(rows_list) + 1, alias, "concept_class", canonical, "active", 100])

    alias_to_canonical_df = pd.DataFrame.from_records(rows_list, columns=column_names)

    con.register("temp_alias_to_canonical_df", alias_to_canonical_df)
    con.execute("CREATE TABLE IF NOT EXISTS alias_to_canonical_df AS SELECT * FROM temp_alias_to_canonical_df")
    con.execute("CREATE INDEX IF NOT EXISTS idx_alias_to_canonical_df_alias ON alias_to_canonical_df(alias);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_alias_to_canonical_df_target_kind ON alias_to_canonical_df(target_kind);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_alias_to_canonical_df_canonical ON alias_to_canonical_df(canonical);")

    # create sixth dataframe mapping from each term to its associated concept class
    # columns: term, concept_class, status (always 'active'), priority (always 100)
    column_names = ["term", "concept_class", "status", "priority"]
    rows_list = []

    # iterate through master_list and get all term->concept class mappings
    term_to_cc_set_mappings = {}
    for master_list_i in range(len(norm_obj.master_list)):
        for c_ent in norm_obj.master_list[master_list_i][7]:
            if c_ent["normalized_term"] not in term_to_cc_set_mappings.keys():
                term_to_cc_set_mappings[c_ent["normalized_term"]] = set(c_ent["concept_classes"])

    # now use this dict to populate term_concept_class_map_df
    for term, concept_class_set in term_to_cc_set_mappings.items():
        for concept_class in concept_class_set:
            rows_list.append([term, concept_class, "active", 100])

    term_concept_class_map_df = pd.DataFrame.from_records(rows_list, columns=column_names)

    con.register("temp_term_concept_class_map_df", term_concept_class_map_df)
    con.execute("CREATE TABLE IF NOT EXISTS term_concept_class_map_df AS SELECT * FROM temp_term_concept_class_map_df")
    con.execute("CREATE INDEX IF NOT EXISTS idx_term_concept_class_map_df_term ON term_concept_class_map_df(term);")
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_term_concept_class_map_df_concept_class ON term_concept_class_map_df(concept_class);"
    )

    # save & close duckdb connection
    con.close()

    # now generate precomputed alias embeddings and store to faiss_out_path & vector_ids_out_path
    if len(alias_to_canonical_df) != len(set(alias_to_canonical_df["alias_id"])):
        raise ValueError("every alias must have a unique id in precomputed embedding files")

    alias_to_canonical_df = alias_to_canonical_df[alias_to_canonical_df["alias"].str.len() >= norm_obj.min_alias_len]
    X = (
        norm_obj.emb_model.encode(
            alias_to_canonical_df["alias"].tolist(), batch_size=256, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        if norm_obj.device == "cuda"
        else norm_obj.emb_model.encode(
            alias_to_canonical_df["alias"].tolist(), normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
    )
    faiss.normalize_L2(X)

    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, faiss_out_path)
    np.save(vector_ids_out_path, alias_to_canonical_df["alias_id"].to_numpy(np.int64))
