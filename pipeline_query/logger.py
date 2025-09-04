"""
logger.py

Utility module for logging query payloads into a Postgres database.

Note: **the logger is commented out in `main.py`**. If you want to activate logging, you must remove the comment in `main.py` where `log_payload(...)` is called.

"""

import os, json
from typing import Any, Dict, Iterable, Optional
import numpy as np
import pandas as pd
import psycopg
from config import settings
from datetime import datetime, timezone
import math


DDL = """
create table if not exists query_logs (
  request_id        uuid primary key,
  user_query        text not null,
  request_timestamp timestamptz not null,
  run_status        text not null,
  payload_json      jsonb not null,
  created_at        timestamptz not null default now()
);
"""

def _coerce_ts(ts: Any) -> datetime:
	"""
	Normalize a timestamp-like input into a timezone-aware UTC datetime.

	Parameters
	----------
	ts : Any
		May be a datetime, ISO8601 string, or other.

	Returns
	-------
	datetime
		UTC-aware datetime object. Falls back to `now()` if unparseable.
	"""
	if isinstance(ts, datetime):
		return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
	if isinstance(ts, str):
		# normalize trailing Z to +00:00 for fromisoformat
		try:
			return datetime.fromisoformat(ts.replace("Z", "+00:00"))
		except ValueError:
			pass
	return datetime.now(timezone.utc)
	
def _clean(o: Any) -> Any:
	"""Convert payload to strict-JSON-safe types (no NaN/Inf, no pandas/numpy dtypes)."""
	if o is None:
		return None
	if isinstance(o, (bool, int, str)):
		return o
	if isinstance(o, float):
		return None if (math.isnan(o) or math.isinf(o)) else o
	if isinstance(o, (np.integer,)):
		return int(o)
	if isinstance(o, (np.floating,)):
		f = float(o); return None if (math.isnan(f) or math.isinf(f)) else f
	if isinstance(o, (np.bool_,)):
		return bool(o)
	if isinstance(o, (datetime,)):
		return o.isoformat()
	if isinstance(o, pd.Timestamp):
		return o.to_pydatetime().isoformat()
	if isinstance(o, dict):
		return {k: _clean(v) for k, v in o.items()}
	if isinstance(o, (list, tuple, set)):
		return [_clean(v) for v in o]
	if isinstance(o, pd.Series):
		return _clean(o.to_dict())
	if isinstance(o, pd.DataFrame):
		return {"_type": "dataframe_summary", "rows": int(len(o)), "cols": [str(c) for c in o.columns]}
	return o  
	
def log_payload(payload: Dict[str, Any]) -> Optional[int]:
	"""
	Insert a cleaned payload into the `query_logs` Postgres table.

	Parameters
	----------
	payload : Dict[str, Any]
		Dictionary containing query details. Must include:
		- `request_id` : UUID
		- `user_query` : str
		- `request_timestamp` : datetime or str
		- `run_status` : str

	Returns
	-------
	Optional[int]
		The current total row count in `query_logs`, or None if logging
		is disabled (no DSN) or if an error occurred.
	"""
	dsn = settings.params.pg_dsn
	if not dsn:
		return None

	request_id  = payload.get("request_id")
	user_query  = payload.get("user_query", "")
	request_ts  = _coerce_ts(payload.get("request_timestamp"))
	run_status  = payload.get("run_status", "unknown")

	# Sanitize to strict JSON
	clean_payload = _clean(payload)

	try:
		with psycopg.connect(dsn) as conn, conn.cursor() as cur:
			cur.execute(DDL)
			# Let psycopg encode JSON; no explicit ::jsonb cast needed
			pj = json.dumps(clean_payload, allow_nan=False)  # converts NaN/Inf to error; ensure you cleaned them
			cur.execute(
				"""
				insert into query_logs (
					request_id, user_query, request_timestamp, run_status, payload_json
				) values (%s, %s, %s, %s, %s::jsonb)
				on conflict (request_id) do nothing
				""",
				(request_id, user_query, request_ts, run_status, pj),
			)
			cur.execute("select count(*) from query_logs;")
			return int(cur.fetchone()[0])
	except Exception:
		return None