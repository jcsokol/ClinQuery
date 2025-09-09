# logging_setup.py
import logging
import os
import sys


def setup_logging(default_level: str = "INFO") -> None:
    """
    Configure root logging to print to the console (stdout) only—no log files. Intended for use by BOTH pipeline_ingest and pipeline_query:
    """
    level_name = os.getenv("LOG_LEVEL", default_level).upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)

    root.addHandler(handler)
    root.setLevel(level)
