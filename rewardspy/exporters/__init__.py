"""Persistence for reward history.

The in-memory store is bounded (10k records by default), so long runs need the
full history written to disk. JSONL is the streaming sink, written one record at
a time as calls happen. CSV and Parquet are batch exports, produced from a store
or from a JSONL file: CSV when you want a spreadsheet-friendly table, Parquet
when the history is large and you want typed, compressed columns.
"""

from .csv import write_csv
from .jsonl import JSONLExporter, read_jsonl
from .parquet import write_parquet

__all__ = ["JSONLExporter", "read_jsonl", "write_csv", "write_parquet"]
