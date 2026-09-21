"""Data layer: canonical schema and warehouse access helpers."""

from .schema import (
    BRANDS,
    CHANNELS,
    COUNTRIES,
    DDL_STATEMENTS,
    DIVISIONS,
    KPI_METADATA,
    REGIONS,
    SKUS,
    TABLE_NAMES,
    WAREHOUSES,
    TableSummary,
)
from .warehouse import Warehouse, connect

__all__ = [
    "BRANDS",
    "CHANNELS",
    "COUNTRIES",
    "DDL_STATEMENTS",
    "DIVISIONS",
    "KPI_METADATA",
    "REGIONS",
    "SKUS",
    "TABLE_NAMES",
    "TableSummary",
    "Warehouse",
    "WAREHOUSES",
    "connect",
]
