"""Canonical schema definitions & dimension constants for AuraBrite.

Kept as a single source of truth so the generator, the SQL agent, and the
KPI metadata table all agree.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

DIVISIONS = {
    "DIV-PC": "Personal Care",
    "DIV-HC": "Home Care",
    "DIV-NF": "Nutrition & Foods",
}

BRANDS = [
    # brand_id, brand_name, division_id, subcategory
    ("BR-AGL", "AuraGlow", "DIV-PC", "Skincare"),
    ("BR-DEN", "Dentafresh", "DIV-PC", "Oral Care"),
    ("BR-CLX", "CleanX", "DIV-HC", "Home Care"),
    ("BR-NVT", "NutriVita", "DIV-NF", "Health Foods"),
]

SKUS = [
    # sku_id, sku_name, brand_id, pack_size, list_price_usd
    ("SKU-AGL-SRM-030", "AuraGlow Serum 30ml", "BR-AGL", "30ml", 24.90),
    ("SKU-AGL-DCM-050", "AuraGlow Day Cream 50g", "BR-AGL", "50g", 18.50),
    ("SKU-AGL-NCM-030", "AuraGlow Niacinamide Booster 30ml", "BR-AGL", "30ml", 22.00),
    ("SKU-DEN-TPT-100", "Dentafresh Whitening Toothpaste 100g", "BR-DEN", "100g", 4.20),
    ("SKU-DEN-MWH-250", "Dentafresh Mouthwash 250ml", "BR-DEN", "250ml", 5.80),
    ("SKU-CLX-DSH-750", "CleanX Dish Liquid 750ml", "BR-CLX", "750ml", 3.60),
    ("SKU-CLX-FLR-1L",  "CleanX Floor Cleaner 1L",  "BR-CLX", "1L",    4.80),
    ("SKU-CLX-LAU-2L",  "CleanX Laundry Detergent 2L", "BR-CLX", "2L", 9.90),
    ("SKU-NVT-OAT-1KG", "NutriVita Oats 1kg", "BR-NVT", "1kg", 6.40),
    ("SKU-NVT-PRT-500", "NutriVita Plant Protein 500g", "BR-NVT", "500g", 21.00),
]

REGIONS = {
    "REG-EMEA":  "EMEA",
    "REG-APAC":  "APAC",
    "REG-NA":    "North America",
    "REG-LATAM": "LATAM",
}

COUNTRIES = [
    # market_id, country, region_id
    ("MKT-UK",  "United Kingdom", "REG-EMEA"),
    ("MKT-DE",  "Germany",        "REG-EMEA"),
    ("MKT-IN",  "India",          "REG-APAC"),
    ("MKT-SG",  "Singapore",      "REG-APAC"),
    ("MKT-US",  "United States",  "REG-NA"),
    ("MKT-BR",  "Brazil",         "REG-LATAM"),
]

CHANNELS = [
    # channel_id, channel_name
    ("CH-MT",   "Modern Trade"),
    ("CH-GT",   "Traditional / General Trade"),
    ("CH-EC",   "E-Commerce"),
    ("CH-QC",   "Quick-Commerce"),
]

WAREHOUSES = [
    # warehouse_id, market_id, name
    ("WH-UK-LON", "MKT-UK", "London DC"),
    ("WH-DE-HAM", "MKT-DE", "Hamburg DC"),
    ("WH-IN-MUM", "MKT-IN", "Mumbai DC"),
    ("WH-SG-JUR", "MKT-SG", "Singapore Jurong DC"),
    ("WH-US-DAL", "MKT-US", "Dallas DC"),
    ("WH-BR-SPO", "MKT-BR", "São Paulo DC"),
]

# ---------------------------------------------------------------------------
# KPI metadata (aliases & business definitions)
# ---------------------------------------------------------------------------

KPI_METADATA = [
    # kpi_id, kpi_name, aliases, definition, formula, unit
    (
        "KPI-GR",
        "Gross Revenue",
        "revenue|sales|top line|gross sales",
        "Total invoiced revenue before discounts, returns, and rebates.",
        "SUM(fact_sales.gross_revenue_usd)",
        "USD",
    ),
    (
        "KPI-NR",
        "Net Revenue",
        "net sales|nsv|net turnover",
        "Gross revenue minus discounts, returns and trade rebates.",
        "SUM(fact_sales.net_revenue_usd)",
        "USD",
    ),
    (
        "KPI-VOL",
        "Volume Units",
        "units|units sold|volume",
        "Number of consumer units sold.",
        "SUM(fact_sales.volume_units)",
        "units",
    ),
    (
        "KPI-GM",
        "Gross Margin %",
        "gm|gross margin|margin",
        "(Net Revenue - COGS) / Net Revenue.",
        "1 - SUM(cogs_usd) / NULLIF(SUM(net_revenue_usd),0)",
        "percent",
    ),
    (
        "KPI-DIO",
        "Days of Inventory Outstanding",
        "dio|days on hand|doh|days of stock",
        "Average number of days inventory is held before being sold.",
        "AVG(fact_inventory.days_of_inventory_outstanding)",
        "days",
    ),
    (
        "KPI-OOS",
        "Out-of-Stock Events",
        "oos|stockouts|out of stock",
        "Count of unique SKU-warehouse-day OOS incidents.",
        "SUM(fact_inventory.out_of_stock_events)",
        "count",
    ),
]

# ---------------------------------------------------------------------------
# DDL — pure SQL, works on both DuckDB and SQLite
# ---------------------------------------------------------------------------

DDL_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS dim_product (
        sku_id        TEXT PRIMARY KEY,
        sku_name      TEXT NOT NULL,
        brand_id      TEXT NOT NULL,
        brand_name    TEXT NOT NULL,
        division_id   TEXT NOT NULL,
        division_name TEXT NOT NULL,
        subcategory   TEXT NOT NULL,
        pack_size     TEXT NOT NULL,
        list_price_usd DOUBLE NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS dim_geography (
        market_id  TEXT PRIMARY KEY,
        country    TEXT NOT NULL,
        region_id  TEXT NOT NULL,
        region     TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS dim_channel (
        channel_id   TEXT PRIMARY KEY,
        channel_name TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS dim_warehouse (
        warehouse_id TEXT PRIMARY KEY,
        market_id    TEXT NOT NULL,
        name         TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS dim_kpi_metadata (
        kpi_id     TEXT PRIMARY KEY,
        kpi_name   TEXT NOT NULL,
        aliases    TEXT NOT NULL,
        definition TEXT NOT NULL,
        formula    TEXT NOT NULL,
        unit       TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS fact_sales (
        date              DATE NOT NULL,
        sku_id            TEXT NOT NULL,
        brand_id          TEXT NOT NULL,
        market_id         TEXT NOT NULL,
        channel_id        TEXT NOT NULL,
        gross_revenue_usd DOUBLE NOT NULL,
        net_revenue_usd   DOUBLE NOT NULL,
        volume_units      BIGINT NOT NULL,
        cogs_usd          DOUBLE NOT NULL,
        discount_rate     DOUBLE NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS fact_inventory (
        date                         DATE NOT NULL,
        sku_id                       TEXT NOT NULL,
        warehouse_id                 TEXT NOT NULL,
        stock_on_hand                BIGINT NOT NULL,
        days_of_inventory_outstanding DOUBLE NOT NULL,
        out_of_stock_events          INTEGER NOT NULL
    );
    """,
]


@dataclass(frozen=True)
class TableSummary:
    name: str
    rows: int


TABLE_NAMES = [
    "dim_product",
    "dim_geography",
    "dim_channel",
    "dim_warehouse",
    "dim_kpi_metadata",
    "fact_sales",
    "fact_inventory",
]
