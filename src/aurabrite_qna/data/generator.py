"""Deterministic synthetic-data generator for AuraBrite Consumer Brands.

The generator seeds a fully reproducible warehouse **and** writes narrative
Markdown documents whose claims match the numbers in the warehouse. This is
what makes the QnA system honest — the answer to "why did APAC sales dip
in Q4 2025?" is *both* in the numbers and in the supply-chain narrative.

The overlaps to preserve:
    * APAC sales for AuraGlow and Dentafresh drop ~14% in Q4 2025.
    * Root cause is a Singapore port strike (Oct 20 – Nov 12 2025) plus a
      global niacinamide shortage — documented in the Supply-Chain report.
    * Consumer-Insights Q1 2026 explains the +11% E-commerce lift for
      NutriVita by a shift to clean-label products.
    * Tesco/Walmart/Amazon rebate rates match the modelled discount rates.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from .schema import (
    BRANDS,
    CHANNELS,
    COUNTRIES,
    DDL_STATEMENTS,
    DIVISIONS,
    KPI_METADATA,
    SKUS,
    TABLE_NAMES,
    WAREHOUSES,
    TableSummary,
)
from .warehouse import Warehouse

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Business tuning constants — these ARE the ground truth. Documents below
# must match them.
# ---------------------------------------------------------------------------

BASE_MONTHLY_UNITS = {  # per SKU × channel (before market weighting)
    "SKU-AGL-SRM-030": 4_800,
    "SKU-AGL-DCM-050": 6_200,
    "SKU-AGL-NCM-030": 3_400,
    "SKU-DEN-TPT-100": 22_000,
    "SKU-DEN-MWH-250": 9_500,
    "SKU-CLX-DSH-750": 18_000,
    "SKU-CLX-FLR-1L":  11_000,
    "SKU-CLX-LAU-2L":  14_500,
    "SKU-NVT-OAT-1KG": 16_000,
    "SKU-NVT-PRT-500": 3_800,
}

MARKET_WEIGHT = {
    "MKT-UK": 1.10, "MKT-DE": 1.20, "MKT-IN": 1.35,
    "MKT-SG": 0.55, "MKT-US": 1.55, "MKT-BR": 0.75,
}

CHANNEL_MIX = {
    "CH-MT": 0.42, "CH-GT": 0.24, "CH-EC": 0.22, "CH-QC": 0.12,
}

CHANNEL_DISCOUNT = {  # matches Retail Partnership Notes
    "CH-MT": 0.14,   # Modern Trade — Tesco/Walmart avg rebate
    "CH-GT": 0.09,   # Traditional Trade
    "CH-EC": 0.18,   # E-Commerce — Amazon co-op & promotions
    "CH-QC": 0.22,   # Quick-Commerce — heavy funded promos
}

COGS_RATIO = {  # cogs / gross_revenue
    "BR-AGL": 0.38,   # premium skincare, richer margin
    "BR-DEN": 0.52,
    "BR-CLX": 0.55,
    "BR-NVT": 0.48,
}

# Long-run growth (annualised) baked into the seasonal engine.
BRAND_TREND = {
    "BR-AGL": 0.08,   # +8% YoY
    "BR-DEN": 0.03,
    "BR-CLX": 0.02,
    "BR-NVT": 0.11,   # +11% — the clean-label story
}

# --- Special "events" the narrative documents also describe ----------------
SG_PORT_STRIKE = (date(2025, 10, 20), date(2025, 11, 12))   # ~14% APAC dip
NIACINAMIDE_SHORTAGE = (date(2025, 10, 1), date(2025, 12, 31))  # AuraGlow raw material
EU_PALM_OIL_CRUNCH = (date(2025, 11, 15), date(2026, 1, 31))    # Dentafresh EU
NUTRIVITA_EC_LIFT = (date(2026, 1, 1), date(2026, 6, 30))       # +11% EC lift Q1-Q2 2026


@dataclass
class GenerationReport:
    tables: list[TableSummary]
    documents: list[Path]

    def as_text(self) -> str:
        lines = ["Warehouse tables:"]
        lines += [f"  • {t.name:<20s} {t.rows:>7,} rows" for t in self.tables]
        lines.append("Documents:")
        lines += [f"  • {p.name}" for p in self.documents]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_all(
    warehouse: Warehouse,
    docs_dir: Path,
    seed: int = 20260921,
) -> GenerationReport:
    """Populate warehouse + write narrative documents. Idempotent."""

    rng = random.Random(seed)
    _reset_schema(warehouse)
    _load_dimensions(warehouse)
    sales_rows, inv_rows = _build_facts(rng)
    _bulk_insert(warehouse, sales_rows, inv_rows)

    table_summaries = _summarise(warehouse)

    docs_dir.mkdir(parents=True, exist_ok=True)
    doc_paths = _write_documents(docs_dir)

    return GenerationReport(tables=table_summaries, documents=doc_paths)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _reset_schema(wh: Warehouse) -> None:
    for tbl in TABLE_NAMES:
        wh.execute(f"DROP TABLE IF EXISTS {tbl};")
    for stmt in DDL_STATEMENTS:
        wh.execute(stmt)


def _load_dimensions(wh: Warehouse) -> None:
    # dim_product
    prod_rows = []
    for sku_id, sku_name, brand_id, pack, price in SKUS:
        b_row = next(b for b in BRANDS if b[0] == brand_id)
        div_id = b_row[2]
        prod_rows.append(
            (sku_id, sku_name, brand_id, b_row[1], div_id, DIVISIONS[div_id], b_row[3], pack, price)
        )
    wh.executemany(
        "INSERT INTO dim_product VALUES (?,?,?,?,?,?,?,?,?);", prod_rows
    )

    wh.executemany(
        "INSERT INTO dim_geography VALUES (?,?,?,?);",
        [(m, c, r, _region_name(r)) for (m, c, r) in COUNTRIES],
    )
    wh.executemany("INSERT INTO dim_channel VALUES (?,?);", CHANNELS)
    wh.executemany("INSERT INTO dim_warehouse VALUES (?,?,?);", WAREHOUSES)
    wh.executemany("INSERT INTO dim_kpi_metadata VALUES (?,?,?,?,?,?);", KPI_METADATA)


def _region_name(region_id: str) -> str:
    from .schema import REGIONS
    return REGIONS[region_id]


# ---- fact tables -----------------------------------------------------------

def _months() -> list[date]:
    out: list[date] = []
    d = date(2024, 1, 1)
    end = date(2026, 6, 1)
    while d <= end:
        out.append(d)
        # advance by one month
        year = d.year + (1 if d.month == 12 else 0)
        month = 1 if d.month == 12 else d.month + 1
        d = date(year, month, 1)
    return out


def _seasonality(month: int, brand_id: str) -> float:
    """Deterministic per-brand seasonality (Jan=1 .. Dec=12)."""
    # Skincare peaks in winter, home care in spring, foods in Q4 (holidays).
    if brand_id == "BR-AGL":
        base = 1.0 + 0.18 * math.cos((month - 1) * math.pi / 6.0)
    elif brand_id == "BR-DEN":
        base = 1.0 + 0.06 * math.sin(month * math.pi / 6.0)
    elif brand_id == "BR-CLX":
        base = 1.0 + 0.10 * math.sin((month - 3) * math.pi / 6.0)
    elif brand_id == "BR-NVT":
        base = 1.0 + 0.14 * math.cos((month - 11) * math.pi / 6.0)
    else:
        base = 1.0
    return max(0.6, base)


def _trend_multiplier(brand_id: str, month_index: int) -> float:
    """month_index is 0 for 2024-01, ~29 for 2026-06."""
    monthly = BRAND_TREND[brand_id] / 12.0
    return (1.0 + monthly) ** month_index


def _event_multiplier(brand_id: str, market_id: str, d: date) -> float:
    """Return a >0 multiplier reflecting supply-chain / demand events."""
    region_id = next(r for m, _, r in COUNTRIES if m == market_id)
    mult = 1.0

    # Singapore port strike Oct-Nov 2025 → -14% APAC sales for skincare + oral
    if region_id == "REG-APAC" and _in_range(d, SG_PORT_STRIKE):
        if brand_id in ("BR-AGL", "BR-DEN"):
            mult *= 0.86  # exact -14%

    # Global niacinamide shortage → AuraGlow Niacinamide SKU crimped globally
    if brand_id == "BR-AGL" and _in_range(d, NIACINAMIDE_SHORTAGE):
        mult *= 0.93  # -7% brand-wide

    # EU palm oil crunch → Dentafresh Europe -9%
    if brand_id == "BR-DEN" and region_id == "REG-EMEA" and _in_range(d, EU_PALM_OIL_CRUNCH):
        mult *= 0.91

    return mult


def _in_range(d: date, span: tuple[date, date]) -> bool:
    return span[0] <= d <= span[1]


def _build_facts(rng: random.Random) -> tuple[list[tuple], list[tuple]]:
    months = _months()
    sales_rows: list[tuple] = []
    inv_rows: list[tuple] = []

    for i, month in enumerate(months):
        for sku_id, _, brand_id, _, list_price in SKUS:
            base_units = BASE_MONTHLY_UNITS[sku_id]
            for market_id in [m for m, _, _ in COUNTRIES]:
                mw = MARKET_WEIGHT[market_id]
                event_mult = _event_multiplier(brand_id, market_id, month)
                trend = _trend_multiplier(brand_id, i)
                season = _seasonality(month.month, brand_id)

                for channel_id, mix in CHANNEL_MIX.items():
                    # NutriVita e-commerce clean-label lift Q1-Q2 2026
                    ec_lift = 1.0
                    if (
                        brand_id == "BR-NVT"
                        and channel_id == "CH-EC"
                        and _in_range(month, NUTRIVITA_EC_LIFT)
                    ):
                        ec_lift = 1.11  # +11% exactly

                    noise = 1.0 + rng.uniform(-0.03, 0.03)
                    units = (
                        base_units * mw * mix * season * trend * event_mult * ec_lift * noise
                    )
                    units_i = max(0, int(round(units)))

                    gross = units_i * list_price
                    discount = CHANNEL_DISCOUNT[channel_id] * (0.95 + rng.uniform(0, 0.10))
                    net = gross * (1.0 - discount)
                    cogs = gross * COGS_RATIO[brand_id]
                    sales_rows.append(
                        (
                            month.isoformat(),
                            sku_id,
                            brand_id,
                            market_id,
                            channel_id,
                            round(gross, 2),
                            round(net, 2),
                            units_i,
                            round(cogs, 2),
                            round(discount, 4),
                        )
                    )

            # inventory per warehouse (1 warehouse per market)
            for wh_id, wh_mkt, _ in WAREHOUSES:
                # Stock on hand is the target 45 DIO worth of units for that market
                mw = MARKET_WEIGHT[wh_mkt]
                monthly_demand = base_units * mw
                target_stock = int(monthly_demand * 1.5)  # ~45 DIO
                # OOS events spike during supply events for affected brand/region
                region_id = next(r for m, _, r in COUNTRIES if m == wh_mkt)
                oos_events = 0
                dio = 45.0 + rng.uniform(-4, 4)
                if brand_id in ("BR-AGL", "BR-DEN") and region_id == "REG-APAC" and _in_range(month, SG_PORT_STRIKE):
                    target_stock = int(target_stock * 0.55)
                    oos_events = rng.randint(6, 14)
                    dio = rng.uniform(18, 26)
                if brand_id == "BR-AGL" and _in_range(month, NIACINAMIDE_SHORTAGE):
                    target_stock = int(target_stock * 0.80)
                    oos_events = max(oos_events, rng.randint(2, 6))
                if brand_id == "BR-DEN" and region_id == "REG-EMEA" and _in_range(month, EU_PALM_OIL_CRUNCH):
                    target_stock = int(target_stock * 0.78)
                    oos_events = max(oos_events, rng.randint(1, 5))

                inv_rows.append(
                    (
                        month.isoformat(),
                        sku_id,
                        wh_id,
                        max(0, target_stock),
                        round(dio, 2),
                        oos_events,
                    )
                )

    return sales_rows, inv_rows


def _bulk_insert(wh: Warehouse, sales: list[tuple], inv: list[tuple]) -> None:
    wh.executemany(
        "INSERT INTO fact_sales VALUES (?,?,?,?,?,?,?,?,?,?);", sales
    )
    wh.executemany(
        "INSERT INTO fact_inventory VALUES (?,?,?,?,?,?);", inv
    )


def _summarise(wh: Warehouse) -> list[TableSummary]:
    out = []
    for t in TABLE_NAMES:
        r = wh.query(f"SELECT COUNT(*) FROM {t};")
        out.append(TableSummary(name=t, rows=int(r.rows[0][0])))
    return out


# ---------------------------------------------------------------------------
# Documents — narrative that stays numerically consistent with the facts
# ---------------------------------------------------------------------------

def _write_documents(docs_dir: Path) -> list[Path]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, body in _document_bodies().items():
        p = docs_dir / name
        p.write_text(body, encoding="utf-8")
        written.append(p)
    return written


def _document_bodies() -> dict[str, str]:
    return {
        "AuraGlow_Brand_Strategy_2025_2026.md": _doc_auraglow_strategy(),
        "Supply_Chain_Disruption_Report_Q4_2025.md": _doc_supply_chain(),
        "Retail_Partnership_Notes_Tesco_Walmart_Amazon.md": _doc_retail_partnership(),
        "Consumer_Insights_Q1_2026.md": _doc_consumer_insights(),
        "KPI_Glossary.md": _doc_kpi_glossary(),
    }


def _doc_auraglow_strategy() -> str:
    return """# AuraGlow — Brand Strategy 2025 → 2026

**Owner:** Global Skincare CMO Office · **Classification:** Internal · Confidential

## 1. Strategic Ambition
AuraGlow enters 2026 as AuraBrite's fastest-growing Personal Care brand, targeting
**+8% YoY net revenue growth** on the back of the *Niacinamide Booster* franchise
and the reformulated *Day Cream 50g*. The brand's positioning ("clinical clarity,
crafted daily") anchors on efficacy claims validated in the Q3 2025 consumer study.

## 2. SKU Rationalisation
The 2026 plan retains three hero SKUs and delists two long-tail variants (legacy
Rose Toner 100ml, and the discontinued Vitamin-E Oil 15ml). The active SKU roster
is:

| SKU                                  | Pack   | List (USD) | Role                |
|--------------------------------------|--------|------------|---------------------|
| AuraGlow Serum 30ml                  | 30ml   | 24.90      | Hero, entry-premium |
| AuraGlow Day Cream 50g               | 50g    | 18.50      | Volume driver       |
| AuraGlow Niacinamide Booster 30ml    | 30ml   | 22.00      | Growth engine       |

## 3. Campaign Spending
Working-media investment increases from **USD 42M (2025) to USD 56M (2026)** with
70% weighted to digital and creator-led activations in APAC and NA. Trade
investment is held flat as a % of net revenue (14% MT, 18% EC).

## 4. Brand Sentiment Snapshot
Net Brand Sentiment (measured on the AuraBrite Social Listening panel):

- 2024 average: **+38 pts**
- 2025 Q3: **+44 pts** (peak; niacinamide launch)
- 2025 Q4: **+31 pts** — *dip driven by APAC stock-outs* (see Supply-Chain
  Disruption Report Q4 2025).
- 2026 Q1: recovery to **+40 pts**.

## 5. Growth Targets by Region
- APAC: +12% NR (2026 vs 2025) — contingent on niacinamide supply normalising.
- EMEA: +6% NR — driven by Modern Trade share gains at Tesco and dm-drogerie.
- NA: +9% NR — accelerated by Amazon Premium Beauty programme.
- LATAM: +4% NR — foundation-building year.

## 6. Risks
1. Continued niacinamide shortage (see supply report) — could erode APAC targets.
2. Competitive premiumisation in North America.
3. Discount creep in E-Commerce past the 18% guardrail.
"""


def _doc_supply_chain() -> str:
    return """# Global Supply-Chain Disruption Report — Q4 2025

**Prepared by:** AuraBrite Global Supply Chain COE · **Date:** 15 December 2025
**Distribution:** Executive Committee, Regional Presidents, Category GMs

## Executive Summary
Q4 2025 saw two concurrent supply shocks impacting the Personal Care division:

1. A **logistics strike at the Port of Singapore (20 October – 12 November 2025)**
   which disrupted downstream distribution for the entire APAC hub.
2. A **global niacinamide active-ingredient shortage** driven by an upstream
   fermentation-plant outage in South Korea (October – December 2025).

Combined, these events removed an estimated **USD 14.6M of net revenue** from the
quarter and produced measurable stock-outs in six markets.

## Impact on APAC (AuraGlow & Dentafresh)
Sales in APAC for **AuraGlow** and **Dentafresh** contracted by approximately
**14%** during the strike window versus the pre-strike baseline. The Mumbai and
Singapore DCs recorded a combined **32 out-of-stock events** across the affected
SKUs during the period. Trade partners in India (Reliance Smart, DMart) and
Singapore (FairPrice) escalated stock complaints in weeks 43–46.

## Niacinamide Shortage (Global)
The niacinamide shortage impacted the AuraGlow franchise globally with an
approximate **7% brand-wide unit loss** for October–December 2025. The most
affected SKU is *AuraGlow Niacinamide Booster 30ml*. Alternative sourcing from
a European supplier is in qualification and is expected to restore full supply
by end of Q1 2026.

## EMEA Palm-Oil Crunch (Dentafresh)
A tightening of RSPO-certified palm-oil supply from Indonesia caused a
**~9% Dentafresh EMEA volume decline** for the period **15 November 2025 to
31 January 2026**. UK and Germany warehouses (London DC, Hamburg DC) reported
low but non-critical stock levels.

## Recovery Actions
- Dual-sourcing for niacinamide (Rotterdam supplier under contract from Feb 2026).
- Air-freight prioritisation for AuraGlow Serum into Singapore Jurong DC.
- Reserved slot bookings on APAC container lanes through end-Q1 2026.

## Outlook
Barring further disruption, APAC volumes are expected to fully normalise by
February 2026 and revert to plan by Q2 2026.
"""


def _doc_retail_partnership() -> str:
    return """# Retail Partnership Notes — Tesco · Walmart · Amazon

**Owner:** Global Key-Account Team · **Cycle:** JBP 2026 · **Last Updated:** Feb 2026

## 1. Tesco (United Kingdom, Modern Trade)
- Fiscal alignment: February – January (Tesco calendar).
- **Base rebate: 14%** on net invoice for Personal Care (Modern Trade rate).
- Category performance FY25: AuraGlow +9% value / Dentafresh +2% value.
- JBP asks 2026: shelf reset for the AuraGlow *Niacinamide Booster*; joint
  investment in the *Day Cream 50g* endcap in Q1.
- Risk: continued Dentafresh EMEA supply constraints (Q4 2025 palm-oil crunch).

## 2. Walmart (United States, Modern Trade)
- Fiscal alignment: February – January.
- **Base rebate: 14% MT rate**, plus a 2% growth-accelerator kicker on brands
  that exceed +8% YoY net sales.
- CleanX is the top volume brand at Walmart (Laundry Detergent 2L is the #1 SKU
  in the aisle by units).
- Category performance FY25: AuraGlow +10% value / NutriVita +12% value.
- JBP asks 2026: Rollback event for CleanX Dish Liquid 750ml in Q2; premium
  endcap for AuraGlow Serum 30ml in NA Prestige Beauty pilot stores.

## 3. Amazon (Global E-Commerce)
- Contract is a global co-op agreement, not per country.
- **E-Commerce rebate & funded-promo bundle: 18%** (matches internal EC discount
  guardrail set by CMO office).
- Category performance FY25: NutriVita +14% (subscribe-and-save), AuraGlow +11%,
  Dentafresh flat.
- JBP asks 2026: Prime Day exclusive AuraGlow bundle; investment in the
  *NutriVita Plant Protein 500g* subscription ladder.

## 4. Quick-Commerce Partners (Regional)
- Blinkit (India), Getir (EMEA legacy), Gopuff (US): funded promo intensity is
  ~**22%** — highest across channels — reflecting the acquisition-heavy phase of
  the QC channel.
"""


def _doc_consumer_insights() -> str:
    return """# Consumer Insights — Q1 2026

**Prepared by:** AuraBrite Consumer & Market Insights · **Date:** April 2026

## 1. Headline Findings
- **Clean-label demand accelerating in Health Foods.** 68% of surveyed
  consumers in NA and 61% in EMEA now list "clean ingredient list" as a
  top-3 purchase driver for cereals and plant proteins — up 12 pts vs 2024.
- This shift is reflected in a **~11% E-Commerce sales lift for NutriVita
  during Q1–Q2 2026**, concentrated in the *Oats 1kg* and *Plant Protein 500g*
  SKUs (Amazon Subscribe & Save cohort).
- **AuraGlow brand health rebounded to +40 Net Sentiment** in Q1 2026 after
  the Q4 2025 APAC dip; niacinamide messaging resonates strongly with the
  25–34 female cohort.

## 2. Consumer Sentiment by Brand (Q1 2026)
| Brand      | Net Sentiment | Notable driver                             |
|------------|---------------|--------------------------------------------|
| AuraGlow   | +40           | Efficacy proof, dermatologist endorsements |
| NutriVita  | +47           | Clean-label, sustainability packaging      |
| CleanX     | +18           | Price-value perception                     |
| Dentafresh | +22           | Whitening efficacy; -3 pts vs Q4 2025      |

## 3. Shift Toward Clean-Label Alternatives
Qualitative panels indicate that NutriVita is meaningfully benefiting from
retailer-led clean-label programmes (Whole Foods 365, Amazon Climate Pledge
Friendly). Renaming the "Oats 1kg" to "Steel-cut Oats 1kg — no additives" is
under consideration for Q3 2026.

## 4. Watch-outs
- Dentafresh EMEA sentiment softening on the back of supply issues.
- Quick-Commerce shoppers show lower brand loyalty; heavy promo intensity
  translates into repeat-rate erosion vs Modern Trade.

## 5. Recommendations
1. Double-down on clean-label positioning for NutriVita — extend claim to the
   *Oats 1kg* SKU packaging.
2. Rebuild Dentafresh EMEA share of voice once palm-oil supply normalises.
3. Cap Quick-Commerce promo intensity at 20% to protect long-run margins.
"""


def _doc_kpi_glossary() -> str:
    lines = [
        "# AuraBrite KPI Glossary",
        "",
        "The definitive reference for KPI names, aliases, and formulas used across",
        "AuraBrite reporting. This file is also loaded into `dim_kpi_metadata` in the",
        "warehouse so the SQL agent can resolve fuzzy business terms.",
        "",
        "| KPI | Aliases | Definition | Formula | Unit |",
        "| --- | ------- | ---------- | ------- | ---- |",
    ]
    for kpi_id, name, aliases, definition, formula, unit in KPI_METADATA:
        lines.append(
            f"| **{name}** ({kpi_id}) | {aliases} | {definition} | `{formula}` | {unit} |"
        )
    return "\n".join(lines) + "\n"
