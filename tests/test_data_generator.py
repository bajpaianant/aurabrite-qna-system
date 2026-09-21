"""Sanity checks on the synthetic-data generator's *narrative-consistency* claims.

If any of these fail, either the generator was tweaked without updating the
narrative documents, or vice versa. Both must move together.
"""

from __future__ import annotations

from aurabrite_qna.data import connect


def test_all_tables_populated(sample_env):
    with connect(sample_env["warehouse_path"]) as wh:
        for table, min_rows in [
            ("dim_product", 10),
            ("dim_geography", 6),
            ("dim_channel", 4),
            ("dim_kpi_metadata", 5),
            ("fact_sales", 5000),
            ("fact_inventory", 1000),
        ]:
            rows = wh.query(f"SELECT COUNT(*) FROM {table}").rows[0][0]
            assert rows >= min_rows, f"{table} has too few rows: {rows}"


def test_documents_written(sample_env):
    docs = list(sample_env["docs_dir"].glob("*.md"))
    assert len(docs) >= 5
    names = {d.name for d in docs}
    assert "Supply_Chain_Disruption_Report_Q4_2025.md" in names
    assert "AuraGlow_Brand_Strategy_2025_2026.md" in names


def test_apac_dip_actually_happens(sample_env):
    """The Singapore port strike must depress AuraGlow+Dentafresh APAC volumes in
    Nov 2025 vs the same month a year earlier (trend-adjusted)."""
    with connect(sample_env["warehouse_path"]) as wh:
        r = wh.query(
            """
            SELECT strftime(s.date, '%Y-%m') AS ym,
                   SUM(s.net_revenue_usd) AS nsv
            FROM fact_sales s JOIN dim_geography g USING (market_id)
            WHERE s.brand_id IN ('BR-AGL','BR-DEN')
              AND g.region_id = 'REG-APAC'
              AND (s.date = DATE '2024-11-01' OR s.date = DATE '2025-11-01')
            GROUP BY 1 ORDER BY 1
            """
        )
    by_month = {row[0]: row[1] for row in r.rows}
    # Strike-affected Nov 2025 should be below Nov 2024 despite +8% underlying
    # brand trend for AuraGlow. The multiplier is 0.86 → we allow generous margin.
    assert by_month["2025-11"] < by_month["2024-11"] * 1.0, (
        f"Expected Nov-2025 APAC (AGL+DEN) < Nov-2024 due to strike; "
        f"got 2024={by_month['2024-11']:.0f} 2025={by_month['2025-11']:.0f}"
    )


def test_nutrivita_ecommerce_lift(sample_env):
    """H1 2026 EC NutriVita revenue must be > H1 2025 EC revenue (clean-label lift)."""
    with connect(sample_env["warehouse_path"]) as wh:
        r = wh.query(
            """
            SELECT
              SUM(CASE WHEN date BETWEEN DATE '2025-01-01' AND DATE '2025-06-01'
                       THEN net_revenue_usd ELSE 0 END) AS h1_2025,
              SUM(CASE WHEN date BETWEEN DATE '2026-01-01' AND DATE '2026-06-01'
                       THEN net_revenue_usd ELSE 0 END) AS h1_2026
            FROM fact_sales
            WHERE brand_id = 'BR-NVT' AND channel_id = 'CH-EC'
        """
        )
    h1_2025, h1_2026 = r.rows[0]
    assert h1_2026 > h1_2025, "NutriVita EC H1 2026 should exceed H1 2025."
