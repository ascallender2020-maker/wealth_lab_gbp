from __future__ import annotations

import sys
from pathlib import Path

# Ensure 'src/' is on sys.path when running from a fresh clone (no package install required)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from wealth_lab.config import DEFAULT_UNIVERSE, DEFAULT_BT
from wealth_lab.pipeline import ARTIFACTS_DIR, build_all_and_save, load_artifacts


REPORTS_DIR = Path("reports")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_html(path: Path, title: str, body_html: str) -> None:
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 28px; }}
    a {{ color: #0b63ce; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .card {{ border: 1px solid #e6e6e6; border-radius: 10px; padding: 16px; }}
    img {{ max-width: 100%; height: auto; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #eee; padding: 8px; text-align: left; font-size: 14px; }}
    th {{ background: #fafafa; }}
    .muted {{ color: #666; font-size: 14px; }}
  </style>
</head>
<body>
{body_html}
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def save_line_chart_png(series: pd.Series, out: Path, title: str) -> None:
    plt.figure()
    series.plot()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def save_df_plot_png(df: pd.DataFrame, out: Path, title: str) -> None:
    plt.figure()
    df.plot()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def main() -> None:
    ensure_dir(REPORTS_DIR)

    # Build artifacts if missing
    if not (Path(ARTIFACTS_DIR) / "saa_backtest.parquet").exists():
        build_all_and_save(DEFAULT_UNIVERSE, DEFAULT_BT, ARTIFACTS_DIR)

    d = load_artifacts(Path(ARTIFACTS_DIR))

    # ---- SAA ----
    bt = d["saa_backtest"]
    nav_png = REPORTS_DIR / "saa_nav.png"
    save_line_chart_png(bt["portfolio_nav"], nav_png, "SAA backtest NAV (monthly)")

    w_cols = [c for c in bt.columns if c.startswith("w_")]
    weights = bt[w_cols].copy()
    weights.columns = [c[2:] for c in weights.columns]
    w_png = REPORTS_DIR / "saa_weights.png"
    save_df_plot_png(weights, w_png, "SAA weights over time")

    latest_w = weights.iloc[-1].sort_values(ascending=False).to_frame("weight")
    latest_html = latest_w.to_html(float_format=lambda x: f"{x:.2%}")

    write_html(
        REPORTS_DIR / "saa.html",
        "SAA Optimiser",
        f"""
<h1>SAA optimiser (monthly)</h1>
<p class="muted">Universe: {", ".join(DEFAULT_UNIVERSE.tickers)} | Lookback: {DEFAULT_BT.lookback_months} months</p>
<div class="grid">
  <div class="card">
    <h2>NAV</h2>
    <img src="{nav_png.name}" alt="NAV">
  </div>
  <div class="card">
    <h2>Weights</h2>
    <img src="{w_png.name}" alt="Weights">
  </div>
</div>
<div class="card" style="margin-top:18px">
  <h2>Latest weights</h2>
  {latest_html}
</div>
<p><a href="index.html">Back to index</a></p>
""",
    )

    # ---- Tax ----
    trades = d["tax_trades"]
    sold = d["tax_sold_lots"]
    tax_liab = float(d["tax_summary"].get("tax_liability", 0.0))

    write_html(
        REPORTS_DIR / "tax_rebal.html",
        "Tax-aware rebalancing",
        f"""
<h1>Tax-aware rebalancing (demo)</h1>
<p class="muted">Estimated tax liability (demo): <b>£{tax_liab:,.0f}</b></p>
<div class="grid">
  <div class="card">
    <h2>Trade list</h2>
    {trades.to_html(index=False)}
  </div>
  <div class="card">
    <h2>Lots selected to sell</h2>
    {sold.to_html(index=False)}
  </div>
</div>
<p><a href="index.html">Back to index</a></p>
""",
    )

    # ---- Segmentation ----
    profiles = d["segment_profiles"]
    clients = d["clients"]
    seg_counts = clients["segment"].value_counts().sort_index()
    seg_png = REPORTS_DIR / "segments.png"
    plt.figure()
    seg_counts.plot(kind="bar")
    plt.title("Client segments (counts)")
    plt.tight_layout()
    plt.savefig(seg_png, dpi=160)
    plt.close()

    write_html(
        REPORTS_DIR / "segmentation.html",
        "Client segmentation",
        f"""
<h1>Client segmentation</h1>
<p class="muted">Client attributes are synthetic (publish-safe). Pipeline is real.</p>
<div class="grid">
  <div class="card">
    <h2>Segment sizes</h2>
    <img src="{seg_png.name}" alt="Segments">
  </div>
  <div class="card">
    <h2>Segment profiles (median)</h2>
    {profiles.to_html()}
  </div>
</div>
<div class="card" style="margin-top:18px">
  <h2>Sample clients</h2>
  {clients.sample(min(30, len(clients)), random_state=DEFAULT_BT.seed).to_html(index=False)}
</div>
<p><a href="index.html">Back to index</a></p>
""",
    )

    # ---- Attribution + Factors ----
    attr = d["attribution"]
    attr_png = REPORTS_DIR / "attribution.png"
    save_df_plot_png(attr[["allocation", "selection", "interaction", "active"]], attr_png, "Attribution effects")

    alpha = d["factor_alpha"]
    betas = d["factor_betas"]
    alpha_png = REPORTS_DIR / "alpha.png"
    save_line_chart_png(alpha, alpha_png, "Rolling alpha")

    betas_png = REPORTS_DIR / "betas.png"
    save_df_plot_png(betas, betas_png, "Rolling factor betas")

    write_html(
        REPORTS_DIR / "attribution_factors.html",
        "Attribution + Factors",
        f"""
<h1>Attribution and factor exposure</h1>
<p class="muted">Attribution is a demo vs equal-weight benchmark; factors are Fama-French 3 factors (monthly) where available.</p>
<div class="grid">
  <div class="card">
    <h2>Attribution effects</h2>
    <img src="{attr_png.name}" alt="Attribution">
  </div>
  <div class="card">
    <h2>Rolling alpha</h2>
    <img src="{alpha_png.name}" alt="Alpha">
  </div>
  <div class="card">
    <h2>Rolling betas</h2>
    <img src="{betas_png.name}" alt="Betas">
  </div>
  <div class="card">
    <h2>Latest betas</h2>
    {betas.tail(1).to_html(float_format=lambda x: f"{x:.2f}")}
  </div>
</div>
<p><a href="index.html">Back to index</a></p>
""",
    )

    # ---- Index ----
    write_html(
        REPORTS_DIR / "index.html",
        "Wealth Lab — Director View",
        """
<h1>Wealth Lab (GBP) — Director View</h1>
<p class="muted">Static reports generated from public data. Use the links below.</p>
<div class="grid">
  <div class="card"><h2>📊 SAA optimiser</h2><p><a href="saa.html">Open report</a></p></div>
  <div class="card"><h2>🧾 Tax-aware rebalancing</h2><p><a href="tax_rebal.html">Open report</a></p></div>
  <div class="card"><h2>👥 Client segmentation</h2><p><a href="segmentation.html">Open report</a></p></div>
  <div class="card"><h2>🧩 Attribution + factors</h2><p><a href="attribution_factors.html">Open report</a></p></div>
</div>
""",
    )

    print("Reports written to ./reports (open reports/index.html)")


if __name__ == "__main__":
    main()
