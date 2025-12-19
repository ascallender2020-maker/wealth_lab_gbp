from __future__ import annotations

import sys
from pathlib import Path

# Ensure 'src/' is on sys.path when running from a fresh clone (no package install required)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from pathlib import Path

import pandas as pd
import streamlit as st

from wealth_lab.config import DEFAULT_UNIVERSE, DEFAULT_BT
from wealth_lab.pipeline import (
    ARTIFACTS_DIR,
    build_all_and_save,
    load_artifacts,
    load_monthly_returns,
    build_saa_outputs,
    build_tax_demo,
    build_segmentation_outputs,
    build_attribution_demo,
    build_factor_outputs,
)
from wealth_lab.data.prices_yf import fetch_adj_close


st.set_page_config(
    page_title="Wealth Lab (GBP)",
    page_icon="📈",
    layout="wide",
)

st.title("Wealth Lab (GBP)")
st.caption(
    "Public-data prototype: SAA optimiser (monthly), tax-aware rebalancing, client segmentation, and attribution/factor exposure."
)

# ---- Mode selection ----
mode = st.sidebar.radio("View mode", ["Director Mode", "Analyst Mode"], index=0)

use_saved = st.sidebar.checkbox("Use precomputed artifacts", value=True)
art_dir = Path(ARTIFACTS_DIR)

# ---- Data / artifacts helpers ----
@st.cache_data(show_spinner=False)
def _build_on_the_fly():
    returns = load_monthly_returns(DEFAULT_UNIVERSE)
    saa_bt = build_saa_outputs(returns, DEFAULT_BT)

    w_cols = [c for c in saa_bt.columns if c.startswith("w_")]
    weights = saa_bt[w_cols].copy()
    weights.columns = [c[2:] for c in weights.columns]

    prices = fetch_adj_close(list(DEFAULT_UNIVERSE.tickers), start=str(returns.index.min().date()))
    px_last = prices.resample("M").last().iloc[-1]
    w_last = weights.iloc[-1]

    trades, tax_liab, sold_table = build_tax_demo(px_last, w_last, seed=DEFAULT_BT.seed)

    clients, profiles = build_segmentation_outputs(seed=DEFAULT_BT.seed)
    attribution = build_attribution_demo(returns.loc[weights.index], weights)

    alpha, betas = build_factor_outputs(saa_bt["portfolio_return"], window=36)

    return {
        "returns": returns,
        "saa_backtest": saa_bt,
        "tax_trades": trades,
        "tax_sold_lots": sold_table,
        "tax_summary": {"tax_liability": float(tax_liab)},
        "clients": clients,
        "segment_profiles": profiles,
        "attribution": attribution,
        "factor_alpha": alpha,
        "factor_betas": betas,
    }


def get_data():
    if use_saved and art_dir.exists() and (art_dir / "saa_backtest.parquet").exists():
        try:
            return load_artifacts(art_dir)
        except Exception:
            # Fall back gracefully
            return _build_on_the_fly()
    return _build_on_the_fly()


def director_landing():
    st.subheader("Director Mode")
    st.write(
        "Click a tile to view a model. This mode uses defaults and hides most controls."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        go_saa = st.button("📊 Strategic Asset Allocation", use_container_width=True)
    with c2:
        go_tax = st.button("🧾 Tax-aware Rebalancing", use_container_width=True)
    with c3:
        go_seg = st.button("👥 Client Segmentation", use_container_width=True)
    with c4:
        go_att = st.button("🧩 Attribution + Factors", use_container_width=True)

    # Persist selection in session state
    if "director_page" not in st.session_state:
        st.session_state.director_page = "saa"

    if go_saa:
        st.session_state.director_page = "saa"
    if go_tax:
        st.session_state.director_page = "tax"
    if go_seg:
        st.session_state.director_page = "seg"
    if go_att:
        st.session_state.director_page = "att"

    return st.session_state.director_page


def render_saa(d):
    st.header("Strategic Asset Allocation (monthly)")
    bt = d["saa_backtest"]
    st.metric("Backtest months", int(bt.shape[0]))
    st.line_chart(bt["portfolio_nav"], height=260)

    w_cols = [c for c in bt.columns if c.startswith("w_")]
    weights = bt[w_cols].copy()
    weights.columns = [c[2:] for c in weights.columns]

    st.subheader("Latest weights")
    latest = weights.iloc[-1].sort_values(ascending=False).to_frame("weight")
    st.dataframe(latest.style.format({"weight": "{:.2%}"}), use_container_width=True)

    st.subheader("Weight history")
    st.area_chart(weights, height=260)


def render_tax(d):
    st.header("Tax-aware Rebalancing (demo)")
    trades = d["tax_trades"].copy()
    tax_liab = float(d["tax_summary"].get("tax_liability", 0.0))
    st.metric("Estimated CG tax liability (demo)", f"£{tax_liab:,.0f}")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Trade list")
        st.dataframe(trades, use_container_width=True)
    with c2:
        st.subheader("Lots selected to sell (explainability)")
        sold = d["tax_sold_lots"]
        st.dataframe(sold, use_container_width=True)


def render_seg(d):
    st.header("Client segmentation")
    st.caption("Client attributes are synthetic (publish-safe). The pipeline is real.")
    profiles = d["segment_profiles"]
    st.subheader("Segment profiles (median feature values)")
    st.dataframe(profiles, use_container_width=True)

    clients = d["clients"]
    st.subheader("Segment sizes")
    counts = clients["segment"].value_counts().sort_index()
    st.bar_chart(counts, height=240)

    st.subheader("Sample clients")
    st.dataframe(clients.sample(min(30, len(clients)), random_state=DEFAULT_BT.seed), use_container_width=True)


def render_att(d):
    st.header("Attribution and factor exposure")
    attr = d["attribution"]
    st.subheader("Brinson-style effects (vs equal-weight benchmark; demo)")
    st.line_chart(attr[["allocation", "selection", "interaction", "active"]], height=260)

    alpha = d["factor_alpha"]
    betas = d["factor_betas"]
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Rolling alpha")
        st.line_chart(alpha, height=260)
    with c2:
        st.subheader("Rolling factor betas")
        st.line_chart(betas, height=260)


# ---- Main app flow ----
if mode == "Analyst Mode":
    st.sidebar.markdown("---")
    if st.sidebar.button("Build / refresh artifacts", help="Downloads data and stores precomputed outputs in ./artifacts"):
        with st.spinner("Building artifacts..."):
            build_all_and_save(DEFAULT_UNIVERSE, DEFAULT_BT, ARTIFACTS_DIR)
        st.success("Artifacts built. Toggle 'Use precomputed artifacts' to load them.")

    tabs = st.tabs(["SAA", "Tax", "Segmentation", "Attribution + Factors"])
    d = get_data()

    with tabs[0]:
        render_saa(d)
    with tabs[1]:
        render_tax(d)
    with tabs[2]:
        render_seg(d)
    with tabs[3]:
        render_att(d)

else:
    page = director_landing()
    d = get_data()

    st.markdown("---")
    if page == "saa":
        render_saa(d)
    elif page == "tax":
        render_tax(d)
    elif page == "seg":
        render_seg(d)
    else:
        render_att(d)
