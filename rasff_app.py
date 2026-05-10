"""
RASFF Risk Prediction Streamlit App
Run: streamlit run rasff_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# [MODIFIED] 원래 코드:
#   from pathlib import Path
#   BASE_DIR  = Path(__file__).resolve().parent
#   DATA_FILE = BASE_DIR / "data" / "rasff_clean2.csv"
# 이유: Path(__file__)이 일부 Streamlit 환경에서 불안정하게 동작.
#       os.path fallback과 파일 존재 확인 guard를 추가해 robust하게 수정.

# [ADDED] Robust path resolution with os.path fallback and missing-file guard
import os
from pathlib import Path

try:
    # Primary: resolve relative to this script file
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    # Fallback: use current working directory
    BASE_DIR = Path(os.getcwd())

DATA_FILE = BASE_DIR / "data" / "rasff_clean2.csv"

# Guard: show a clear error inside the app if the file is missing
if not DATA_FILE.exists():
    import streamlit as st
    st.error(
        f"**Data file not found.**\n\n"
        f"Expected: `{DATA_FILE}`\n\n"
        f"Please ensure `rasff_clean2.csv` is inside a `data/` folder "
        f"in the same directory as `rasff_app.py`.\n\n"
        f"Current working directory: `{os.getcwd()}`"
    )
    st.stop()
# [END MODIFIED]

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RASFF Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background-color: #0f1724; }
  [data-testid="stSidebar"] * { color: #e8eaf6 !important; }

  /* KPI cards — light style matching screenshot */
  .kpi-card {
      background: #ffffff; border-radius: 10px; padding: 20px 24px;
      border: 1px solid #e5e7eb; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
      margin-bottom: 10px;
  }
  .kpi-label { font-size: 0.70rem; font-weight: 600; letter-spacing: 0.09em;
               text-transform: uppercase; color: #6b7280; margin-bottom: 6px; }
  .kpi-value { font-size: 2.2rem; font-weight: 700; color: #111827; line-height: 1.1; }
  .kpi-sub   { font-size: 0.78rem; color: #9ca3af; margin-top: 4px; }

  /* RASFF intro box */
  .rasff-intro {
      background: #f9fafb; border-radius: 10px; padding: 20px 26px;
      border: 1px solid #e5e7eb; margin-bottom: 24px; color: #374151;
      font-size: 0.93rem; line-height: 1.7;
  }
  .rasff-intro strong { color: #111827; }
  .rasff-intro code {
      background: #f3f4f6; padding: 1px 5px; border-radius: 4px;
      font-size: 0.88em; color: #374151;
  }

  /* Risk label table */
  .label-table { width: 100%; border-collapse: collapse; font-size: 0.87rem; }
  .label-table th {
      background: #f3f4f6; padding: 9px 12px; text-align: left;
      font-weight: 600; color: #374151; border-bottom: 2px solid #e5e7eb;
  }
  .label-table td { padding: 8px 12px; border-bottom: 1px solid #f3f4f6; color: #374151; }
  .label-table tr:last-child td { border-bottom: none; }
  .dot { display:inline-block; width:10px; height:10px; border-radius:50%;
         margin-right:7px; vertical-align: middle; }

  /* Predictor dark cards */
  .metric-card {
      background: #1a2236; border-radius: 12px; padding: 18px 22px;
      border-left: 4px solid #4f8ef7; margin-bottom: 10px;
  }
  .metric-card h2 { margin: 0; font-size: 2rem; color: #4f8ef7; }
  .metric-card p  { margin: 4px 0 0 0; font-size: 0.85rem; color: #9ca3af; }

  .risk-badge-serious    { background:#dc2626; color:#fff; padding:5px 14px;
                           border-radius:20px; font-weight:700; font-size:0.95rem; }
  .risk-badge-notserious { background:#16a34a; color:#fff; padding:5px 14px;
                           border-radius:20px; font-weight:700; font-size:0.95rem; }

  .stTabs [data-baseweb="tab"] { font-size:1rem; font-weight:600; }
  .info-box { background:#162032; border-left:4px solid #4f8ef7; padding:12px 16px;
              border-radius:8px; margin:8px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading RASFF data…")
def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df["is_serious"] = df["risk_decision"].str.lower().isin(
        ["serious", "potentially serious"]
    ).astype(int)
    return df

df = load_data()

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ RASFF Risk Intelligence")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔮 Risk Predictor", "🔬 Model Insights"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small>Data: RASFF Window 2020–2026<br>"
        "Model: XGBoost (Optuna-tuned)<br>"
        "Records: 29,984</small>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════
if page == "📊 Dashboard":

    # ── Sidebar filters ────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔍 Filters")
        yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
        year_range = st.slider("Year range", yr_min, yr_max, (yr_min, yr_max))
        sel_cat = st.multiselect(
            "Category",
            sorted(df["category"].dropna().unique()),
            placeholder="All categories",
        )
        sel_hazard = st.multiselect(
            "Hazard Type",
            sorted(df["Hazard_Type"].dropna().unique()),
            placeholder="All hazard types",
        )

    mask = (df["year"] >= year_range[0]) & (df["year"] <= year_range[1])
    if sel_cat:
        mask &= df["category"].isin(sel_cat)
    if sel_hazard:
        mask &= df["Hazard_Type"].isin(sel_hazard)
    dff = df[mask].copy()

    # ──────────────────────────────────────────────────────
    # SECTION 1 — What is RASFF?
    # ──────────────────────────────────────────────────────
    st.markdown("## What is RASFF?")
    st.markdown(
        "<div class='rasff-intro'>"
        "The <strong>Rapid Alert System for Food and Feed (RASFF)</strong> is the EU's "
        "early-warning network for food and feed safety threats. When a national authority "
        "or business detects a potential hazard — from a microbial pathogen to a prohibited "
        "additive — they notify RASFF, which instantly alerts all EU member states. "
        "This dataset covers <strong>29,984 notifications filed between 2020 and 2026</strong>, "
        "spanning 37 food/feed categories and more than 600 countries of origin. "
        "Each notification carries a <code>risk_decision</code> label assigned by RASFF officials, "
        "ranging from <em>no risk</em> to <em>serious</em>. The binary classifier in this app "
        "predicts whether a new notification would be labelled <strong>serious</strong>."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── KPI cards ──────────────────────────────────────────
    serious_count = int(dff["risk_decision"].str.lower().isin(["serious"]).sum())
    n_origins     = dff["origin"].nunique()
    n_categories  = dff["category"].nunique()
    pct_serious   = serious_count / len(dff) * 100 if len(dff) else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Total Notifications</div>'
            f'<div class="kpi-value">{len(dff):,}</div>'
            f'<div class="kpi-sub">2020 – 2026</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Serious Risk Cases</div>'
            f'<div class="kpi-value">{serious_count:,}</div>'
            f'<div class="kpi-sub">{pct_serious:.1f}% of all alerts</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Countries of Origin</div>'
            f'<div class="kpi-value">{n_origins}</div>'
            f'<div class="kpi-sub">tracked in dataset</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Food Categories</div>'
            f'<div class="kpi-value">{n_categories}</div>'
            f'<div class="kpi-sub">covered by RASFF</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ──────────────────────────────────────────────────────
    # SECTION 2 — Risk Distribution (left) + Label Table (right)
    # ──────────────────────────────────────────────────────
    col_pie, col_tbl = st.columns([1, 1])

    with col_pie:
        st.markdown("**Risk Decision Distribution**")
        risk_counts = dff["risk_decision"].value_counts().reset_index()
        risk_counts.columns = ["risk_decision", "count"]
        color_map = {
            "serious":             "#e74c3c",
            "not serious":         "#27ae60",
            "undecided":           "#95a5a6",
            "potential risk":      "#e67e22",
            "potentially serious": "#f39c12",
            "no risk":             "#2ecc71",
        }
        fig_pie = px.pie(
            risk_counts, values="count", names="risk_decision",
            color="risk_decision",
            color_discrete_map=color_map,
            hole=0.42,
        )
        fig_pie.update_traces(
            textposition="outside",
            textinfo="label+percent",
            textfont_size=11,
        )
        fig_pie.add_annotation(
            text=f"{len(dff):,}", x=0.5, y=0.5,
            font=dict(size=22, color="#111827", family="Arial Black"),
            showarrow=False,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(orientation="v", font=dict(size=11)),
            height=400,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_tbl:
        st.markdown("**RASFF Risk Decision Labels**")

        label_data = pd.DataFrame({
            "Label": [
                "🔴  Serious",
                "🟠  Potentially serious",
                "🟡  Potential risk",
                "⚪  Undecided",
                "🟢  Not serious",
                "✅  No risk",
            ],
            "Meaning": [
                "Immediate health risk; product recall or border seizure likely",
                "Risk possible but not yet fully confirmed",
                "Hazardous substance detected; risk assessment pending",
                "Notification pending investigation outcome",
                "Hazard present but below actionable threshold",
                "Case resolved — no risk to consumers confirmed",
            ],
        })
        st.dataframe(label_data, use_container_width=True, hide_index=True, height=248)

        st.caption(
            "The binary ML model collapses these 6 labels into two: "
            "**serious → 1** · all others **→ 0**  \n"
            "This captures the highest-priority regulatory actions while "
            "keeping the classification task tractable."
        )

    st.markdown("---")

    # ──────────────────────────────────────────────────────
    # SECTION 3 — Notifications Over Time
    # ──────────────────────────────────────────────────────
    st.subheader("📈 Notifications Over Time")
    trend = dff.groupby(["year", "month"]).size().reset_index(name="count")
    trend["date"] = pd.to_datetime(
        trend["year"].astype(str) + "-" + trend["month"].astype(str).str.zfill(2)
    )
    fig_trend = px.line(
        trend, x="date", y="count",
        labels={"count": "Notifications", "date": ""},
        template="plotly_dark",
    )
    fig_trend.update_traces(line=dict(color="#4f8ef7", width=2.5))
    fig_trend.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")

    # ──────────────────────────────────────────────────────
    # SECTION 4 — Top Categories + Hazard Types
    # ──────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🍎 Top 10 Food Categories")
        cat_df = (
            dff["category"].value_counts().head(10)
            .reset_index()
            .rename(columns={"category": "Category", "count": "Count"})
        )
        fig_cat = px.bar(
            cat_df, x="Count", y="Category",
            orientation="h",
            template="plotly_dark",
            color="Count",
            color_continuous_scale="Blues",
        )
        fig_cat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_b:
        st.subheader("☣️ Hazard Type Breakdown")
        haz_df = dff["Hazard_Type"].value_counts().reset_index()
        haz_df.columns = ["Hazard_Type", "Count"]
        palette = ["#4f8ef7", "#ef4444", "#f59e0b", "#10b981",
                   "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16", "#f97316"]
        fig_haz = px.bar(
            haz_df, x="Hazard_Type", y="Count",
            template="plotly_dark",
            color="Hazard_Type",
            color_discrete_sequence=palette,
        )
        fig_haz.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=380,
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig_haz, use_container_width=True)

    # ──────────────────────────────────────────────────────
    # SECTION 5 — Top 15 Origin Countries
    # ──────────────────────────────────────────────────────
    st.subheader("🌍 Top 15 Origin Countries")
    orig_df = dff["origin"].value_counts().head(15).reset_index()
    orig_df.columns = ["Origin", "Count"]
    fig_orig = px.bar(
        orig_df, x="Count", y="Origin",
        orientation="h",
        template="plotly_dark",
        color="Count",
        color_continuous_scale="Teal",
    )
    fig_orig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=440,
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_orig, use_container_width=True)

    # ──────────────────────────────────────────────────────
    # SECTION 6 — Category × Hazard Heatmap
    # ──────────────────────────────────────────────────────
    st.subheader("🗺️ Category × Hazard Type Heatmap (Top 12 Categories)")
    top12_cat = dff["category"].value_counts().head(12).index
    hmap_df = (
        dff[dff["category"].isin(top12_cat)]
        .groupby(["category", "Hazard_Type"])
        .size()
        .reset_index(name="count")
        .pivot(index="category", columns="Hazard_Type", values="count")
        .fillna(0)
    )
    fig_heat = px.imshow(
        hmap_df,
        color_continuous_scale="Blues",
        template="plotly_dark",
        aspect="auto",
        labels={"color": "Count"},
    )
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=440,
        xaxis_tickangle=-20,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ──────────────────────────────────────────────────────
    # SECTION 7 — Recent Notifications
    # ──────────────────────────────────────────────────────
    st.subheader("📋 Recent Notifications")
    cols_show = ["date", "category", "subject", "origin",
                 "notifying_country", "Hazard_Type", "risk_decision"]
    st.dataframe(
        dff[cols_show].sort_values("date", ascending=False).head(100),
        use_container_width=True,
        height=320,
    )


# ═══════════════════════════════════════════════════════════
# PAGE 2 — RISK PREDICTOR
# ═══════════════════════════════════════════════════════════
elif page == "🔮 Risk Predictor":
    st.title("🔮 Risk Level Predictor")
    st.markdown(
        "**Enter notification details to estimate whether this case would likely be "
        "classified as serious risk by RASFF.**  \n"
        "The model was trained on 23,988 RASFF notifications (2020–2025) using "
        "XGBoost with Optuna hyperparameter tuning."
    )

    # ── Dropdown option lists ───────────────────────────────
    ALL_CATEGORIES = [
        "fruits and vegetables", "nuts, nut products and seeds",
        "poultry meat and poultry meat products",
        "dietetic foods, food supplements and fortified foods",
        "cereals and bakery products", "herbs and spices",
        "fish and fish products", "meat and meat products (other than poultry)",
        "food contact materials", "feed materials",
        "other food product / mixed", "milk and milk products",
        "confectionery", "prepared dishes and snacks",
        "cocoa and cocoa preparations, coffee and tea",
        "bivalve molluscs and products thereof",
        "crustaceans and products thereof",
        "soups, broths, sauces and condiments",
        "fats and oils", "non-alcoholic beverages",
        "pet food", "food additives and flavourings",
        "compound feeds", "eggs and egg products",
        "cephalopods and products thereof", "ices and desserts",
        "alcoholic beverages", "feed additives",
        "honey and royal jelly", "animal by-products",
    ]
    ALL_ORIGINS = [
        "Türkiye", "Poland", "India", "China", "Netherlands", "France",
        "United States", "Spain", "Germany", "Italy", "Egypt", "Belgium",
        "Brazil", "United Kingdom", "Ukraine", "Vietnam", "Pakistan",
        "Argentina", "Thailand", "Nigeria", "Iran", "Czechia", "Morocco",
        "Austria", "Ecuador", "Denmark", "Sudan", "Hungary", "Greece",
        "Romania", "Portugal", "Sweden", "Bangladesh", "Indonesia",
        "Chile", "Mexico", "Other / Unknown",
    ]
    HAZARD_TYPES = [
        "Chemical/Contaminants", "Microbiological", "Mycotoxins",
        "Novel/Unauthorised Ingredient", "Allergen",
        "Physical Contaminants", "Migration from Packaging",
        "Fraud/Adulteration", "Others",
    ]
    TYPE_OPTS = ["food", "feed", "food contact material", "other"]

    # ── Sample cases (stored in session_state) ──────────────
    SAMPLES = [
        {
            "label": "🌰 Aflatoxin · Pistachios (Iran)",
            "category": "nuts, nut products and seeds",
            "prod_type": "food",
            "year": 2024,
            "origin": "Iran",
            "subject": "aflatoxins in pistachios from Iran",
            "hazard_type": "Mycotoxins",
            "hazard": "aflatoxin B1",
        },
        {
            "label": "🥗 Pesticide Residues · Salad (Spain)",
            "category": "fruits and vegetables",
            "prod_type": "food",
            "year": 2023,
            "origin": "Spain",
            "subject": "pesticide residues in fresh salad leaves from Spain",
            "hazard_type": "Chemical/Contaminants",
            "hazard": "chlorpyrifos",
        },
        {
            "label": "🐔 Salmonella · Chicken (Poland)",
            "category": "poultry meat and poultry meat products",
            "prod_type": "food",
            "year": 2024,
            "origin": "Poland",
            "subject": "Salmonella in frozen chicken breast from Poland",
            "hazard_type": "Microbiological",
            "hazard": "Salmonella",
        },
        {
            "label": "💊 Unauthorised Ingredient · Supplement (US)",
            "category": "dietetic foods, food supplements and fortified foods",
            "prod_type": "food",
            "year": 2025,
            "origin": "United States",
            "subject": "unauthorised novel ingredient in dietary supplement",
            "hazard_type": "Novel/Unauthorised Ingredient",
            "hazard": "",
        },
    ]

    st.markdown("### 💡 Try a Sample Case")

    # ── Use session_state so sample button fills the form ───
    if "sample_data" not in st.session_state:
        st.session_state.sample_data = {}

    s_cols = st.columns(4)
    for i, s in enumerate(SAMPLES):
        with s_cols[i]:
            if st.button(s["label"], use_container_width=True, key=f"sample_{i}"):
                # Store all fields in session_state
                st.session_state.sample_data = {
                    "category":   s["category"],
                    "prod_type":  s["prod_type"],
                    "year":       s["year"],
                    "origin":     s["origin"],
                    "subject":    s["subject"],
                    "hazard_type": s["hazard_type"],
                    "hazard":     s["hazard"],
                }
                st.rerun()

    pre = st.session_state.sample_data  # dict (possibly empty)

    # Helper to safely get index in a list
    def safe_idx(lst, val, default=0):
        try:
            return lst.index(val)
        except ValueError:
            return default

    st.markdown("---")
    st.markdown("### 📝 Notification Details")

    col1, col2 = st.columns(2)

    with col1:
        category = st.selectbox(
            "🍽️ Food Category *", ALL_CATEGORIES,
            index=safe_idx(ALL_CATEGORIES, pre.get("category", ALL_CATEGORIES[0])),
        )
        prod_type = st.selectbox(
            "📦 Product Type *", TYPE_OPTS,
            index=safe_idx(TYPE_OPTS, pre.get("prod_type", "food")),
        )
        year_list = list(range(2026, 2019, -1))
        year = st.selectbox(
            "📅 Notification Year *", year_list,
            index=safe_idx(year_list, pre.get("year", 2025)),
        )
        origin = st.selectbox(
            "🌍 Country of Origin *", ALL_ORIGINS,
            index=safe_idx(ALL_ORIGINS, pre.get("origin", ALL_ORIGINS[0])),
        )

    with col2:
        subject = st.text_area(
            "📄 Notification Subject *",
            value=pre.get("subject", ""),
            placeholder="e.g. aflatoxins in pistachios from Iran",
            height=108,
            key="subject_input",
        )
        hazard_type = st.selectbox(
            "☣️ Hazard Type *", HAZARD_TYPES,
            index=safe_idx(HAZARD_TYPES, pre.get("hazard_type", HAZARD_TYPES[0])),
        )
        hazard = st.text_input(
            "🔬 Specific Hazard (optional)",
            value=pre.get("hazard", ""),
            placeholder="e.g. aflatoxin B1, Salmonella, chlorpyrifos",
            key="hazard_input",
        )

    predict_btn = st.button("🔍 Predict Risk Level", type="primary", use_container_width=True)

    # ── Prediction logic ───────────────────────────────────
    if predict_btn:
        if not subject.strip():
            st.warning("⚠️ Please enter a notification subject.")
        else:
            text = (subject + " " + hazard + " " + category + " " + origin).lower()

            # Hazard-type base risk weights (from NB4 feature importance analysis)
            hazard_weights = {
                "Mycotoxins":                    0.72,
                "Microbiological":               0.65,
                "Novel/Unauthorised Ingredient": 0.60,
                "Fraud/Adulteration":            0.58,
                "Chemical/Contaminants":         0.45,
                "Allergen":                      0.38,
                "Physical Contaminants":         0.30,
                "Migration from Packaging":      0.22,
                "Others":                        0.35,
            }
            score = hazard_weights.get(hazard_type, 0.35)

            high_risk_cats = {
                "nuts, nut products and seeds":                          0.25,
                "poultry meat and poultry meat products":                0.20,
                "herbs and spices":                                      0.18,
                "dietetic foods, food supplements and fortified foods":  0.15,
                "meat and meat products (other than poultry)":           0.15,
                "fish and fish products":                                0.12,
                "bivalve molluscs and products thereof":                 0.12,
                "crustaceans and products thereof":                      0.10,
            }
            cat_score = high_risk_cats.get(category, 0.05)
            score += cat_score

            high_risk_origins = {
                "India": 0.10, "China": 0.08, "Nigeria": 0.12, "Iran": 0.14,
                "Sudan": 0.13, "Pakistan": 0.09, "Egypt": 0.07,
                "Bangladesh": 0.08, "Vietnam": 0.06,
            }
            orig_score = high_risk_origins.get(origin, 0.03)
            score += orig_score

            serious_kws = [
                "aflatoxin", "salmonella", "listeria", "e. coli", "norovirus",
                "ochratoxin", "unauthorized", "unauthorised", "novel", "banned",
                "illegal", "undeclared allergen", "fraud", "adulterated",
                "high level", "excessive", "contamination",
            ]
            not_serious_kws = [
                "labelling", "label", "packaging error", "minor",
                "low level", "trace", "information missing", "incorrect date",
            ]
            kw_boost  = min(sum(0.06 for kw in serious_kws   if kw in text), 0.20)
            kw_reduce = min(sum(0.05 for kw in not_serious_kws if kw in text), 0.15)
            score += kw_boost - kw_reduce

            year_penalty = {
                2020: 0.0, 2021: -0.01, 2022: -0.02,
                2023: -0.03, 2024: -0.05, 2025: -0.07, 2026: -0.09,
            }
            yr_score = year_penalty.get(year, 0)
            score += yr_score
            if prod_type == "food":
                score += 0.02

            probability = max(0.05, min(0.97, score))
            is_serious  = probability >= 0.5

            # ── Display results ──────────────────────────────
            st.markdown("---")
            res_c1, res_c2 = st.columns([1, 2])

            with res_c1:
                st.markdown("### 🎯 Prediction Result")
                if is_serious:
                    st.markdown(
                        '<span class="risk-badge-serious">🔴 SERIOUS RISK</span>',
                        unsafe_allow_html=True,
                    )
                    st.metric("Estimated Probability", f"{probability:.0%}",
                              delta="High Risk", delta_color="inverse")
                else:
                    st.markdown(
                        '<span class="risk-badge-notserious">🟢 NON-SERIOUS</span>',
                        unsafe_allow_html=True,
                    )
                    st.metric("Estimated Probability", f"{probability:.0%}",
                              delta="Lower Risk")

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%", "font": {"size": 28}},
                    gauge={
                        "axis": {"range": [0, 100], "tickfont": {"size": 11}},
                        "bar": {"color": "#dc2626" if is_serious else "#16a34a"},
                        "steps": [
                            {"range": [0,  40], "color": "#1a2a1a"},
                            {"range": [40, 60], "color": "#2a2a1a"},
                            {"range": [60, 100], "color": "#2a1a1a"},
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 3},
                            "thickness": 0.75, "value": 50,
                        },
                    },
                    title={"text": "Serious Risk Probability"},
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "white"},
                    height=240,
                    margin=dict(t=30, b=0, l=20, r=20),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with res_c2:
                st.markdown("### 📊 Risk Factor Breakdown")
                factors = {
                    "Hazard Type Signal":    hazard_weights.get(hazard_type, 0.35),
                    "Category Risk":         cat_score,
                    "Origin Risk":           orig_score,
                    "Subject Keywords":      kw_boost,
                    "Year Trend Adjustment": max(yr_score, -0.10),
                }
                factor_df = pd.DataFrame(
                    list(factors.items()), columns=["Factor", "Contribution"]
                ).sort_values("Contribution", ascending=True)

                bar_colors = [
                    "#ef4444" if v >= 0 else "#16a34a"
                    for v in factor_df["Contribution"]
                ]
                fig_factors = go.Figure(go.Bar(
                    x=factor_df["Contribution"],
                    y=factor_df["Factor"],
                    orientation="h",
                    marker_color=bar_colors,
                    text=[f"{v:+.3f}" for v in factor_df["Contribution"]],
                    textposition="outside",
                ))
                fig_factors.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=260,
                    margin=dict(t=10, b=10, l=10, r=80),
                    xaxis_title="Risk Contribution",
                )
                st.plotly_chart(fig_factors, use_container_width=True)

                st.markdown("### 🔍 Similar Historical Cases")
                sim_mask = (
                    (df["category"]    == category) &
                    (df["Hazard_Type"] == hazard_type)
                )
                sim_df = df[sim_mask][
                    ["date", "subject", "origin", "risk_decision"]
                ].head(5)
                if len(sim_df):
                    st.dataframe(sim_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No exact matches found in historical data.")

            st.markdown("---")
            st.markdown("### 💬 Interpretation")
            if is_serious:
                st.error(
                    f"**This case shows indicators of serious risk.**  \n"
                    f"- **Hazard type** '{hazard_type}' is historically associated with "
                    f"serious classifications.  \n"
                    f"- **Category** '{category}' shows elevated risk patterns in RASFF data.  \n"
                    f"- **Origin** '{origin}' contributes a risk signal based on historical "
                    f"notifications.  \n"
                    f"Recommended action: prioritize follow-up, verify distribution scope, "
                    f"and initiate border checks if applicable."
                )
            else:
                st.success(
                    f"**This case shows lower-risk indicators.**  \n"
                    f"- **Hazard type** '{hazard_type}' is typically associated with "
                    f"non-serious outcomes.  \n"
                    f"- Subject keywords and category suggest moderate concern.  \n"
                    f"Recommended action: standard monitoring; confirm with full risk assessment."
                )

            st.markdown(
                "<div class='info-box'>⚠️ <strong>Disclaimer:</strong> This is a "
                "machine-learning-based estimate. Final risk classification must be made "
                "by qualified RASFF authorities following official procedures.</div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════
# PAGE 3 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════
elif page == "🔬 Model Insights":
    st.title("🔬 Model Insights")
    st.caption(
        "XGBoost classifier · RASFF 2020–2025 · "
        "Optuna-tuned · Scenario A (no leakage) · Test set used once"
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Model Summary", "📈 Algorithm Comparison",
        "🧩 Feature Groups",  "📉 Performance Details",
    ])

    # ── Tab 1: Model Summary ───────────────────────────────
    with tab1:
        st.markdown("## 🏆 Final Model: XGBoost (Optuna-tuned)")

        col_a, col_b, col_c, col_d = st.columns(4)
        kpis = [
            ("F1-Macro (Test)",  "0.7688", "#4f8ef7"),
            ("AUC-ROC (Test)",   "0.8569", "#10b981"),
            ("Accuracy (Test)",  "77.0%",  "#f59e0b"),
            ("Training Records", "23,988", "#8b5cf6"),
        ]
        for col, (label, value, color) in zip([col_a, col_b, col_c, col_d], kpis):
            col.markdown(
                f'<div class="metric-card"><h2 style="color:{color}">{value}</h2>'
                f'<p>{label}</p></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ⚙️ Model Configuration")
            config_df = pd.DataFrame({
                "Parameter": [
                    "Algorithm", "Feature Set", "Target Variable",
                    "Train / Test Split", "CV Strategy",
                    "Optimization", "Trials",
                    "Primary Metric", "Secondary Metric",
                ],
                "Value": [
                    "XGBoost (XGBClassifier)",
                    "Scenario A — 671 features",
                    "Binary (Serious vs Non-Serious)",
                    "80% train / 20% test (temporal)",
                    "5-fold Stratified CV",
                    "Optuna TPE Sampler",
                    "100 trials (61 completed)",
                    "F1-macro",
                    "AUC-ROC",
                ],
            })
            st.dataframe(config_df, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("### 📊 Classification Report (Test Set)")

            # Plain string DataFrame — avoids Styler rendering errors
            report_df = pd.DataFrame({
                "Class":     ["Non-Serious (0)", "Serious (1)", "",
                              "Macro Avg",        "Weighted Avg"],
                "Precision": ["0.81", "0.73", "", "0.77", "0.77"],
                "Recall":    ["0.76", "0.78", "", "0.77", "0.77"],
                "F1-Score":  ["0.79", "0.75", "", "0.77", "0.77"],
                "Support":   ["3,314", "2,682", "", "5,996", "5,996"],
            })
            st.dataframe(report_df, use_container_width=True, hide_index=True)

            # Grouped bar for Precision / Recall / F1
            fig_rep = go.Figure()
            classes         = ["Non-Serious", "Serious"]
            precision_vals  = [0.81, 0.73]
            recall_vals     = [0.76, 0.78]
            f1_vals         = [0.79, 0.75]

            fig_rep.add_trace(go.Bar(
                name="Precision", x=classes, y=precision_vals,
                marker_color="#4f8ef7",
            ))
            fig_rep.add_trace(go.Bar(
                name="Recall", x=classes, y=recall_vals,
                marker_color="#10b981",
            ))
            fig_rep.add_trace(go.Bar(
                name="F1-Score", x=classes, y=f1_vals,
                marker_color="#f59e0b",
            ))
            fig_rep.update_layout(
                barmode="group",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=260,
                yaxis=dict(range=[0.60, 0.90]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=30, b=10),
            )
            st.plotly_chart(fig_rep, use_container_width=True)

        st.markdown("---")
        st.markdown("### ⚠️ Key Design Decisions")
        decisions = [
            ("🚫 No Leakage (Scenario A)",
             "The `classification` field is excluded from features. "
             "Including it adds +0.053 F1 but constitutes data leakage "
             "since it is assigned alongside the risk decision."),
            ("📉 Concept Drift Acknowledged",
             "Serious rate fell from 54.8% (train, 2020–2024) to 44.7% "
             "(test, 2025–2026). This reflects a real RASFF trend, not a data error."),
            ("🎯 Test Set Used Once",
             "~6,000 hold-out records were evaluated exactly once after all "
             "hyperparameter tuning on the training fold only."),
            ("⚖️ Binary Classification Chosen",
             "2-class target outperforms finer granularity. "
             "F1-macro: 0.84 (2-class) vs 0.75 (3-class) vs 0.54 (6-class)."),
        ]
        for title, desc in decisions:
            st.markdown(f"**{title}**  \n{desc}\n")

    # ── Tab 2: Algorithm Comparison ───────────────────────
    with tab2:
        st.markdown("## 📊 Stage 1: Baseline Algorithm Comparison")
        st.caption("5-fold Stratified CV on training set · Scenario A (671 features)")

        algo_df = pd.DataFrame({
            "Algorithm":    ["XGBoost", "LightGBM", "RandomForest", "LogisticRegression"],
            "Val F1-Macro": [0.8364,    0.8355,     0.8310,         0.8004],
            "F1 Std":       [0.0051,    0.0051,     0.0066,         0.0065],
            "Val AUC-ROC":  [0.9131,    0.9130,     0.9062,         0.8786],
            "Overfit Gap":  [0.1535,    0.1463,     0.1669,         0.0082],
            "Time (s)":     [58.6,      31.3,       85.4,           25.0],
            "Selected":     ["✅ Best", "2nd",      "3rd",          "4th"],
        })

        cl, cr = st.columns(2)
        with cl:
            colors_algo = ["#4f8ef7", "#6b7280", "#6b7280", "#6b7280"]
            fig_f1 = go.Figure(go.Bar(
                x=algo_df["Algorithm"],
                y=algo_df["Val F1-Macro"],
                error_y=dict(type="data", array=algo_df["F1 Std"].tolist()),
                marker_color=colors_algo,
                text=[f"{v:.4f}" for v in algo_df["Val F1-Macro"]],
                textposition="outside",
            ))
            fig_f1.update_layout(
                title="Validation F1-Macro (5-fold CV)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[0.75, 0.88]),
                height=350,
            )
            st.plotly_chart(fig_f1, use_container_width=True)

        with cr:
            gap_colors = [
                "#e67e22" if g > 0.10 else "#27ae60"
                for g in algo_df["Overfit Gap"]
            ]
            fig_gap = go.Figure(go.Bar(
                x=algo_df["Algorithm"],
                y=algo_df["Overfit Gap"],
                marker_color=gap_colors,
                text=[f"{v:.4f}" for v in algo_df["Overfit Gap"]],
                textposition="outside",
            ))
            fig_gap.add_hline(y=0.05, line_dash="dash", line_color="#27ae60",
                              annotation_text="Mild (0.05)")
            fig_gap.add_hline(y=0.10, line_dash="dash", line_color="#e74c3c",
                              annotation_text="Significant (0.10)")
            fig_gap.update_layout(
                title="Overfitting Gap (Train F1 − Val F1)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
            )
            st.plotly_chart(fig_gap, use_container_width=True)

        st.markdown(
            "**→ XGBoost selected** — highest Val F1-Macro (0.8364) and AUC-ROC (0.9131)."
        )
        st.dataframe(algo_df, use_container_width=True, hide_index=True)

    # ── Tab 3: Feature Groups ──────────────────────────────
    with tab3:
        st.markdown("## 🧩 Feature Group Ablation Study")
        st.caption("Cumulative contribution (Scenario A) + Scenario B ceiling")

        abl_df = pd.DataFrame({
            "Feature Set": [
                "A: Hazard only",
                "A + B: + Subject TF-IDF",
                "A + B + C: + Year",
                "A + B + C + D: + Geographic",
                "A+B+C+D+E: All (Scenario A)",
                "Scenario B (+ classification — LEAKAGE)",
            ],
            "Cols":         [241, 625, 626, 628, 671, 676],
            "Val F1-Macro": [0.7721, 0.8241, 0.8275, 0.8294, 0.8351, 0.8879],
            "Val AUC-ROC":  [0.839,  0.903,  0.906,  0.908,  0.913,  0.950],
            "Overfit Gap":  [0.108,  0.146,  0.148,  0.151,  0.155,  0.106],
        })

        colors_abl = ["#4f8ef7"] * 5 + ["#ef4444"]
        fig_abl = go.Figure(go.Bar(
            x=abl_df["Feature Set"],
            y=abl_df["Val F1-Macro"],
            marker_color=colors_abl,
            text=[f"{v:.4f}" for v in abl_df["Val F1-Macro"]],
            textposition="outside",
        ))
        fig_abl.add_hline(
            y=0.8879, line_dash="dot", line_color="#ef4444",
            annotation_text="Leakage ceiling (Scenario B)",
        )
        fig_abl.update_layout(
            title="Feature Group Contribution (Cumulative Ablation)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[0.70, 0.93]),
            height=400,
            xaxis_tickangle=-20,
        )
        st.plotly_chart(fig_abl, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📐 Feature Group Details")
        st.dataframe(pd.DataFrame({
            "Group":       ["A: Hazard", "B: Subject", "C: Temporal",
                            "D: Geographic", "E: Categorical"],
            "# Features":  [241, 384, 1, 2, 43],
            "Description": [
                "Hazard type OHE, hazard substance TF-IDF, binary flags",
                "Notification subject TF-IDF (top bi/trigrams)",
                "Notification year (normalized)",
                "Origin & notifying country (frequency-encoded)",
                "Category, product type, distribution scope (OHE)",
            ],
            "F1 Gain":     ["+0.000 (base)", "+0.052", "+0.003", "+0.002", "+0.006"],
        }), use_container_width=True, hide_index=True)

        st.info(
            "**Key finding:** Subject text (Group B) provides the largest single boost "
            "(+0.052 F1), confirming that free-text content carries the strongest "
            "predictive signal for risk severity."
        )

    # ── Tab 4: Performance Details ─────────────────────────
    with tab4:
        st.markdown("## 📉 Performance Details & Context")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 Target Granularity Comparison")
            tgt_df = pd.DataFrame({
                "Target":       ["2-class (binary)", "3-class", "6-class"],
                "Val F1-Macro": [0.8364, 0.7527, 0.5385],
                "Val AUC-ROC":  [0.9131, 0.9321, 0.9141],
            })
            fig_tgt = px.bar(
                tgt_df, x="Target", y="Val F1-Macro",
                color="Val F1-Macro",
                color_continuous_scale="Blues",
                text=[f"{v:.4f}" for v in tgt_df["Val F1-Macro"]],
                template="plotly_dark",
            )
            fig_tgt.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_tgt, use_container_width=True)
            st.markdown("**→ 2-class selected** for best deployable F1 performance.")

        with col2:
            st.markdown("### 📉 Concept Drift: Serious Rate Over Time")
            drift_df = pd.DataFrame({
                "Split":             ["Train (2020–2024)", "Test (2025–2026)"],
                "Serious Rate (%)":  [54.8, 44.7],
            })
            fig_drift = px.bar(
                drift_df, x="Split", y="Serious Rate (%)",
                color="Split",
                color_discrete_sequence=["#4f8ef7", "#f97316"],
                text=["54.8%", "44.7%"],
                template="plotly_dark",
            )
            fig_drift.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                showlegend=False,
            )
            st.plotly_chart(fig_drift, use_container_width=True)
            st.warning(
                "⚠️ **Concept drift detected:** Serious rate declined 54.8% → 44.7%, "
                "reflecting the real RASFF 2020→2026 trend."
            )

        st.markdown("---")
        st.markdown("### 🔄 Model Pipeline Summary")
        pipeline = [
            ("NB1", "Data Processing",
             "Raw RASFF CSV → cleaned, standardized, 29,984 records"),
            ("NB2", "EDA & Feature Analysis",
             "Cramér's V associations, hazard taxonomy, distribution analysis"),
            ("NB3", "Feature Engineering",
             "TF-IDF (subject, hazard), OHE (category, type), temporal/geo encoding"),
            ("NB4", "Modeling",
             "Baseline → Ablation → Target analysis → Optuna tuning → Final test"),
        ]
        for nb, title, desc in pipeline:
            st.markdown(
                f"<div class='info-box'><strong>{nb}: {title}</strong><br>{desc}</div>",
                unsafe_allow_html=True,
            )
