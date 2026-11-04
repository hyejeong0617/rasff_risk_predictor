"""Portfolio explorer. Uses frozen artifacts; never trains or recalculates metrics."""
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from rasff_core import DATA, ARTIFACTS, load_prepared, predict_bundle

st.set_page_config(page_title="RASFF | Retrospective Label Classification", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
/* Presentation only: the app uses the light Streamlit theme configured at repo root. */
.stApp, [data-testid="stAppViewContainer"] {background:#F5F8FB;color:#20344A;}
.block-container {max-width:1480px!important;width:100%!important;padding:2rem clamp(1.1rem,3.5vw,3.5rem) 3.5rem!important;}
h1,h2,h3 {letter-spacing:-.035em;color:#172F49;}
h1 {font-size:clamp(2rem,3vw,2.65rem)!important;font-weight:760!important;}
h2,h3 {font-weight:700!important;}
p,li {line-height:1.58;}
.kicker {font-size:.78rem;font-weight:750;letter-spacing:.12em;text-transform:uppercase;color:#087B89;margin:.15rem 0 .65rem;}
.intro {font-size:1.06rem;color:#516579;max-width:940px;line-height:1.65;margin:.2rem 0 .7rem;}
.hero {background:linear-gradient(115deg,#FFFFFF 0%,#EAF5F7 100%);border:1px solid #DBE8EF;border-radius:16px;padding:1.5rem 1.75rem;margin:.4rem 0 1.35rem;}
.hero h1 {margin:.1rem 0 .4rem;}
.hero .scope {font-size:.86rem;color:#506579;margin-top:.8rem;}
[data-testid="stMetric"] {background:#FFFFFF;border:1px solid #DDE7EF;border-radius:12px;padding:.8rem 1rem;box-shadow:0 2px 8px rgba(24,51,74,.035);}
[data-testid="stMetricValue"] {color:#143E55;font-size:1.8rem!important;font-weight:750;}
[data-testid="stMetricLabel"] {color:#536779;}
[data-testid="stPlotlyChart"] {background:#FFFFFF;border:1px solid #E2EAF1;border-radius:12px;padding:.55rem;}
[data-testid="stExpander"] {border:1px solid #DCE7EE;background:#FFFFFF;border-radius:12px;}
[data-testid="stForm"] {border:1px solid #DAE6EE!important;background:#FFFFFF;border-radius:12px;padding:1rem;}
[data-testid="stDataFrame"] {border-radius:9px;overflow:hidden;}
[data-testid="stTabs"] button {font-weight:650;}
[data-testid="stTabs"] [aria-selected="true"] {color:#087B89!important;}
.stButton button[kind="primary"],.stFormSubmitButton button[kind="primary"] {border-radius:9px;font-weight:700;}
.section-note {color:#587083;font-size:.88rem;margin:.25rem 0 1rem;}
@media(max-width:700px){
  .block-container {padding:1.15rem .85rem 2rem!important;}
  .hero {padding:1.1rem 1rem;}
  h1 {font-size:1.9rem!important;}
  [data-testid="stMetricValue"] {font-size:1.4rem!important;}
}
</style>""", unsafe_allow_html=True)

LABELS = {"category": "Product category", "type": "Product type",
          "notifying_country": "Notifying country", "classification": "Notification classification",
          "risk_decision": "Recorded risk decision", "origin": "Origin"}
COLORS = ["#167D8D", "#344A72", "#9B7134", "#6C658B", "#537967", "#945B64"]
REPO = "https://github.com/hyejeong0617/rasff_risk_predictor"


@st.cache_data(show_spinner=False)
def data(stamp):
    return load_prepared()


@st.cache_resource(show_spinner=False)
def load_model(stamp):
    return joblib.load(ARTIFACTS / "final_bundle.joblib")


@st.cache_data(show_spinner=False)
def read_json(name, stamp):
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def artifact_json(name):
    return read_json(name, (ARTIFACTS / name).stat().st_mtime_ns)


def chart(fig, key):
    # Explicit white chart canvas keeps axes legible even in a user's dark theme.
    fig.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                      font=dict(color="#17344A", size=13), colorway=COLORS,
                      margin=dict(l=12, r=16, t=30, b=20),
                      legend=dict(orientation="h", y=-.25, title_text=""),
                      hoverlabel=dict(bgcolor="#FFFFFF", font_color="#17344A"))
    st.plotly_chart(fig, width="stretch", theme=None, key=key,
                    config={"displayModeBar": False, "scrollZoom": False})


def explore():
    st.subheader("Explore RASFF notifications")
    st.markdown('<div class="section-note">Explore recorded notifications by time, product, origin and official decision label.</div>', unsafe_allow_html=True)
    try:
        d = data((DATA / "prepared.csv").stat().st_mtime_ns)
    except (OSError, ValueError, KeyError):
        st.error("The prepared notification data are unavailable. Restore data/prepared.csv from the repository. Other tabs remain available.")
        return
    st.caption(f"Historical snapshot · {len(d):,} notifications · {d.date.min():%d %b %Y}–{d.date.max():%d %b %Y} · all supplied product types")
    with st.expander("Filter notifications", expanded=True):
        with st.form("filters"):
            left, right = st.columns(2)
            years = left.multiselect("Years", sorted(d.date.dt.year.unique()),
                                     default=sorted(d.date.dt.year.unique()), key="years")
            dates = right.date_input("Notification date range", value=(d.date.min().date(), d.date.max().date()),
                                     min_value=d.date.min().date(), max_value=d.date.max().date(), key="dates")
            selections = {}
            for i, col in enumerate(["type", "category", "notifying_country", "classification", "risk_decision"]):
                selections[col] = (left if i % 2 == 0 else right).multiselect(
                    LABELS[col], sorted(d[col].dropna().unique()), key="filter_" + col)
            origins = right.multiselect("Origin countries (any match)",
                sorted({x.strip() for s in d.origin for x in s.split(",") if x.strip()}), key="origins")
            query = st.text_input("Search subject or hazards", help="Literal text, case-insensitive; no regular expressions.", key="query")
            st.caption("Blank category/country filters include all values. Clearing Years selects no records. Apply filters to update the view.")
            st.form_submit_button("Apply filters", type="primary")
    if not isinstance(dates, (tuple, list)) or len(dates) != 2:
        st.info("Select both a start and an end date, then apply filters.")
        return
    f = d[d.date.dt.year.isin(years)]
    f = f[(f.date >= pd.Timestamp(dates[0])) & (f.date < pd.Timestamp(dates[1]) + pd.Timedelta(days=1))]
    for col, values in selections.items():
        if values:
            f = f[f[col].isin(values)]
    if origins:
        selected_origins = set(origins)
        f = f[f.origin.map(lambda s: bool(set(x.strip() for x in s.split(",")) & selected_origins))]
    if query:
        f = f[f.subject.str.contains(query, case=False, regex=False) | f.hazards.str.contains(query, case=False, regex=False)]
    a, b, c = st.columns(3)
    a.metric("Selected notifications", f"{len(f):,}")
    b.metric("Recorded serious share", f"{f.target.mean():.1%}" if len(f) else "—")
    c.metric("Missing original hazards", f"{int(f.hazards.eq('').sum()):,}")
    st.caption("These are counts of recorded notifications, not population risk, inspection rates or country safety.")
    if f.empty:
        st.info("No notifications match. Broaden the filters or clear the search and apply again.")
        return
    st.markdown("#### Notifications over time")
    monthly = f.assign(month=f.date.dt.to_period("M").dt.to_timestamp()).groupby(["month", "risk_decision"]).size().reset_index(name="notifications")
    fig = px.bar(monthly, x="month", y="notifications", color="risk_decision",
                 color_discrete_map=dict(zip(sorted(d.risk_decision.unique()), COLORS)),
                 labels={"month": "Notification month", "notifications": "Notifications", "risk_decision": "Recorded decision"})
    fig.update_layout(height=390, bargap=0.17)
    chart(fig, "monthly")
    st.markdown("#### Most frequently recorded values")
    dimension = st.selectbox("Group by", list(LABELS), format_func=LABELS.get, key="dimension")
    counts = f[dimension].replace("", "(missing)").value_counts().head(12).sort_values().rename_axis("value").reset_index(name="notifications")
    # Numeric axis avoids clipping long category names on narrow screens; full text stays in hover.
    counts["short_label"] = counts.value.map(lambda x: x if len(x) <= 38 else x[:35] + "…")
    counts["position"] = range(len(counts))
    fig = px.bar(counts, x="notifications", y="position", orientation="h", custom_data=["value"],
                 color_discrete_sequence=[COLORS[0]], labels={"notifications": "Notifications", "position": ""})
    fig.update_traces(hovertemplate="%{customdata[0]}<br>%{x:,} notifications<extra></extra>")
    fig.update_yaxes(tickvals=counts.position.tolist(), ticktext=counts.short_label.tolist())
    fig.update_layout(height=max(330, len(counts) * 37), margin=dict(l=15,r=25,t=25,b=40))
    chart(fig, "ranking")
    st.caption("Top 12 values within the filtered records. " + ("Origin values are recorded country combinations, not individual-country risk estimates." if dimension == "origin" else "Counts reflect notifications, not relative risk."))
    if st.checkbox("Show records and CSV download", key="show_records"):
        visible = ["reference", "date", "subject", "category", "type", "origin", "notifying_country", "hazards", "classification", "risk_decision"]
        st.caption(f"Preview: first {min(len(f), 500):,} of {len(f):,} selected records. CSV includes all selected records and original prepared columns.")
        st.dataframe(f[visible].head(500), width="stretch", hide_index=True)
        st.download_button("Download all selected records", f.to_csv(index=False).encode("utf-8-sig"),
                           "rasff_filtered.csv", "text/csv", key="download")


def predictor():
    st.subheader("Predict the recorded label")
    st.markdown('<div class="section-note">A demonstration of the frozen model, not a food-safety decision or forward prediction.</div>', unsafe_allow_html=True)
    st.write("Enter notification information to estimate whether its recorded risk decision would be **serious** rather than one of the other known labels.")
    try:
        bundle = load_model((ARTIFACTS / "final_bundle.joblib").stat().st_mtime_ns)
        options = artifact_json("input_options.json")
        expected = {"subject", "origin", "category", "type", "notifying_country", "hazards"}
        if set(bundle["columns"]) != expected:
            raise ValueError("Unexpected prediction contract")
    except Exception:
        st.error("The saved prediction bundle or input options could not be loaded. Restore artifacts/final_bundle.joblib and artifacts/input_options.json and use the pinned dependencies. No model is trained by this app.")
        return
    st.caption("Six inputs · frozen Logistic Regression model · Explorer filters do not change the predictor")
    with st.form("prediction"):
        row = {}
        row["subject"] = st.text_area("Notification subject", placeholder="e.g. Salmonella in sesame seeds from India",
            help="The original notification description. Required to submit; example text is illustrative.", key="pred_subject")
        row["hazards"] = st.text_area("Original recorded hazards", placeholder="e.g. salmonella",
            help="Use the original hazard text. Leave blank if missing; missing does not mean safe.", key="pred_hazards")
        left, right = st.columns(2)
        row["category"] = left.selectbox("Product category", [""] + options["category"],
            format_func=lambda x: x or "Not supplied", key="pred_category")
        row["type"] = right.selectbox("Product type", [""] + options["type"],
            format_func=lambda x: x or "Not supplied", key="pred_type")
        countries = sorted({x.strip() for s in options["origin"] for x in s.split(",") if x.strip()})
        row["origin"] = ",".join(left.multiselect("Origin countries", countries,
            help="Select all recorded origins, if known.", key="pred_origin"))
        row["notifying_country"] = right.selectbox("Notifying country", [""] + options["notifying_country"],
            format_func=lambda x: x or "Not supplied", help="The country that submitted the notification, which may differ from its origin.", key="pred_notifying")
        st.caption("Dropdown values come from development data. Public snapshot fields may include later updates; this is not a validated pre-decision forecast.")
        submitted = st.form_submit_button("Predict recorded label", type="primary")
    if submitted:
        st.session_state.pop("prediction_result", None)
        if not row["subject"].strip():
            st.warning("Enter a notification subject before predicting.")
        else:
            try:
                probability = float(predict_bundle(bundle, pd.DataFrame([row]))[0])
                st.session_state.prediction_result = (probability, row.copy(), bundle["threshold"])
            except Exception:
                st.error("Prediction could not be completed with the saved bundle. Check the pinned runtime dependencies.")
    if "prediction_result" in st.session_state:
        probability, submitted_row, threshold = st.session_state.prediction_result
        st.divider()
        st.caption("Result for the last submitted inputs. Submit again after editing the form.")
        st.metric("Estimated P(recorded serious)", f"{probability:.1%}")
        st.write("**Predicted recorded label:** " + ("serious" if probability >= threshold else "Other recorded labels — not safe"))
        st.caption(f"Fixed decision threshold: {threshold:.2f}. Historical calibration may not transfer to future records.")
        st.info("This is not the probability of food harm. A negative prediction does not mean safe: the other-label group includes undecided, potential risk and potentially serious records.")
        if any(not str(v).strip() for k, v in submitted_row.items() if k != "hazards"):
            st.warning("Some inputs were not supplied. Interpret this estimate with caution.")
        with st.expander("Review submitted inputs"):
            st.json(submitted_row)


def evaluation():
    st.subheader("Model & evaluation")
    st.markdown('<div class="section-note">Held-out retrospective evaluation from the saved report; no metrics are recomputed here.</div>', unsafe_allow_html=True)
    try:
        report = artifact_json("test_report.json")
        scores = report["calibrated"]
        st.write(f"**Logistic Regression · {report['candidate']['name']} · sigmoid calibration**")
        st.caption(f"Frozen final test · {report['test_rows']:,} notifications · {report['test_dates'][0][:10]} to {report['test_dates'][1][:10]}")
        for col, (key, label) in zip(st.columns(4), [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("roc_auc", "ROC-AUC"), ("serious_recall", "Serious recall")]):
            col.metric(label, f"{scores[key]:.3f}")
        st.caption("Read directly from the saved test report; scores are not recalculated in the app.")
        if "confusion_matrix" in report:
            st.markdown("#### Where the classifier was right and wrong")
            st.dataframe(pd.DataFrame(report["confusion_matrix"], index=["Actual other labels", "Actual serious"],
                columns=["Predicted other labels", "Predicted serious"]), width="stretch")
            st.caption("Rows = recorded labels; columns = model predictions. Other labels do not mean safe.")
        with st.expander("All saved evaluation metrics"):
            st.dataframe(pd.DataFrame({k: report[k] for k in ["calibrated", "uncalibrated", "prior_baseline"]}).rename(
                columns={"calibrated": "Final calibrated model", "uncalibrated": "Before calibration", "prior_baseline": "Prior baseline"}), width="stretch")
    except (OSError, ValueError, KeyError, TypeError):
        st.error("The frozen evaluation report is unavailable or incomplete. Restore artifacts/test_report.json; no substitute metrics are calculated.")
    st.markdown("#### How the temporal evaluation works")
    st.write("Forward cross-validation within training selects the candidate by mean macro F1. Validation is diagnostic, then joins training for the final pipeline fit. The next block fits sigmoid calibration; the final block supplies the frozen test results.")
    try:
        splits = artifact_json("split_manifest.json")
        table = pd.DataFrame([{"Block": name.title(), "Records": block["rows"], "First day": block["first"][:10], "Last day": block["last"][:10]} for name, block in splits.items()])
        st.dataframe(table, width="stretch", hide_index=True)
    except (OSError, ValueError, KeyError, TypeError):
        st.info("The saved split manifest is unavailable. Restore artifacts/split_manifest.json to view dates and counts.")
    st.markdown("#### Scientific boundaries")
    st.markdown("""- **Retrospective labels, not actual harm:** 1 = recorded serious; 0 = all five other known labels, including uncertain and potentially serious records.
- **Timing is unverified:** public fields may have been updated after the original decision. Temporal separation does not rule out related incidents across blocks.
- **Inherited provenance:** the clean snapshot retains earlier manual edits and exclusions. Notification counts lack a population or inspection denominator.
- **Limited transfer evidence:** no external or prospective validation; historical calibration does not guarantee future reliability.
- **Feature scope:** classification and generated hazard fields are excluded from the primary model. The classification ablation is a separate validation diagnostic.""")
    st.markdown(f"[Inspect the frozen report]({REPO}/blob/main/RASFF_Rebuild/artifacts/test_report.json) · [Read the evidence audit]({REPO}/blob/main/RASFF_Rebuild/audit/REPORT.md)")


st.markdown("""<div class="hero">
<div class="kicker">Scientific AI Builder / Food Safety Data Science</div>
<h1>RASFF Risk Predictor</h1>
<div class="intro"><strong>Retrospective label classification</strong> · Explore historical EU RASFF notifications, test a saved machine-learning model, and inspect its validation evidence.</div>
<div class="scope">Research portfolio demo · Recorded <strong>serious</strong> labels, not actual food-harm probabilities or regulatory decisions.</div>
</div>""", unsafe_allow_html=True)
explorer_tab, predictor_tab, evaluation_tab = st.tabs(["Explore RASFF Notifications", "Predict Recorded Label", "Model & Evaluation"])
with explorer_tab:
    explore()
with predictor_tab:
    predictor()
with evaluation_tab:
    evaluation()
st.divider()
st.caption("Historical evidence · Transparent methods · Bounded scientific interpretation")
st.markdown(f"[Source repository]({REPO})")
