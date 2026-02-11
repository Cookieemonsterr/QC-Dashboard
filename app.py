
import re
import html
import time
from collections import defaultdict
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_PATH = "istep_data.csv"
REPORT_PATH = "Evaluation Report Istep - Report.csv"

# ============================================================
# PAGE
# ============================================================
st.set_page_config(page_title="QC Scores Dashboard (iStep)", page_icon="✅", layout="wide")

CSS = """
<style>
.block-container{ padding-top: 2.1rem !important; }

html[data-theme="dark"]{
  --card-bg: rgba(255,255,255,.06);
  --card-border: rgba(255,255,255,.12);
  --muted: rgba(255,255,255,.65);
  --shadow: 0 10px 25px rgba(0,0,0,.28);
  --hr: rgba(255,255,255,.10);
}
html[data-theme="light"]{
  --card-bg: rgba(0,0,0,.03);
  --card-border: rgba(0,0,0,.10);
  --muted: rgba(0,0,0,.55);
  --shadow: 0 10px 25px rgba(0,0,0,.10);
  --hr: rgba(0,0,0,.08);
}

.qc-title{ font-size:1.75rem; font-weight:900; letter-spacing:.2px; margin:.1rem 0 .15rem 0; }
.qc-sub{ color: var(--muted); margin-top:-.15rem; }

.kpi{
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: var(--shadow);
}
.kpi .label{ font-size:.9rem; color: var(--muted); margin-bottom:2px; }
.kpi .value{ font-size:2.05rem; font-weight:900; }

hr{ border:none; border-top:1px solid var(--hr); margin:.8rem 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================
def normalize_header(h: str) -> str:
    h = "" if h is None else str(h)
    h = html.unescape(h)
    h = re.sub(r"\s+", " ", h).strip()
    return h

def make_unique(cols):
    seen = defaultdict(int)
    out = []
    for c in cols:
        seen[c] += 1
        out.append(c if seen[c] == 1 else f"{c}__{seen[c]}")
    return out

def safe_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _to_pct(series: pd.Series) -> pd.Series:
    if series is None:
        return series
    s = series.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return s
    if (s.dropna() <= 1.5).mean() > 0.7:
        s = s * 100
    return s

def mean_pct(x: pd.Series):
    x = pd.to_numeric(x, errors="coerce")
    return None if x.dropna().empty else float(x.mean())

def kpi_card(label: str, value: str):
    st.markdown(
        f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )

def first_non_empty(series: pd.Series):
    s = series.dropna().astype(str).str.strip()
    s = s[(s != "") & (s.str.lower() != "nan")]
    return s.iloc[0] if len(s) else pd.NA

def unique_join(series: pd.Series, sep=" | "):
    s = series.dropna().astype(str).str.strip()
    s = s[(s != "") & (s.str.lower() != "nan")]
    uniq = list(dict.fromkeys(s.tolist()))
    return sep.join(uniq) if uniq else pd.NA

# ============================================================
# TICKET TYPE FROM SUBJECT (your original logic)
# ============================================================
def map_ticket_type_from_subject(subject: str) -> str:
    s = ("" if subject is None else str(subject)).strip().lower()
    if s.startswith("outlet catalogue update request"):
        return "Update Tickets"
    if s.startswith("new brand setup"):
        return "Build Tickets"
    if s.startswith("new outlet setup for existing brand"):
        return "Existing Tickets"
    return "Other"

# ============================================================
# BASE COLLAPSE: 1 row per Reference ID
# ============================================================
def collapse_base(df: pd.DataFrame) -> pd.DataFrame:
    ref = safe_col(df, ["Reference ID", "Ticket ID"])
    if not ref:
        return df

    subject = safe_col(df, ["Subject", "Ticket Type"])
    c_city = safe_col(df, ["Catalogue City", "City"])
    c_market = safe_col(df, ["Catalogue Market", "Market"])
    c_dt = safe_col(df, ["Catalogue Date & Time", "Ticket Creation Time"])
    s_dt = safe_col(df, ["Studio Date & Time", "Ticket Creation Time__2"])

    c_agent = safe_col(df, ["Catalogue Name", "Catalogue Agent Name"])
    s_agent = safe_col(df, ["Studio Name", "Studio Agent Name"])

    agg = {}
    if subject: agg[subject] = first_non_empty
    if c_city: agg[c_city] = first_non_empty
    if c_market: agg[c_market] = first_non_empty
    if c_dt: agg[c_dt] = first_non_empty
    if s_dt: agg[s_dt] = first_non_empty
    if c_agent: agg[c_agent] = unique_join
    if s_agent: agg[s_agent] = first_non_empty

    # keep other columns too (first non-empty)
    for col in df.columns:
        if col == ref or col in agg:
            continue
        agg[col] = first_non_empty

    out = df.groupby(ref, as_index=False).agg(agg)
    out.rename(columns={ref: "Reference ID"}, inplace=True)
    return out

# ============================================================
# REPORT CARD RULES (your message)
# ============================================================
UPDATE_FORMS = {
    "Price Update",
    "Update scorecard",
    "Location Update",
    "Tags Update",
    "Operational Hours Update",
}
STUDIO_FORMS = {"Studio scorecard", "Images Update"}

BUILD_MAIN_FORM = "New Points/ Builds"
EXISTING_MAIN_FORM = "Existing Company and Brand - New Outlet"
CATALOGUE_SCORECARD_FORM = "Catalogue scorecard"

def build_scores_from_report(report: pd.DataFrame, ref_to_type: dict[str, str]) -> pd.DataFrame:
    ref = safe_col(report, ["Reference ID"])
    form = safe_col(report, ["Form Name"])
    score = safe_col(report, ["Score"])
    sb_cat = safe_col(report, ["Sent Back To Catalog"])
    name = safe_col(report, ["Name", "Agent Name"])
    user_id = safe_col(report, ["User Id", "User ID", "Agent User Id"])

    if not ref or not form or not score:
        raise RuntimeError("Report CSV must include: Reference ID, Form Name, Score")

    r = report.copy()
    r["__ref__"] = r[ref].astype(str).str.strip()
    r["__form__"] = r[form].astype(str).str.strip()
    r["__score__"] = _to_pct(r[score])
    r["__sent_back__"] = pd.to_numeric(r[sb_cat], errors="coerce").fillna(0) if sb_cat else 0

    r["__agent__"] = (r[name].astype(str).str.strip() if name else "")
    r["__uid__"] = (r[user_id].astype(str).str.strip() if user_id else "")

    r["__ticket_type__"] = r["__ref__"].map(ref_to_type).fillna("Other")

    # ✅ De-dupe duplicates within same ticket/card/agent
    # This matches: "take everything unless duplicated in the same ticket"
    r = r.drop_duplicates(subset=["__ref__", "__form__", "__agent__", "__uid__", "__score__", "__sent_back__"])

    # --------------------------
    # Catalogue QC rows
    # --------------------------
    is_catalog_update = (r["__ticket_type__"] == "Update Tickets") & (r["__form__"].isin(UPDATE_FORMS))

    is_catalog_build = (r["__ticket_type__"] == "Build Tickets") & (
        (r["__form__"] == BUILD_MAIN_FORM) | (r["__form__"] == CATALOGUE_SCORECARD_FORM)
    )

    is_catalog_existing = (r["__ticket_type__"] == "Existing Tickets") & (
        (r["__form__"] == EXISTING_MAIN_FORM) | (r["__form__"] == CATALOGUE_SCORECARD_FORM)
    )

    is_catalog = is_catalog_update | is_catalog_build | is_catalog_existing

    # --------------------------
    # Studio QC rows
    # --------------------------
    is_studio = r["__form__"].isin(STUDIO_FORMS)

    # --------------------------
    # Ticket Score (Total QC) rows
    # --------------------------
    is_ticket_build = (r["__ticket_type__"] == "Build Tickets") & (r["__form__"] == BUILD_MAIN_FORM)
    is_ticket_update = (r["__ticket_type__"] == "Update Tickets") & (r["__form__"] == "Update scorecard")
    is_ticket_existing = (r["__ticket_type__"] == "Existing Tickets") & (r["__form__"] == EXISTING_MAIN_FORM)
    is_ticket = is_ticket_build | is_ticket_update | is_ticket_existing

    cat = (
        r.loc[is_catalog]
        .groupby("__ref__", as_index=False)["__score__"]
        .mean()
        .rename(columns={"__score__": "catalog_score_pct"})
    )

    stu = (
        r.loc[is_studio]
        .groupby("__ref__", as_index=False)["__score__"]
        .mean()
        .rename(columns={"__score__": "studio_score_pct"})
    )

    ticket = (
        r.loc[is_ticket]
        .sort_values(["__ref__"])
        .groupby("__ref__", as_index=False)["__score__"]
        .apply(lambda s: s.dropna().iloc[0] if len(s.dropna()) else pd.NA)
        .rename(columns={"__score__": "ticket_score_pct"})
    )

    sent_back = (
        r.groupby("__ref__", as_index=False)["__sent_back__"]
        .sum()
        .rename(columns={"__sent_back__": "sent_back_total"})
    )

    out = sent_back.merge(cat, on="__ref__", how="left").merge(stu, on="__ref__", how="left").merge(ticket, on="__ref__", how="left")
    out["sent_back_total"] = pd.to_numeric(out["sent_back_total"], errors="coerce").fillna(0).astype(int)
    return out

# ============================================================
# LOAD
# ============================================================
st.sidebar.markdown("## Data")
if st.sidebar.button("🔄 Refresh"):
    st.cache_data.clear()

@st.cache_data(show_spinner=False, ttl=600)
def load_all():
    # Base
    base_raw = pd.read_excel(BASE_PATH, sheet_name=0)
    base_raw.columns = [normalize_header(c) for c in base_raw.columns]
    base_raw.columns = make_unique(base_raw.columns)
    base = collapse_base(base_raw)

    subj_col = safe_col(base, ["Subject", "Ticket Type"])
    base["ticket_type_raw"] = base[subj_col].astype(str) if subj_col else ""
    base["ticket_type"] = base["ticket_type_raw"].apply(map_ticket_type_from_subject)

    ref_to_type = dict(zip(base["Reference ID"].astype(str).str.strip(), base["ticket_type"]))

    # Report
    report = pd.read_csv(REPORT_PATH, encoding="utf-8", low_memory=False)
    report.columns = [normalize_header(c) for c in report.columns]
    report.columns = make_unique(report.columns)

    scores = build_scores_from_report(report, ref_to_type)

    # Merge
    base["__ref__"] = base["Reference ID"].astype(str).str.strip()
    merged = base.merge(scores, on="__ref__", how="left")

    # Filters fields
    city_col = safe_col(merged, ["Catalogue City", "City"])
    market_col = safe_col(merged, ["Catalogue Market", "Market"])
    c_agent_col = safe_col(merged, ["Catalogue Name", "Catalogue Agent Name"])
    s_agent_col = safe_col(merged, ["Studio Name", "Studio Agent Name"])

    merged["ticket_id"] = merged["Reference ID"].astype(str)
    merged["city"] = merged[city_col] if city_col else pd.NA
    merged["market"] = merged[market_col] if market_col else pd.NA
    merged["catalog_agent"] = merged[c_agent_col] if c_agent_col else pd.NA
    merged["studio_agent"] = merged[s_agent_col] if s_agent_col else pd.NA

    # Datetime from base
    col_cat_dt = safe_col(merged, ["Catalogue Date & Time", "Ticket Creation Time"])
    col_stu_dt = safe_col(merged, ["Studio Date & Time", "Ticket Creation Time__2"])
    merged["dt"] = pd.to_datetime(merged[col_cat_dt], errors="coerce") if col_cat_dt else pd.NaT
    if col_stu_dt:
        merged["dt"] = merged["dt"].fillna(pd.to_datetime(merged[col_stu_dt], errors="coerce"))

    merged["date"] = merged["dt"].dt.date

    # Total QC = Ticket Score
    merged["total_qc_pct"] = merged["ticket_score_pct"]

    # remove "Unknown" city values (so they don't appear in charts)
    merged.loc[merged["city"].astype(str).str.strip().str.lower() == "unknown", "city"] = pd.NA

    return merged

t0 = time.time()
with st.spinner("Loading…"):
    df = load_all()
st.caption(f"Loaded in {time.time()-t0:.2f}s")

# ============================================================
# FILTERS
# ============================================================
st.sidebar.markdown("## Filters")
view_mode = st.sidebar.radio("Ticket Type View", ["All", "Build Tickets", "Update Tickets", "Existing Tickets"], index=0)

min_dt = df["dt"].min()
max_dt = df["dt"].max()
default_range = ((min_dt.date() if pd.notna(min_dt) else None), (max_dt.date() if pd.notna(max_dt) else None))
date_range = st.sidebar.date_input("Date range", value=default_range)

cities = sorted([c for c in df["city"].dropna().astype(str).unique().tolist() if c.strip() and c.lower() != "nan"])
sel_cities = st.sidebar.multiselect("City", cities, default=[])

markets = sorted([m for m in df["market"].dropna().astype(str).unique().tolist() if m.strip() and m.lower() != "nan"])
sel_markets = st.sidebar.multiselect("Market", markets, default=[])

ticket_id_search = st.sidebar.text_input("Ticket ID contains", value="")

score_type = st.sidebar.selectbox("Score Type (charts)", ["Total QC (Ticket Score)", "Catalogue QC", "Studio QC"], index=0)
score_col = {
    "Total QC (Ticket Score)": "ticket_score_pct",
    "Catalogue QC": "catalog_score_pct",
    "Studio QC": "studio_score_pct",
}[score_type]

# APPLY FILTERS
f = df.copy()
if view_mode != "All":
    f = f[f["ticket_type"] == view_mode]

if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    if start: f = f[f["date"] >= start]
    if end: f = f[f["date"] <= end]

if sel_cities:
    f = f[f["city"].isin(sel_cities)]
if sel_markets:
    f = f[f["market"].isin(sel_markets)]
if ticket_id_search.strip():
    f = f[f["ticket_id"].str.contains(ticket_id_search.strip(), case=False, na=False)]

# ============================================================
# HEADER (fixed, no broken markdown)
# ============================================================
left, right = st.columns([0.78, 0.22], vertical_alignment="bottom")
with left:
    st.markdown('<div class="qc-title">QC Scores Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="qc-sub">Scores from Report CSV (cards) • Ticket type from Subject • De-duped per ticket/card/agent</div>', unsafe_allow_html=True)

with right:
    st.download_button(
        "⬇️ Download filtered CSV",
        data=f.to_csv(index=False).encode("utf-8"),
        file_name="qc_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("<hr/>", unsafe_allow_html=True)

# ============================================================
# KPI ROW
# ============================================================
k1, k2, k3, k4, k5, k6 = st.columns(6)

catalog_avg = mean_pct(f["catalog_score_pct"])
studio_avg = mean_pct(f["studio_score_pct"])
total_avg = mean_pct(f["ticket_score_pct"])
sent_back_total = int(pd.to_numeric(f["sent_back_total"], errors="coerce").fillna(0).sum())
low_perf = int((pd.to_numeric(f["ticket_score_pct"], errors="coerce") < 90).fillna(False).sum())

with k1: kpi_card("Catalogue QC", "—" if catalog_avg is None else f"{catalog_avg:.2f}%")
with k2: kpi_card("Studio QC", "—" if studio_avg is None else f"{studio_avg:.2f}%")
with k3: kpi_card("Total QC (Ticket Score)", "—" if total_avg is None else f"{total_avg:.2f}%")
with k4: kpi_card("Low performers (<90%)", f"{low_perf:,}")
with k5: kpi_card("Sent Back (total)", f"{sent_back_total:,}")
with k6: kpi_card("Tickets (filtered)", f"{len(f):,}")

st.markdown("<br/>", unsafe_allow_html=True)

# ============================================================
# TABLE + SCORE CHANGE
# ============================================================
a, b = st.columns([0.62, 0.38])

with a:
    st.markdown("### Tickets")
    cols = [
        "ticket_id", "ticket_type", "ticket_type_raw",
        "catalog_agent", "catalog_score_pct",
        "studio_agent", "studio_score_pct",
        "ticket_score_pct",
        "city", "market", "dt",
        "sent_back_total",
    ]
    cols = [c for c in cols if c in f.columns]

    st.dataframe(
        f[cols].sort_values("dt", ascending=False),
        use_container_width=True,
        height=460,
        column_config={
            "catalog_score_pct": st.column_config.NumberColumn("Catalogue QC", format="%.2f%%"),
            "studio_score_pct": st.column_config.NumberColumn("Studio QC", format="%.2f%%"),
            "ticket_score_pct": st.column_config.NumberColumn("Ticket Score (Total QC)", format="%.2f%%"),
            "dt": st.column_config.DatetimeColumn("Date & Time"),
        },
    )

with b:
    st.markdown("### Score Change")
    t = f.dropna(subset=["dt"]).copy()
    if t.empty:
        st.info("No datetime values available for this filter.")
    else:
        t["date_only"] = t["dt"].dt.date
        daily = (
            t.groupby("date_only", as_index=False)[["catalog_score_pct", "studio_score_pct", "ticket_score_pct"]]
            .mean(numeric_only=True)
            .sort_values("date_only")
        )
        fig = px.line(daily, x="date_only", y=score_col, markers=True)
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ============================================================
# SMALL CHARTS
# ============================================================
c1, c2, c3 = st.columns([0.34, 0.33, 0.33])

with c1:
    st.markdown("### Ticket type split")
    tt = f["ticket_type"].fillna("Other").value_counts().reset_index()
    tt.columns = ["ticket_type", "count"]
    fig = px.pie(tt, names="ticket_type", values="count", hole=0.55)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("### Score distribution")
    s = pd.to_numeric(f[score_col], errors="coerce").dropna()
    if s.empty:
        st.info("No score values available.")
    else:
        fig = px.histogram(s, nbins=20)
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Score (%)", yaxis_title="Tickets")
        st.plotly_chart(fig, use_container_width=True)

with c3:
    st.markdown("### Tickets by City (no Unknown)")
    city_counts = (
        f["city"]
        .dropna()
        .astype(str)
        .value_counts()
        .reset_index()
    )
    city_counts.columns = ["city", "ticket_count"]
    fig = px.bar(city_counts.head(20), x="city", y="ticket_count")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
