
import re
import html
import time
from collections import defaultdict
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_PATH = "istep_data.csv"
REPORT_PATH = "Evaluation Report Istep - Report.csv"
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
    # if values look like 0-1, treat as fraction
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
# TICKET TYPE FROM SUBJECT (same logic)
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
# COLLAPSE OVERALL: 1 row per Reference ID (for ticket fields)
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

    for col in df.columns:
        if col == ref or col in agg:
            continue
        agg[col] = first_non_empty

    out = df.groupby(ref, as_index=False).agg(agg)
    out.rename(columns={ref: "Reference ID"}, inplace=True)
    return out

# ============================================================
# FORM RULES (exactly like your process)
# ============================================================
UPDATE_FORMS = {
    "Price Update",
    "Update scorecard",
    "Location Update",
    "Tags Update",
    "Operational Hours Update",
}
STUDIO_FORMS = {"Studio scorecard", "Images Update"}

BUILD_MAIN_FORMS = {"New Points/ Builds"}  # as in your sheet
CATALOGUE_SCORECARD_FORM = "Catalogue scorecard"

EXISTING_MAIN_FORMS = {
    "Existing Company and Brand - New Outlet",
    "Existing Company and Brand - New Outlet for Shops",
    "Existing Company - New Brand and New outlet",
}

def form_bucket(ticket_type: str, form_name: str) -> str | None:
    """Return which score bucket this row belongs to."""
    tt = (ticket_type or "").strip()
    fn = (form_name or "").strip()

    # Studio bucket (independent of ticket type)
    if fn in STUDIO_FORMS:
        return "Studio QC"

    # Total QC (Ticket Score) bucket
    if (tt == "Build Tickets") and (fn in BUILD_MAIN_FORMS):
        return "Total QC"
    if (tt == "Update Tickets") and (fn == "Update scorecard"):
        return "Total QC"
    if (tt == "Existing Tickets") and (fn in EXISTING_MAIN_FORMS):
        return "Total QC"

    # Catalogue bucket
    if (tt == "Update Tickets") and (fn in UPDATE_FORMS):
        return "Catalogue QC"
    if (tt == "Build Tickets") and (fn in BUILD_MAIN_FORMS or fn == CATALOGUE_SCORECARD_FORM):
        return "Catalogue QC"
    if (tt == "Existing Tickets") and (fn in EXISTING_MAIN_FORMS or fn == CATALOGUE_SCORECARD_FORM):
        return "Catalogue QC"

    return None

# ============================================================
# LOAD
# ============================================================
st.sidebar.markdown("## Data")
if st.sidebar.button("🔄 Refresh"):
    st.cache_data.clear()

@st.cache_data(show_spinner=False, ttl=600)
def load_all():
    # ---------
    # Overall (ticket fields)
    # ---------
    base_raw = pd.read_csv(BASE_PATH, low_memory=False)
    base_raw.columns = [normalize_header(c) for c in base_raw.columns]
    base_raw.columns = make_unique(base_raw.columns)
    base = collapse_base(base_raw)

    subj_col_base = safe_col(base, ["Subject", "Ticket Type"])
    base["ticket_type_raw"] = base[subj_col_base].astype(str) if subj_col_base else ""
    base["ticket_type"] = base["ticket_type_raw"].apply(map_ticket_type_from_subject)

    base["__ref__"] = base["Reference ID"].astype(str).str.strip()

    # helper fields for filtering
    city_col = safe_col(base, ["Catalogue City", "City"])
    market_col = safe_col(base, ["Catalogue Market", "Market"])
    c_agent_col = safe_col(base, ["Catalogue Name", "Catalogue Agent Name"])
    s_agent_col = safe_col(base, ["Studio Name", "Studio Agent Name"])

    base["ticket_id"] = base["Reference ID"].astype(str)
    base["city"] = base[city_col] if city_col else pd.NA
    base["market"] = base[market_col] if market_col else pd.NA
    base["catalog_agent"] = base[c_agent_col] if c_agent_col else pd.NA
    base["studio_agent"] = base[s_agent_col] if s_agent_col else pd.NA

    base.loc[base["city"].astype(str).str.strip().str.lower() == "unknown", "city"] = pd.NA

    ref_to_type = dict(zip(base["__ref__"], base["ticket_type"]))

    # ---------
    # Evaluation Report (ROW-LEVEL scores)
    # ---------
    report = pd.read_csv(REPORT_PATH, encoding="utf-8", low_memory=False)
    report.columns = [normalize_header(c) for c in report.columns]
    report.columns = make_unique(report.columns)

    ref_col = safe_col(report, ["Reference ID"])
    form_col = safe_col(report, ["Form Name"])
    score_col = safe_col(report, ["Score"])
    sb_col = safe_col(report, ["Sent Back To Catalog"])
    name_col = safe_col(report, ["Name", "Agent Name"])
    uid_col = safe_col(report, ["User Id", "User ID", "Agent User Id"])
    dt_col = safe_col(report, ["Date & Time", "Date&Time", "Date Time"])

    if not ref_col or not form_col or not score_col:
        raise RuntimeError("Evaluation Report must include: Reference ID, Form Name, Score")

    r = report.copy()
    r["__ref__"] = r[ref_col].astype(str).str.strip()
    r["form_name"] = r[form_col].astype(str).str.strip()
    r["score_pct"] = _to_pct(r[score_col])
    r["sent_back"] = pd.to_numeric(r[sb_col], errors="coerce").fillna(0) if sb_col else 0

    r["agent_name"] = (r[name_col].astype(str).str.strip() if name_col else "")
    r["agent_uid"] = (r[uid_col].astype(str).str.strip() if uid_col else "")

    # ticket type mapping (from OVERALL subject logic)
    r["ticket_type"] = r["__ref__"].map(ref_to_type).fillna("Other")

    # De-dupe within same ticket/card/agent (your intention)
    r = r.drop_duplicates(subset=["__ref__", "form_name", "agent_name", "agent_uid", "score_pct", "sent_back"])

    # Bucket each row into Total/Catalog/Studio
    r["score_bucket"] = r.apply(lambda x: form_bucket(x["ticket_type"], x["form_name"]), axis=1)

    # Parse datetime from Evaluation Report (this is what you filter in Google Sheet)
    r["dt"] = pd.to_datetime(r[dt_col], errors="coerce") if dt_col else pd.NaT
    r["date"] = r["dt"].dt.date

    # Join OVERALL fields onto every evaluation row
    rows = r.merge(base, on="__ref__", how="left", suffixes=("", "_base"))

    # Per-ticket summaries (for the Tickets table)
    per_ticket_catalog = (
        rows[rows["score_bucket"] == "Catalogue QC"]
        .groupby("__ref__", as_index=False)["score_pct"].mean()
        .rename(columns={"score_pct": "catalog_score_pct"})
    )
    per_ticket_studio = (
        rows[rows["score_bucket"] == "Studio QC"]
        .groupby("__ref__", as_index=False)["score_pct"].mean()
        .rename(columns={"score_pct": "studio_score_pct"})
    )
    per_ticket_total = (
        rows[rows["score_bucket"] == "Total QC"]
        .groupby("__ref__", as_index=False)["score_pct"]
        .apply(lambda s: s.dropna().iloc[0] if len(s.dropna()) else pd.NA)
        .rename(columns={"score_pct": "ticket_score_pct"})
    )
    per_ticket_sent_back = (
        rows.groupby("__ref__", as_index=False)["sent_back"].sum().rename(columns={"sent_back": "sent_back_total"})
    )

    tickets = (
        base.merge(per_ticket_sent_back, on="__ref__", how="left")
            .merge(per_ticket_catalog, on="__ref__", how="left")
            .merge(per_ticket_studio, on="__ref__", how="left")
            .merge(per_ticket_total, on="__ref__", how="left")
    )
    tickets["sent_back_total"] = pd.to_numeric(tickets["sent_back_total"], errors="coerce").fillna(0).astype(int)

    return rows, tickets

t0 = time.time()
with st.spinner("Loading…"):
    rows_df, tickets_df = load_all()
st.caption(f"Loaded in {time.time()-t0:.2f}s")

# ============================================================
# FILTERS
# ============================================================
st.sidebar.markdown("## Filters")

view_mode = st.sidebar.radio(
    "Ticket Type View",
    ["All", "Build Tickets", "Update Tickets", "Existing Tickets"],
    index=0
)

# IMPORTANT: date range based on evaluation report dt
min_dt = rows_df["dt"].min()
max_dt = rows_df["dt"].max()
default_range = ((min_dt.date() if pd.notna(min_dt) else None), (max_dt.date() if pd.notna(max_dt) else None))
date_range = st.sidebar.date_input("Date range (Evaluation Date)", value=default_range)

cities = sorted([c for c in rows_df["city"].dropna().astype(str).unique().tolist() if c.strip() and c.lower() != "nan"])
sel_cities = st.sidebar.multiselect("City", cities, default=[])

markets = sorted([m for m in rows_df["market"].dropna().astype(str).unique().tolist() if m.strip() and m.lower() != "nan"])
sel_markets = st.sidebar.multiselect("Market", markets, default=[])

ticket_id_search = st.sidebar.text_input("Ticket ID contains", value="")

score_type = st.sidebar.selectbox(
    "Score Type (avg + charts)",
    ["Total QC (Ticket Score)", "Catalogue QC", "Studio QC"],
    index=0
)
bucket = {
    "Total QC (Ticket Score)": "Total QC",
    "Catalogue QC": "Catalogue QC",
    "Studio QC": "Studio QC",
}[score_type]

# APPLY FILTERS (ROW-LEVEL for scoring)
rf = rows_df.copy()

# Score bucket filter (this is your "Form Name filter" logic)
rf = rf[rf["score_bucket"] == bucket]

if view_mode != "All":
    rf = rf[rf["ticket_type"] == view_mode]

if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    if start:
        rf = rf[rf["date"] >= start]
    if end:
        rf = rf[rf["date"] <= end]

if sel_cities:
    rf = rf[rf["city"].isin(sel_cities)]
if sel_markets:
    rf = rf[rf["market"].isin(sel_markets)]
if ticket_id_search.strip():
    rf = rf[rf["ticket_id"].astype(str).str.contains(ticket_id_search.strip(), case=False, na=False)]

# Tickets table should follow SAME high-level filters (but ticket-level)
tf = tickets_df.copy()
if view_mode != "All":
    tf = tf[tf["ticket_type"] == view_mode]
if sel_cities:
    tf = tf[tf["city"].isin(sel_cities)]
if sel_markets:
    tf = tf[tf["market"].isin(sel_markets)]
if ticket_id_search.strip():
    tf = tf[tf["ticket_id"].astype(str).str.contains(ticket_id_search.strip(), case=False, na=False)]

# For tickets date filter: use evaluation dates from rf (keep tickets that appear in filtered rows)
if isinstance(date_range, tuple) and len(date_range) == 2:
    allowed_refs = set(rf["__ref__"].dropna().astype(str))
    tf = tf[tf["__ref__"].astype(str).isin(allowed_refs)]

# ============================================================
# HEADER
# ============================================================
left, right = st.columns([0.78, 0.22], vertical_alignment="bottom")
with left:
    st.markdown('<div class="qc-title">QC Scores Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="qc-sub">Average QC Scores</div>',
        unsafe_allow_html=True
    )

with right:
    st.download_button(
        "⬇️ Download filtered score rows",
        data=rf.to_csv(index=False).encode("utf-8"),
        file_name="qc_score_rows_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("<hr/>", unsafe_allow_html=True)

# ============================================================
# KPI ROW (ROW-LEVEL AVG — MATCHES YOUR SHEET)
# ============================================================
k1, k2, k3, k4, k5, k6 = st.columns(6)

avg_score = mean_pct(rf["score_pct"])
tickets_count = rf["__ref__"].nunique()
rows_count = len(rf)
sent_back_total = int(pd.to_numeric(rf["sent_back"], errors="coerce").fillna(0).sum())
low_perf = int((pd.to_numeric(rf["score_pct"], errors="coerce") < 90).fillna(False).sum())

with k1: kpi_card("Average score", "—" if avg_score is None else f"{avg_score:.2f}%")
with k2: kpi_card("Sent Back", f"{sent_back_total:,}")
with k3: kpi_card("Mode", bucket)

st.markdown("<br/>", unsafe_allow_html=True)

# ============================================================
# TABLES + CHARTS
# ============================================================
a, b = st.columns([0.62, 0.38])

with a:
    st.markdown("### Tickets")
    cols = [
        "ticket_id", "ticket_type",
        "catalog_agent", "catalog_score_pct",
        "studio_agent", "studio_score_pct",
        "ticket_score_pct",
        "city", "market",
        "sent_back_total",
    ]
    cols = [c for c in cols if c in tf.columns]
    st.dataframe(
        tf[cols].sort_values("ticket_id", ascending=False),
        use_container_width=True,
        height=460,
        column_config={
            "catalog_score_pct": st.column_config.NumberColumn("Catalogue QC", format="%.2f%%"),
            "studio_score_pct": st.column_config.NumberColumn("Studio QC", format="%.2f%%"),
            "ticket_score_pct": st.column_config.NumberColumn("Ticket Score (Total QC)", format="%.2f%%"),
        },
    )

with b:
    st.markdown("### Score Change")
    t = rf.dropna(subset=["dt"]).copy()
    if t.empty:
        st.info("No evaluation datetime values available for this filter.")
    else:
        t["date_only"] = t["dt"].dt.date
        daily = (
            t.groupby("date_only", as_index=False)["score_pct"]
            .mean(numeric_only=True)
            .sort_values("date_only")
        )
        fig = px.line(daily, x="date_only", y="score_pct", markers=True)
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

c1, c2, c3 = st.columns([0.34, 0.33, 0.33])

with c1:
    st.markdown("### Ticket type split")
    tt = rf["ticket_type"].fillna("Other").value_counts().reset_index()
    tt.columns = ["ticket_type", "count"]
    fig = px.pie(tt, names="ticket_type", values="count", hole=0.55)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("### Score distribution")
    s = pd.to_numeric(rf["score_pct"], errors="coerce").dropna()
    if s.empty:
        st.info("No score values available.")
    else:
        fig = px.histogram(s, nbins=20)
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Score (%)", yaxis_title="Rows")
        st.plotly_chart(fig, use_container_width=True)

with c3:
    st.markdown("### Score rows by City")
    city_counts = rf["city"].dropna().astype(str).value_counts().reset_index()
    city_counts.columns = ["city", "row_count"]
    fig = px.bar(city_counts.head(20), x="city", y="row_count")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

