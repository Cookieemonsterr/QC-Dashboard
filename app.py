import re
import html
import time
from collections import defaultdict

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_PATH = "istep_data.csv"
REPORT_PATH = "Evaluation Report Istep - Report.csv"

st.set_page_config(page_title="QC Scores Dashboard", page_icon="✅", layout="wide")

CSS = """
<style>
.block-container{ padding-top: 1.8rem !important; }

html[data-theme="dark"]{
  --card-bg: rgba(255,255,255,.05);
  --card-border: rgba(255,255,255,.10);
  --muted: rgba(255,255,255,.68);
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

.qc-title{ font-size:2.0rem; font-weight:900; letter-spacing:.2px; margin:.0rem 0 .25rem 0; }

.kpi{
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: var(--shadow);
}
.kpi .label{ font-size:.95rem; color: var(--muted); margin-bottom:4px; }
.kpi .value{ font-size:2.05rem; font-weight:900; line-height:1.1; }

hr{ border:none; border-top:1px solid var(--hr); margin:.9rem 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ============================================================
# FAST CSV CACHE (no UI change)
# ============================================================
@st.cache_data(show_spinner=False, ttl=1800)
def read_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

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

def safe_col(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def to_pct(series: pd.Series) -> pd.Series:
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

# Ticket type from Subject (Overall)
def map_ticket_type_from_subject(subject: str) -> str:
    s = ("" if subject is None else str(subject)).strip().lower()
    if s.startswith("outlet catalogue update request"):
        return "Update Tickets"
    if s.startswith("new brand setup"):
        return "Build Tickets"
    if s.startswith("new outlet setup for existing brand"):
        return "Existing Tickets"
    return "Other"

def collapse_overall(df: pd.DataFrame) -> pd.DataFrame:
    ref = safe_col(df, ["Reference ID", "Ticket ID"])
    if not ref:
        return df

    subject = safe_col(df, ["Subject"])
    c_city = safe_col(df, ["Catalogue City", "City"])
    c_market = safe_col(df, ["Catalogue Market", "Market"])
    c_agent = safe_col(df, ["Catalogue Name", "Catalogue Agent Name"])
    s_agent = safe_col(df, ["Studio Name", "Studio Agent Name"])

    agg = {}
    if subject: agg[subject] = first_non_empty
    if c_city: agg[c_city] = first_non_empty
    if c_market: agg[c_market] = first_non_empty
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
# BUCKETS
# ============================================================
UPDATE_FORMS = {
    "Price Update",
    "Update scorecard",
    "Location Update",
    "Tags Update",
    "Operational Hours Update",
}
STUDIO_FORMS = {"Studio scorecard", "Images Update"}

BUILD_MAIN_FORMS = {"New Points/ Builds"}
CATALOGUE_SCORECARD_FORM = "Catalogue scorecard"

EXISTING_MAIN_FORMS = {
    "Existing Company and Brand - New Outlet",
    "Existing Company and Brand - New Outlet for Shops",
    "Existing Company - New Brand and New outlet",
}

def bucket_for_row(ticket_type: str, form_name: str):
    tt = (ticket_type or "").strip()
    fn = (form_name or "").strip()

    if fn in STUDIO_FORMS:
        return "Studio"

    if (tt == "Build Tickets") and (fn in BUILD_MAIN_FORMS):
        return "Total"
    if (tt == "Update Tickets") and (fn == "Update scorecard"):
        return "Total"
    if (tt == "Existing Tickets") and (fn in EXISTING_MAIN_FORMS):
        return "Total"

    if (tt == "Update Tickets") and (fn in UPDATE_FORMS):
        return "Catalogue"
    if tt in {"Build Tickets", "Existing Tickets"} and (
        fn in BUILD_MAIN_FORMS or fn == CATALOGUE_SCORECARD_FORM or fn in EXISTING_MAIN_FORMS
    ):
        return "Catalogue"

    return None

# ============================================================
# DATA
# ============================================================
st.sidebar.markdown("## Data")
if st.sidebar.button("Refresh data now"):
    st.cache_data.clear()

@st.cache_data(show_spinner=False, ttl=600)
def load_data():
    # ✅ cached reads (faster)
    overall = read_csv_cached(BASE_PATH).copy()
    overall.columns = make_unique([normalize_header(c) for c in overall.columns])
    overall = collapse_overall(overall)

    subj_col = safe_col(overall, ["Subject"])
    overall["ticket_type_raw"] = overall[subj_col].astype(str) if subj_col else ""
    overall["ticket_type"] = overall["ticket_type_raw"].apply(map_ticket_type_from_subject)

    overall["__ref__"] = overall["Reference ID"].astype(str).str.strip()

    city_col = safe_col(overall, ["Catalogue City", "City"])
    market_col = safe_col(overall, ["Catalogue Market", "Market"])
    c_agent_col = safe_col(overall, ["Catalogue Name", "Catalogue Agent Name"])
    s_agent_col = safe_col(overall, ["Studio Name", "Studio Agent Name"])

    overall["city"] = overall[city_col] if city_col else pd.NA
    overall["market"] = overall[market_col] if market_col else pd.NA
    overall["catalog_agent"] = overall[c_agent_col] if c_agent_col else pd.NA
    overall["studio_agent"] = overall[s_agent_col] if s_agent_col else pd.NA

    # Evaluation Report (SOURCE OF TRUTH FOR SCORES)
    ev = read_csv_cached(REPORT_PATH).copy()
    ev.columns = make_unique([normalize_header(c) for c in ev.columns])

    ref_col = safe_col(ev, ["Reference ID"])
    form_col = safe_col(ev, ["Form Name"])
    score_col = safe_col(ev, ["Score"])
    sb_col = safe_col(ev, ["Sent Back To Catalog"])
    dt_col = safe_col(ev, ["Date & Time", "Date Time", "Date&Time"])

    if not ref_col or not form_col or not score_col:
        raise RuntimeError("Evaluation report must contain: Reference ID, Form Name, Score")

    ev["__ref__"] = ev[ref_col].astype(str).str.strip()
    ev["form_name"] = ev[form_col].astype(str).str.strip()
    ev["score_pct"] = to_pct(ev[score_col])
    ev["sent_back_catalog"] = pd.to_numeric(ev[sb_col], errors="coerce").fillna(0) if sb_col else 0
    ev["dt"] = pd.to_datetime(ev[dt_col], errors="coerce") if dt_col else pd.NaT
    ev["date"] = ev["dt"].dt.date

    ref_to_type = dict(zip(overall["__ref__"], overall["ticket_type"]))
    ev["ticket_type"] = ev["__ref__"].map(ref_to_type).fillna("Other")

    ev["bucket"] = ev.apply(lambda r: bucket_for_row(r["ticket_type"], r["form_name"]), axis=1)

    ev["is_studio"] = ev["form_name"].isin(STUDIO_FORMS)
    ev["is_non_studio"] = ~ev["is_studio"]

    rows = ev.merge(overall, on="__ref__", how="left", suffixes=("", "_overall"))

    # ✅ make filters faster (no change to UI/numbers)
    for c in ["ticket_type", "form_name", "city", "market", "catalog_agent", "studio_agent"]:
        if c in rows.columns:
            rows[c] = rows[c].astype("category")

    # Ticket-level table scores
    per_cat = (
        rows[rows["is_non_studio"]]
        .groupby("__ref__", as_index=False)["score_pct"]
        .mean()
        .rename(columns={"score_pct": "catalog_qc"})
    )
    per_stu = (
        rows[rows["is_studio"]]
        .groupby("__ref__", as_index=False)["score_pct"]
        .mean()
        .rename(columns={"score_pct": "studio_qc"})
    )
    per_tot = (
        rows[rows["bucket"] == "Total"]
        .groupby("__ref__", as_index=False)["score_pct"]
        .mean()
        .rename(columns={"score_pct": "total_qc"})
    )

    sent_back_cat_events = (
        rows.groupby("__ref__", as_index=False)["sent_back_catalog"]
        .sum()
        .rename(columns={"sent_back_catalog": "sent_back_to_catalog_events"})
    )

    tickets = (
        overall.merge(sent_back_cat_events, on="__ref__", how="left")
        .merge(per_cat, on="__ref__", how="left")
        .merge(per_stu, on="__ref__", how="left")
        .merge(per_tot, on="__ref__", how="left")
    )

    tickets["ticket_id"] = tickets["Reference ID"].astype(str)
    tickets["sent_back_to_catalog_events"] = pd.to_numeric(
        tickets["sent_back_to_catalog_events"], errors="coerce"
    ).fillna(0).astype(int)

    # ✅ PRECOMPUTE SENT-BACK FLAGS ONCE (numbers identical, faster later)
    sb_cat_flag_all = (
        rows[rows["is_non_studio"]]
        .groupby("__ref__")["sent_back_catalog"]
        .max()
        .fillna(0)
        .gt(0)
    )
    sb_stu_flag_all = (
        rows[rows["is_studio"]]
        .groupby("__ref__")["sent_back_catalog"]
        .max()
        .fillna(0)
        .gt(0)
    )

    # ✅ PRECOMPUTE DAILY PER TICKET (keeps same math, faster later)
    daily_per_ticket = (
        rows.dropna(subset=["dt"])
        .assign(date_only=lambda d: d["dt"].dt.date)
        .groupby(["__ref__", "date_only"], as_index=False)["score_pct"]
        .mean()
    )

    return rows, tickets, sb_cat_flag_all, sb_stu_flag_all, daily_per_ticket

t0 = time.time()
with st.spinner("Loading…"):
    rows_df, tickets_df, sb_cat_flag_all, sb_stu_flag_all, daily_per_ticket = load_data()
st.caption(f"Loaded in {time.time()-t0:.2f}s")

# ============================================================
# FILTERS
# ============================================================
st.sidebar.markdown("## Filters")

ticket_view = st.sidebar.radio(
    "Ticket Type View", ["All", "Build Tickets", "Update Tickets", "Existing Tickets"], index=0
)

min_dt = rows_df["dt"].min()
max_dt = rows_df["dt"].max()
default_range = (min_dt.date() if pd.notna(min_dt) else None, max_dt.date() if pd.notna(max_dt) else None)
date_range = st.sidebar.date_input("Date range", value=default_range)

cities = sorted([c for c in rows_df["city"].dropna().astype(str).unique() if c and c.lower() != "nan"])
markets = sorted([m for m in rows_df["market"].dropna().astype(str).unique() if m and m.lower() != "nan"])
catalog_agents = sorted([a for a in rows_df["catalog_agent"].dropna().astype(str).unique() if a and a.lower() != "nan"])
studio_agents = sorted([a for a in rows_df["studio_agent"].dropna().astype(str).unique() if a and a.lower() != "nan"])

sel_city = st.sidebar.multiselect("City", cities, default=[])
sel_market = st.sidebar.multiselect("Market", markets, default=[])
sel_catalog_agent = st.sidebar.multiselect("Catalogue Agent", catalog_agents, default=[])
sel_studio_agent = st.sidebar.multiselect("Studio Agent", studio_agents, default=[])

rf = rows_df.copy()

if ticket_view != "All":
    rf = rf[rf["ticket_type"] == ticket_view]

if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    if start:
        rf = rf[rf["date"] >= start]
    if end:
        rf = rf[rf["date"] <= end]

if sel_city:
    rf = rf[rf["city"].isin(sel_city)]
if sel_market:
    rf = rf[rf["market"].isin(sel_market)]
if sel_catalog_agent:
    rf = rf[rf["catalog_agent"].isin(sel_catalog_agent)]
if sel_studio_agent:
    rf = rf[rf["studio_agent"].isin(sel_studio_agent)]

allowed_refs = set(rf["__ref__"].dropna().astype(str))
tf = tickets_df[tickets_df["__ref__"].astype(str).isin(allowed_refs)].copy()

# ============================================================
# HEADER
# ============================================================
top_left, top_right = st.columns([0.72, 0.28], vertical_alignment="center")
with top_left:
    st.markdown('<div class="qc-title">QC Scores Dashboard</div>', unsafe_allow_html=True)

with top_right:
    st.download_button(
        "⬇️ Download filtered CSV",
        data=rf.to_csv(index=False).encode("utf-8"),
        file_name="qc_filtered_rows.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("<hr/>", unsafe_allow_html=True)

# ============================================================
# KPIs
# ============================================================
k1, k2, k3, k4, k5 = st.columns(5)

cat_rows = rf[rf["is_non_studio"]].copy()
stu_rows = rf[rf["is_studio"]].copy()
tot_rows = rf[rf["bucket"] == "Total"].copy()

cat_avg = mean_pct(cat_rows["score_pct"])
stu_avg = mean_pct(stu_rows["score_pct"])
tot_avg = mean_pct(tot_rows["score_pct"])

# ✅ faster: just count within filtered refs using precomputed flags
tf_ref = tf["__ref__"].astype(str)
sent_back_catalog_tickets = int(tf_ref.map(sb_cat_flag_all).fillna(False).sum())
sent_back_studio_tickets  = int(tf_ref.map(sb_stu_flag_all).fillna(False).sum())

with k1: kpi_card("Catalog QC", "—" if cat_avg is None else f"{cat_avg:.2f}%")
with k2: kpi_card("Studio QC", "—" if stu_avg is None else f"{stu_avg:.2f}%")
with k3: kpi_card("Total QC (Ticket)", "—" if tot_avg is None else f"{tot_avg:.2f}%")
with k4: kpi_card("Sent Back → Catalog (Tickets)", f"{int(sent_back_catalog_tickets):,}")
with k5: kpi_card("Sent Back → Studio (Tickets)", f"{int(sent_back_studio_tickets):,}")

st.markdown("<br/>", unsafe_allow_html=True)

# ============================================================
# INSIGHTS
# ============================================================
st.markdown("## Insights")
c1, c2, c3 = st.columns([0.34, 0.33, 0.33])

with c1:
    st.markdown("### Ticket type split")
    tt = tf["ticket_type"].fillna("Other").value_counts().reset_index()
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
        fig = px.histogram(x=s, nbins=20)
        fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Score (%)",
            yaxis_title="Rows",
        )
        st.plotly_chart(fig, use_container_width=True)

with c3:
    st.markdown("### Tickets by City")
    city_series = tf["city"].dropna().astype(str).str.strip()
    city_series = city_series[
        (city_series != "")
        & (city_series.str.lower() != "nan")
        & (city_series.str.lower() != "unknown")
    ]
    city_counts = city_series.value_counts().reset_index()
    city_counts.columns = ["city", "ticket_count"]

    if city_counts.empty:
        st.info("No city data available.")
    else:
        fig = px.bar(city_counts.head(20), x="city", y="ticket_count")
        fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="",
            yaxis_title="Tickets",
        )
        st.plotly_chart(fig, use_container_width=True)

# ✅ faster: map flags (no groupby here)
tf["Sent Back → Catalog"] = tf_ref.map(sb_cat_flag_all).fillna(False).map(lambda x: "Yes" if x else "No")
tf["Sent Back → Studio"]  = tf_ref.map(sb_stu_flag_all).fillna(False).map(lambda x: "Yes" if x else "No")

# ============================================================
# TICKETS + SCORE CHANGE
# ============================================================
left, right = st.columns([0.62, 0.38])

with left:
    st.markdown("## Tickets")
    show_cols = [
        "ticket_id",
        "ticket_type",
        "ticket_type_raw",
        "city",
        "market",
        "catalog_agent",
        "studio_agent",
        "catalog_qc",
        "studio_qc",
        "total_qc",
        "Sent Back → Catalog",
        "Sent Back → Studio",
        "sent_back_to_catalog_events",
    ]
    show_cols = [c for c in show_cols if c in tf.columns]
    st.dataframe(
        tf[show_cols].sort_values("ticket_id", ascending=False),
        use_container_width=True,
        height=520,
        column_config={
            "catalog_qc": st.column_config.NumberColumn("Catalog QC", format="%.2f%%"),
            "studio_qc": st.column_config.NumberColumn("Studio QC", format="%.2f%%"),
            "total_qc": st.column_config.NumberColumn("Total QC", format="%.2f%%"),
        },
    )

with right:
    st.markdown("## Score Change")

    # ✅ fast: use precomputed daily_per_ticket, then aggregate only the filtered refs
    dpt = daily_per_ticket[daily_per_ticket["__ref__"].astype(str).isin(allowed_refs)].copy()
    if dpt.empty:
        st.info("No evaluation datetime values available for this filter.")
    else:
        daily = (
            dpt.groupby("date_only", as_index=False)["score_pct"]
               .mean()
               .sort_values("date_only")
        )

        fig = px.line(daily, x="date_only", y="score_pct", markers=True)
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ============================================================
# SUMMARIES
# ============================================================
st.markdown("## Catalogue QC — Avg Score by Market & Catalogue Agent")
cat_summary = (
    cat_rows.dropna(subset=["score_pct"])
    .groupby(["market", "catalog_agent"], as_index=False)
    .agg(
        avg_score=("score_pct", "mean"),
        tickets=("__ref__", "nunique"),
    )
    .sort_values(["avg_score", "tickets"], ascending=[False, False])
)

st.dataframe(
    cat_summary,
    use_container_width=True,
    height=420,
    column_config={"avg_score": st.column_config.NumberColumn("Avg Score", format="%.2f%%")},
)

st.markdown("## Studio QC — Avg Score by Market & Studio Agent")
stu_summary = (
    stu_rows.dropna(subset=["score_pct"])
    .groupby(["market", "studio_agent"], as_index=False)
    .agg(
        avg_score=("score_pct", "mean"),
        tickets=("__ref__", "nunique"),
    )
    .sort_values(["avg_score", "tickets"], ascending=[False, False])
)

st.dataframe(
    stu_summary,
    use_container_width=True,
    height=420,
    column_config={"avg_score": st.column_config.NumberColumn("Avg Score", format="%.2f%%")},
)
