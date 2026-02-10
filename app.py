
import io
import re
import html
from collections import defaultdict

import requests
import pandas as pd
import plotly.express as px
import streamlit as st

SHEET_ID = "1rQHlDgQC5mZQ00fPVz20h4KEFbmsosE2"

XLSX_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=xlsx"
CSV_URL  = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

# ============================================================
# PAGE
# ============================================================
st.set_page_config(page_title="QC Scores Dashboard (iStep)", page_icon="✅", layout="wide")

# ============================================================
# LIGHT/DARK SAFE UI
# ============================================================
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
.kpi .delta{ font-size:.9rem; color: var(--muted); }

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
    if " : " in h:
        h = h.split(" : ", 1)[1].strip()
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

# ✅ FIXED: handles "100%" properly
def _to_pct(series: pd.Series) -> pd.Series:
    if series is None:
        return series
    s = series.astype(str)
    s = s.str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
    s = pd.to_numeric(s, errors="coerce")

    if s.dropna().empty:
        return s

    # if values look like 0..1, convert to %
    if (s.dropna() <= 1.5).mean() > 0.7:
        s = s * 100
    return s

def mean_pct(x: pd.Series):
    x = pd.to_numeric(x, errors="coerce")
    return None if x.dropna().empty else float(x.mean())

def kpi_card(label: str, value: str, delta: str | None = None):
    d = f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
    if delta:
        d += f'<div class="delta">{delta}</div>'
    d += "</div>"
    st.markdown(d, unsafe_allow_html=True)

# ============================================================
# ✅ TICKET TYPE RULES
# ============================================================
def map_ticket_type(raw: str) -> str:
    s = ("" if raw is None else str(raw)).strip()
    low = s.lower()
    if low.startswith("outlet catalogue update request"):
        return "Update Tickets"
    if low.startswith("new brand setup"):
        return "Build Tickets"
    if low.startswith("new outlet setup for existing brand"):
        return "Existing Tickets"
    return "Other"

# ============================================================
# ✅ COLLAPSE TO 1 ROW PER TICKET (FIXED FOR YOUR REAL DATA)
# ============================================================
def first_non_empty(series: pd.Series):
    s = series.dropna().astype(str).str.strip()
    s = s[(s != "") & (s.str.lower() != "nan")]
    return s.iloc[0] if len(s) else pd.NA

def unique_join(series: pd.Series, sep=" | "):
    s = series.dropna().astype(str).str.strip()
    s = s[(s != "") & (s.str.lower() != "nan")]
    uniq = list(dict.fromkeys(s.tolist()))
    return sep.join(uniq) if uniq else pd.NA

def mean_numeric(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce")
    return pd.NA if s.dropna().empty else float(s.mean())

def max_numeric(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce")
    return pd.NA if s.dropna().empty else float(s.max())

def collapse_tickets(df: pd.DataFrame) -> pd.DataFrame:
    ref = safe_col(df, ["Reference ID", "Ticket ID"])
    if not ref:
        return df

    # Explicit columns (your sheet names)
    c_agent = safe_col(df, ["Catalogue Name", "Catalogue Agent Name"])
    s_agent = safe_col(df, ["Studio Name", "Studio Agent Name"])

    c_score = safe_col(df, ["Catalogue Score", "Catalogue Agent QC Score", "Catalogue Agent QC Score__2"])
    s_score = safe_col(df, ["Studio Score", "Studio Agent QC Score", "Studio Agent QC Score__2"])
    t_score = safe_col(df, ["Ticket Score"])

    c_sb = safe_col(df, ["Catalogue Sent Back To Catalog", "Sent back to catalog"])
    s_sb = safe_col(df, ["Studio Sent Back To Catalog", "Sent back to catalog__2"])

    agg = {}

    # Agents
    if c_agent: agg[c_agent] = unique_join
    if s_agent: agg[s_agent] = first_non_empty  # only one studio agent

    # Scores
    if c_score: agg[c_score] = mean_numeric      # many catalogue rows → average
    if s_score: agg[s_score] = first_non_empty   # usually only one row has it
    if t_score: agg[t_score] = first_non_empty   # ticket-level

    # ✅ Sent back is duplicated in catalogue rows → use MAX (NOT SUM)
    if c_sb: agg[c_sb] = max_numeric
    if s_sb: agg[s_sb] = max_numeric

    # Everything else: keep first real value so details don’t disappear
    for col in df.columns:
        if col == ref or col in agg:
            continue
        agg[col] = first_non_empty

    out = df.groupby(ref, as_index=False).agg(agg)
    out.rename(columns={ref: "Reference ID"}, inplace=True)
    return out

def _http_get(url: str) -> requests.Response:
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
    return requests.get(url, headers=headers, timeout=120, allow_redirects=True)

@st.cache_data(show_spinner=False, ttl=600)
def load_sheet_df() -> pd.DataFrame:
    r = _http_get(XLSX_URL)
    if r.status_code == 200 and len(r.content) > 1000:
        try:
            return pd.read_excel(io.BytesIO(r.content), sheet_name=0)
        except Exception:
            pass

    r2 = _http_get(CSV_URL)
    if r2.status_code == 200 and len(r2.content) > 10:
        return pd.read_csv(io.BytesIO(r2.content))

    raise RuntimeError(
        f"Could not download sheet. XLSX={r.status_code} CSV={r2.status_code}. "
        "Make sure sharing is 'Anyone with the link (Viewer)'."
    )

@st.cache_data(show_spinner=False, ttl=600)
def load_clean_df() -> pd.DataFrame:
    df = load_sheet_df()

    # normalize headers + keep duplicates
    df.columns = [normalize_header(c) for c in df.columns]
    df.columns = make_unique(df.columns)

    # collapse multi-row tickets
    df = collapse_tickets(df)

    # normalize again after collapse
    df.columns = [normalize_header(c) for c in df.columns]
    df.columns = make_unique(df.columns)

    col_ref = safe_col(df, ["Reference ID", "Ticket ID"])
    col_subject = safe_col(df, ["Subject", "Ticket Type"])

    col_ticket_score = safe_col(df, ["Ticket Score"])
    col_cat_score = safe_col(df, ["Catalogue Score", "Catalogue Agent QC Score"])
    col_stu_score = safe_col(df, ["Studio Score", "Studio Agent QC Score"])

    col_cat_agent = safe_col(df, ["Catalogue Name", "Catalogue Agent Name"])
    col_stu_agent = safe_col(df, ["Studio Name", "Studio Agent Name"])

    col_cat_sb = safe_col(df, ["Catalogue Sent Back To Catalog", "Sent back to catalog"])
    col_stu_sb = safe_col(df, ["Studio Sent Back To Catalog", "Sent back to catalog__2"])

    col_cat_dt = safe_col(df, ["Catalogue Date & Time", "Ticket Creation Time"])
    col_stu_dt = safe_col(df, ["Studio Date & Time", "Ticket Creation Time__2"])

    col_city = safe_col(df, ["Catalogue City", "Studio City", "City", "City__2"])
    col_market = safe_col(df, ["Catalogue Market", "Studio Market", "Market", "Market__2"])

    df["ticket_id"] = df[col_ref].astype(str) if col_ref else df.index.astype(str)
    df["ticket_type_raw"] = df[col_subject].astype(str) if col_subject else ""
    df["ticket_type"] = df["ticket_type_raw"].apply(map_ticket_type)

    # Scores (fixed parsing)
    df["ticket_score_pct"] = _to_pct(df[col_ticket_score]) if col_ticket_score else pd.NA
    df["catalog_score_pct"] = _to_pct(df[col_cat_score]) if col_cat_score else pd.NA
    df["studio_score_pct"] = _to_pct(df[col_stu_score]) if col_stu_score else pd.NA

    # Agents
    df["catalog_agent"] = df[col_cat_agent] if col_cat_agent else pd.NA
    df["studio_agent"] = df[col_stu_agent] if col_stu_agent else pd.NA

    # Sent back (now correct because collapse used MAX)
    df["catalog_sent_back"] = pd.to_numeric(df[col_cat_sb], errors="coerce").fillna(0) if col_cat_sb else 0
    df["studio_sent_back"]  = pd.to_numeric(df[col_stu_sb], errors="coerce").fillna(0) if col_stu_sb else 0

    # City/Market
    df["city"] = df[col_city] if col_city else pd.NA
    df["market"] = df[col_market] if col_market else pd.NA

    # Datetime
    cat_dt = pd.to_datetime(df[col_cat_dt], errors="coerce") if col_cat_dt else pd.NaT
    stu_dt = pd.to_datetime(df[col_stu_dt], errors="coerce") if col_stu_dt else pd.NaT
    df["dt"] = cat_dt.fillna(stu_dt)

    df["date"] = df["dt"].dt.date
    df["month"] = df["dt"].dt.to_period("M").astype(str)
    df["week"] = df["dt"].dt.isocalendar().week.astype("Int64")
    df["day"] = df["dt"].dt.day_name()

    # ✅ YOUR RULES:
    # Catalog QC = Catalogue Score
    # Studio QC  = Studio Score
    # Total QC   = Ticket Score
    df["total_qc_pct"] = df["ticket_score_pct"]

    return df

# ============================================================
# SIDEBAR + LOAD
# ============================================================
st.sidebar.markdown("## Data")
if st.sidebar.button("🔄 Refresh data now"):
    st.cache_data.clear()

try:
    df = load_clean_df()
except Exception as e:
    st.error("Data download failed.")
    st.code(str(e))
    st.stop()

# ============================================================
# FILTERS
# ============================================================
st.sidebar.markdown("## Filters")

view_mode = st.sidebar.radio(
    "Ticket Type View",
    ["All", "Build Tickets", "Update Tickets", "Existing Tickets"],
    index=0,
)

min_dt = df["dt"].min()
max_dt = df["dt"].max()
default_range = (
    (min_dt.date() if pd.notna(min_dt) else None),
    (max_dt.date() if pd.notna(max_dt) else None),
)
date_range = st.sidebar.date_input("Date range", value=default_range)

cities = sorted([c for c in df["city"].dropna().astype(str).unique().tolist() if c.strip() and c.lower() != "nan"])
sel_cities = st.sidebar.multiselect("City", cities, default=[])

markets = sorted([m for m in df["market"].dropna().astype(str).unique().tolist() if m.strip() and m.lower() != "nan"])
sel_markets = st.sidebar.multiselect("Market", markets, default=[])

cat_agents = sorted([a for a in df["catalog_agent"].dropna().astype(str).unique().tolist() if a.strip() and a.lower() != "nan"])
sel_cat_agents = st.sidebar.multiselect("Catalogue Agent", cat_agents, default=[])

stu_agents = sorted([a for a in df["studio_agent"].dropna().astype(str).unique().tolist() if a.strip() and a.lower() != "nan"])
sel_stu_agents = st.sidebar.multiselect("Studio Agent", stu_agents, default=[])

ticket_id_search = st.sidebar.text_input("Ticket ID contains", value="")

score_type = st.sidebar.selectbox(
    "Score Type",
    ["Ticket Score (Total QC)", "Catalogue Score", "Studio Score"],
    index=0,
)
score_col = {
    "Ticket Score (Total QC)": "ticket_score_pct",
    "Catalogue Score": "catalog_score_pct",
    "Studio Score": "studio_score_pct",
}[score_type]

# ============================================================
# APPLY FILTERS
# ============================================================
f = df.copy()

if view_mode != "All":
    f = f[f["ticket_type"] == view_mode]

if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    if start:
        f = f[f["date"] >= start]
    if end:
        f = f[f["date"] <= end]

if sel_cities:
    f = f[f["city"].isin(sel_cities)]
if sel_markets:
    f = f[f["market"].isin(sel_markets)]
if sel_cat_agents:
    f = f[f["catalog_agent"].isin(sel_cat_agents)]
if sel_stu_agents:
    f = f[f["studio_agent"].isin(sel_stu_agents)]
if ticket_id_search.strip():
    f = f[f["ticket_id"].str.contains(ticket_id_search.strip(), case=False, na=False)]

# ============================================================
# HEADER
# ============================================================
left, right = st.columns([0.78, 0.22], vertical_alignment="bottom")
with left:
    st.markdown('<div class="qc-title">QC Scores Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        unsafe_allow_html=True,
    )
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
# KPI ROW (SEPARATE SENT BACKS)
# ============================================================
k1, k2, k3, k4, k5, k6 = st.columns(6)

catalog_avg = mean_pct(f["catalog_score_pct"])       # Catalogue Score
studio_avg  = mean_pct(f["studio_score_pct"])        # Studio Score
total_avg   = mean_pct(f["ticket_score_pct"])        # Ticket Score = Total QC

sent_back_catalog = int(pd.to_numeric(f["catalog_sent_back"], errors="coerce").fillna(0).sum())
sent_back_studio  = int(pd.to_numeric(f["studio_sent_back"], errors="coerce").fillna(0).sum())
low_perf = int((pd.to_numeric(f["ticket_score_pct"], errors="coerce") < 90).fillna(False).sum())

with k1: kpi_card("Catalog QC", "—" if catalog_avg is None else f"{catalog_avg:.2f}%")
with k2: kpi_card("Studio QC", "—" if studio_avg is None else f"{studio_avg:.2f}%")
with k3: kpi_card("Total QC (Ticket)", "—" if total_avg is None else f"{total_avg:.2f}%")
with k4: kpi_card("Sent Back → Catalog", f"{sent_back_catalog:,}")
with k5: kpi_card("Sent Back → Studio", f"{sent_back_studio:,}")

st.markdown("<br/>", unsafe_allow_html=True)


a, b = st.columns([0.62, 0.38])

with a:
    st.markdown("### Tickets")
    show_cols = [
        "ticket_id", "ticket_type", "ticket_type_raw",
        "catalog_agent", "catalog_score_pct",
        "studio_agent", "studio_score_pct",
        "ticket_score_pct",
        "city", "market", "dt",
        "catalog_sent_back", "studio_sent_back",
    ]
    show_cols = [c for c in show_cols if c in f.columns]
    tbl = f[show_cols].sort_values("dt", ascending=False)

    st.dataframe(
        tbl,
        use_container_width=True,
        height=460,
        column_config={
            "catalog_score_pct": st.column_config.NumberColumn("Catalogue Score", format="%.2f%%"),
            "studio_score_pct": st.column_config.NumberColumn("Studio Score", format="%.2f%%"),
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
            t.groupby("date_only", as_index=False)[
                ["catalog_score_pct", "studio_score_pct", "ticket_score_pct"]
            ]
            .mean(numeric_only=True)
            .sort_values("date_only")
        )
        fig = px.line(daily, x="date_only", y=score_col, markers=True)
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

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
        fig = px.histogram(x=s, nbins=20)
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Score (%)", yaxis_title="Tickets")
        st.plotly_chart(fig, use_container_width=True)

with c3:
    st.markdown("### Tickets by City")
    city_counts = (
        f["city"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: (s != "") & (s.str.lower() != "nan") & (s.str.lower() != "unknown")]
        .value_counts()
        .reset_index()
    )
    city_counts.columns = ["city", "ticket_count"]

    if city_counts.empty:
        st.info("No city data available (after excluding Unknown).")
    else:
        fig = px.bar(city_counts.head(20), x="city", y="ticket_count")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ============================================================
# AGENTS
# ============================================================
st.markdown("## Agent Performance")

p1, p2 = st.columns([0.52, 0.48])

with p1:
    st.markdown("### Catalogue agents")
    ca = (
        f.groupby("catalog_agent", dropna=False)
        .agg(
            tickets=("ticket_id", "count"),
            avg_catalog=("catalog_score_pct", "mean"),
            avg_ticket=("ticket_score_pct", "mean"),
            sent_back=("catalog_sent_back", "sum"),
        )
        .reset_index()
        .sort_values(["avg_ticket", "tickets"], ascending=[True, False])
    )
    ca["low_perf_flag"] = (ca["avg_ticket"] < 90).fillna(False)
    st.dataframe(
        ca,
        use_container_width=True,
        height=360,
        column_config={
            "catalog_agent": st.column_config.TextColumn("Catalogue Agent(s)"),
            "tickets": st.column_config.NumberColumn("Tickets"),
            "avg_catalog": st.column_config.NumberColumn("Avg Catalogue Score", format="%.2f%%"),
            "avg_ticket": st.column_config.NumberColumn("Avg Ticket Score (Total QC)", format="%.2f%%"),
            "sent_back": st.column_config.NumberColumn("Sent Back → Catalog"),
            "low_perf_flag": st.column_config.CheckboxColumn("Low performer (<90%)"),
        },
    )

with p2:
    st.markdown("### Studio agents")
    sa = (
        f.groupby("studio_agent", dropna=False)
        .agg(
            tickets=("ticket_id", "count"),
            avg_studio=("studio_score_pct", "mean"),
            avg_ticket=("ticket_score_pct", "mean"),
            sent_back=("studio_sent_back", "sum"),
        )
        .reset_index()
        .sort_values(["avg_ticket", "tickets"], ascending=[True, False])
    )
    sa["low_perf_flag"] = (sa["avg_ticket"] < 90).fillna(False)
    st.dataframe(
        sa,
        use_container_width=True,
        height=360,
        column_config={
            "studio_agent": st.column_config.TextColumn("Studio Agent"),
            "tickets": st.column_config.NumberColumn("Tickets"),
            "avg_studio": st.column_config.NumberColumn("Avg Studio Score", format="%.2f%%"),
            "avg_ticket": st.column_config.NumberColumn("Avg Ticket Score (Total QC)", format="%.2f%%"),
            "sent_back": st.column_config.NumberColumn("Sent Back → Studio"),
            "low_perf_flag": st.column_config.CheckboxColumn("Low performer (<90%)"),
        },
    )

with st.expander("🔎 Debug: subjects that became Other (should be near zero)"):
    other_vals = (
        df[df["ticket_type"] == "Other"]["ticket_type_raw"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    st.dataframe(other_vals.value_counts().reset_index().head(120), use_container_width=True)
