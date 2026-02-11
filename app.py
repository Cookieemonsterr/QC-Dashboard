import io
import os
import re
import html
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

.qc-title{ font-size:1.85rem; font-weight:900; letter-spacing:.2px; margin:.1rem 0 .35rem 0; }
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
    return h

def safe_num(x):
    return pd.to_numeric(x, errors="coerce")

def clean_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def kpi_card(label: str, value: str, delta: str | None = None):
    d = f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
    if delta:
        d += f'<div class="delta">{delta}</div>'
    d += "</div>"
    st.markdown(d, unsafe_allow_html=True)

def find_local_file(candidates: list[str]) -> str | None:
    for f in candidates:
        if os.path.exists(f):
            return f
    return None

def read_eval_csv(path_or_url: str) -> pd.DataFrame:
    if path_or_url.startswith("http"):
        return pd.read_csv(path_or_url)
    return pd.read_csv(path_or_url)

def read_istep_table(path_or_url: str) -> pd.DataFrame:
    # Istep Data (1).xls is actually HTML-exported, so read_html works best.
    if path_or_url.startswith("http"):
        if path_or_url.lower().endswith(".csv"):
            return pd.read_csv(path_or_url)
        tables = pd.read_html(path_or_url)
        return tables[0]

    # local
    lower = path_or_url.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path_or_url)

    # html-xls / html
    tables = pd.read_html(path_or_url)
    return tables[0]


# ============================================================
# TICKET TYPE RULES (from Subject)
# ============================================================
def map_ticket_type(subject: str) -> str:
    s = clean_str(subject).lower()
    if s.startswith("outlet catalogue update request"):
        return "Update Tickets"
    if s.startswith("new brand setup"):
        return "Build Tickets"
    if s.startswith("new outlet setup for existing brand"):
        return "Existing Tickets"
    return "Other"

# ============================================================
# FORM NAME CLASSIFICATION (from Evaluation Report)
# ============================================================
def norm_form(x: str) -> str:
    s = clean_str(x)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

STUDIO_FORMS = {norm_form("Studio scorecard"), norm_form("Images Update")}
UPDATE_FORMS = {
    norm_form("Update scorecard"),
    norm_form("Price Update"),
    norm_form("Location Update"),
    norm_form("Tags Update"),
    norm_form("Operational Hours Update"),
}
BUILD_FORMS = {norm_form("New Points/ Builds"), norm_form("New Points/Builds"), norm_form("New Points Builds")}
EXISTING_FORMS = {norm_form("Existing Company and Brand - New Outlet")}
CATALOGUE_SCORECARD = {norm_form("Catalogue scorecard")}

CATALOG_ALL_FORMS = UPDATE_FORMS | BUILD_FORMS | EXISTING_FORMS | CATALOGUE_SCORECARD


# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data(show_spinner=False, ttl=600)
def load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    # ---- Evaluation report (scores source)
    eval_path = find_local_file(EVAL_LOCAL_FILES)
    if eval_path:
        eval_df = read_eval_csv(eval_path)
    elif EVAL_RAW_URL:
        eval_df = read_eval_csv(EVAL_RAW_URL)
    else:
        raise RuntimeError(
            "Couldn't find Evaluation Report file. Put 'Evaluation Report Istep - Report.csv' in repo root "
            "or set EVAL_RAW_URL."
        )

    # ---- Istep base (meta source)
    istep_path = find_local_file(ISTEP_LOCAL_FILES)
    if istep_path:
        istep_df = read_istep_table(istep_path)
    elif ISTEP_RAW_URL:
        istep_df = read_istep_table(ISTEP_RAW_URL)
    else:
        raise RuntimeError(
            "Couldn't find Istep Data file. Put 'Istep Data (1).xls' in repo root (HTML export) "
            "or set ISTEP_RAW_URL."
        )

    # normalize headers
    eval_df.columns = [normalize_header(c) for c in eval_df.columns]
    istep_df.columns = [normalize_header(c) for c in istep_df.columns]
    return eval_df, istep_df


# ============================================================
# BUILD TICKET-LEVEL TABLE (NO DUPLICATE INFLATION)
# ============================================================
@st.cache_data(show_spinner=False, ttl=600)
def build_ticket_table() -> pd.DataFrame:
    eval_df, istep_df = load_sources()

    # --- Evaluation Report: per-ticket, per-form aggregated (dedupe card duplication)
    needed = ["Reference ID", "Subject", "Name", "User Id", "Sent Back To Catalog", "Score", "Date & Time", "Form Name"]
    missing = [c for c in needed if c not in eval_df.columns]
    if missing:
        raise RuntimeError(f"Evaluation report is missing columns: {missing}")

    e = eval_df.copy()
    e["Reference ID"] = e["Reference ID"].astype(str).str.strip()
    e["form_norm"] = e["Form Name"].apply(norm_form)
    e["score_num"] = safe_num(e["Score"])
    e["sent_back_num"] = safe_num(e["Sent Back To Catalog"]).fillna(0)

    # Deduplicate duplicates inside same ticket+form (agents/cards duplication):
    # - score: mean per form (per ticket)
    # - sent_back: max per form (per ticket) so duplicates don't inflate
    per_form = (
        e.groupby(["Reference ID", "form_norm"], as_index=False)
        .agg(
            form_score=("score_num", "mean"),
            form_sent_back=("sent_back_num", "max"),
        )
    )

    # Helper to compute mean score across a set of forms, per ticket
    def mean_forms(ticket_id: str, form_set: set[str]) -> float | None:
        s = per_form[(per_form["Reference ID"] == ticket_id) & (per_form["form_norm"].isin(form_set))]["form_score"]
        s = pd.to_numeric(s, errors="coerce").dropna()
        return None if s.empty else float(s.mean())

    def sum_sent_back(ticket_id: str, form_set: set[str]) -> int:
        s = per_form[(per_form["Reference ID"] == ticket_id) & (per_form["form_norm"].isin(form_set))]["form_sent_back"]
        s = pd.to_numeric(s, errors="coerce").dropna()
        return int(s.sum()) if not s.empty else 0

    # --- Istep base: collapse duplicates to ticket-level for meta + ticket score + agents + city/market + sent back (dedup by agent)
    base = istep_df.copy()
    if "Reference ID" not in base.columns:
        raise RuntimeError("Istep Data is missing 'Reference ID' column.")

    base["Reference ID"] = base["Reference ID"].astype(str).str.strip()

    # Normalize subject + ticket score
    if "Subject" in base.columns:
        base["Subject"] = base["Subject"].astype(str)

    # Ticket score should be used as Total QC (your rule)
    base["Ticket Score"] = safe_num(base.get("Ticket Score", pd.NA))

    # Datetime: prefer Catalogue Date & Time then Studio Date & Time
    base["cat_dt"] = pd.to_datetime(base.get("Catalogue Date & Time", pd.NaT), errors="coerce")
    base["stu_dt"] = pd.to_datetime(base.get("Studio Date & Time", pd.NaT), errors="coerce")
    base["dt"] = base["cat_dt"].fillna(base["stu_dt"])

    # Agent names (unique join)
    def unique_join(series, sep=" | "):
        s = series.dropna().astype(str).map(str.strip)
        s = s[(s != "") & (s.str.lower() != "nan")]
        uniq = list(dict.fromkeys(s.tolist()))
        return sep.join(uniq) if uniq else pd.NA

    # Sent back dedupe PER TICKET PER AGENT (this is what you described)
    # Catalogue Sent Back To Catalog
    if "Catalogue Name" in base.columns and "Catalogue Sent Back To Catalog" in base.columns:
        cat_sb = base[["Reference ID", "Catalogue Name", "Catalogue Sent Back To Catalog"]].copy()
        cat_sb["Catalogue Name"] = cat_sb["Catalogue Name"].astype(str).str.strip()
        cat_sb["sb"] = safe_num(cat_sb["Catalogue Sent Back To Catalog"]).fillna(0)
        cat_sb = cat_sb[cat_sb["Catalogue Name"] != ""]
        cat_sb_agg = cat_sb.groupby(["Reference ID", "Catalogue Name"], as_index=False)["sb"].max()
        cat_sb_ticket = cat_sb_agg.groupby("Reference ID", as_index=False)["sb"].sum().rename(columns={"sb": "catalog_sent_back"})
    else:
        cat_sb_ticket = pd.DataFrame({"Reference ID": base["Reference ID"].unique(), "catalog_sent_back": 0})

    # Studio Sent Back To Catalog
    if "Studio Name" in base.columns and "Studio Sent Back To Catalog" in base.columns:
        stu_sb = base[["Reference ID", "Studio Name", "Studio Sent Back To Catalog"]].copy()
        stu_sb["Studio Name"] = stu_sb["Studio Name"].astype(str).str.strip()
        stu_sb["sb"] = safe_num(stu_sb["Studio Sent Back To Catalog"]).fillna(0)
        stu_sb = stu_sb[stu_sb["Studio Name"] != ""]
        stu_sb_agg = stu_sb.groupby(["Reference ID", "Studio Name"], as_index=False)["sb"].max()
        stu_sb_ticket = stu_sb_agg.groupby("Reference ID", as_index=False)["sb"].sum().rename(columns={"sb": "studio_sent_back"})
    else:
        stu_sb_ticket = pd.DataFrame({"Reference ID": base["Reference ID"].unique(), "studio_sent_back": 0})

    # Ticket-level collapse
    agg = {
        "Subject": "first",
        "Ticket Score": "first",
        "dt": "first",
        "Catalogue City": "first",
        "Catalogue Market": "first",
        "Studio City": "first",
        "Studio Market": "first",
    }

    if "Catalogue Name" in base.columns:
        agg["Catalogue Name"] = unique_join
    if "Studio Name" in base.columns:
        agg["Studio Name"] = unique_join

    ticket = base.groupby("Reference ID", as_index=False).agg(agg)

    # Pick city/market: prefer Catalogue then Studio
    ticket["city"] = ticket.get("Catalogue City", pd.NA).fillna(ticket.get("Studio City", pd.NA))
    ticket["market"] = ticket.get("Catalogue Market", pd.NA).fillna(ticket.get("Studio Market", pd.NA))

    ticket["ticket_type"] = ticket["Subject"].apply(map_ticket_type)

    # Date helper fields
    ticket["date"] = pd.to_datetime(ticket["dt"], errors="coerce").dt.date
    ticket["month"] = pd.to_datetime(ticket["dt"], errors="coerce").dt.to_period("M").astype(str)
    ticket["week"] = pd.to_datetime(ticket["dt"], errors="coerce").dt.isocalendar().week.astype("Int64")
    ticket["day"] = pd.to_datetime(ticket["dt"], errors="coerce").dt.day_name()

    # Merge sent back per ticket (from Istep base, deduped per agent)
    ticket = ticket.merge(cat_sb_ticket, on="Reference ID", how="left")
    ticket = ticket.merge(stu_sb_ticket, on="Reference ID", how="left")
    ticket["catalog_sent_back"] = safe_num(ticket["catalog_sent_back"]).fillna(0).astype(int)
    ticket["studio_sent_back"] = safe_num(ticket["studio_sent_back"]).fillna(0).astype(int)

    # --- Scores per ticket (ONLY from Evaluation Report)
    ids = ticket["Reference ID"].astype(str).tolist()

    # studio score = mean of STUDIO forms only
    studio_scores = {}
    catalog_scores_update = {}
    catalog_scores_build = {}
    catalog_scores_existing = {}

    for tid in ids:
        studio_scores[tid] = mean_forms(tid, STUDIO_FORMS)

        # catalog score depends on ticket type (your rules)
        # update tickets: update forms + catalogue scorecard (if exists)
        catalog_scores_update[tid] = mean_forms(tid, UPDATE_FORMS | CATALOGUE_SCORECARD)

        # build tickets: build forms + catalogue scorecard
        catalog_scores_build[tid] = mean_forms(tid, BUILD_FORMS | CATALOGUE_SCORECARD)

        # existing tickets: existing forms + catalogue scorecard
        catalog_scores_existing[tid] = mean_forms(tid, EXISTING_FORMS | CATALOGUE_SCORECARD)

    ticket["studio_score_pct"] = ticket["Reference ID"].map(studio_scores)
    ticket["catalog_score_update_pct"] = ticket["Reference ID"].map(catalog_scores_update)
    ticket["catalog_score_build_pct"] = ticket["Reference ID"].map(catalog_scores_build)
    ticket["catalog_score_existing_pct"] = ticket["Reference ID"].map(catalog_scores_existing)

    # Choose the catalog score based on ticket type (THIS is the “correct filtering” you do manually)
    def choose_catalog_score(row):
        ttype = row["ticket_type"]
        if ttype == "Update Tickets":
            return row["catalog_score_update_pct"]
        if ttype == "Build Tickets":
            return row["catalog_score_build_pct"]
        if ttype == "Existing Tickets":
            return row["catalog_score_existing_pct"]
        # fallback: any catalog form
        return row["catalog_score_update_pct"] or row["catalog_score_build_pct"] or row["catalog_score_existing_pct"]

    ticket["catalog_score_pct"] = ticket.apply(choose_catalog_score, axis=1)

    # Total QC should be Ticket Score (your rule)
    ticket["ticket_score_pct"] = safe_num(ticket["Ticket Score"])

    # Friendly columns
    ticket.rename(
        columns={
            "Reference ID": "ticket_id",
            "Catalogue Name": "catalog_agents",
            "Studio Name": "studio_agents",
        },
        inplace=True,
    )

    return ticket


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## Data")

if st.sidebar.button("🔄 Refresh data now"):
    st.cache_data.clear()

try:
    df = build_ticket_table()
except Exception as e:
    st.error("Data load / build failed.")
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

min_dt = pd.to_datetime(df["dt"], errors="coerce").min()
max_dt = pd.to_datetime(df["dt"], errors="coerce").max()
default_range = (
    (min_dt.date() if pd.notna(min_dt) else None),
    (max_dt.date() if pd.notna(max_dt) else None),
)
date_range = st.sidebar.date_input("Date range", value=default_range)

cities = sorted([c for c in df["city"].dropna().astype(str).unique().tolist() if c.strip() and c.lower() != "nan"])
sel_cities = st.sidebar.multiselect("City", cities, default=[])

markets = sorted([m for m in df["market"].dropna().astype(str).unique().tolist() if m.strip() and m.lower() != "nan"])
sel_markets = st.sidebar.multiselect("Market", markets, default=[])

ticket_id_search = st.sidebar.text_input("Ticket ID contains", value="")

score_type = st.sidebar.selectbox(
    "Score Type",
    ["Total QC (Ticket Score)", "Catalog QC", "Studio QC"],
    index=0,
)
score_col = {
    "Total QC (Ticket Score)": "ticket_score_pct",
    "Catalog QC": "catalog_score_pct",
    "Studio QC": "studio_score_pct",
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
        f = f[pd.to_datetime(f["date"], errors="coerce") >= start]
    if end:
        f = f[pd.to_datetime(f["date"], errors="coerce") <= end]

if sel_cities:
    f = f[f["city"].isin(sel_cities)]
if sel_markets:
    f = f[f["market"].isin(sel_markets)]
if ticket_id_search.strip():
    f = f[f["ticket_id"].astype(str).str.contains(ticket_id_search.strip(), case=False, na=False)]

# ============================================================
# HEADER (FIXED — no broken st.markdown)
# ============================================================
left, right = st.columns([0.78, 0.22], vertical_alignment="bottom")
with left:
    st.markdown('<div class="qc-title">QC Scores Dashboard</div>', unsafe_allow_html=True)
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
# KPI ROW (ticket-level accurate)
# ============================================================
k1, k2, k3, k4, k5 = st.columns(5)

catalog_avg = pd.to_numeric(f["catalog_score_pct"], errors="coerce").dropna().mean()
studio_avg = pd.to_numeric(f["studio_score_pct"], errors="coerce").dropna().mean()
total_avg = pd.to_numeric(f["ticket_score_pct"], errors="coerce").dropna().mean()

sent_back_catalog = int(pd.to_numeric(f["catalog_sent_back"], errors="coerce").fillna(0).sum())
sent_back_studio = int(pd.to_numeric(f["studio_sent_back"], errors="coerce").fillna(0).sum())

with k1: kpi_card("Catalog QC", "—" if pd.isna(catalog_avg) else f"{catalog_avg:.2f}%")
with k2: kpi_card("Studio QC", "—" if pd.isna(studio_avg) else f"{studio_avg:.2f}%")
with k3: kpi_card("Total QC (Ticket)", "—" if pd.isna(total_avg) else f"{total_avg:.2f}%")
with k4: kpi_card("Sent Back → Catalog", f"{sent_back_catalog:,}")
with k5: kpi_card("Sent Back → Studio", f"{sent_back_studio:,}")

st.markdown("<br/>", unsafe_allow_html=True)

# ============================================================
# TABLE + TREND
# ============================================================
a, b = st.columns([0.62, 0.38])

with a:
    st.markdown("### Tickets (ticket-level, deduped)")
    show_cols = [
        "ticket_id",
        "ticket_type",
        "Subject",
        "catalog_agents",
        "catalog_score_pct",
        "studio_agents",
        "studio_score_pct",
        "ticket_score_pct",
        "city",
        "market",
        "dt",
        "catalog_sent_back",
        "studio_sent_back",
    ]
    show_cols = [c for c in show_cols if c in f.columns]
    tbl = f[show_cols].sort_values("dt", ascending=False)

    st.dataframe(
        tbl,
        use_container_width=True,
        height=520,
        column_config={
            "catalog_score_pct": st.column_config.NumberColumn("Catalog QC", format="%.2f%%"),
            "studio_score_pct": st.column_config.NumberColumn("Studio QC", format="%.2f%%"),
            "ticket_score_pct": st.column_config.NumberColumn("Ticket Score", format="%.2f%%"),
            "dt": st.column_config.DatetimeColumn("Date & Time"),
        },
    )

with b:
    st.markdown("### Score Change (safe)")
    t = f.dropna(subset=["dt"]).copy()
    if t.empty:
        st.info("No datetime values available for this filter.")
    else:
        t["date_only"] = pd.to_datetime(t["dt"], errors="coerce").dt.date
        t = t.dropna(subset=["date_only"])

        # build daily safely
        if score_col not in t.columns:
            st.info("Selected score not available.")
        else:
            daily = (
                t.groupby("date_only", as_index=False)[score_col]
                .mean(numeric_only=True)
                .sort_values("date_only")
            )

            ydata = pd.to_numeric(daily[score_col], errors="coerce").dropna()
            if ydata.empty:
                st.info("No values available to plot for this score type.")
            else:
                fig = px.line(daily, x="date_only", y=score_col, markers=True)
                fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10),
                                  xaxis_title="", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ============================================================
# CHARTS
# ============================================================
c1, c2, c3 = st.columns([0.34, 0.33, 0.33])

with c1:
    st.markdown("### Ticket type split")
    tt = f["ticket_type"].fillna("Other").value_counts().reset_index()
    tt.columns = ["ticket_type", "count"]
    fig = px.pie(tt, names="ticket_type", values="count", hole=0.55)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("### Score distribution")
    s = pd.to_numeric(f[score_col], errors="coerce").dropna()
    if s.empty:
        st.info("No score values available.")
    else:
        fig = px.histogram(s, nbins=20)
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Score (%)", yaxis_title="Tickets")
        st.plotly_chart(fig, use_container_width=True)

with c3:
    st.markdown("### Tickets by City (NO Unknown)")
    city_series = f["city"].dropna().astype(str).str.strip()
    city_series = city_series[(city_series != "") & (city_series.str.lower() != "unknown") & (city_series.str.lower() != "nan")]
    if city_series.empty:
        st.info("No city values available.")
    else:
        city_counts = city_series.value_counts().reset_index()
        city_counts.columns = ["city", "ticket_count"]
        fig = px.bar(city_counts.head(20), x="city", y="ticket_count")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# DEBUG
# ============================================================
with st.expander("🔎 Debug: show how Catalog score is chosen per ticket"):
    dbg = f[["ticket_id", "ticket_type", "catalog_score_update_pct", "catalog_score_build_pct", "catalog_score_existing_pct", "catalog_score_pct"]].copy()
    st.dataframe(dbg.sort_values("ticket_type"), use_container_width=True)
