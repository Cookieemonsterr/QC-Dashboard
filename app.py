import re
import html
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================
# FILES
# =========================
BASE_PATH = "istep_data.csv"
REPORT_PATH = "Evaluation Report Istep - Report.csv"

# KPI source names (folder or <name>.zip)
SRC_BUILD_TICKETS = "Build Tickets"
SRC_UPDATE_TICKETS = "Update Tickets"
SRC_TICKETS_BY_CITY = "Tickets by City"
SRC_TICKETS_BY_MARKET = "Tickets by Market"
SRC_BUILD_TICKETS_BY_MARKET = "Build Tickets by Market"
SRC_UPDATE_TICKETS_BY_MARKET = "Update Tickets by Market"
SRC_CATALOG_AGENTS_SCORES = "Catalog Agents Scores"
SRC_STUDIO_AGENTS_SCORES = "Studio Agents Scores"

# ── Catalog Mistakes: fixed filenames (xlsx or csv), place next to app.py
MISTAKES_FILES = {
    "UAE – All":       "catalog_mistakes_uae_all",
    "Jordan – All":    "catalog_mistakes_jordan_all",
    "UAE – Build":     "catalog_mistakes_uae_build",
    "UAE – Update":    "catalog_mistakes_uae_update",
    "Jordan – Build":  "catalog_mistakes_jordan_build",
    "Jordan – Update": "catalog_mistakes_jordan_update",
}

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
.small-muted{ color: var(--muted); font-size: .9rem; }

.section-title {
  font-size: 1.35rem;
  font-weight: 700;
  margin: 1rem 0 0.5rem 0;
}
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
        c = "" if c is None else str(c)
        c = normalize_header(c)
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

def kpi_card(label: str, value: str):
    st.markdown(
        f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )

def first_non_empty(series: pd.Series):
    s = series.dropna().astype(str).str.strip()
    s = s[(s != "") & (s.str.lower() != "nan")]
    return s.iloc[0] if len(s) else pd.NA

def file_exists_or_stop(path: str, label: str):
    if not Path(path).exists():
        st.error(f"Missing: **{path}** ({label}). Put it in the SAME repo/folder as app.py.")
        st.stop()

def read_kpi_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, header=[0, 1])
        if isinstance(df.columns, pd.MultiIndex):
            cols = []
            for a, b in df.columns:
                a = normalize_header(a)
                b = normalize_header(b)
                if "Unnamed" in a:
                    a = ""
                if "Unnamed" in b:
                    b = ""
                name = " ".join([x for x in [a, b] if x]).strip()
                cols.append(name if name else f"col_{len(cols)}")
            df.columns = make_unique(cols)
            df = df.dropna(axis=1, how="all")
            return df
    except Exception:
        pass

    df = pd.read_csv(path)
    df.columns = make_unique(df.columns)
    df = df.loc[:, [c for c in df.columns if not c.lower().startswith("unnamed")]]
    df = df.dropna(axis=1, how="all")
    return df

def extract_month_from_filename(stem: str) -> str:
    s = stem.strip()
    m = re.search(r"(20\d{2})[-_ ]?([01]\d)", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m2 = re.search(r"(20\d{2})([01]\d)", s)
    if m2:
        return f"{m2.group(1)}-{m2.group(2)}"
    return s

def resolve_source(name: str) -> tuple[str, Path]:
    d = Path(name)
    z = Path(f"{name}.zip")
    if d.exists() and d.is_dir():
        return ("dir", d)
    if z.exists() and z.is_file():
        return ("zip", z)
    z2 = Path(name)
    if z2.exists() and z2.is_file() and z2.suffix.lower() == ".zip":
        return ("zip", z2)
    st.error(
        f"Missing KPI source: **{name}**. I looked for a folder named `{name}/` or a zip named `{name}.zip`."
    )
    st.stop()

def ensure_zip_extracted(zip_path: Path, extract_root: Path) -> Path:
    out_dir = extract_root / zip_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    if list(out_dir.rglob("*.csv")):
        return out_dir
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
    except zipfile.BadZipFile:
        st.error(f"Bad zip file: **{zip_path.name}**. Re-zip it normally (ZIP).")
        st.stop()
    return out_dir

@st.cache_data(show_spinner=False, ttl=600)
def read_kpi_source(name: str) -> pd.DataFrame:
    kind, path = resolve_source(name)
    if kind == "dir":
        base = path
    else:
        extract_root = Path(".kpi_cache")
        extract_root.mkdir(exist_ok=True)
        base = ensure_zip_extracted(path, extract_root)

    csv_files = sorted(list(base.rglob("*.csv")))
    if not csv_files:
        st.error(f"No CSVs found inside **{name}** ({'folder' if kind=='dir' else 'zip'}).")
        st.stop()

    dfs = []
    for f in csv_files:
        d = read_kpi_csv(str(f))
        d["__month__"] = extract_month_from_filename(f.stem)
        d["__source_file__"] = f.name
        dfs.append(d)

    out = pd.concat(dfs, ignore_index=True, sort=False)
    out.columns = make_unique(out.columns)
    return out

def metrics_from_report(df: pd.DataFrame, mode: str, level: str) -> pd.DataFrame | None:
    lvl_col = safe_col(df, [level, level.lower(), level.upper()])
    tickets_col = safe_col(
        df,
        [
            f"Total Tickets Resolved {mode}",
            "Total Tickets Resolved",
            f"Tickets {mode}",
            "Tickets",
            "Ticket Count",
            "ticket_count",
            "Total Tickets",
        ],
    )
    score_col = safe_col(
        df,
        [
            f"Total Average QC Score {mode}",
            "Total Average QC Score",
            f"Average QC Score {mode}",
            "Average QC Score",
            "Avg QC Score",
        ],
    )

    if not lvl_col:
        return None

    cols = [lvl_col]
    if tickets_col:
        cols.append(tickets_col)
    if score_col:
        cols.append(score_col)

    out = df[cols].copy()
    rename_map = {lvl_col: level}
    if tickets_col:
        rename_map[tickets_col] = "tickets"
    if score_col:
        rename_map[score_col] = "avg_score"
    out = out.rename(columns=rename_map)

    out[level] = out[level].astype(str).str.strip()
    out = out[~out[level].str.lower().isin(["total", "unknown", "nan", ""])]

    if "tickets" in out.columns:
        out["tickets"] = pd.to_numeric(out["tickets"], errors="coerce").fillna(0).astype(int)
    if "avg_score" in out.columns:
        out["avg_score"] = pd.to_numeric(out["avg_score"], errors="coerce")

    out = out.dropna(axis=0, how="all")
    return out

def combine_kpis_by_level(raw_df: pd.DataFrame, mode: str, level: str) -> pd.DataFrame:
    m = metrics_from_report(raw_df, mode, level)
    if m is None or m.empty:
        return pd.DataFrame(columns=[level, "tickets", "avg_score"])

    if "tickets" not in m.columns:
        m["tickets"] = 0

    m["tickets"] = pd.to_numeric(m["tickets"], errors="coerce").fillna(0).astype(int)

    if "avg_score" not in m.columns:
        g = m.groupby(level, as_index=False).agg(tickets=("tickets", "sum"))
        return g

    m["avg_score"] = pd.to_numeric(m["avg_score"], errors="coerce")
    m = m.dropna(subset=[level]).copy()
    m[level] = m[level].astype(str).str.strip()
    m = m[m[level].ne("")]

    m["_w"] = m["tickets"].clip(lower=0)
    m["_wx"] = m["avg_score"] * m["_w"]

    g = m.groupby(level, as_index=False).agg(
        tickets=("tickets", "sum"),
        _w=("_w", "sum"),
        _wx=("_wx", "sum"),
    )
    g["avg_score"] = g.apply(lambda r: (r["_wx"] / r["_w"]) if r["_w"] > 0 else None, axis=1)
    g = g.drop(columns=["_w", "_wx"])
    return g

def total_tickets_from_combined(df: pd.DataFrame) -> int | None:
    if df is None or df.empty or "tickets" not in df.columns:
        return None
    return int(pd.to_numeric(df["tickets"], errors="coerce").fillna(0).sum())

def avg_score_from_combined_with_filter(df: pd.DataFrame, level: str, selected: list[str]) -> float | None:
    if df is None or df.empty or "avg_score" not in df.columns:
        return None
    tmp = df.copy()
    if selected:
        tmp = tmp[tmp[level].isin(selected)]
    tmp = tmp.dropna(subset=["avg_score"]).copy()
    if tmp.empty:
        return None
    if "tickets" in tmp.columns and tmp["tickets"].sum() > 0:
        return float((tmp["avg_score"] * tmp["tickets"]).sum() / tmp["tickets"].sum())
    return float(tmp["avg_score"].mean())

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

    agg = {}
    if subject: agg[subject] = first_non_empty
    if c_city: agg[c_city] = first_non_empty
    if c_market: agg[c_market] = first_non_empty

    for col in df.columns:
        if col == ref or col in agg:
            continue
        agg[col] = first_non_empty

    out = df.groupby(ref, as_index=False).agg(agg)
    out.rename(columns={ref: "Reference ID"}, inplace=True)
    return out

# ============================================================
# FORMS / BUCKETS
# ============================================================
STUDIO_FORMS = {"Studio scorecard", "Images Update"}
BUILD_MAIN_FORMS = {"New Points/ Builds"}
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

    return None

# ============================================================
# CATALOG MISTAKES LOADER  (fixed filenames next to app.py)
# ============================================================
def _normalize_mistakes_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    if "Attribute" in df.columns:
        df["Attribute"] = df["Attribute"].astype(str).str.replace(r"\n", " ", regex=True).str.strip()
    if "Failure %" in df.columns:
        df["Failure %"] = (
            df["Failure %"].astype(str).str.replace("%", "", regex=False).str.strip()
        )
        df["Failure %"] = pd.to_numeric(df["Failure %"], errors="coerce")
    if "Total Failures" in df.columns:
        df["Total Failures"] = pd.to_numeric(df["Total Failures"], errors="coerce")
    return df

@st.cache_data(show_spinner=False, ttl=600)
def load_mistakes_file(stem: str) -> pd.DataFrame | None:
    """Try <stem>.xlsx then <stem>.csv. Returns None if neither exists."""
    for suffix in (".xlsx", ".xls", ".csv"):
        p = Path(stem + suffix)
        if not p.exists():
            continue
        try:
            df = pd.read_excel(p) if suffix in (".xlsx", ".xls") else pd.read_csv(p)
            return _normalize_mistakes_df(df)
        except Exception as e:
            st.warning(f"Could not read {p.name}: {e}")
    return None
# ============================================================
# UI: Sidebar
# ============================================================
st.sidebar.markdown("## Data")
if st.sidebar.button("Refresh data now"):
    st.cache_data.clear()

mode = st.sidebar.radio("Report mode (for KPI files)", ["Monthly", "Weekly"], index=0)

# ============================================================
# LOAD CORE DATA
# ============================================================
@st.cache_data(show_spinner=False, ttl=600)
def load_data(mode: str):
    file_exists_or_stop(BASE_PATH, "overall / metadata")
    file_exists_or_stop(REPORT_PATH, "evaluation report (scores)")

    # ---- Overall
    overall = pd.read_csv(BASE_PATH, low_memory=False)
    overall.columns = make_unique(overall.columns)
    overall = collapse_overall(overall)

    subj_col = safe_col(overall, ["Subject"])
    overall["ticket_type_raw"] = overall[subj_col].astype(str) if subj_col else ""
    overall["ticket_type"] = overall["ticket_type_raw"].apply(map_ticket_type_from_subject)
    overall["__ref__"] = overall["Reference ID"].astype(str).str.strip()

    city_col = safe_col(overall, ["Catalogue City", "City"])
    market_col = safe_col(overall, ["Catalogue Market", "Market"])
    overall["city"] = overall[city_col] if city_col else pd.NA
    overall["market"] = overall[market_col] if market_col else pd.NA

    overall_small = overall[["__ref__", "Reference ID", "Subject", "ticket_type_raw", "ticket_type", "city", "market"]].copy()

    # ---- Evaluation report
    ev = pd.read_csv(REPORT_PATH, low_memory=False)
    ev.columns = make_unique(ev.columns)

    ref_col = safe_col(ev, ["Reference ID"])
    name_col = safe_col(ev, ["Name"])
    form_col = safe_col(ev, ["Form Name"])
    score_col = safe_col(ev, ["Score"])
    sb_col = safe_col(ev, ["Back To Cat", "Sent Back To Cat", "Sent Back To Catalog"])
    dt_col = safe_col(ev, ["Date & Time", "Date Time", "Date&Time"])

    if not ref_col or not form_col or not score_col:
        raise RuntimeError("Evaluation report must contain: Reference ID, Form Name, Score")
    if not name_col:
        raise RuntimeError("Evaluation report must contain: Name")

    ev = ev.copy()
    ev["__ref__"] = ev[ref_col].astype(str).str.strip()
    ev["agent_name"] = ev[name_col].astype(str).str.strip()
    ev["form_name"] = ev[form_col].astype(str).str.strip()
    ev["score_pct"] = to_pct(ev[score_col])

    if sb_col:
        ev["sent_back_catalog"] = pd.to_numeric(ev[sb_col], errors="coerce").fillna(0)
    else:
        ev["sent_back_catalog"] = 0

    ev["dt"] = pd.to_datetime(ev[dt_col], errors="coerce") if dt_col else pd.NaT
    ev["date"] = ev["dt"].dt.date

    ref_to_type = dict(zip(overall_small["__ref__"], overall_small["ticket_type"]))
    ev["ticket_type"] = ev["__ref__"].map(ref_to_type).fillna("Other")

    ev["bucket"] = ev.apply(lambda r: bucket_for_row(r["ticket_type"], r["form_name"]), axis=1)
    ev["is_studio"] = ev["form_name"].isin(STUDIO_FORMS)
    ev["is_non_studio"] = ~ev["is_studio"]

    rows = ev.merge(
        overall_small[["__ref__", "Reference ID", "Subject", "ticket_type_raw", "city", "market"]],
        on="__ref__",
        how="left",
        suffixes=("", "_overall"),
    )
    rows["market"] = rows["market"].fillna("Unknown")
    rows["city"] = rows["city"].fillna("Unknown")

    per_cat = rows[rows["is_non_studio"]].groupby("__ref__", as_index=False)["score_pct"].mean().rename(columns={"score_pct": "catalog_qc"})
    per_stu = rows[rows["is_studio"]].groupby("__ref__", as_index=False)["score_pct"].mean().rename(columns={"score_pct": "studio_qc"})
    per_tot = rows[rows["bucket"] == "Total"].groupby("__ref__", as_index=False)["score_pct"].mean().rename(columns={"score_pct": "total_qc"})
    sent_back_events = rows.groupby("__ref__", as_index=False)["sent_back_catalog"].sum().rename(columns={"sent_back_catalog": "sent_back_to_catalog_events"})

    tickets = (
        overall_small.merge(sent_back_events, on="__ref__", how="left")
        .merge(per_cat, on="__ref__", how="left")
        .merge(per_stu, on="__ref__", how="left")
        .merge(per_tot, on="__ref__", how="left")
    )
    tickets["ticket_id"] = tickets["Reference ID"].astype(str)
    tickets["sent_back_to_catalog_events"] = pd.to_numeric(tickets["sent_back_to_catalog_events"], errors="coerce").fillna(0).astype(int)

    city_stats_raw = read_kpi_source(SRC_TICKETS_BY_CITY)
    build_city_raw = read_kpi_source(SRC_BUILD_TICKETS)
    update_city_raw = read_kpi_source(SRC_UPDATE_TICKETS)

    market_stats_raw = read_kpi_source(SRC_TICKETS_BY_MARKET)
    build_market_raw = read_kpi_source(SRC_BUILD_TICKETS_BY_MARKET)
    update_market_raw = read_kpi_source(SRC_UPDATE_TICKETS_BY_MARKET)

    catalog_agents_raw = read_kpi_source(SRC_CATALOG_AGENTS_SCORES)
    studio_agents_raw = read_kpi_source(SRC_STUDIO_AGENTS_SCORES)

    return (
        rows,
        tickets,
        city_stats_raw,
        build_city_raw,
        update_city_raw,
        market_stats_raw,
        build_market_raw,
        update_market_raw,
        catalog_agents_raw,
        studio_agents_raw,
    )

t0 = time.time()
with st.spinner("Loading…"):
    (
        rows_df,
        tickets_df,
        city_stats_raw,
        build_city_raw,
        update_city_raw,
        market_stats_raw,
        build_market_raw,
        update_market_raw,
        catalog_agents_raw,
        studio_agents_raw,
    ) = load_data(mode)
st.caption(f"Loaded in {time.time()-t0:.2f}s")

# ============================================================
# FILTERS
# ============================================================
st.sidebar.markdown("## Filters")

ticket_view = st.sidebar.radio("Ticket Type View", ["All", "Build Tickets", "Update Tickets", "Existing Tickets"], index=0)

rows_df["year_month"] = rows_df["dt"].dt.to_period("M").astype(str)
available_months = sorted([m for m in rows_df["year_month"].dropna().unique() if m and m.lower() != "nan"])
sel_months = st.sidebar.multiselect("Month", available_months, default=[])

min_dt = rows_df["dt"].min()
max_dt = rows_df["dt"].max()
default_range = (min_dt.date() if pd.notna(min_dt) else None, max_dt.date() if pd.notna(max_dt) else None)
date_range = st.sidebar.date_input("Date range", value=default_range)

cities = sorted([c for c in rows_df["city"].dropna().astype(str).unique() if c and c.lower() != "nan"])
markets = sorted([m for m in rows_df["market"].dropna().astype(str).unique() if m and m.lower() != "nan"])

catalog_agents = sorted(rows_df[rows_df["is_non_studio"]]["agent_name"].dropna().astype(str).unique())
studio_agents = sorted(rows_df[rows_df["is_studio"]]["agent_name"].dropna().astype(str).unique())

sel_city = st.sidebar.multiselect("City", cities, default=[])
sel_market = st.sidebar.multiselect("Market", markets, default=[])
sel_catalog_agent = st.sidebar.multiselect("Catalogue Agent (Evaluation Name)", catalog_agents, default=[])
sel_studio_agent = st.sidebar.multiselect("Studio Agent (Evaluation Name)", studio_agents, default=[])

rf = rows_df.copy()
if "ticket_type" not in rf.columns:
    rf["ticket_type"] = "Other"

if sel_months:
    rf = rf[rf["year_month"].isin(sel_months)]

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
    rf = rf[~rf["is_non_studio"] | rf["agent_name"].isin(sel_catalog_agent)]
if sel_studio_agent:
    rf = rf[~rf["is_studio"] | rf["agent_name"].isin(sel_studio_agent)]

allowed_refs = set(rf["__ref__"].dropna().astype(str))
tf = tickets_df[tickets_df["__ref__"].astype(str).isin(allowed_refs)].copy()

def apply_months_to_kpi(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return raw
    if not sel_months:
        return raw
    if "__month__" not in raw.columns:
        return raw
    return raw[raw["__month__"].isin(sel_months)].copy()

city_stats_raw_f = apply_months_to_kpi(city_stats_raw)
build_city_raw_f = apply_months_to_kpi(build_city_raw)
update_city_raw_f = apply_months_to_kpi(update_city_raw)

market_stats_raw_f = apply_months_to_kpi(market_stats_raw)
build_market_raw_f = apply_months_to_kpi(build_market_raw)
update_market_raw_f = apply_months_to_kpi(update_market_raw)

catalog_agents_raw_f = apply_months_to_kpi(catalog_agents_raw)
studio_agents_raw_f = apply_months_to_kpi(studio_agents_raw)

city_stats_df = combine_kpis_by_level(city_stats_raw_f, mode, "City")
build_city_df = combine_kpis_by_level(build_city_raw_f, mode, "City")
update_city_df = combine_kpis_by_level(update_city_raw_f, mode, "City")

market_stats_df = combine_kpis_by_level(market_stats_raw_f, mode, "Market")
build_market_df = combine_kpis_by_level(build_market_raw_f, mode, "Market")
update_market_df = combine_kpis_by_level(update_market_raw_f, mode, "Market")

catalog_agent_scores_df = combine_kpis_by_level(catalog_agents_raw_f, mode, "Name")
studio_agent_scores_df = combine_kpis_by_level(studio_agents_raw_f, mode, "Name")

build_total = total_tickets_from_combined(build_city_df)
update_total = total_tickets_from_combined(update_city_df)

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

cat_avg_kpi = avg_score_from_combined_with_filter(catalog_agent_scores_df, "Name", [])
stu_avg_kpi = avg_score_from_combined_with_filter(studio_agent_scores_df, "Name", [])

if ticket_view == "Update Tickets":
    total_src_city = update_city_df
    total_src_market = update_market_df
elif ticket_view in ["Build Tickets", "Existing Tickets"]:
    total_src_city = build_city_df
    total_src_market = build_market_df
else:
    total_src_city = city_stats_df
    total_src_market = market_stats_df

if sel_city:
    tot_avg_kpi = avg_score_from_combined_with_filter(total_src_city, "City", sel_city)
elif sel_market:
    tot_avg_kpi = avg_score_from_combined_with_filter(total_src_market, "Market", sel_market)
else:
    tot_avg_kpi = avg_score_from_combined_with_filter(total_src_city, "City", [])

sent_back_catalog_tickets = cat_rows.groupby("__ref__")["sent_back_catalog"].max().fillna(0).gt(0).sum()
sent_back_studio_tickets = stu_rows.groupby("__ref__")["sent_back_catalog"].max().fillna(0).gt(0).sum()

with k1: kpi_card("Catalog QC", "—" if cat_avg_kpi is None else f"{cat_avg_kpi:.2f}%")
with k2: kpi_card("Studio QC", "—" if stu_avg_kpi is None else f"{stu_avg_kpi:.2f}%")
with k3: kpi_card("Total QC (Ticket)", "—" if tot_avg_kpi is None else f"{tot_avg_kpi:.2f}%")
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

    if sel_city and not build_city_df.empty and not update_city_df.empty:
        b = total_tickets_from_combined(build_city_df[build_city_df["City"].isin(sel_city)])
        u = total_tickets_from_combined(update_city_df[update_city_df["City"].isin(sel_city)])
        st.caption(f"Using KPI exports by City ({mode}) (City filter applied).")
    elif sel_market and not build_market_df.empty and not update_market_df.empty:
        b = total_tickets_from_combined(build_market_df[build_market_df["Market"].isin(sel_market)])
        u = total_tickets_from_combined(update_market_df[update_market_df["Market"].isin(sel_market)])
        st.caption(f"Using KPI exports by Market ({mode}) (Market filter applied).")
    else:
        b = build_total if build_total is not None else 0
        u = update_total if update_total is not None else 0
        st.caption(f"Using KPI export totals ({mode}).")

    tt = pd.DataFrame({"ticket_type": ["Build Tickets", "Update Tickets"], "count": [int(b or 0), int(u or 0)]})
    fig = px.pie(tt, names="ticket_type", values="count", hole=0.55)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("### Avg QC Score by City")

    if ticket_view == "Update Tickets":
        src = update_city_df
        st.caption(f"Update Tickets by City ({mode}).")
    elif ticket_view in ["Build Tickets", "Existing Tickets"]:
        src = build_city_df
        st.caption(f"Build Tickets by City ({mode}).")
    else:
        src = city_stats_df
        st.caption(f"Scores by City ({mode}).")

    if src is None or src.empty or "avg_score" not in src.columns:
        st.info("No city score data found in KPI exports.")
    else:
        cm = src.copy()
        if sel_city:
            cm = cm[cm["City"].isin(sel_city)]
        cm = cm.dropna(subset=["avg_score"]).sort_values("avg_score", ascending=False).head(20)
        if cm.empty:
            st.info("No city score data for current filter.")
        else:
            fig = px.bar(cm, x="City", y="avg_score")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="Avg QC Score")
            st.plotly_chart(fig, use_container_width=True)

with c3:
    st.markdown("### Tickets by City")

    if ticket_view == "Update Tickets":
        src = update_city_df
    elif ticket_view in ["Build Tickets", "Existing Tickets"]:
        src = build_city_df
    else:
        src = city_stats_df

    if src is None or src.empty or "tickets" not in src.columns:
        st.info("No city ticket data found in KPI exports.")
    else:
        cm = src.copy()
        if sel_city:
            cm = cm[cm["City"].isin(sel_city)]
        cm = cm.sort_values("tickets", ascending=False).head(20)
        if cm.empty:
            st.info("No city ticket data for current filter.")
        else:
            fig = px.bar(cm, x="City", y="tickets")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="Tickets")
            st.plotly_chart(fig, use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ============================================================
# MARKET INSIGHTS
# ============================================================
st.markdown("## Market Insights (KPI Exports)")
m1, m2 = st.columns(2)

with m1:
    st.markdown("### Avg QC Score by Market")

    if ticket_view == "Update Tickets":
        src = update_market_df
        st.caption(f"Update Tickets by Market ({mode}).")
    elif ticket_view in ["Build Tickets", "Existing Tickets"]:
        src = build_market_df
        st.caption(f"Build Tickets by Market ({mode}).")
    else:
        src = market_stats_df
        st.caption(f"Scores by Market ({mode}).")

    if src is None or src.empty or "avg_score" not in src.columns:
        st.info("No market score data found in KPI exports.")
    else:
        mm = src.copy()
        if sel_market:
            mm = mm[mm["Market"].isin(sel_market)]
        mm = mm.dropna(subset=["avg_score"]).sort_values("avg_score", ascending=False).head(20)
        if mm.empty:
            st.info("No market score data for current filter.")
        else:
            fig = px.bar(mm, x="Market", y="avg_score")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="Avg QC Score")
            st.plotly_chart(fig, use_container_width=True)

with m2:
    st.markdown("### Tickets by Market")

    if ticket_view == "Update Tickets":
        src = update_market_df
    elif ticket_view in ["Build Tickets", "Existing Tickets"]:
        src = build_market_df
    else:
        src = market_stats_df

    if src is None or src.empty or "tickets" not in src.columns:
        st.info("No market ticket data found in KPI exports.")
    else:
        mm = src.copy()
        if sel_market:
            mm = mm[mm["Market"].isin(sel_market)]
        mm = mm.sort_values("tickets", ascending=False).head(20)
        if mm.empty:
            st.info("No market ticket data for current filter.")
        else:
            fig = px.bar(mm, x="Market", y="tickets")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="Tickets")
            st.plotly_chart(fig, use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ============================================================
# TABLE + LINE CHART
# ============================================================
left, right = st.columns([0.62, 0.38])

cat_flag = cat_rows.groupby("__ref__")["sent_back_catalog"].max().fillna(0).gt(0)
stu_flag = stu_rows.groupby("__ref__")["sent_back_catalog"].max().fillna(0).gt(0)

tf["Sent Back → Catalog"] = tf["__ref__"].map(cat_flag).fillna(False).map(lambda x: "Yes" if x else "No")
tf["Sent Back → Studio"] = tf["__ref__"].map(stu_flag).fillna(False).map(lambda x: "Yes" if x else "No")

with left:
    st.markdown("## Tickets")
    show_cols = [
        "ticket_id",
        "ticket_type",
        "ticket_type_raw",
        "city",
        "market",
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
    t = rf.dropna(subset=["dt"]).copy()
    if t.empty:
        st.info("No evaluation datetime values available for this filter.")
    else:
        t["date_only"] = t["dt"].dt.date
        daily = t.groupby("date_only", as_index=False)["score_pct"].mean().sort_values("date_only")
        fig = px.line(daily, x="date_only", y="score_pct", markers=True)
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ============================================================
# ★ DAILY TRENDS  (from Evaluation Report)
# ============================================================
st.markdown("## 📅 Daily Trends")

daily_df = rf.dropna(subset=["dt"]).copy()
daily_df["date_only"] = daily_df["dt"].dt.date

if daily_df.empty:
    st.info("No evaluation data available for the current filters.")
else:
    d_col1, d_col2 = st.columns([0.5, 0.5])

    all_forms = sorted(daily_df["form_name"].dropna().unique().tolist())
    with d_col1:
        sel_forms = st.multiselect(
            "Filter by Form Type",
            all_forms,
            default=[],
            key="daily_forms",
            help="Leave empty to include all forms",
        )
    with d_col2:
        # Agent multiselect for spotlight — top 10 pre-selected by volume
        all_daily_agents = sorted(daily_df["agent_name"].dropna().unique().tolist())
        top10_default = daily_df["agent_name"].value_counts().head(10).index.tolist()
        sel_daily_agents = st.multiselect(
            "Highlight Agents (leave empty = all as one line)",
            all_daily_agents,
            default=[],
            key="daily_agents",
            help="Select specific agents to plot individual lines. Leave empty for an overall average.",
        )

    ddf = daily_df.copy()
    if sel_forms:
        ddf = ddf[ddf["form_name"].isin(sel_forms)]

    if ddf.empty:
        st.info("No data for selected form types.")
    else:
        # ── Overall avg score line chart (always shown)
        st.markdown("### Avg QC Score by Day")

        if not sel_daily_agents:
            # Single overall line
            agg = ddf.groupby("date_only", as_index=False)["score_pct"].mean().sort_values("date_only")
            fig = px.line(
                agg, x="date_only", y="score_pct", markers=True,
                labels={"date_only": "Date", "score_pct": "Avg Score (%)"},
            )
            fig.update_traces(line_color="#4f8ef7", line_width=2.5, marker_size=7)
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Per-agent lines for selected agents + overall dashed reference
            overall_agg = ddf.groupby("date_only", as_index=False)["score_pct"].mean().sort_values("date_only")
            agent_agg = (
                ddf[ddf["agent_name"].isin(sel_daily_agents)]
                .groupby(["date_only", "agent_name"], as_index=False)["score_pct"].mean()
                .sort_values("date_only")
            )

            fig = px.line(
                agent_agg, x="date_only", y="score_pct", color="agent_name",
                markers=True,
                labels={"date_only": "Date", "score_pct": "Avg Score (%)", "agent_name": "Agent"},
            )
            # Add overall as a dashed grey reference line
            fig.add_scatter(
                x=overall_agg["date_only"],
                y=overall_agg["score_pct"],
                mode="lines",
                name="Overall Avg",
                line=dict(color="grey", dash="dash", width=1.5),
            )
            fig.update_layout(
                height=440,
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(font_size=11),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Per-agent daily summary table (only when agents selected)
        if sel_daily_agents:
            st.markdown("### Agent Daily Summary")
            agent_daily = (
                ddf[ddf["agent_name"].isin(sel_daily_agents)]
                .groupby(["date_only", "agent_name"], as_index=False)
                .agg(
                    evaluations=("__ref__", "count"),
                    unique_tickets=("__ref__", "nunique"),
                    avg_score=("score_pct", "mean"),
                    min_score=("score_pct", "min"),
                    max_score=("score_pct", "max"),
                )
                .sort_values(["date_only", "agent_name"], ascending=[False, True])
            )
            st.dataframe(
                agent_daily,
                use_container_width=True,
                height=420,
                column_config={
                    "date_only": st.column_config.DateColumn("Date"),
                    "agent_name": "Agent",
                    "evaluations": st.column_config.NumberColumn("Evaluations"),
                    "unique_tickets": st.column_config.NumberColumn("Unique Tickets"),
                    "avg_score": st.column_config.NumberColumn("Avg Score", format="%.2f%%"),
                    "min_score": st.column_config.NumberColumn("Min", format="%.2f%%"),
                    "max_score": st.column_config.NumberColumn("Max", format="%.2f%%"),
                },
            )
            st.download_button(
                "⬇️ Download agent daily data",
                data=agent_daily.to_csv(index=False).encode("utf-8"),
                file_name="agent_daily_scores.csv",
                mime="text/csv",
            )
        else:
            # Overall daily table when no specific agents selected
            st.markdown("### Daily Summary")
            daily_summary = (
                ddf.groupby("date_only", as_index=False)
                .agg(
                    evaluations=("__ref__", "count"),
                    unique_tickets=("__ref__", "nunique"),
                    avg_score=("score_pct", "mean"),
                    min_score=("score_pct", "min"),
                    max_score=("score_pct", "max"),
                    agents=("agent_name", "nunique"),
                )
                .sort_values("date_only", ascending=False)
            )
            st.dataframe(
                daily_summary,
                use_container_width=True,
                height=400,
                column_config={
                    "date_only": st.column_config.DateColumn("Date"),
                    "evaluations": st.column_config.NumberColumn("Evaluations"),
                    "unique_tickets": st.column_config.NumberColumn("Unique Tickets"),
                    "avg_score": st.column_config.NumberColumn("Avg Score", format="%.2f%%"),
                    "min_score": st.column_config.NumberColumn("Min Score", format="%.2f%%"),
                    "max_score": st.column_config.NumberColumn("Max Score", format="%.2f%%"),
                    "agents": st.column_config.NumberColumn("Agents"),
                },
            )
            st.download_button(
                "⬇️ Download daily summary",
                data=daily_summary.to_csv(index=False).encode("utf-8"),
                file_name="daily_summary.csv",
                mime="text/csv",
            )
# ============================================================
# AGENT SCORES
# ============================================================
st.markdown("## Agent Scores")
tab1, tab2 = st.tabs(["Catalog Agents Scores", "Studio Agents Scores"])

def render_agent_scores(df: pd.DataFrame, title_hint: str):
    if df is None or df.empty:
        st.info(f"{title_hint}: no data found.")
        return

    a = df.copy()
    if "Name" not in a.columns:
        st.warning(f"{title_hint}: Name column not found. Showing raw preview.")
        st.dataframe(a.head(50), use_container_width=True)
        return

    view_cols = ["Name"]
    if "tickets" in a.columns:
        view_cols.append("tickets")
    if "avg_score" in a.columns:
        view_cols.append("avg_score")

    a = a[view_cols].copy()
    a = a.rename(columns={"tickets": "Tickets", "avg_score": "Avg QC Score"})

    if "Tickets" in a.columns:
        a["Tickets"] = pd.to_numeric(a["Tickets"], errors="coerce").fillna(0).astype(int)
    if "Avg QC Score" in a.columns:
        a["Avg QC Score"] = pd.to_numeric(a["Avg QC Score"], errors="coerce")

    if "Avg QC Score" in a.columns:
        a = a.dropna(subset=["Avg QC Score"]).sort_values("Avg QC Score", ascending=False)

    st.dataframe(a, use_container_width=True, height=520)

with tab1:
    render_agent_scores(catalog_agent_scores_df, "Catalog Agents Scores")

with tab2:
    render_agent_scores(studio_agent_scores_df, "Studio Agents Scores")

st.markdown("<br/>", unsafe_allow_html=True)

# ============================================================
# ★ CATALOG MISTAKES  (folder / zip sources)
# ============================================================
st.markdown("## 🔍 Catalog Mistakes")
)

# Load all available files — silently skip missing ones
mistakes_data: dict[str, pd.DataFrame] = {}
for label, stem in MISTAKES_FILES.items():
    df_m = load_mistakes_file(stem)
    if df_m is not None and not df_m.empty:
        mistakes_data[label] = df_m

if not mistakes_data:
    st.info(
        "No catalog mistakes files found yet. "
        "Place your files next to `app.py` using these names: "
        + ", ".join(f"`{s}.xlsx`" for s in MISTAKES_FILES.values())
    )
else:
    tab_labels = list(mistakes_data.keys())
    mistake_tabs = st.tabs(tab_labels)

    for i, label in enumerate(tab_labels):
        with mistake_tabs[i]:
            df_m = mistakes_data[label].copy()

            has_attr = "Attribute" in df_m.columns
            has_failures = "Total Failures" in df_m.columns
            has_pct = "Failure %" in df_m.columns

            if not has_attr:
                st.dataframe(df_m, use_container_width=True)
                continue

            # Controls row
            ctrl1, ctrl2, ctrl3 = st.columns([0.35, 0.35, 0.30])
            with ctrl1:
                top_n = st.slider("Show top N attributes", 5, min(50, len(df_m)), 15, key=f"topn_{i}")
            with ctrl2:
                sort_options = []
                if has_failures: sort_options.append("Total Failures")
                if has_pct: sort_options.append("Failure %")
                sort_by = st.radio("Sort by", sort_options, horizontal=True, key=f"sortby_{i}")
            with ctrl3:
                chart_type = st.radio("Chart", ["Horizontal Bar", "Bar"], horizontal=True, key=f"chart_{i}")

            sort_col = sort_by if sort_by in df_m.columns else (sort_options[0] if sort_options else "Total Failures")
            df_plot = df_m.dropna(subset=[sort_col]).sort_values(sort_col, ascending=False).head(top_n)

            # KPI cards
            if has_failures and has_pct:
                total_fail = int(df_m["Total Failures"].sum())
                top_row = df_m.sort_values("Total Failures", ascending=False).iloc[0]
                top_attr = top_row["Attribute"]
                top_pct = top_row["Failure %"]

                km1, km2, km3 = st.columns(3)
                with km1:
                    kpi_card("Total Failures", f"{total_fail:,}")
                with km2:
                    kpi_card("Top Failure Attribute", str(top_attr)[:40])
                with km3:
                    kpi_card("Top Failure Rate", f"{top_pct:.2f}%" if top_pct is not None else "—")
                st.markdown("<br/>", unsafe_allow_html=True)

            y_col = sort_col
            y_label = sort_col

            if chart_type == "Bar":
                fig = px.bar(
                    df_plot, x="Attribute", y=y_col, text=y_col,
                    labels={"Attribute": "", y_col: y_label},
                    color=y_col, color_continuous_scale="Reds",
                )
                fig.update_traces(
                    texttemplate="%{text:.1f}" + ("%" if "%" in y_label else ""),
                    textposition="outside",
                )
                fig.update_layout(
                    height=440, margin=dict(l=10, r=10, t=30, b=120),
                    coloraxis_showscale=False, xaxis_tickangle=-35,
                )
            else:
                df_plot_h = df_plot.sort_values(y_col, ascending=True)
                fig = px.bar(
                    df_plot_h, x=y_col, y="Attribute", orientation="h", text=y_col,
                    labels={"Attribute": "", y_col: y_label},
                    color=y_col, color_continuous_scale="Reds",
                )
                fig.update_traces(
                    texttemplate="%{text:.1f}" + ("%" if "%" in y_label else ""),
                    textposition="outside",
                )
                fig.update_layout(
                    height=max(400, top_n * 30),
                    margin=dict(l=10, r=90, t=30, b=10),
                    coloraxis_showscale=False,
                )

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 Full data table"):
                st.dataframe(df_m, use_container_width=True, height=400)
                st.download_button(
                    f"⬇️ Download {label}",
                    data=df_m.to_csv(index=False).encode("utf-8"),
                    file_name=f"mistakes_{label.lower().replace(' ', '_').replace('–','_')}.csv",
                    mime="text/csv",
                    key=f"dl_mistakes_{i}",
                )
