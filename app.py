import re
import html
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# =========================
# FILES (must be in same folder as app.py)
# =========================
BASE_PATH = "istep_data.csv"
REPORT_PATH = "Evaluation Report Istep - Report.csv"

CITY_STATS_PATH = "Tickets by City.csv"
BUILD_TICKETS_PATH = "Build Tickets.csv"
UPDATE_TICKETS_PATH = "Update Tickets.csv"

CATALOG_AGENT_SCORES_PATH = "Catalog Agents Scores.csv"
STUDIO_AGENT_SCORES_PATH = "Studio Agents Scores.csv"

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

def file_exists_or_stop(path: str, label: str):
    if not Path(path).exists():
        st.error(f"Missing file: **{path}** ({label}). Put it in the SAME folder as app.py.")
        st.stop()

def read_kpi_csv(path: str) -> pd.DataFrame:
    """
    Reads KPI exports robustly:
    - If 2-row header exists → flatten
    - Else normal CSV → normalize headers, drop 'Unnamed', force unique
    """
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

def extract_total_tickets_from_report(df: pd.DataFrame, mode: str = "Monthly") -> int | None:
    """
    Extract total tickets from Build/Update report:
    - Prefer a 'Total' row if exists
    - Else sum the tickets column
    """
    candidates = [
        f"Total Tickets Resolved {mode}",
        "Total Tickets Resolved",
        f"Total Tickets {mode}",
        "Total Tickets",
        f"Tickets {mode}",
        "Tickets",
        "Ticket Count",
        "ticket_count",
    ]
    tcol = safe_col(df, candidates)

    if not tcol:
        num_cols = []
        for c in df.columns:
            s = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
            if s.notna().sum() > 0:
                num_cols.append(c)
        if not num_cols:
            return None
        tcol = num_cols[0]

    key_col = safe_col(df, ["Name", "City", "Market"])
    if key_col:
        mask_total = df[key_col].astype(str).str.strip().str.lower().eq("total")
        if mask_total.any():
            v = pd.to_numeric(
                df.loc[mask_total, tcol].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            ).dropna()
            if len(v):
                return int(v.iloc[0])

    vals = pd.to_numeric(df[tcol].astype(str).str.replace(",", "", regex=False), errors="coerce").dropna()
    if vals.empty:
        return None
    return int(vals.sum())

def city_metrics_from_report(df: pd.DataFrame, mode: str) -> pd.DataFrame | None:
    """
    Returns: City | tickets | avg_score (if columns exist)
    Works for Tickets by City / Build Tickets / Update Tickets reports.
    """
    city_col = safe_col(df, ["City", "city", "CITY"])
    tickets_col = safe_col(
        df,
        [
            f"Total Tickets Resolved {mode}",
            "Total Tickets Resolved",
            f"Tickets {mode}",
            "Tickets",
            "Ticket Count",
            "ticket_count",
        ],
    )
    score_col = safe_col(
        df,
        [
            f"Total Average QC Score {mode}",
            "Total Average QC Score",
            f"Average QC Score {mode}",
            "Average QC Score",
        ],
    )

    if not city_col:
        return None

    cols = [city_col]
    if tickets_col:
        cols.append(tickets_col)
    if score_col:
        cols.append(score_col)

    out = df[cols].copy()

    rename_map = {city_col: "City"}
    if tickets_col:
        rename_map[tickets_col] = "tickets"
    if score_col:
        rename_map[score_col] = "avg_score"
    out = out.rename(columns=rename_map)

    out["City"] = out["City"].astype(str).str.strip()
    out = out[~out["City"].str.lower().isin(["total", "unknown", "nan", ""])]

    if "tickets" in out.columns:
        out["tickets"] = pd.to_numeric(out["tickets"], errors="coerce").fillna(0).astype(int)
    if "avg_score" in out.columns:
        out["avg_score"] = pd.to_numeric(out["avg_score"], errors="coerce")

    out = out.dropna(axis=0, how="all")
    return out

# Ticket type from Subject (overall export)
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

    # Total bucket (keep as you want)
    if (tt == "Build Tickets") and (fn in BUILD_MAIN_FORMS):
        return "Total"
    if (tt == "Update Tickets") and (fn == "Update scorecard"):
        return "Total"
    if (tt == "Existing Tickets") and (fn in EXISTING_MAIN_FORMS):
        return "Total"

    return None

# ============================================================
# DATA
# ============================================================
st.sidebar.markdown("## Data")
if st.sidebar.button("Refresh data now"):
    st.cache_data.clear()

mode = st.sidebar.radio("Report mode (for City/Build/Update/Agent files)", ["Monthly", "Weekly"], index=0)

@st.cache_data(show_spinner=False, ttl=600)
def load_data(mode: str):
    file_exists_or_stop(BASE_PATH, "overall / metadata")
    file_exists_or_stop(REPORT_PATH, "evaluation report (scores)")

    file_exists_or_stop(CITY_STATS_PATH, "tickets by city")
    file_exists_or_stop(BUILD_TICKETS_PATH, "build tickets")
    file_exists_or_stop(UPDATE_TICKETS_PATH, "update tickets")
    file_exists_or_stop(CATALOG_AGENT_SCORES_PATH, "catalog agent scores")
    file_exists_or_stop(STUDIO_AGENT_SCORES_PATH, "studio agent scores")

    # ---- Overall (only for subject/city/market + ticket type mapping)
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

    # ---- Evaluation report (SCORES SOURCE OF TRUTH)
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

    # Ticket type mapping from overall (critical for filters + Total logic)
    ref_to_type = dict(zip(overall_small["__ref__"], overall_small["ticket_type"]))
    ev["ticket_type"] = ev["__ref__"].map(ref_to_type).fillna("Other")

    # Buckets + flags
    ev["bucket"] = ev.apply(lambda r: bucket_for_row(r["ticket_type"], r["form_name"]), axis=1)
    ev["is_studio"] = ev["form_name"].isin(STUDIO_FORMS)
    ev["is_non_studio"] = ~ev["is_studio"]  # Catalog QC = everything except studio

    # Merge in city/market/subject without overwriting ticket_type
    rows = ev.merge(
        overall_small[["__ref__", "Reference ID", "Subject", "ticket_type_raw", "city", "market"]],
        on="__ref__",
        how="left",
        suffixes=("", "_overall"),
    )

    if "ticket_type" not in rows.columns:
        rows["ticket_type"] = "Other"
    rows["market"] = rows["market"].fillna("Unknown")
    rows["city"] = rows["city"].fillna("Unknown")

    # Ticket-level (for table)
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

    # Source of truth files
    city_stats = read_kpi_csv(CITY_STATS_PATH)
    build_df = read_kpi_csv(BUILD_TICKETS_PATH)   # includes Existing
    update_df = read_kpi_csv(UPDATE_TICKETS_PATH)

    build_total = extract_total_tickets_from_report(build_df, mode=mode)
    update_total = extract_total_tickets_from_report(update_df, mode=mode)

    catalog_agent_scores = read_kpi_csv(CATALOG_AGENT_SCORES_PATH)
    studio_agent_scores = read_kpi_csv(STUDIO_AGENT_SCORES_PATH)

    return (
        rows,
        tickets,
        city_stats,
        build_df,
        update_df,
        build_total,
        update_total,
        catalog_agent_scores,
        studio_agent_scores,
    )

t0 = time.time()
with st.spinner("Loading…"):
    (
        rows_df,
        tickets_df,
        city_stats_df,
        build_df,
        update_df,
        build_total,
        update_total,
        catalog_agent_scores_df,
        studio_agent_scores_df,
    ) = load_data(mode)
st.caption(f"Loaded in {time.time()-t0:.2f}s")

# ============================================================
# FILTERS
# ============================================================
st.sidebar.markdown("## Filters")

ticket_view = st.sidebar.radio("Ticket Type View", ["All", "Build Tickets", "Update Tickets", "Existing Tickets"], index=0)

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

filters_applied = any([ticket_view != "All", sel_city, sel_market, sel_catalog_agent, sel_studio_agent]) or (
    isinstance(date_range, tuple) and len(date_range) == 2 and (date_range[0] is not None or date_range[1] is not None)
)

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
# KPIs (Scores from Evaluation Report ONLY)
# ============================================================
k1, k2, k3, k4, k5 = st.columns(5)

cat_rows = rf[rf["is_non_studio"]].copy()
stu_rows = rf[rf["is_studio"]].copy()
tot_rows = rf[rf["bucket"] == "Total"].copy()

cat_avg = mean_pct(cat_rows["score_pct"])
stu_avg = mean_pct(stu_rows["score_pct"])
tot_avg = mean_pct(tot_rows["score_pct"])

sent_back_catalog_tickets = cat_rows.groupby("__ref__")["sent_back_catalog"].max().fillna(0).gt(0).sum()
sent_back_studio_tickets = stu_rows.groupby("__ref__")["sent_back_catalog"].max().fillna(0).gt(0).sum()

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

# -------------------------
# Ticket type split
# -------------------------
with c1:
    st.markdown("### Ticket type split")

    if (not filters_applied) and (build_total is not None) and (update_total is not None):
        # Build_total already INCLUDES Existing (as you said)
        counts = {
            "Build Tickets (incl. Existing)": int(build_total),
            "Update Tickets": int(update_total),
        }

        # Add only non-(Build/Update/Existing) buckets from evaluation-derived tickets
        rest = tf["ticket_type"].fillna("Other").value_counts().to_dict()
        for k, v in rest.items():
            if k in ["Build Tickets", "Update Tickets", "Existing Tickets"]:
                continue
            counts[k] = int(v)

        tt = pd.DataFrame({"ticket_type": list(counts.keys()), "count": list(counts.values())})
        st.caption(f"Using Build/Update totals from KPI exports ({mode}).")
    else:
        tt = tf["ticket_type"].fillna("Other").value_counts().reset_index()
        tt.columns = ["ticket_type", "count"]
        st.caption("Filtered view: split recalculated from filtered tickets.")

    fig = px.pie(tt, names="ticket_type", values="count", hole=0.55)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Avg QC Score by City (source-of-truth)
# -------------------------
with c2:
    st.markdown("### Avg QC Score by City")

    # Choose report source based on ticket filter
    if ticket_view == "Update Tickets":
        src = update_df
        st.caption(f"Source: Update Tickets report ({mode}).")
    elif ticket_view in ["Build Tickets", "Existing Tickets"]:
        # Existing is included in build report (your rule)
        src = build_df
        st.caption(f"Source: Build Tickets report (includes Existing) ({mode}).")
    else:
        src = city_stats_df
        st.caption(f"Source: Tickets by City report ({mode}).")

    cm = city_metrics_from_report(src, mode)

    if cm is None or "avg_score" not in cm.columns:
        st.info("Couldn’t detect the Avg QC Score column in this report export.")
    else:
        cm = cm.dropna(subset=["avg_score"]).sort_values("avg_score", ascending=False).head(20)
        if cm.empty:
            st.info("No city score data found.")
        else:
            fig = px.bar(cm, x="City", y="avg_score")
            fig.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="",
                yaxis_title="Avg QC Score",
            )
            st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Tickets by City (source-of-truth)
# -------------------------
with c3:
    st.markdown("### Tickets by City")

    if ticket_view == "Update Tickets":
        src = update_df
        st.caption(f"Source: Update Tickets report ({mode}).")
    elif ticket_view in ["Build Tickets", "Existing Tickets"]:
        src = build_df
        st.caption(f"Source: Build Tickets report (includes Existing) ({mode}).")
    else:
        src = city_stats_df
        st.caption(f"Source: Tickets by City report ({mode}).")

    cm = city_metrics_from_report(src, mode)

    if cm is None or "tickets" not in cm.columns:
        st.info("Couldn’t detect the Tickets column in this report export.")
    else:
        cm = cm.sort_values("tickets", ascending=False).head(20)
        if cm.empty:
            st.info("No city ticket data found.")
        else:
            fig = px.bar(cm, x="City", y="tickets")
            fig.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="",
                yaxis_title="Tickets",
            )
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
# AGENT SCORES (SOURCE OF TRUTH ONLY)
# ============================================================
st.markdown("## Agent Scores")

tab1, tab2 = st.tabs(["Catalog Agents Scores", "Studio Agents Scores"])

def render_agent_scores(df: pd.DataFrame, title_hint: str):
    a = df.copy()
    name_c = safe_col(a, ["Name", "Agent", "agent", "Agent Name"])
    tickets_c = safe_col(a, [f"Total Tickets Resolved {mode}", "Total Tickets Resolved", "Tickets", "Ticket Count"])
    score_c = safe_col(a, [f"Total Average QC Score {mode}", "Total Average QC Score", "Avg QC Score", "Average QC Score"])

    if not name_c or not score_c:
        st.warning(f"{title_hint}: columns not recognized. Keep original export format.")
        st.dataframe(a.head(50), use_container_width=True)
        return

    view_cols = [c for c in [name_c, tickets_c, score_c] if c]
    a = a[view_cols].copy()

    new_cols = ["Name"]
    if tickets_c:
        new_cols.append("Tickets")
    new_cols.append("Avg QC Score")
    a.columns = make_unique(new_cols)

    a["Avg QC Score"] = pd.to_numeric(a["Avg QC Score"], errors="coerce")
    if "Tickets" in a.columns:
        a["Tickets"] = pd.to_numeric(a["Tickets"], errors="coerce")

    a = a.dropna(subset=["Avg QC Score"]).sort_values("Avg QC Score", ascending=False)
    st.dataframe(a, use_container_width=True, height=520)

with tab1:
    render_agent_scores(catalog_agent_scores_df, "Catalog Agents Scores")

with tab2:
    render_agent_scores(studio_agent_scores_df, "Studio Agents Scores")
