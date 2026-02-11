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
# CARD CLASSIFICATION (THIS IS YOUR LAW)
# ============================================================
STUDIO_CARDS = {"Studio scorecard", "Images Update"}

UPDATE_CARDS = {
    "Price Update",
    "Update scorecard",
    "Location Update",
    "Tags Update",
    "Operational Hours Update",
}

BUILD_CARDS = {"New Points/ Builds"}
EXISTING_CARDS = {"Existing Company and Brand - New Outlet"}

# This appears in build/existing sometimes and should count as CATALOG for those ticket types
CATALOGUE_SCORECARD = "Catalogue scorecard"

def ticket_type_from_cards(card_set: set[str]) -> str:
    # Determine ticket type purely from card presence
    if any(c in BUILD_CARDS for c in card_set):
        return "Build Tickets"
    if any(c in EXISTING_CARDS for c in card_set):
        return "Existing Tickets"
    if any(c in UPDATE_CARDS for c in card_set):
        return "Update Tickets"
    return "Other"

def is_catalog_card(form: str, ticket_type: str) -> bool:
    # Catalog QC rules:
    # - Update tickets: update cards are catalog-side
    # - Build tickets: build cards + catalogue scorecard are catalog-side
    # - Existing tickets: existing cards + catalogue scorecard are catalog-side
    if ticket_type == "Update Tickets":
        return form in UPDATE_CARDS
    if ticket_type == "Build Tickets":
        return (form in BUILD_CARDS) or (form == CATALOGUE_SCORECARD)
    if ticket_type == "Existing Tickets":
        return (form in EXISTING_CARDS) or (form == CATALOGUE_SCORECARD)
    return False

def is_studio_card(form: str) -> bool:
    return form in STUDIO_CARDS

# ============================================================
# LOAD
# ============================================================
st.sidebar.markdown("## Data")
if st.sidebar.button("🔄 Refresh"):
    st.cache_data.clear()

@st.cache_data(show_spinner=False, ttl=600)
def load_report() -> pd.DataFrame:
    r = pd.read_csv(REPORT_PATH, encoding="utf-8", low_memory=False)
    r.columns = [normalize_header(c) for c in r.columns]
    r.columns = make_unique(r.columns)
    return r

@st.cache_data(show_spinner=False, ttl=600)
def load_base_meta() -> pd.DataFrame:
    # base might be csv or xls(x) — try csv first then excel
    try:
        b = pd.read_csv(BASE_PATH, low_memory=False)
    except Exception:
        b = pd.read_excel(BASE_PATH, sheet_name=0)

    b.columns = [normalize_header(c) for c in b.columns]
    b.columns = make_unique(b.columns)
    return b

def collapse_base_meta(b: pd.DataFrame) -> pd.DataFrame:
    ref = safe_col(b, ["Reference ID", "Ticket ID"])
    if not ref:
        raise RuntimeError("Base file must include Reference ID")

    subject = safe_col(b, ["Subject", "Ticket Type"])
    city = safe_col(b, ["Catalogue City", "City"])
    market = safe_col(b, ["Catalogue Market", "Market"])
    c_agent = safe_col(b, ["Catalogue Name", "Catalogue Agent Name"])
    s_agent = safe_col(b, ["Studio Name", "Studio Agent Name"])
    c_dt = safe_col(b, ["Catalogue Date & Time", "Ticket Creation Time"])
    s_dt = safe_col(b, ["Studio Date & Time", "Ticket Creation Time__2"])

    agg = {}
    if subject: agg[subject] = first_non_empty
    if city: agg[city] = first_non_empty
    if market: agg[market] = first_non_empty
    if c_agent: agg[c_agent] = unique_join
    if s_agent: agg[s_agent] = first_non_empty
    if c_dt: agg[c_dt] = first_non_empty
    if s_dt: agg[s_dt] = first_non_empty

    out = b.groupby(ref, as_index=False).agg(agg)
    out.rename(columns={ref: "Reference ID"}, inplace=True)
    return out

def build_ticket_scores(report: pd.DataFrame) -> pd.DataFrame:
    ref = safe_col(report, ["Reference ID"])
    form = safe_col(report, ["Form Name"])
    score = safe_col(report, ["Score"])
    sent_back = safe_col(report, ["Sent Back To Catalog"])
    agent = safe_col(report, ["Name", "Agent Name"])
    uid = safe_col(report, ["User Id", "User ID", "Agent User Id"])

    if not ref or not form or not score:
        raise RuntimeError("Report CSV must include: Reference ID, Form Name, Score")

    r = report.copy()
    r["ref"] = r[ref].astype(str).str.strip()
    r["form"] = r[form].astype(str).str.strip()
    r["score_pct"] = _to_pct(r[score])
    r["sent_back"] = pd.to_numeric(r[sent_back], errors="coerce").fillna(0) if sent_back else 0
    r["agent"] = r[agent].astype(str).str.strip() if agent else ""
    r["uid"] = r[uid].astype(str).str.strip() if uid else ""

    # de-dupe duplicates inside same ticket/card/agent
    r = r.drop_duplicates(subset=["ref", "form", "agent", "uid", "score_pct", "sent_back"])

    # ticket type by card presence
    card_sets = r.groupby("ref")["form"].apply(lambda x: set(x.dropna().tolist())).to_dict()
    r["ticket_type"] = r["ref"].map(lambda x: ticket_type_from_cards(card_sets.get(x, set())))

    # compute per ticket
    def agg_ticket(g: pd.DataFrame) -> pd.Series:
        ttype = g["ticket_type"].iloc[0]
        # Studio avg from studio cards only
        studio_vals = g.loc[g["form"].map(is_studio_card), "score_pct"]
        studio_avg = studio_vals.mean() if studio_vals.dropna().size else pd.NA

        # Catalog avg from catalog-relevant cards (depends on ticket type)
        catalog_vals = g.loc[g["form"].map(lambda f: is_catalog_card(f, ttype)), "score_pct"]
        catalog_avg = catalog_vals.mean() if catalog_vals.dropna().size else pd.NA

        # Total ticket score:
        # - Build: from Build card
        # - Existing: from Existing card
        # - Update: from Update scorecard
        if ttype == "Build Tickets":
            ticket_vals = g.loc[g["form"].isin(BUILD_CARDS), "score_pct"]
        elif ttype == "Existing Tickets":
            ticket_vals = g.loc[g["form"].isin(EXISTING_CARDS), "score_pct"]
        elif ttype == "Update Tickets":
            ticket_vals = g.loc[g["form"] == "Update scorecard", "score_pct"]
        else:
            ticket_vals = pd.Series(dtype=float)

        ticket_score = ticket_vals.dropna().iloc[0] if ticket_vals.dropna().size else pd.NA

        sent_total = int(pd.to_numeric(g["sent_back"], errors="coerce").fillna(0).sum())

        return pd.Series(
            {
                "ticket_type": ttype,
                "catalog_score_pct": catalog_avg,
                "studio_score_pct": studio_avg,
                "ticket_score_pct": ticket_score,
                "sent_back_total": sent_total,
            }
        )

    out = r.groupby("ref", as_index=False).apply(agg_ticket)
    # groupby apply returns weird index sometimes
    out = out.reset_index(drop=True).rename(columns={"ref": "Reference ID"})
    return out

# ============================================================
# BUILD FINAL DF
# ============================================================
t0 = time.time()
with st.spinner("Loading…"):
    rep = load_report()
    scores = build_ticket_scores(rep)

    base_raw = load_base_meta()
    base = collapse_base_meta(base_raw)

    df = base.merge(scores, on="Reference ID", how="left")

    # normalize meta fields
    subj_col = safe_col(df, ["Subject", "Ticket Type"])
    city_col = safe_col(df, ["Catalogue City", "City"])
    market_col = safe_col(df, ["Catalogue Market", "Market"])
    c_dt = safe_col(df, ["Catalogue Date & Time", "Ticket Creation Time"])
    s_dt = safe_col(df, ["Studio Date & Time", "Ticket Creation Time__2"])

    df["ticket_id"] = df["Reference ID"].astype(str)
    df["ticket_type_raw"] = df[subj_col].astype(str) if subj_col else ""
    df["city"] = df[city_col] if city_col else pd.NA
    df["market"] = df[market_col] if market_col else pd.NA

    df["dt"] = pd.to_datetime(df[c_dt], errors="coerce") if c_dt else pd.NaT
    if s_dt:
        df["dt"] = df["dt"].fillna(pd.to_datetime(df[s_dt], errors="coerce"))
    df["date"] = df["dt"].dt.date

    # Remove Unknown city
    df.loc[df["city"].astype(str).str.strip().str.lower() == "unknown", "city"] = pd.NA

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

score_type = st.sidebar.selectbox("Score Type (charts)", ["Ticket Score", "Catalogue QC", "Studio QC"], index=0)
score_col = {
    "Ticket Score": "ticket_score_pct",
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
# HEADER
# ============================================================
left, right = st.columns([0.78, 0.22], vertical_alignment="bottom")
with left:
    st.markdown('<div class="qc-title">QC Scores Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="qc-sub">ALL SCORES from Evaluation Report CSV • Ticket type from cards • Studio/Catalog computed per your rules</div>', unsafe_allow_html=True)
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
# KPIs
# ============================================================
k1, k2, k3, k4, k5, k6 = st.columns(6)

catalog_avg = mean_pct(f["catalog_score_pct"])
studio_avg = mean_pct(f["studio_score_pct"])
ticket_avg = mean_pct(f["ticket_score_pct"])
sent_back_total = int(pd.to_numeric(f["sent_back_total"], errors="coerce").fillna(0).sum())

with k1: kpi_card("Catalogue QC", "—" if catalog_avg is None else f"{catalog_avg:.2f}%")
with k2: kpi_card("Studio QC", "—" if studio_avg is None else f"{studio_avg:.2f}%")
with k3: kpi_card("Ticket Score (Total)", "—" if ticket_avg is None else f"{ticket_avg:.2f}%")
with k4: kpi_card("Tickets", f"{len(f):,}")
with k5: kpi_card("Sent Back (total)", f"{sent_back_total:,}")
with k6: kpi_card("View", view_mode)

st.markdown("<br/>", unsafe_allow_html=True)

# ============================================================
# TABLE + SCORE CHANGE
# ============================================================
a, b = st.columns([0.62, 0.38])

with a:
    st.markdown("### Tickets")
    cols = [
        "ticket_id", "ticket_type", "ticket_type_raw",
        "catalog_score_pct", "studio_score_pct", "ticket_score_pct",
        "city", "market", "dt", "sent_back_total"
    ]
    cols = [c for c in cols if c in f.columns]
    st.dataframe(
        f[cols].sort_values("dt", ascending=False),
        use_container_width=True,
        height=460,
        column_config={
            "catalog_score_pct": st.column_config.NumberColumn("Catalogue QC", format="%.2f%%"),
            "studio_score_pct": st.column_config.NumberColumn("Studio QC", format="%.2f%%"),
            "ticket_score_pct": st.column_config.NumberColumn("Ticket Score", format="%.2f%%"),
            "dt": st.column_config.DatetimeColumn("Date & Time"),
        },
    )

with b:
    st.markdown("### Score Change")
    t = f.dropna(subset=["dt"]).copy()

    if t.empty:
        st.info("No datetime values available for this filter.")
    else:
        t["date_only"] = pd.to_datetime(t["dt"], errors="coerce").dt.date
        t = t.dropna(subset=["date_only"])

        score_cols_all = ["catalog_score_pct", "studio_score_pct", "ticket_score_pct"]
        score_cols_present = [c for c in score_cols_all if c in t.columns]

        if not score_cols_present:
            st.info("No score columns available.")
        else:
            daily = (
                t.groupby("date_only", as_index=False)[score_cols_present]
                .mean(numeric_only=True)
                .sort_values("date_only")
            )

            # choose y column safely
            ycol = score_col if score_col in daily.columns else score_cols_present[0]

            # if still missing somehow, stop safely
            if ycol not in daily.columns:
                st.info("No data available for the selected score type.")
            else:
                ydata = pd.to_numeric(daily[ycol], errors="coerce").dropna()
                if ydata.empty:
                    st.info("No values available to plot for this score type.")
                else:
                    fig = px.line(daily, x="date_only", y=ycol, markers=True)
                    fig.update_layout(
                        height=460,
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="",
                        yaxis_title="",
                    )
                    st.plotly_chart(fig, use_container_width=True)
