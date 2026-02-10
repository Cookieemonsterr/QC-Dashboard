# app.py
# QC Scores Dashboard (iStep-style) — Auto-load from Google Sheets (no uploads), robust download,
# collapses multi-row tickets to 1 row per Reference ID, Build/Update/Existing filters, light+dark safe UI.
#
# Run locally:  streamlit run app.py
# Deploy: Streamlit Cloud + requirements.txt (below)

import io
import re
import html
from collections import defaultdict

import requests
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# ✅ YOUR GOOGLE SHEET (PUBLIC VIEWER REQUIRED)
# ============================================================
SHEET_ID = "1rQHlDgQC5mZQ00fPVz20h4KEFbmsosE2"

# Try XLSX first (best), fallback to CSV if Google blocks XLSX export
XLSX_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=xlsx"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

# ============================================================
# PAGE
# ============================================================
st.set_page_config(page_title="QC Scores Dashboard (iStep)", page_icon="✅", layout="wide")

# ============================================================
# LIGHT/DARK SAFE UI (no dark-only styling)
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

.qc-title{ font-size:1.75rem; font-weight:900; letter-spacing:.2px; margin:.1rem 0 .35rem 0; }
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
    # If header looks like "X : Y" keep Y (your mapping style)
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

def _to_pct(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
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

def _clean_str(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    return s

def _norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _norm_city(s: str) -> str:
    s = _norm_spaces(_clean_str(s))
    if not s:
        return ""
    mapping = {
        "AbuDhabi": "Abu Dhabi",
        "Abu dhabi": "Abu Dhabi",
        "AL AIN": "Al Ain",
        "Alain": "Al Ain",
        "AL SHARJAH": "Sharjah",
        "Sharjah ": "Sharjah",
    }
    return mapping.get(s, s)

def _safe_num(x) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (int, float)) and not pd.isna(x):
        return float(x)
    s = _clean_str(x)
    if not s:
        return 0.0
    s = s.replace("%", "").strip()
    try:
        return float(s)
    except:
        return 0.0

# ============================================================
# ✅ TICKET TYPE RULES (NO "UNKNOWN" TYPES)
# ============================================================
def map_ticket_type(raw: str) -> str:
    s = _norm_spaces(_clean_str(raw))
    low = s.lower()
    if low.startswith("outlet catalogue update request"):
        return "Update Tickets"
    if low.startswith("new brand setup"):
        # covers (large)/(small)
        return "Build Tickets"
    if low.startswith("new outlet setup for existing brand"):
        return "Existing Tickets"
    return "Other"  # Only if the subject truly doesn't match your 3 rules.

# ============================================================
# ✅ COLLAPSE “SAME TICKET IN MULTIPLE ROWS” → 1 ROW PER TICKET
# ============================================================
def first_non_empty(series: pd.Series):
    s = series.dropna().astype(str).map(_clean_str)
    s = s[s != ""]
    return s.iloc[0] if len(s) else pd.NA

def unique_join(series: pd.Series, sep=" | "):
    s = series.dropna().astype(str).map(_clean_str)
    s = s[s != ""]
    uniq = list(dict.fromkeys(s.tolist()))
    return sep.join(uniq) if uniq else pd.NA

def collapse_tickets(df: pd.DataFrame) -> pd.DataFrame:
    ref = safe_col(df, ["Reference ID", "Ticket ID"])
    if not ref:
        return df

    subject = safe_col(df, ["Subject", "Ticket Type"])
    tscore = safe_col(df, ["Ticket Score"])

    c_agent = safe_col(df, ["Catalogue Agent Name", "Catalogue Name"])
    s_agent = safe_col(df, ["Studio Agent Name", "Studio Name"])

    c_city = safe_col(df, ["Catalogue City", "City"])
    c_market = safe_col(df, ["Catalogue Market", "Market"])
    c_score = safe_col(df, ["Catalogue Score", "Catalogue Agent QC Score", "Catalogue Agent QC Score__2"])
    c_dt = safe_col(df, ["Catalogue Date & Time", "Ticket Creation Time"])

    s_city = safe_col(df, ["Studio City", "City__2"])
    s_market = safe_col(df, ["Studio Market", "Market__2"])
    s_score = safe_col(df, ["Studio Score", "Studio Agent QC Score", "Studio Agent QC Score__2"])
    s_dt = safe_col(df, ["Studio Date & Time", "Ticket Creation Time__2"])

    c_sb = safe_col(df, ["Catalogue Sent Back To Catalog", "Sent back to catalog"])
    s_sb = safe_col(df, ["Studio Sent Back To Catalog", "Sent back to catalog__2"])

    agg = {}

    # 1-value fields → take first non-empty
    for col in [subject, tscore, c_city, c_market, c_score, c_dt, s_city, s_market, s_score, s_dt]:
        if col:
            agg[col] = first_non_empty

    # sent back → sum
    if c_sb:
        agg[c_sb] = lambda x: pd.to_numeric(x, errors="coerce").fillna(0).sum()
    if s_sb:
        agg[s_sb] = lambda x: pd.to_numeric(x, errors="coerce").fillna(0).sum()

    # multi-people fields → collect unique
    if c_agent:
        agg[c_agent] = unique_join
    if s_agent:
        agg[s_agent] = unique_join

    out = df.groupby(ref, as_index=False).agg(agg)
    out.rename(columns={ref: "Reference ID"}, inplace=True)
    return out

# ============================================================
# ✅ MISTAKE / DEDUCTION LOGIC (CALC)
# Rule: if Ticket Score == 100% → mistakes = 0 even if deduction columns have values
# ============================================================
UPDATE_UNIT_POINTS = {
    "Spelling Mistake": 5,
    "profile update": 5,
    "Calories": 3,
    "Categories": 7,
    "Item Name": 5,
    "Missing Items": 10,
    "Missing items": 10,
    "Item Price": 15,
    "Description": 5,
    "Addon (Min :Max)": 10,
    "Addon - Missing/Wrong option name": 10,
    "Addon Price": 15,
    "Translation": 5,
    "Item Operational Hours": 10,
}

STUDIO_UNIT_POINTS = {
    "Hero Image": 30,
    "Image without Text/Price/Code/Logo/Graphics": 15,
    "Image Uploaded to its Proper Item/Images weren't uploaded": 20,
    "Logo Missing": 15,
    "Correct Image Size/Dimensions": 10,
    "Images Pixelated": 10,
}

LOCATION_UPDATE_UNIT_POINTS = {
    "Location Pin": 25,
    "City": 25,
    "Zone Name/Zone Extensions": 25,
    "Zone Name & Zone Extensions": 25,
    "Zone Name & Extensional Zones": 25,
    "Full Address": 25,
}

# (We keep Build/Existing maps minimal here for safety; you can expand anytime)
BUILD_EXISTING_GENERIC = {
    "Location Pin": 10,
    "City": 2,
    "Full Address": 2,
    "Operational Hours": 9,
    "Login ID": 2,
    "Phone Number": 2,
    "Email": 2,
    "Role": 2,
    "Name": 2,
}

def _pick_unit_map(ticket_type: str) -> dict:
    if ticket_type == "Update Tickets":
        merged = {}
        merged.update(UPDATE_UNIT_POINTS)
        merged.update(STUDIO_UNIT_POINTS)
        merged.update(LOCATION_UPDATE_UNIT_POINTS)
        return merged
    if ticket_type == "Build Tickets":
        return BUILD_EXISTING_GENERIC
    if ticket_type == "Existing Tickets":
        return BUILD_EXISTING_GENERIC
    return {}

def compute_deductions_and_mistakes(row: pd.Series, ticket_type: str, ticket_score_pct: float):
    # If 100% ticket score: ignore everything
    if ticket_score_pct >= 100:
        return 0.0, 0

    unit_map = _pick_unit_map(ticket_type)
    if not unit_map:
        return 0.0, 0

    deducted = 0.0
    mistakes = 0

    for col, unit in unit_map.items():
        if col in row.index:
            v = _safe_num(row[col])
            if v > 0:
                deducted += v
                m = int(round(v / float(unit)))
                if m == 0:
                    m = 1
                mistakes += m

    return deducted, mistakes

# ============================================================
# ✅ ROBUST DOWNLOAD (XLSX → CSV) + FRIENDLY ERROR BOX
# ============================================================
def _http_get(url: str) -> requests.Response:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "*/*",
    }
    return requests.get(url, headers=headers, timeout=120, allow_redirects=True)

@st.cache_data(show_spinner=False, ttl=600)
def load_sheet_df() -> pd.DataFrame:
    # 1) XLSX
    r = _http_get(XLSX_URL)
    if r.status_code == 200 and len(r.content) > 1000:
        try:
            return pd.read_excel(io.BytesIO(r.content), sheet_name=0)
        except Exception:
            pass  # fallback to CSV

    # 2) CSV
    r2 = _http_get(CSV_URL)
    if r2.status_code == 200 and len(r2.content) > 10:
        return pd.read_csv(io.BytesIO(r2.content))

    # 3) Fail with clear message
    def preview(resp: requests.Response) -> str:
        try:
            txt = resp.text
            return (txt[:500] + "…") if len(txt) > 500 else txt
        except Exception:
            return "<could not decode response text>"

    msg = f"""
Could not download the Google Sheet.

XLSX_URL status: {r.status_code}
CSV_URL status:  {r2.status_code}

Fix:
- Open the sheet → Share → General access → set to "Anyone with the link" (Viewer)

If it’s already public, you might be rate-limited (429). Wait a minute then press Refresh.
"""
    raise RuntimeError(msg + "\n\n--- XLSX preview ---\n" + preview(r) + "\n\n--- CSV preview ---\n" + preview(r2))

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

    # map columns (robust candidates)
    col_ref = safe_col(df, ["Reference ID", "Ticket ID"])
    col_subject = safe_col(df, ["Subject", "Ticket Type"])
    col_ticket_score = safe_col(df, ["Ticket Score"])

    col_cat_agent = safe_col(df, ["Catalogue Agent Name", "Catalogue Name"])
    col_stu_agent = safe_col(df, ["Studio Agent Name", "Studio Name"])

    col_cat_score = safe_col(df, ["Catalogue Score", "Catalogue Agent QC Score"])
    col_stu_score = safe_col(df, ["Studio Score", "Studio Agent QC Score"])

    col_cat_sb = safe_col(df, ["Catalogue Sent Back To Catalog", "Sent back to catalog"])
    col_stu_sb = safe_col(df, ["Studio Sent Back To Catalog", "Sent back to catalog__2"])

    col_cat_dt = safe_col(df, ["Catalogue Date & Time", "Ticket Creation Time"])
    col_stu_dt = safe_col(df, ["Studio Date & Time", "Ticket Creation Time__2"])

    # city/market candidates
    col_cat_city = safe_col(df, ["Catalogue City"])
    col_stu_city = safe_col(df, ["Studio City"])
    col_city_any = safe_col(df, ["City"])
    col_cat_market = safe_col(df, ["Catalogue Market"])
    col_stu_market = safe_col(df, ["Studio Market"])
    col_market_any = safe_col(df, ["Market"])

    # standardized columns used by dashboard
    df["ticket_id"] = df[col_ref].astype(str) if col_ref else df.index.astype(str)
    df["ticket_type_raw"] = df[col_subject].astype(str) if col_subject else ""
    df["ticket_type"] = df["ticket_type_raw"].apply(map_ticket_type)

    df["ticket_score_pct"] = _to_pct(df[col_ticket_score]) if col_ticket_score else pd.NA
    df["catalog_score_pct"] = _to_pct(df[col_cat_score]) if col_cat_score else pd.NA
    df["studio_score_pct"] = _to_pct(df[col_stu_score]) if col_stu_score else pd.NA

    # Agents: fix None -> Unassigned
    df["catalog_agent"] = (df[col_cat_agent] if col_cat_agent else pd.NA).astype("string")
    df["studio_agent"] = (df[col_stu_agent] if col_stu_agent else pd.NA).astype("string")
    df["catalog_agent"] = df["catalog_agent"].fillna("").map(_clean_str)
    df["studio_agent"] = df["studio_agent"].fillna("").map(_clean_str)
    df.loc[df["catalog_agent"] == "", "catalog_agent"] = "Unassigned"
    df.loc[df["studio_agent"] == "", "studio_agent"] = "Unassigned"

    df["catalog_sent_back"] = pd.to_numeric(df[col_cat_sb], errors="coerce").fillna(0) if col_cat_sb else 0
    df["studio_sent_back"] = pd.to_numeric(df[col_stu_sb], errors="coerce").fillna(0) if col_stu_sb else 0

    # Datetime
    cat_dt = pd.to_datetime(df[col_cat_dt], errors="coerce") if col_cat_dt else pd.NaT
    stu_dt = pd.to_datetime(df[col_stu_dt], errors="coerce") if col_stu_dt else pd.NaT
    df["dt"] = cat_dt.fillna(stu_dt)
    df["date"] = df["dt"].dt.date
    df["month"] = df["dt"].dt.to_period("M").astype(str)
    df["week"] = df["dt"].dt.isocalendar().week.astype("Int64")
    df["day"] = df["dt"].dt.day_name()

    # ✅ City: resolve from best available (Catalogue City → Studio City → City)
    cat_city = df[col_cat_city] if col_cat_city else pd.NA
    stu_city = df[col_stu_city] if col_stu_city else pd.NA
    any_city = df[col_city_any] if col_city_any else pd.NA

    resolved_city = []
    for a, b, c in zip(cat_city, stu_city, any_city):
        aa = _norm_city(a)
        bb = _norm_city(b)
        cc = _norm_city(c)
        resolved_city.append(aa or bb or cc or "Unknown")
    df["city"] = resolved_city

    # Market: prefer Catalogue Market → Studio Market → Market
    cat_m = df[col_cat_market] if col_cat_market else pd.NA
    stu_m = df[col_stu_market] if col_stu_market else pd.NA
    any_m = df[col_market_any] if col_market_any else pd.NA
    resolved_market = []
    for a, b, c in zip(cat_m, stu_m, any_m):
        aa = _clean_str(a)
        bb = _clean_str(b)
        cc = _clean_str(c)
        resolved_market.append(aa or bb or cc or "")
    df["market"] = resolved_market

    # Total qc score
    df["total_qc_pct"] = df[["catalog_score_pct", "studio_score_pct"]].mean(axis=1, skipna=True)

    # ✅ Deducted Points + Mistakes (calc)
    # (Only uses columns that exist; safe.)
    deducted_list = []
    mistakes_list = []
    ticket_score_series = pd.to_numeric(df["ticket_score_pct"], errors="coerce").fillna(0)

    for idx, row in df.iterrows():
        ttype = row.get("ticket_type", "Other")
        tscore = float(ticket_score_series.loc[idx]) if idx in ticket_score_series.index else 0.0
        ded, mis = compute_deductions_and_mistakes(row, ttype, tscore)
        deducted_list.append(ded)
        mistakes_list.append(mis)

    df["deducted_points_calc"] = deducted_list
    df["mistakes_calc"] = mistakes_list

    return df

# ============================================================
# SIDEBAR + LOAD
# ============================================================
st.sidebar.markdown("## Data")
st.sidebar.caption("Auto-loads from Google Sheet (no uploads).")

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

exclude_unassigned = st.sidebar.checkbox("Exclude Unassigned agents from leaderboards", value=True)

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
    "Score Type (Trend + Distribution)",
    ["Total QC Score", "Catalog Agent QC Score", "Studio Agent QC Score", "Ticket Score"],
    index=0,
)
score_col = {
    "Total QC Score": "total_qc_pct",
    "Catalog Agent QC Score": "catalog_score_pct",
    "Studio Agent QC Score": "studio_score_pct",
    "Ticket Score": "ticket_score_pct",
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
        f'<div class="qc-sub">Cleaned: <b>1 row per ticket</b> • Showing <b>{len(f):,}</b> tickets</div>',
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
# KPI ROW
# ============================================================
k1, k2, k3, k4, k5, k6 = st.columns(6)

catalog_avg = mean_pct(f["catalog_score_pct"])
studio_avg = mean_pct(f["studio_score_pct"])
total_avg = mean_pct(f["total_qc_pct"])
ticket_avg = mean_pct(f["ticket_score_pct"])

sent_back_total = int(
    pd.to_numeric(f["catalog_sent_back"], errors="coerce").fillna(0).sum()
    + pd.to_numeric(f["studio_sent_back"], errors="coerce").fillna(0).sum()
)
low_perf = int((pd.to_numeric(f["total_qc_pct"], errors="coerce") < 90).fillna(False).sum())

mistakes_sum = int(pd.to_numeric(f["mistakes_calc"], errors="coerce").fillna(0).sum())
deducted_sum = float(pd.to_numeric(f["deducted_points_calc"], errors="coerce").fillna(0).sum())

with k1: kpi_card("Catalog QC", "—" if catalog_avg is None else f"{catalog_avg:.2f}%")
with k2: kpi_card("Studio QC", "—" if studio_avg is None else f"{studio_avg:.2f}%")
with k3: kpi_card("Total QC", "—" if total_avg is None else f"{total_avg:.2f}%")
with k4: kpi_card("Ticket Score", "—" if ticket_avg is None else f"{ticket_avg:.2f}%")
with k5: kpi_card("Tickets", f"{len(f):,}", f"{low_perf:,} under 90%")
with k6: kpi_card("Mistakes", f"{mistakes_sum:,}", f"Deducted: {deducted_sum:.0f} pts")

st.markdown("<br/>", unsafe_allow_html=True)

# ============================================================
# MAIN: TABLE + TREND
# ============================================================
a, b = st.columns([0.62, 0.38])

with a:
    st.markdown("### Tickets (clean)")
    show_cols = [
        "ticket_id", "ticket_type", "ticket_type_raw",
        "catalog_agent", "catalog_score_pct",
        "studio_agent", "studio_score_pct",
        "total_qc_pct", "ticket_score_pct",
        "mistakes_calc", "deducted_points_calc",
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
            "catalog_score_pct": st.column_config.NumberColumn("Catalog Score", format="%.2f%%"),
            "studio_score_pct": st.column_config.NumberColumn("Studio Score", format="%.2f%%"),
            "total_qc_pct": st.column_config.NumberColumn("Total QC", format="%.2f%%"),
            "ticket_score_pct": st.column_config.NumberColumn("Ticket Score", format="%.2f%%"),
            "mistakes_calc": st.column_config.NumberColumn("Mistakes (calc)"),
            "deducted_points_calc": st.column_config.NumberColumn("Deducted (calc)"),
            "dt": st.column_config.DatetimeColumn("Date & Time"),
        },
    )

with b:
    st.markdown("### Trend")
    t = f.dropna(subset=["dt"]).copy()
    if t.empty:
        st.info("No datetime values available for this filter.")
    else:
        t["date_only"] = t["dt"].dt.date
        daily = (
            t.groupby("date_only", as_index=False)[
                ["catalog_score_pct", "studio_score_pct", "total_qc_pct", "ticket_score_pct"]
            ]
            .mean(numeric_only=True)
            .sort_values("date_only")
        )
        fig = px.line(daily, x="date_only", y=score_col, markers=True)
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ============================================================
# WOW SECTION: DISTRIBUTIONS + SPLITS
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
    st.markdown("### Tickets by City")
    city_counts = f["city"].fillna("Unknown").astype(str).value_counts().reset_index()
    city_counts.columns = ["city", "ticket_count"]
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
    ca_src = f.copy()
    if exclude_unassigned:
        ca_src = ca_src[ca_src["catalog_agent"] != "Unassigned"]

    ca = (
        ca_src.groupby("catalog_agent", dropna=False)
        .agg(
            tickets=("ticket_id", "count"),
            avg_catalog=("catalog_score_pct", "mean"),
            avg_total=("total_qc_pct", "mean"),
            sent_back=("catalog_sent_back", "sum"),
            mistakes=("mistakes_calc", "sum"),
        )
        .reset_index()
        .sort_values(["avg_total", "tickets"], ascending=[True, False])
    )
    ca["low_perf_flag"] = (ca["avg_total"] < 90).fillna(False)
    st.dataframe(
        ca,
        use_container_width=True,
        height=360,
        column_config={
            "catalog_agent": st.column_config.TextColumn("Catalogue Agent"),
            "tickets": st.column_config.NumberColumn("Tickets"),
            "avg_catalog": st.column_config.NumberColumn("Avg Catalog", format="%.2f%%"),
            "avg_total": st.column_config.NumberColumn("Avg Total", format="%.2f%%"),
            "sent_back": st.column_config.NumberColumn("Sent Back"),
            "mistakes": st.column_config.NumberColumn("Mistakes (calc)"),
            "low_perf_flag": st.column_config.CheckboxColumn("Low performer (<90%)"),
        },
    )

with p2:
    st.markdown("### Studio agents")
    sa_src = f.copy()
    if exclude_unassigned:
        sa_src = sa_src[sa_src["studio_agent"] != "Unassigned"]

    sa = (
        sa_src.groupby("studio_agent", dropna=False)
        .agg(
            tickets=("ticket_id", "count"),
            avg_studio=("studio_score_pct", "mean"),
            avg_total=("total_qc_pct", "mean"),
            sent_back=("studio_sent_back", "sum"),
            mistakes=("mistakes_calc", "sum"),
        )
        .reset_index()
        .sort_values(["avg_total", "tickets"], ascending=[True, False])
    )
    sa["low_perf_flag"] = (sa["avg_total"] < 90).fillna(False)
    st.dataframe(
        sa,
        use_container_width=True,
        height=360,
        column_config={
            "studio_agent": st.column_config.TextColumn("Studio Agent"),
            "tickets": st.column_config.NumberColumn("Tickets"),
            "avg_studio": st.column_config.NumberColumn("Avg Studio", format="%.2f%%"),
            "avg_total": st.column_config.NumberColumn("Avg Total", format="%.2f%%"),
            "sent_back": st.column_config.NumberColumn("Sent Back"),
            "mistakes": st.column_config.NumberColumn("Mistakes (calc)"),
            "low_perf_flag": st.column_config.CheckboxColumn("Low performer (<90%)"),
        },
    )

# ============================================================
# DEBUG: show why anything became "Other" or why city Unknown
# ============================================================
with st.expander("🔎 Debug: subjects that became Other (should be near zero)"):
    other_vals = (
        df[df["ticket_type"] == "Other"]["ticket_type_raw"]
        .fillna("")
        .astype(str)
        .map(_norm_spaces)
    )
    st.dataframe(other_vals.value_counts().reset_index().head(120), use_container_width=True)

with st.expander("🔎 Debug: why city is Unknown"):
    unknown_rows = df[df["city"] == "Unknown"].copy()
    st.write(f"Unknown city rows: {len(unknown_rows):,}")
    cols = [c for c in ["ticket_id", "ticket_type_raw", "city", "market", "dt"] if c in unknown_rows.columns]
    st.dataframe(unknown_rows[cols].head(200), use_container_width=True)
