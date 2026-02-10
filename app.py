# app.py
import re
import html
from collections import defaultdict

import pandas as pd
import plotly.express as px
import streamlit as st

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="QC Scores Dashboard (iStep)",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# ✅ CSS that works in LIGHT + DARK
# (does NOT force white text)
# =========================
CUSTOM_CSS = """
<style>
/* Fix "cut from top" */
.block-container{ padding-top: 2.2rem !important; }

/* Theme-aware variables (Streamlit sets html[data-theme]) */
html[data-theme="dark"]{
  --qc-card-bg: rgba(255,255,255,.06);
  --qc-card-border: rgba(255,255,255,.12);
  --qc-muted: rgba(255,255,255,.65);
  --qc-shadow: 0 10px 25px rgba(0,0,0,.28);
  --qc-hr: rgba(255,255,255,.10);
}
html[data-theme="light"]{
  --qc-card-bg: rgba(0,0,0,.03);
  --qc-card-border: rgba(0,0,0,.10);
  --qc-muted: rgba(0,0,0,.55);
  --qc-shadow: 0 10px 25px rgba(0,0,0,.10);
  --qc-hr: rgba(0,0,0,.08);
}

.qc-title{
  font-size: 1.75rem;
  font-weight: 900;
  letter-spacing: .2px;
  line-height: 1.1;
  margin: .15rem 0 .35rem 0;
}
.qc-sub{ color: var(--qc-muted); margin-top: -.15rem; }

.kpi{
  background: var(--qc-card-bg);
  border: 1px solid var(--qc-card-border);
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: var(--qc-shadow);
}
.kpi .label{ font-size: .9rem; color: var(--qc-muted); margin-bottom: 2px; }
.kpi .value{ font-size: 2.05rem; font-weight: 900; }
.kpi .delta{ font-size: .9rem; color: var(--qc-muted); }

hr{
  border: none;
  border-top: 1px solid var(--qc-hr);
  margin: .8rem 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================
# Header normalization + duplicates
# =========================
def normalize_header(h: str) -> str:
    h = "" if h is None else str(h)
    h = html.unescape(h)
    h = re.sub(r"\s+", " ", h).strip()
    # If header came like "X : Y" keep Y
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

def base_col(c: str) -> str:
    return re.sub(r"__\d+$", "", c)

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

# =========================
# ✅ Ticket type mapping (STARTS WITH rules)
# =========================
def map_ticket_type(raw: str) -> str:
    s = "" if raw is None else str(raw)
    s = s.strip()
    s_low = s.lower()

    if s_low in ("", "nan", "none", "null"):
        return "Other"

    if s_low.startswith("outlet catalogue update request"):
        return "Update Tickets"

    if s_low.startswith("new brand setup (large)") or s_low.startswith("new brand setup (small)"):
        return "Build Tickets"

    if s_low.startswith("new outlet setup for existing brand"):
        return "Existing Tickets"

    return "Other"

# =========================
# ✅ Points per 1 mistake (per ticket type)
# =========================
FIELD_POINTS_BUILD = {
    "Adding or linking Company Details": 4,
    "Name": 1,
    "Role": 1,
    "Phone Number": 2,
    "Email": 2,
    "Send Related Info": 2,
    "Certificates Address": 1,
    "TRN": 2,
    "Trade/Commercial License Copy": 1,
    "VAT/Tax Registration Certificate Copy": 1,
    "Contract File Uploaded": 2,
    "Platform, logistic Payments fees": 5,
    "Platform, logistic & Payments fees": 5,
    "Outlet Contract Type": 2,
    "Outlet & Contract Type": 2,
    "Contract Date": 1,
    "Bank Details": 5,
    "Brand Name": 2,
    "Profile Name": 4,
    "Tags": 5,
    "Food Prep Time": 1,
    "Order Prep Time": 1,
    "Discovery Radius": 4,
    "Delivery Charge": 2,
    "Minimum Order Value": 1,
    "Price Per Person": 1,
    "Business Type": 1,
    "Direct/AM/Sales POC": 1,
    "Direct/AM/Sales": 1,
    "Outlet Tax Registered": 2,
    "Tax on Commission": 4,
    "Delivery Type & Target acceptance time": 2,
    "Delivery Type & Target Acceptance Time": 2,
    "MIF File Uploaded": 2,
    "Location Pin": 8,
    "City": 2,
    "Zone Name & Extensional Zones": 5,
    "Zone Name & Zone Extensions": 5,
    "Zone Name/Zone Extensions": 5,
    "Full Address": 1,
    "Send Delivery Related Info": 2,
    "Priority Numbers For Live Order Issues": 4,
    "Operational Hours": 8,
    "Login ID": 2,
    "Select a Role": 1,
}

FIELD_POINTS_UPDATE = {
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
    "Hero Image": 30,
    "Image without Text/Price/Code/Logo/Graphics": 15,
    "Image Uploaded to its Proper Item/Images weren't uploaded": 20,
    "Logo Missing": 15,
    "Correct Image Size/Dimensions": 10,
    "Images Pixelated": 10,
    "Location Pin": 25,
    "City": 25,
    "Zone Name/Zone Extensions": 25,
    "Zone Name & Zone Extensions": 25,
    "Zone Name & Extensional Zones": 25,
    "Full Address": 25,
}

FIELD_POINTS_EXISTING = {
    "Adding or linking Company Details": 5,
    "Profile Name": 5,
    "Tags": 5,
    "Food Prep Time": 2,
    "Discovery Radius": 5,
    "Delivery Charge": 2,
    "Minimum Order Value": 2,
    "Price Per Person": 2,
    "Business Type": 1,
    "Direct/AM/Sales POC": 1,
    "Outlet Tax Registered": 2,
    "Tax on Commission": 5,
    "Delivery Type & Target acceptance time": 2,
    "Delivery Type & Target Acceptance Time": 2,
    "MIF File Uploaded": 2,
    "Location Pin": 10,
    "City": 2,
    "Zone Name & Extensional Zones": 7,
    "Zone Name & Zone Extensions": 7,
    "Zone Name/Zone Extensions": 7,
    "Full Address": 2,
    "Name": 2,
    "Role": 2,
    "Phone Number": 2,
    "Email": 2,
    "Send Delivery Related Info": 2,
    "Priority Numbers For Live Order Issues": 5,
    "Operational Hours": 9,
    "Login ID": 2,
    "Select a Role": 2,
    "Contract File Uploaded": 3,
    "Platform, logistic Payments fees": 6,
    "Platform, logistic & Payments fees": 6,
    "Outlet Contract Type": 3,
    "Outlet & Contract Type": 3,
    "Contract Date": 2,
    "Correct Brand Linked": 2,
}

POINT_MAP_BY_TYPE = {
    "Build Tickets": FIELD_POINTS_BUILD,
    "Update Tickets": FIELD_POINTS_UPDATE,
    "Existing Tickets": FIELD_POINTS_EXISTING,
    "Other": {},
}

# Buckets for visuals
BUCKETS_BY_TYPE = {
    "Build Tickets": {
        "Company": [
            "Adding or linking Company Details","Name","Role","Phone Number","Email",
            "Send Related Info","Certificates Address","TRN",
            "Trade/Commercial License Copy","VAT/Tax Registration Certificate Copy"
        ],
        "Contracts": [
            "Contract File Uploaded","Platform, logistic Payments fees","Platform, logistic & Payments fees",
            "Outlet Contract Type","Outlet & Contract Type","Contract Date","Bank Details"
        ],
        "Profile": [
            "Brand Name","Profile Name","Tags","Food Prep Time","Order Prep Time","Discovery Radius",
            "Delivery Charge","Minimum Order Value","Price Per Person","Business Type","Direct/AM/Sales POC",
            "Direct/AM/Sales","Outlet Tax Registered","Tax on Commission",
            "Delivery Type & Target acceptance time","Delivery Type & Target Acceptance Time","MIF File Uploaded"
        ],
        "Address": ["Location Pin","City","Zone Name & Extensional Zones","Zone Name & Zone Extensions","Zone Name/Zone Extensions","Full Address"],
        "Contact": ["Send Delivery Related Info","Priority Numbers For Live Order Issues","Operational Hours"],
        "Login": ["Login ID","Select a Role"],
    },
    "Update Tickets": {
        "Catalogue / Update": [
            "Spelling Mistake","profile update","Calories","Categories","Item Name","Missing Items","Missing items",
            "Item Price","Description","Addon (Min :Max)","Addon - Missing/Wrong option name","Addon Price",
            "Translation","Item Operational Hours"
        ],
        "Studio": [
            "Hero Image","Image without Text/Price/Code/Logo/Graphics",
            "Image Uploaded to its Proper Item/Images weren't uploaded","Logo Missing",
            "Correct Image Size/Dimensions","Images Pixelated"
        ],
        "Location": ["Location Pin","City","Zone Name/Zone Extensions","Zone Name & Zone Extensions","Zone Name & Extensional Zones","Full Address"],
    },
    "Existing Tickets": {
        "Company": ["Adding or linking Company Details"],
        "Profile": [
            "Profile Name","Tags","Food Prep Time","Discovery Radius","Delivery Charge","Minimum Order Value",
            "Price Per Person","Business Type","Direct/AM/Sales POC","Outlet Tax Registered","Tax on Commission",
            "Delivery Type & Target acceptance time","Delivery Type & Target Acceptance Time","MIF File Uploaded"
        ],
        "Address": ["Location Pin","City","Zone Name & Extensional Zones","Zone Name & Zone Extensions","Zone Name/Zone Extensions","Full Address"],
        "Contact": ["Name","Role","Phone Number","Email","Send Delivery Related Info","Priority Numbers For Live Order Issues","Operational Hours"],
        "Login": ["Login ID","Select a Role"],
        "Contracts / Brand": [
            "Contract File Uploaded","Platform, logistic Payments fees","Platform, logistic & Payments fees",
            "Outlet Contract Type","Outlet & Contract Type","Contract Date","Correct Brand Linked"
        ],
    },
    "Other": {},
}

def compute_mistakes_type_aware(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    all_known_fields = set(FIELD_POINTS_BUILD) | set(FIELD_POINTS_UPDATE) | set(FIELD_POINTS_EXISTING)
    all_point_cols = [c for c in out.columns if base_col(c) in all_known_fields]

    for c in all_point_cols:
        out[c + " (mistakes)"] = pd.NA

    out["Total Deduction Points"] = 0.0
    out["Total Mistakes"] = 0

    all_bucket_names = set()
    for _, buckets in BUCKETS_BY_TYPE.items():
        all_bucket_names |= set(buckets.keys())
    for b in sorted(all_bucket_names):
        out[b + " (mistakes)"] = 0

    for ttype, subset in out.groupby("ticket_type", dropna=False):
        points_map = POINT_MAP_BY_TYPE.get(ttype, {})
        buckets = BUCKETS_BY_TYPE.get(ttype, {})
        idx = subset.index
        if not points_map:
            continue

        cols = [c for c in all_point_cols if base_col(c) in points_map]
        if not cols:
            continue

        pts = out.loc[idx, cols].apply(pd.to_numeric, errors="coerce").fillna(0)

        for c in cols:
            denom = points_map[base_col(c)]
            out.loc[idx, c + " (mistakes)"] = (pts[c] / denom).round(0)

        out.loc[idx, "Total Deduction Points"] = pts.sum(axis=1)
        out.loc[idx, "Total Mistakes"] = (
            out.loc[idx, [c + " (mistakes)" for c in cols]]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .sum(axis=1)
            .round(0)
        )

        for bname, fields in buckets.items():
            mcols = [c + " (mistakes)" for c in cols if base_col(c) in fields]
            if mcols:
                out.loc[idx, bname + " (mistakes)"] = (
                    out.loc[idx, mcols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1).round(0)
                )

    # cast
    out["Total Mistakes"] = pd.to_numeric(out["Total Mistakes"], errors="coerce").fillna(0).astype("Int64")
    out["Total Deduction Points"] = pd.to_numeric(out["Total Deduction Points"], errors="coerce").fillna(0)

    for c in all_point_cols:
        mc = c + " (mistakes)"
        out[mc] = pd.to_numeric(out[mc], errors="coerce").fillna(0).astype("Int64")

    for b in sorted(all_bucket_names):
        bc = b + " (mistakes)"
        out[bc] = pd.to_numeric(out[bc], errors="coerce").fillna(0).astype("Int64")

    return out

# =========================
# ✅ Load data (CSV OR Excel)
# =========================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    p = str(path).lower()

    if p.endswith(".csv"):
        # UTF-8-SIG fixes weird BOM headers
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    else:
        df = pd.read_excel(path, sheet_name=0)

    df.columns = [normalize_header(c) for c in df.columns]
    df.columns = make_unique(df.columns)

    # core columns
    col_ticket_id = safe_col(df, ["Ticket ID", "Reference ID"])
    col_subject = safe_col(df, ["Ticket Type", "Subject"])
    col_ticket_score = safe_col(df, ["Ticket Score"])

    col_cat_agent = safe_col(df, ["Catalogue Agent Name", "Catalogue Name"])
    col_stu_agent = safe_col(df, ["Studio Agent Name", "Studio Name"])

    col_cat_score = safe_col(df, ["Catalogue Agent QC Score", "Catalogue Score"])
    col_stu_score = safe_col(df, ["Studio Agent QC Score", "Studio Score"])

    col_cat_sb = safe_col(df, ["Catalogue Sent Back To Catalog", "Sent back to catalog"])
    col_stu_sb = safe_col(df, ["Studio Sent Back To Catalog", "Sent back to catalog__2"])

    col_cat_dt = safe_col(df, ["Ticket Creation Time", "Catalogue Date & Time"])
    col_stu_dt = safe_col(df, ["Ticket Creation Time__2", "Studio Date & Time"])

    col_city = safe_col(df, ["Catalogue City", "Studio City", "City"])
    col_market = safe_col(df, ["Catalogue Market", "Studio Market", "Market"])

    df["ticket_id"] = df[col_ticket_id].astype(str) if col_ticket_id else df.index.astype(str)

    df["ticket_type_raw"] = df[col_subject].astype(str) if col_subject else ""
    df["ticket_type"] = df["ticket_type_raw"].apply(map_ticket_type)

    df["ticket_score_pct"] = _to_pct(df[col_ticket_score]) if col_ticket_score else pd.NA
    df["catalog_score_pct"] = _to_pct(df[col_cat_score]) if col_cat_score else pd.NA
    df["studio_score_pct"] = _to_pct(df[col_stu_score]) if col_stu_score else pd.NA

    df["catalog_agent"] = df[col_cat_agent] if col_cat_agent else pd.NA
    df["studio_agent"] = df[col_stu_agent] if col_stu_agent else pd.NA

    df["catalog_sent_back"] = pd.to_numeric(df[col_cat_sb], errors="coerce") if col_cat_sb else 0
    df["studio_sent_back"] = pd.to_numeric(df[col_stu_sb], errors="coerce") if col_stu_sb else 0

    df["city"] = df[col_city] if col_city else pd.NA
    df["market"] = df[col_market] if col_market else pd.NA

    cat_dt = pd.to_datetime(df[col_cat_dt], errors="coerce") if col_cat_dt else pd.NaT
    stu_dt = pd.to_datetime(df[col_stu_dt], errors="coerce") if col_stu_dt else pd.NaT
    df["dt"] = cat_dt.fillna(stu_dt)

    df["date"] = df["dt"].dt.date
    df["month"] = df["dt"].dt.to_period("M").astype(str)
    df["week"] = df["dt"].dt.isocalendar().week.astype("Int64")
    df["day"] = df["dt"].dt.day_name()

    df["total_qc_pct"] = df[["catalog_score_pct", "studio_score_pct"]].mean(axis=1, skipna=True)

    df = compute_mistakes_type_aware(df)
    return df

# =========================
# Sidebar: Data source + filters
# =========================
st.sidebar.markdown("## Data Source")

uploaded = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx", "xls"])
DEFAULT_PATH = "Istep Data (1).csv"  # <- change this to your real csv filename if different

if uploaded is not None:
    # streamlit uploaded file object works with pandas directly
    data_src = uploaded
    src_name = uploaded.name
else:
    data_src = DEFAULT_PATH
    src_name = DEFAULT_PATH

st.sidebar.caption(f"Using: **{src_name}**")

st.sidebar.markdown("## Filters")

df = load_data(data_src)

min_dt = df["dt"].min()
max_dt = df["dt"].max()
default_range = (
    (min_dt.date() if pd.notna(min_dt) else None),
    (max_dt.date() if pd.notna(max_dt) else None),
)

date_range = st.sidebar.date_input("Select date range", value=default_range)

# ✅ Ticket type view (All / Build / Update / Existing)
view_mode = st.sidebar.radio(
    "Ticket Type View",
    ["All", "Build Tickets", "Update Tickets", "Existing Tickets"],
    index=0,
)

months = sorted([m for m in df["month"].dropna().unique().tolist() if m not in ("NaT", "nan")])
sel_months = st.sidebar.multiselect("Month", months, default=[])

weeks = sorted([int(w) for w in df["week"].dropna().unique().tolist()])
sel_weeks = st.sidebar.multiselect("Week", weeks, default=[])

days = [d for d in ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"] if d in df["day"].dropna().unique()]
sel_days = st.sidebar.multiselect("Day", days, default=[])

cities = sorted([c for c in df["city"].dropna().unique().tolist() if str(c).strip() and str(c).lower() != "nan"])
sel_cities = st.sidebar.multiselect("City", cities, default=[])

markets = sorted([m for m in df["market"].dropna().unique().tolist() if str(m).strip() and str(m).lower() != "nan"])
sel_markets = st.sidebar.multiselect("Market", markets, default=[])

cat_agents = sorted([a for a in df["catalog_agent"].dropna().unique().tolist() if str(a).strip() and str(a).lower() != "nan"])
sel_cat_agents = st.sidebar.multiselect("Catalog Agent name", cat_agents, default=[])

stu_agents = sorted([a for a in df["studio_agent"].dropna().unique().tolist() if str(a).strip() and str(a).lower() != "nan"])
sel_stu_agents = st.sidebar.multiselect("Studio Agent Name", stu_agents, default=[])

ticket_id_search = st.sidebar.text_input("Ticket ID contains", value="")

score_type = st.sidebar.selectbox(
    "Score Type",
    ["Total QC Score", "Catalog Agent QC Score", "Studio Agent QC Score", "Ticket Score"],
    index=0,
)

score_col = {
    "Total QC Score": "total_qc_pct",
    "Catalog Agent QC Score": "catalog_score_pct",
    "Studio Agent QC Score": "studio_score_pct",
    "Ticket Score": "ticket_score_pct",
}[score_type]

# =========================
# Apply filters
# =========================
f = df.copy()

# ✅ Apply ticket type view
if view_mode != "All":
    f = f[f["ticket_type"] == view_mode]

if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    if start:
        f = f[f["date"] >= start]
    if end:
        f = f[f["date"] <= end]

if sel_months:
    f = f[f["month"].isin(sel_months)]
if sel_weeks:
    f = f[f["week"].isin(sel_weeks)]
if sel_days:
    f = f[f["day"].isin(sel_days)]
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

# =========================
# Header
# =========================
left, right = st.columns([0.78, 0.22], vertical_alignment="bottom")
with left:
    st.markdown('<div class="qc-title">QC Scores Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="qc-sub">iStep export • {len(f):,} tickets (filtered)</div>', unsafe_allow_html=True)
with right:
    st.download_button(
        "⬇️ Download filtered CSV",
        data=f.to_csv(index=False).encode("utf-8"),
        file_name="qc_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("<hr/>", unsafe_allow_html=True)

# =========================
# KPI row
# =========================
k1, k2, k3, k4, k5, k6 = st.columns([1, 1, 1, 1, 1, 1])

catalog_avg = mean_pct(f["catalog_score_pct"])
studio_avg = mean_pct(f["studio_score_pct"])
total_avg = mean_pct(f["total_qc_pct"])

total_tickets = len(f)
sent_back = int(
    pd.to_numeric(f["catalog_sent_back"], errors="coerce").fillna(0).sum()
    + pd.to_numeric(f["studio_sent_back"], errors="coerce").fillna(0).sum()
)
low_perf = int((pd.to_numeric(f["total_qc_pct"], errors="coerce") < 90).fillna(False).sum())
total_mistakes = int(pd.to_numeric(f["Total Mistakes"], errors="coerce").fillna(0).sum())

with k1: kpi_card("Catalog Agent QC Score", "—" if catalog_avg is None else f"{catalog_avg:.2f}%")
with k2: kpi_card("Studio Agent QC Score", "—" if studio_avg is None else f"{studio_avg:.2f}%")
with k3: kpi_card("Total QC Score", "—" if total_avg is None else f"{total_avg:.2f}%")
with k4: kpi_card("Total Tickets", f"{total_tickets:,}", f"{low_perf:,} low performers (<90%)")
with k5: kpi_card("Tickets Sent Back", f"{sent_back:,}")
with k6: kpi_card("Total Mistakes", f"{total_mistakes:,}")

st.markdown("<br/>", unsafe_allow_html=True)

# =========================
# Tickets table + trend
# =========================
a, b = st.columns([0.62, 0.38])

with a:
    st.markdown("### Tickets")
    show_cols = [
        "ticket_id","ticket_type","ticket_type_raw",
        "catalog_agent","catalog_score_pct",
        "studio_agent","studio_score_pct",
        "total_qc_pct","ticket_score_pct",
        "Total Mistakes","Total Deduction Points",
        "city","market","dt",
    ]
    existing = [c for c in show_cols if c in f.columns]
    table = f[existing].sort_values("dt", ascending=False)
    st.dataframe(
        table,
        use_container_width=True,
        height=440,
        column_config={
            "catalog_score_pct": st.column_config.NumberColumn("Catalog Score", format="%.2f%%"),
            "studio_score_pct": st.column_config.NumberColumn("Studio Score", format="%.2f%%"),
            "total_qc_pct": st.column_config.NumberColumn("Total QC", format="%.2f%%"),
            "ticket_score_pct": st.column_config.NumberColumn("Ticket Score", format="%.2f%%"),
            "dt": st.column_config.DatetimeColumn("Date & Time"),
        },
    )

with b:
    st.markdown("### Trend")
    t = f.dropna(subset=["dt"]).copy()
    if t.empty:
        st.info("No datetime values available in the current filter.")
    else:
        t["date_only"] = t["dt"].dt.date
        daily = (
            t.groupby("date_only", as_index=False)[["catalog_score_pct","studio_score_pct","total_qc_pct","ticket_score_pct"]]
            .mean(numeric_only=True)
            .sort_values("date_only")
        )
        ycol = {
            "Total QC Score": "total_qc_pct",
            "Catalog Agent QC Score": "catalog_score_pct",
            "Studio Agent QC Score": "studio_score_pct",
            "Ticket Score": "ticket_score_pct",
        }[score_type]
        fig = px.line(daily, x="date_only", y=ycol, markers=True)
        fig.update_layout(height=440, margin=dict(l=10,r=10,t=10,b=10), xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

# =========================
# City count + score distribution + ticket type split
# =========================
c1, c2, c3 = st.columns([0.34, 0.33, 0.33])

with c1:
    st.markdown("### Ticket count by City")
    city_counts = f["city"].fillna("Unknown").astype(str).value_counts().reset_index()
    city_counts.columns = ["city", "ticket_count"]
    fig = px.bar(city_counts.head(20), x="city", y="ticket_count")
    fig.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10), xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown(f"### {score_type} distribution")
    s = pd.to_numeric(f[score_col], errors="coerce").dropna()
    if s.empty:
        st.info("No score values available for this selection.")
    else:
        fig = px.histogram(s, nbins=20)
        fig.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10), xaxis_title="Score (%)", yaxis_title="Tickets")
        st.plotly_chart(fig, use_container_width=True)

with c3:
    st.markdown("### Ticket type split")
    tt = f["ticket_type"].fillna("Other").value_counts().reset_index()
    tt.columns = ["ticket_type", "count"]
    fig = px.pie(tt, names="ticket_type", values="count", hole=0.55)
    fig.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

# =========================
# Mistakes Breakdown
# =========================
st.markdown("## Mistakes Breakdown")

bucket_name_set = set().union(*[set(b.keys()) for b in BUCKETS_BY_TYPE.values()])
bucket_cols = [c for c in f.columns if c.endswith(" (mistakes)") and c.replace(" (mistakes)", "") in bucket_name_set]

if bucket_cols:
    bucket_totals = f[bucket_cols].sum(numeric_only=True).reset_index()
    bucket_totals.columns = ["bucket", "mistakes"]
    bucket_totals["bucket"] = bucket_totals["bucket"].str.replace(" (mistakes)", "", regex=False)
    bucket_totals = bucket_totals.sort_values("mistakes", ascending=False)

    m1, m2 = st.columns([0.55, 0.45])
    with m1:
        fig = px.bar(bucket_totals, x="bucket", y="mistakes")
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10), xaxis_title="", yaxis_title="Mistakes")
        st.plotly_chart(fig, use_container_width=True)
    with m2:
        fig = px.pie(bucket_totals, names="bucket", values="mistakes", hole=0.55)
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("### Top Mistake Fields (by count)")
mistake_field_cols = [c for c in f.columns if c.endswith(" (mistakes)") and c not in bucket_cols]
if mistake_field_cols:
    field_totals = (
        f[mistake_field_cols].sum(numeric_only=True).sort_values(ascending=False).head(25).reset_index()
    )
    field_totals.columns = ["field", "mistakes"]
    field_totals["field"] = field_totals["field"].str.replace(" (mistakes)", "", regex=False)
    fig = px.bar(field_totals, x="mistakes", y="field", orientation="h")
    fig.update_layout(height=560, margin=dict(l=10,r=10,t=10,b=10), xaxis_title="Mistakes", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

# =========================
# Agent performance
# =========================
st.markdown("## Agent Performance")

p1, p2 = st.columns([0.52, 0.48])

with p1:
    st.markdown("### Catalog agents (avg score • tickets • sent back • mistakes)")
    ca = (
        f.groupby("catalog_agent", dropna=False)
        .agg(
            tickets=("ticket_id","count"),
            avg_catalog=("catalog_score_pct","mean"),
            avg_total=("total_qc_pct","mean"),
            sent_back=("catalog_sent_back","sum"),
            mistakes=("Total Mistakes","sum"),
        )
        .reset_index()
        .sort_values(["avg_total","tickets"], ascending=[True,False])
    )
    ca["low_perf_flag"] = (ca["avg_total"] < 90).fillna(False)

    st.dataframe(
        ca,
        use_container_width=True,
        height=360,
        column_config={
            "catalog_agent": st.column_config.TextColumn("Catalog Agent"),
            "tickets": st.column_config.NumberColumn("Tickets"),
            "avg_catalog": st.column_config.NumberColumn("Avg Catalog", format="%.2f%%"),
            "avg_total": st.column_config.NumberColumn("Avg Total", format="%.2f%%"),
            "sent_back": st.column_config.NumberColumn("Sent Back"),
            "mistakes": st.column_config.NumberColumn("Mistakes"),
            "low_perf_flag": st.column_config.CheckboxColumn("Low performer (<90%)"),
        },
    )

with p2:
    st.markdown("### Studio agents (avg score • tickets • sent back • mistakes)")
    sa = (
        f.groupby("studio_agent", dropna=False)
        .agg(
            tickets=("ticket_id","count"),
            avg_studio=("studio_score_pct","mean"),
            avg_total=("total_qc_pct","mean"),
            sent_back=("studio_sent_back","sum"),
            mistakes=("Total Mistakes","sum"),
        )
        .reset_index()
        .sort_values(["avg_total","tickets"], ascending=[True,False])
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
            "mistakes": st.column_config.NumberColumn("Mistakes"),
            "low_perf_flag": st.column_config.CheckboxColumn("Low performer (<90%)"),
        },
    )

st.markdown("<hr/>", unsafe_allow_html=True)

# Optional: show what "Other" actually is (so you can kill it completely)
with st.expander("🔎 Debug: raw ticket subjects that became Other"):
    other_vals = (
        df[df["ticket_type"] == "Other"]["ticket_type_raw"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("nan", "")
    )
    st.dataframe(other_vals.value_counts().reset_index().head(50), use_container_width=True)

st.caption(
    "Now supports CSV or Excel + Light/Dark mode automatically. "
    "Ticket Type View uses STARTS WITH rules exactly as you described."
)
