# app.py
# QC Scores Dashboard — iStep-style (Streamlit)
# - Auto-loads from Google Sheets (no uploads)
# - XLSX first (best), fallback to CSV
# - Collapses multi-row tickets -> 1 row per Ticket ID (for charts)
# - Robust Ticket Type mapping (Build/Update/Existing)
# - Smart City resolving
# - 100% ticket score => mistakes forced to 0
# - Light/Dark UI toggle

import re
import io
import math
import html
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
import plotly.express as px

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="QC Scores Dashboard",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ✅ Your Google Sheet ID (must be publicly accessible: Anyone with link = Viewer)
SHEET_ID = "1rQHlDgQC5mZQ00fPVz20h4KEFbmsosE2"
XLSX_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=xlsx"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

# ============================================================
# UI THEME (light/dark safe)
# ============================================================
def inject_css(dark_mode: bool):
    # Streamlit pages sometimes look “blank” in light mode if custom CSS assumes dark background.
    # We force readable colors for both.
    base_bg = "#0E1117" if dark_mode else "#FFFFFF"
    card_bg = "#151A22" if dark_mode else "#F6F7FB"
    text = "#E8EAED" if dark_mode else "#111827"
    subtle = "#9CA3AF" if dark_mode else "#6B7280"
    border = "#2A2F3A" if dark_mode else "#E5E7EB"
    accent = "#3B82F6"

    st.markdown(
        f"""
        <style>
          .block-container {{
            padding-top: 1.2rem !important;
            padding-bottom: 2rem !important;
          }}
          html, body, [data-testid="stAppViewContainer"] {{
            background: {base_bg} !important;
            color: {text} !important;
          }}
          [data-testid="stSidebar"] {{
            background: {card_bg} !important;
            border-right: 1px solid {border} !important;
          }}
          h1,h2,h3,h4,h5,h6, p, span, div, label {{
            color: {text} !important;
          }}
          .qc-card {{
            background: {card_bg};
            border: 1px solid {border};
            border-radius: 14px;
            padding: 14px 16px;
          }}
          .qc-kpi {{
            font-size: 28px;
            font-weight: 800;
            line-height: 1.1;
          }}
          .qc-sub {{
            color: {subtle} !important;
            font-size: 13px;
            margin-top: 6px;
          }}
          .qc-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 12px;
            border: 1px solid {border};
            background: transparent;
          }}
          .qc-accent {{
            color: {accent} !important;
            font-weight: 700;
          }}
          .stDataFrame {{
            border: 1px solid {border} !important;
            border-radius: 12px !important;
            overflow: hidden !important;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# HELPERS
# ============================================================
def to_num(x):
    if pd.isna(x):
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return 0.0
    # handle "100%" etc
    s = s.replace("%", "").strip()
    try:
        return float(s)
    except:
        return 0.0

def to_pct(x):
    v = to_num(x)
    # your file sometimes is already 0-100, sometimes 0-1
    if 0 <= v <= 1:
        v = v * 100
    return float(v)

def clean_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def norm_city(x):
    s = clean_str(x)
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    # Normalize common variations
    mapping = {
        "AbuDhabi": "Abu Dhabi",
        "Abu dhabi": "Abu Dhabi",
        "AL AIN": "Al Ain",
        "Alain": "Al Ain",
        "AL SHARJAH": "Sharjah",
        "Sharjah ": "Sharjah",
    }
    return mapping.get(s, s)

def parse_dt(series: pd.Series):
    # Robust datetime parsing for mixed formats
    return pd.to_datetime(series, errors="coerce", utc=False)

# ============================================================
# TICKET TYPE RULES (YOUR SPEC)
# ============================================================
def classify_ticket_type(subject: str) -> str:
    s = clean_str(subject).lower()
    if not s:
        return "Other"
    # collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()

    if s.startswith("outlet catalogue update request"):
        return "Update Tickets"
    if s.startswith("new brand setup (large)") or s.startswith("new brand setup (small)") or s.startswith("new brand setup"):
        return "Build Tickets"
    if s.startswith("new outlet setup for existing brand"):
        return "Existing Tickets"
    return "Other"

# ============================================================
# MISTAKE POINTS MAPS
# You told me: if column has 4 -> 1 mistake, 8 -> 2, etc => means "deducted points"
# So unit_points = points deducted per 1 mistake for that field.
# We'll compute mistakes = round(deducted_points / unit_points)
# ============================================================
BUILD_UNIT_POINTS = {
    # Company details
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
    # Contracts
    "Contract File Uploaded": 2,
    "Platform, logistic Payments fees": 5,
    "Platform, logistic & Payments fees": 5,
    "Outlet Contract Type": 2,
    "Outlet & Contract Type": 2,
    "Contract Date": 1,
    "Bank Details": 5,
    # Brand/outlet profile
    "Brand Name": 2,
    "Correct Brand Linked": 2,
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
    # Address/contact/login
    "Location Pin": 8,
    "City": 2,
    "Zone Name Extensional Zones": 5,
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
    "Addon (Min :Max)": 10,
    "Addon - Missing/Wrong option name": 10,
    "Addon - Missing/Wrong option name": 10,
    "Addon Price": 15,
    "Translation": 5,
    "Item Operational Hours": 10,
}

STUDIO_UNIT_POINTS = {
    "Hero Image": 30,
    "Image without Text/Price/Code/Logo/Graphics": 15,
    "Image Uploaded to its Proper Item/Images weren't uploaded": 20,
    "Image Uploaded to its Proper Item/Images weren't uploaded": 20,
    "Logo Missing": 15,
    "Correct Image Size/Dimensions": 10,
    "Images Pixelated": 10,
    "Images Pixelated": 10,
}

LOCATION_UPDATE_UNIT_POINTS = {
    "Location Pin": 25,
    "City": 25,
    "Zone Name/Zone Extensions": 25,
    "Zone Name & Zone Extensions": 25,
    "Full Address": 25,
}

EXISTING_UNIT_POINTS = {
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
    "Direct/AM/Sales": 1,
    "Outlet Tax Registered": 2,
    "Tax on Commission": 5,
    "Delivery Type & Target acceptance time": 2,
    "Delivery Type & Target Acceptance Time": 2,
    "MIF File Uploaded": 2,
    # existing address/contact/login variants (from your screenshots)
    "Location Pin": 10,
    "City": 2,
    "Zone Name & Extensional Zones": 7,
    "Full Address": 2,
    "Priority Numbers For Live Order Issues": 5,
    "Operational Hours": 9,
    "Select a Role": 2,
    "Login ID": 2,
    "Phone Number": 2,
    "Email": 2,
    "Role": 2,
    "Name": 2,
}

def pick_unit_map(ticket_type: str) -> dict:
    if ticket_type == "Build Tickets":
        return BUILD_UNIT_POINTS
    if ticket_type == "Update Tickets":
        # Update tickets can include catalog update mistakes + profile + studio — we’ll include both maps safely
        merged = {}
        merged.update(UPDATE_UNIT_POINTS)
        merged.update(STUDIO_UNIT_POINTS)
        return merged
    if ticket_type == "Existing Tickets":
        return EXISTING_UNIT_POINTS
    return {}

def compute_deductions_for_row(row: pd.Series, unit_map: dict) -> tuple[float, float]:
    # returns (deducted_points, mistakes_count)
    deducted = 0.0
    mistakes = 0.0
    for col, unit in unit_map.items():
        if col in row.index:
            v = to_num(row[col])
            if v > 0:
                deducted += v
                # Convert points to mistake count
                m = v / float(unit)
                # If data is slightly off (e.g., 4.1), round safely
                m = int(round(m))
                # If rounding makes 0 but v>0, force at least 1 mistake
                if m == 0:
                    m = 1
                mistakes += m
    return deducted, mistakes

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(show_spinner=False, ttl=600)
def load_remote_sheet(xlsx_url: str, csv_url: str) -> pd.DataFrame:
    # Try XLSX
    try:
        r = requests.get(xlsx_url, timeout=45)
        r.raise_for_status()
        return pd.read_excel(io.BytesIO(r.content), sheet_name=0)
    except Exception:
        # fallback CSV
        r = requests.get(csv_url, timeout=45)
        r.raise_for_status()
        return pd.read_csv(io.BytesIO(r.content))

def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Your headers sometimes have HTML entities
    df = df.copy()
    df.columns = [html.unescape(str(c)).strip() for c in df.columns]
    return df

def build_clean_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = harmonize_columns(df_raw)

    # Try to locate key columns (by your mapping)
    # Primary:
    # Ticket ID / Reference ID
    # Subject
    # Ticket Score
    # Catalogue Agent name
    # Studio Agent Name
    # City fields

    # Rename common variants to a consistent schema if needed
    ren = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ["reference id", "ticket id", "ti cket id", "ticketid", "ti cket id: ticket id"]:
            ren[c] = "Ticket ID"
        if lc.startswith("subject"):
            ren[c] = "Subject"
        if lc == "ticket score":
            ren[c] = "Ticket Score"
        if "catalogue agent" in lc and "name" in lc:
            ren[c] = "Catalogue Agent Name"
        if "studio agent" in lc and "name" in lc:
            ren[c] = "Studio Agent Name"
        if "catalogue city" in lc:
            ren[c] = "Catalogue City"
        if "studio city" in lc:
            ren[c] = "Studio City"
        if lc == "city":
            ren[c] = "City"
        if "catalogue market" in lc:
            ren[c] = "Catalogue Market"
        if "studio market" in lc:
            ren[c] = "Studio Market"
        if "catalogue score" in lc and "qc" in lc:
            ren[c] = "Catalogue Agent QC Score"
        if "studio score" in lc and "qc" in lc:
            ren[c] = "Studio Agent QC Score"
        if "catalogue date" in lc:
            ren[c] = "Catalogue Date & Time"
        if "studio date" in lc:
            ren[c] = "Studio Date & Time"
        if "sent back" in lc and "catalog" in lc and "catalogue" in lc:
            ren[c] = "Catalogue Sent Back"
        if "sent back" in lc and "catalog" in lc and "studio" in lc:
            ren[c] = "Studio Sent Back"

    if ren:
        df = df.rename(columns=ren)

    # Ensure required columns exist (avoid crashing)
    for needed in ["Ticket ID", "Subject", "Ticket Score", "Catalogue Agent Name", "Studio Agent Name"]:
        if needed not in df.columns:
            df[needed] = None

    # Parse scores
    df["Ticket Score %"] = df["Ticket Score"].apply(to_pct)
    df["Catalogue QC %"] = df.get("Catalogue Agent QC Score", pd.Series([None]*len(df))).apply(to_pct)
    df["Studio QC %"] = df.get("Studio Agent QC Score", pd.Series([None]*len(df))).apply(to_pct)

    # Ticket Type
    df["Ticket Type"] = df["Subject"].apply(classify_ticket_type)

    # Resolved City (avoid Unknown if city exists somewhere)
    city_a = df.get("Catalogue City", pd.Series([""] * len(df))).apply(norm_city)
    city_b = df.get("Studio City", pd.Series([""] * len(df))).apply(norm_city)
    city_c = df.get("City", pd.Series([""] * len(df))).apply(norm_city)
    df["Resolved City"] = city_a.where(city_a != "", city_b.where(city_b != "", city_c))
    df["Resolved City"] = df["Resolved City"].fillna("").apply(norm_city)
    df.loc[df["Resolved City"] == "", "Resolved City"] = "Unknown"

    # Fix agent names (None -> Unassigned)
    df["Catalogue Agent Name"] = df["Catalogue Agent Name"].fillna("").apply(clean_str)
    df["Studio Agent Name"] = df["Studio Agent Name"].fillna("").apply(clean_str)
    df.loc[df["Catalogue Agent Name"] == "", "Catalogue Agent Name"] = "Unassigned"
    df.loc[df["Studio Agent Name"] == "", "Studio Agent Name"] = "Unassigned"

    # Datetime fields for filtering
    dt_a = parse_dt(df.get("Catalogue Date & Time", pd.Series([None]*len(df))))
    dt_b = parse_dt(df.get("Studio Date & Time", pd.Series([None]*len(df))))
    df["Ticket Datetime"] = dt_a.fillna(dt_b)
    # If still NaT, fallback to today-like for safety
    df["Ticket Datetime"] = df["Ticket Datetime"].fillna(pd.Timestamp.utcnow())

    # Compute deductions + mistakes (per row)
    ded_list = []
    mis_list = []
    for _, row in df.iterrows():
        unit_map = pick_unit_map(row["Ticket Type"])
        ded, mis = compute_deductions_for_row(row, unit_map)
        # Your rule: 100% ticket score => ignore mistakes completely
        if to_pct(row["Ticket Score"]) >= 100:
            ded, mis = 0.0, 0.0
        ded_list.append(ded)
        mis_list.append(mis)

    df["Deducted Points (calc)"] = ded_list
    df["Mistakes (calc)"] = mis_list

    return df

def collapse_to_ticket_level(df: pd.DataFrame) -> pd.DataFrame:
    # One row per Ticket ID for charts/summary
    # Keep agents as "A, B, C"
    def join_unique(s):
        vals = [clean_str(x) for x in s if clean_str(x)]
        vals = [v for v in vals if v.lower() != "nan"]
        uniq = []
        for v in vals:
            if v not in uniq:
                uniq.append(v)
        return ", ".join(uniq) if uniq else "Unassigned"

    agg = {
        "Subject": "first",
        "Ticket Type": "first",
        "Ticket Datetime": "min",
        "Resolved City": "first",
        "Catalogue Market": "first" if "Catalogue Market" in df.columns else "first",
        "Studio Market": "first" if "Studio Market" in df.columns else "first",
        "Ticket Score %": "mean",
        "Catalogue QC %": "mean",
        "Studio QC %": "mean",
        "Catalogue Sent Back": "max" if "Catalogue Sent Back" in df.columns else "first",
        "Studio Sent Back": "max" if "Studio Sent Back" in df.columns else "first",
        "Deducted Points (calc)": "sum",
        "Mistakes (calc)": "sum",
        "Catalogue Agent Name": join_unique,
        "Studio Agent Name": join_unique,
    }
    # if some keys not in df, remove
    agg = {k: v for k, v in agg.items() if k in df.columns}
    out = df.groupby("Ticket ID", dropna=False, as_index=False).agg(agg)
    out["Ticket Count"] = 1
    return out

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Data")
dark_mode = st.sidebar.toggle("Dark mode", value=True)
inject_css(dark_mode)

refresh = st.sidebar.button("🔄 Refresh data now")

with st.sidebar.expander("Filters", expanded=True):
    ticket_type_filter = st.multiselect(
        "Ticket Type",
        options=["Build Tickets", "Update Tickets", "Existing Tickets", "Other"],
        default=["Build Tickets", "Update Tickets", "Existing Tickets"],
    )
    exclude_unassigned = st.checkbox("Exclude 'Unassigned' from leaderboards", value=True)
    min_score = st.slider("Min Ticket Score", 0, 100, 0)

# ============================================================
# LOAD
# ============================================================
try:
    if refresh:
        load_remote_sheet.clear()

    with st.spinner("Loading data…"):
        raw = load_remote_sheet(XLSX_URL, CSV_URL)
        df = build_clean_df(raw)
        df_tickets = collapse_to_ticket_level(df)

except Exception as e:
    st.error(
        "I couldn't load the sheet. Make sure the Google Sheet is **Public (Anyone with link = Viewer)**.\n\n"
        "Also: If you used a Drive *file* link, it won't work unless it's exported properly as CSV/XLSX.\n\n"
        f"Error (safe): {type(e).__name__}"
    )
    st.stop()

# ============================================================
# APPLY FILTERS
# ============================================================
df_tickets = df_tickets[df_tickets["Ticket Type"].isin(ticket_type_filter)]
df_tickets = df_tickets[df_tickets["Ticket Score %"] >= min_score]

# Date range filter
min_dt = df_tickets["Ticket Datetime"].min()
max_dt = df_tickets["Ticket Datetime"].max()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_dt.date(), max_dt.date()),
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df_tickets = df_tickets[(df_tickets["Ticket Datetime"] >= start_ts) & (df_tickets["Ticket Datetime"] <= end_ts)]

# ============================================================
# HEADER
# ============================================================
st.markdown("## QC Scores Dashboard")
st.markdown(
    f"<span class='qc-badge'>Loaded: <span class='qc-accent'>{len(df_tickets):,}</span> tickets</span>",
    unsafe_allow_html=True,
)

# ============================================================
# KPIs
# ============================================================
total_tickets = len(df_tickets)
avg_ticket = df_tickets["Ticket Score %"].mean() if total_tickets else 0
avg_cat = df_tickets["Catalogue QC %"].mean() if "Catalogue QC %" in df_tickets.columns else 0
avg_studio = df_tickets["Studio QC %"].mean() if "Studio QC %" in df_tickets.columns else 0
total_mistakes = df_tickets["Mistakes (calc)"].sum() if "Mistakes (calc)" in df_tickets.columns else 0
sent_back = 0
if "Catalogue Sent Back" in df_tickets.columns:
    sent_back += pd.to_numeric(df_tickets["Catalogue Sent Back"], errors="coerce").fillna(0).sum()
if "Studio Sent Back" in df_tickets.columns:
    sent_back += pd.to_numeric(df_tickets["Studio Sent Back"], errors="coerce").fillna(0).sum()

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"<div class='qc-card'><div class='qc-kpi'>{total_tickets:,}</div><div class='qc-sub'>Total Tickets</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='qc-card'><div class='qc-kpi'>{avg_ticket:.2f}%</div><div class='qc-sub'>Average Ticket Score</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='qc-card'><div class='qc-kpi'>{avg_cat:.2f}%</div><div class='qc-sub'>Avg Catalogue QC</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='qc-card'><div class='qc-kpi'>{avg_studio:.2f}%</div><div class='qc-sub'>Avg Studio QC</div></div>", unsafe_allow_html=True)
with c5:
    st.markdown(f"<div class='qc-card'><div class='qc-kpi'>{int(total_mistakes):,}</div><div class='qc-sub'>Mistakes (calc, 100% tickets excluded)</div></div>", unsafe_allow_html=True)

st.divider()

# ============================================================
# CHARTS ROW
# ============================================================
left, right = st.columns([1, 1])

with left:
    st.markdown("### Ticket type split")
    type_counts = df_tickets["Ticket Type"].value_counts().reset_index()
    type_counts.columns = ["Ticket Type", "count"]
    fig = px.pie(type_counts, values="count", names="Ticket Type", hole=0.55)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("### Tickets by city")
    city_counts = df_tickets["Resolved City"].value_counts().reset_index()
    city_counts.columns = ["city", "ticket_count"]
    fig = px.bar(city_counts.head(20), x="city", y="ticket_count")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# AGENT TABLES
# ============================================================
st.divider()
st.markdown("### Agent performance")

tab1, tab2 = st.tabs(["Catalogue Agents", "Studio Agents"])

def agent_table(df_in: pd.DataFrame, agent_col: str, score_col: str):
    d = df_in.copy()
    # explode joined list back into multiple agents for fair counting
    d[agent_col] = d[agent_col].fillna("Unassigned")
    d = d.assign(**{agent_col: d[agent_col].astype(str).str.split(r"\s*,\s*")}).explode(agent_col)
    d[agent_col] = d[agent_col].fillna("Unassigned").replace("", "Unassigned")

    if exclude_unassigned:
        d = d[d[agent_col] != "Unassigned"]

    out = (
        d.groupby(agent_col, as_index=False)
        .agg(
            Tickets=("Ticket Count", "sum"),
            Avg_Score=(score_col, "mean"),
            Mistakes=("Mistakes (calc)", "sum"),
        )
        .sort_values(["Tickets", "Avg_Score"], ascending=[False, False])
    )
    out["Avg_Score"] = out["Avg_Score"].fillna(0).map(lambda x: f"{x:.2f}%")
    out["Mistakes"] = out["Mistakes"].fillna(0).astype(int)
    return out

with tab1:
    if "Catalogue Agent Name" in df_tickets.columns:
        t = agent_table(df_tickets, "Catalogue Agent Name", "Catalogue QC %")
        st.dataframe(t, use_container_width=True, height=420)
    else:
        st.info("No Catalogue agent column found.")

with tab2:
    if "Studio Agent Name" in df_tickets.columns:
        t = agent_table(df_tickets, "Studio Agent Name", "Studio QC %")
        st.dataframe(t, use_container_width=True, height=420)
    else:
        st.info("No Studio agent column found.")

# ============================================================
# RAW + TICKET TABLES
# ============================================================
st.divider()
st.markdown("### Ticket table (collapsed)")

cols_show = [
    "Ticket ID",
    "Subject",
    "Ticket Type",
    "Ticket Datetime",
    "Resolved City",
    "Ticket Score %",
    "Catalogue QC %",
    "Studio QC %",
    "Catalogue Agent Name",
    "Studio Agent Name",
    "Mistakes (calc)",
    "Deducted Points (calc)",
]
cols_show = [c for c in cols_show if c in df_tickets.columns]

display_df = df_tickets[cols_show].copy()
display_df = display_df.sort_values("Ticket Datetime", ascending=False)
st.dataframe(display_df, use_container_width=True, height=520)

with st.expander("Show raw rows (per-person rows as in the Excel)", expanded=False):
    # Apply same ticket type + date filters to raw if possible
    df_raw_view = df.copy()
    df_raw_view = df_raw_view[df_raw_view["Ticket Type"].isin(ticket_type_filter)]
    df_raw_view = df_raw_view[df_raw_view["Ticket Score %"] >= min_score]
    df_raw_view = df_raw_view[(df_raw_view["Ticket Datetime"] >= start_ts) & (df_raw_view["Ticket Datetime"] <= end_ts)]
    st.dataframe(df_raw_view, use_container_width=True, height=520)
