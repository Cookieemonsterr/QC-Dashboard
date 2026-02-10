import streamlit as st
import pandas as pd
import numpy as np
import requests
import io

# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="QC Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

SHEET_ID = "1rQHlDgQC5mZQ00fPVz20h4KEFbmsosE2"
DATA_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=xlsx"

# =============================
# DATA LOADER
# =============================
@st.cache_data(ttl=600)
def load_data():
    r = requests.get(DATA_URL, timeout=120)
    r.raise_for_status()
    df = pd.read_excel(io.BytesIO(r.content))
    return df


# =============================
# CLEANING LOGIC
# =============================
def classify_ticket(subject: str) -> str:
    if pd.isna(subject):
        return "Build Tickets"

    s = str(subject).strip().lower()

    if s.startswith("outlet catalogue update request"):
        return "Update Tickets"
    if s.startswith("new brand setup"):
        return "Build Tickets"
    if s.startswith("new outlet setup for existing brand"):
        return "Existing Tickets"

    # fallback → assign to Update (never unknown)
    return "Update Tickets"


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- Ticket Type
    df["Ticket Type"] = df["Subject"].apply(classify_ticket)

    # ---- Fix agents
    df["Studio Name"] = df["Studio Name"].fillna("No Studio Agent")
    df["Catalogue Name"] = df["Catalogue Name"].fillna("No Catalogue Agent")

    # ---- Scores
    df["Ticket Score"] = (
        df["Ticket Score"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .astype(float)
    )

    # ---- Mistake columns (numeric)
    mistake_cols = [
        c for c in df.columns
        if c not in [
            "Reference ID", "Subject", "Ticket Type",
            "Catalogue Name", "Studio Name",
            "Ticket Score"
        ]
    ]

    # Convert mistakes to numeric safely
    for c in mistake_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # ---- RULE: 100% SCORE = ZERO MISTAKES
    df.loc[df["Ticket Score"] == 100, mistake_cols] = 0

    # ---- Total Mistakes
    df["Total Mistakes"] = df[mistake_cols].sum(axis=1)

    return df


# =============================
# LOAD + CLEAN
# =============================
with st.spinner("Loading data..."):
    raw_df = load_data()
    df = clean_df(raw_df)

# =============================
# SIDEBAR FILTERS
# =============================
st.sidebar.header("Filters")

ticket_filter = st.sidebar.multiselect(
    "Ticket Type",
    options=["Build Tickets", "Update Tickets", "Existing Tickets"],
    default=["Build Tickets", "Update Tickets", "Existing Tickets"]
)

df = df[df["Ticket Type"].isin(ticket_filter)]

# =============================
# KPI ROW
# =============================
c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Tickets", len(df))
c2.metric("Build Tickets", (df["Ticket Type"] == "Build Tickets").sum())
c3.metric("Update Tickets", (df["Ticket Type"] == "Update Tickets").sum())
c4.metric("Existing Tickets", (df["Ticket Type"] == "Existing Tickets").sum())

st.divider()

# =============================
# STUDIO AGENT TABLE
# =============================
st.subheader("Studio Agent Performance")

studio_tbl = (
    df.groupby("Studio Name", dropna=False)
    .agg(
        Tickets=("Reference ID", "count"),
        Avg_Studio_Score=("Ticket Score", "mean"),
    )
    .reset_index()
    .sort_values("Tickets", ascending=False)
)

studio_tbl["Avg_Studio_Score"] = studio_tbl["Avg_Studio_Score"].round(2)

st.dataframe(
    studio_tbl,
    use_container_width=True,
    hide_index=True
)

# =============================
# CITY DISTRIBUTION
# =============================
st.subheader("Ticket Distribution by City")

if "City" in df.columns:
    city_chart = df["City"].fillna("Unknown").value_counts().reset_index()
    city_chart.columns = ["City", "Tickets"]
    st.bar_chart(city_chart.set_index("City"))

# =============================
# FOOTER
# =============================
st.caption("QC Dashboard • Auto-loaded from Google Sheets • 100% score rule enforced")
