import os
import io
import pathlib
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Retail Sales Drilldown Dashboard", layout="wide")
st.set_option('client.showErrorDetails', True)

# Paths
BASE_DIR = pathlib.Path(os.getcwd())
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Helpers ----------
REQUIRED_MIN = ["Date", "Sales"]  # must exist or we stop
# Nice-to-have optional columns
OPT_COLS = [
    "Country", "State", "City", "Region", "SubRegion",
    "Store_Type", "Store", "Store_Sqft", "SqFt", "Area_Sqft",
    "Category", "SubCategory", "Product_Line", "Brand", "Product", "SKU", "Profit"
]

def detect_sheet(xls: pd.ExcelFile) -> str:
    # Prefer Sheet2 if present. Else first sheet.
    return "Sheet2" if "Sheet2" in xls.sheet_names else xls.sheet_names[0]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize common column names to a canonical model
    rename_map = {
        "Product Line": "Product_Line",
        "Store Type": "Store_Type",
        "Date old": "Date_old",
        "Brand Name": "Brand",
        "Sub Category": "SubCategory",
        "SquareFeet": "Store_Sqft",
        "Sq Ft": "Store_Sqft",
        "Store Sqft": "Store_Sqft",
        "SQFT": "Store_Sqft",
        "Region Name": "Region",
        "Sub Region": "SubRegion",
    }
    df = df.rename(columns=rename_map)

    # Date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    elif "Date_old" in df.columns:
        df["Date"] = pd.to_datetime(df["Date_old"], errors="coerce")

    # Required checks
    for c in REQUIRED_MIN:
        if c not in df.columns:
            st.error(f"Missing required column: {c}")
            st.stop()

    # Sales numeric
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0.0)

    # Optional numeric
    if "Profit" in df.columns:
        df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce").fillna(0.0)

    # Square feet. Accept several aliases
    sqft_col = None
    for cand in ["Store_Sqft", "SqFt", "Area_Sqft"]:
        if cand in df.columns:
            sqft_col = cand
            break
    if sqft_col is None:
        df["Store_Sqft"] = pd.NA
        sqft_col = "Store_Sqft"

    # Derivations
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["MonthStart"] = df["Date"].values.astype("datetime64[M]")

    # Ensure key dims exist. Fill Unknown
    canon_dims = [
        "Country", "State", "City", "Region", "SubRegion",
        "Store_Type", "Store",
        "Category", "SubCategory", "Product_Line", "Brand", "Product", "SKU"
    ]
    for col in canon_dims:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].astype(str).str.strip()

    # Sales per sqft if sqft available and > 0
    if df[sqft_col].notna().any():
        with pd.option_context('mode.use_inf_as_na', True):
            df["Sales_per_SqFt"] = df["Sales"] / pd.to_numeric(df[sqft_col], errors="coerce")
    else:
        df["Sales_per_SqFt"] = pd.NA

    return df

def agg(df: pd.DataFrame, by_cols, val="Sales"):
    g = df.groupby(by_cols, dropna=False, as_index=False)[val].sum()
    g = g.sort_values(val, ascending=False)
    return g

def build_outputs(df: pd.DataFrame):
    # Geography. Prefer Region/SubRegion when available. Fall back to Country/State/City.
    geo_lvl1 = "Region" if "Region" in df.columns else "Country"
    geo_lvl2 = "SubRegion" if "SubRegion" in df.columns else "State"
    geo_lvl3 = "City"

    national = agg(df, ["Year", "Month", "MonthStart", geo_lvl1])
    regional = agg(df, ["Year", "Month", "MonthStart", geo_lvl1, geo_lvl2])
    area = agg(df, ["Year", "Month", "MonthStart", geo_lvl1, geo_lvl2, geo_lvl3])

    # Retail format wise
    cluster = agg(df, ["Year", "Month", "MonthStart", geo_lvl1, geo_lvl2, "Store_Type"])
    stores = agg(df, ["Year", "Month", "MonthStart", geo_lvl1, geo_lvl2, "Store_Type", "Store"])

    # Category stack
    category = agg(df, ["Year", "Month", "MonthStart", "Category"] if "Category" in df.columns else ["Year", "Month", "MonthStart", "Product_Line"])
    subcategory = agg(df, ["Year", "Month", "MonthStart", "Category", "SubCategory"]) if "SubCategory" in df.columns else None
    product_line = agg(df, ["Year", "Month", "MonthStart", "Product_Line"])
    product = agg(df, ["Year", "Month", "MonthStart", "Product_Line", "Product"])
    brand_profit = None
    if "Profit" in df.columns and "Brand" in df.columns:
        brand_profit = df.groupby(["Year","Month","MonthStart","Brand"], as_index=False).agg(Sales=("Sales","sum"), Profit=("Profit","sum"))
        brand_profit = brand_profit.sort_values("Profit", ascending=False)

    # Sales per SqFt when available
    sps = None
    if "Sales_per_SqFt" in df.columns and df["Sales_per_SqFt"].notna().any():
        sps = df.copy()
        sps["Sales_per_SqFt"] = pd.to_numeric(sps["Sales_per_SqFt"], errors="coerce")
        sps = sps.groupby(
            ["Year","Month","MonthStart", geo_lvl1, geo_lvl2, "Store_Type", "Store"], as_index=False
        ).agg(Sales=("Sales","sum"), Sales_per_SqFt=("Sales_per_SqFt","mean"))
        sps = sps.sort_values("Sales_per_SqFt", ascending=False)

    kpi_monthly = df.groupby(["Year","Month","MonthStart"], as_index=False).agg(
        Total_Sales=("Sales","sum"),
        Total_Profit=("Profit","sum") if "Profit" in df.columns else ("Sales","sum")
    )

    # Flat model for BI
    keep_cols = ["MonthStart","Year","Month", geo_lvl1, geo_lvl2, "City","Store_Type","Store",
                 "Category","SubCategory","Product_Line","Brand","Product","SKU","Sales"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    model_flat = df[keep_cols].copy()

    out = {
        "KPI_Monthly": kpi_monthly,
        "National": national.rename(columns={geo_lvl1: "Lvl1"}),
        "Regional": regional.rename(columns={geo_lvl1: "Lvl1", geo_lvl2: "Lvl2"}),
        "Area": area.rename(columns={geo_lvl1: "Lvl1", geo_lvl2: "Lvl2", geo_lvl3: "Lvl3"}),
        "Cluster": cluster.rename(columns={geo_lvl1: "Lvl1", geo_lvl2: "Lvl2"}),
        "Stores": stores.rename(columns={geo_lvl1: "Lvl1", geo_lvl2: "Lvl2"}),
        "Category": category,
        "Product_Line": product_line,
        "Product": product,
        "Model_Flat": model_flat
    }
    if brand_profit is not None:
        out["Brand_Profit"] = brand_profit
    if sps is not None:
        out["Sales_per_SqFt"] = sps
    return out

def export_outputs(tables: dict):
    # Excel multi-sheet + flat CSV
    excel_path = OUTPUT_DIR / "Retail_Dashboard_Monthly.xlsx"
    csv_path = OUTPUT_DIR / "Retail_Dashboard_Flat.csv"
    with pd.ExcelWriter(excel_path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        for name, tbl in tables.items():
            if name == "Model_Flat":
                continue
            tbl.to_excel(writer, sheet_name=name[:31], index=False)  # Excel sheet name limit
    tables["Model_Flat"].to_csv(csv_path, index=False)
    return excel_path, csv_path

# ---------- UI ----------
st.title("Retail Sales Drilldown Dashboard")
st.caption("Upload a monthly Excel file. The app will process and render KPIs and drilldowns. It will also save outputs to ./output")

st.sidebar.header("Upload New Data File")
uploaded = st.sidebar.file_uploader("Upload Retail Excel (.xlsx)", type=["xlsx"])

# Load data path
df = None
if uploaded:
    # Save uploaded to memory and DataFrame
    data = uploaded.read()
    xls = pd.ExcelFile(io.BytesIO(data))
    sheet = detect_sheet(xls)
    raw = pd.read_excel(io.BytesIO(data), sheet_name=sheet)
    df = normalize_columns(raw)
else:
    # Fallback to existing processed CSV if present
    csv_existing = OUTPUT_DIR / "Retail_Dashboard_Flat.csv"
    if csv_existing.exists():
        df = pd.read_csv(csv_existing)
        if "MonthStart" in df.columns:
            df["MonthStart"] = pd.to_datetime(df["MonthStart"])
    else:
        st.info("No file uploaded yet. Please upload your Excel file in the sidebar.")
        st.stop() # Stop execution if no data is loaded

# Initialize geo_lvl names with defaults
geo_lvl1_name = "Country"
geo_lvl2_name = "State"

# Build outputs and export
if df is not None: # Add a check to ensure df is not None
    tables = build_outputs(df)
    excel_out, csv_out = export_outputs(tables)

    # Update geo_lvl names based on available columns in df
    if "Region" in df.columns:
        geo_lvl1_name = "Region"
    if "SubRegion" in df.columns:
        geo_lvl2_name = "SubRegion"


    # Filters
    st.sidebar.header("Filters")
    # Use geo_lvl1_name and geo_lvl2_name as they reflect the actual column names in df

    lvl1_vals = sorted(df[geo_lvl1_name].unique())
    sel_lvl1 = st.sidebar.selectbox(geo_lvl1_name, lvl1_vals)
    df_geo = df[df[geo_lvl1_name] == sel_lvl1]

    lvl2_vals = ["All"]
    if geo_lvl2_name in df_geo.columns:
        lvl2_vals += sorted(df_geo[geo_lvl2_name].unique())
    sel_lvl2 = st.sidebar.selectbox(geo_lvl2_name, lvl2_vals)
    if sel_lvl2 != "All" and geo_lvl2_name in df_geo.columns:
        df_geo = df_geo[df_geo[geo_lvl2_name] == sel_lvl2]

    pl_vals = sorted(df_geo.get("Product_Line", pd.Series(["Unknown"])).unique())
    sel_pl = st.sidebar.multiselect("Product Line", pl_vals, default=pl_vals)
    df_geo = df_geo[df_geo.get("Product_Line","").isin(sel_pl)] if "Product_Line" in df_geo.columns else df_geo

    # KPIs
    total_sales = float(df_geo["Sales"].sum())
    stores_count = int(df_geo.get("Store", pd.Series(dtype=str)).nunique())
    products_count = int(df_geo.get("Product", pd.Series(dtype=str)).nunique())
    profit_total = float(df_geo["Profit"].sum()) if "Profit" in df_geo.columns else None

    st.subheader("Key Performance Indicators")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Sales", f"{total_sales:,.0f}")
    k2.metric("Active Stores", stores_count)
    k3.metric("Products Sold", products_count)
    if profit_total is not None:
        k4.metric("Total Profit", f"{profit_total:,.0f}")
    else:
        k4.write("Profit not provided")

    # Tabs
    tab_geo, tab_format, tab_category, tab_product, tab_brand, tab_sqft, tab_trend = st.tabs(
        ["National and Regional", "Retail Format", "Category", "Product", "Brand Profit", "Sales per SqFt", "Monthly Trend"]
    )

    with tab_geo:
        st.markdown("**National → Regional → Area**")
        # Use geo_lvl1_name for grouping
        g1 = df.groupby([geo_lvl1_name], as_index=False)["Sales"].sum()
        st.plotly_chart(px.bar(g1, x=geo_lvl1_name, y="Sales", title="National Sales"), use_container_width=True)

        # Use geo_lvl2_name for grouping
        if geo_lvl2_name in df.columns:
            g2 = df[df[geo_lvl1_name] == sel_lvl1].groupby([geo_lvl2_name], as_index=False)["Sales"].sum()
            st.plotly_chart(px.bar(g2, x=geo_lvl2_name, y="Sales", title=f"Regional Sales in {sel_lvl1}"), use_container_width=True)

        if "City" in df.columns and geo_lvl2_name in df.columns and sel_lvl2 != "All":
            g3 = df[(df[geo_lvl1_name] == sel_lvl1) & (df[geo_lvl2_name] == sel_lvl2)].groupby(["City"], as_index=False)["Sales"].sum()
            st.plotly_chart(px.bar(g3, x="City", y="Sales", title=f"Area wise Sales in {sel_lvl2}"), use_container_width=True)

    with tab_format:
        if "Store_Type" in df_geo.columns:
            f1 = df_geo.groupby(["Store_Type"], as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
            st.plotly_chart(px.bar(f1, x="Store_Type", y="Sales", title="Retail format wise Sales"), use_container_width=True)
        else:
            st.info("Store_Type column not found")

    with tab_category:
        # Category or Product_Line
        if "Category" in df_geo.columns and df_geo["Category"].nunique() > 1:
            c1 = df_geo.groupby(["Category"], as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
            st.plotly_chart(px.bar(c1, x="Category", y="Sales", title="Category wise Sales"), use_container_width=True)
            if "SubCategory" in df_geo.columns and df_geo["SubCategory"].nunique() > 1:
                c2 = df_geo.groupby(["Category","SubCategory"], as_index=False)["Sales"].sum()
                st.plotly_chart(px.treemap(c2, path=["Category","SubCategory"], values="Sales", title="SubCategory wise Sales"), use_container_width=True)
        elif "Product_Line" in df_geo.columns:
            pl1 = df_geo.groupby(["Product_Line"], as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
            st.plotly_chart(px.bar(pl1, x="Product_Line", y="Sales", title="Product Line wise Sales"), use_container_width=True)
        else:
            st.info("Category or Product_Line column not found")

    with tab_product:
        if "Product_Line" in df_geo.columns and "Product" in df_geo.columns:
            p1 = df_geo.groupby(["Product_Line","Product"], as_index=False)["Sales"].sum()
            st.plotly_chart(px.treemap(p1, path=["Product_Line","Product"], values="Sales", title="Product wise Sales"), use_container_width=True)
        else:
            st.info("Product fields not found")

    with tab_brand:
        if "Profit" in df_geo.columns and "Brand" in df_geo.columns and df_geo["Brand"].nunique() > 1:
            bp = df_geo.groupby(["Brand"], as_index=False).agg(Sales=("Sales","sum"), Profit=("Profit","sum")).sort_values("Profit", ascending=False)
            st.plotly_chart(px.bar(bp, x="Brand", y="Profit", title="Profit brand wise"), use_container_width=True)
        else:
            st.info("Brand or Profit not available")

    with tab_sqft:
        if "Sales_per_SqFt" in df_geo.columns and df_geo["Sales_per_SqFt"].notna().any():
            sps = df_geo.copy()
            sps["Sales_per_SqFt"] = pd.to_numeric(sps["Sales_per_SqFt"], errors="coerce")
            sps = sps.groupby(["Store"], as_index=False).agg(Sales=("Sales","sum"), Sales_per_SqFt=("Sales_per_SqFt","mean")).sort_values("Sales_per_SqFt", ascending=False).head(25)
            st.plotly_chart(px.bar(sps, x="Store", y="Sales_per_SqFt", title="Sales per SqFt. Top stores"), use_container_width=True)
        else:
            st.info("Store square feet not present. Provide Store_Sqft or SqFt to compute Sales per SqFt")

    with tab_trend:
        m = df_geo.groupby(["MonthStart"], as_index=False)["Sales"].sum()
        st.plotly_chart(px.line(m, x="MonthStart", y="Sales", markers=True, title="Monthly Sales Trend"), use_container_width=True)

    # Downloads
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download processed CSV", (OUTPUT_DIR / "Retail_Dashboard_Flat.csv").read_bytes(), file_name="Retail_Dashboard_Flat.csv")
    with c2:
        st.download_button("Download multi-sheet Excel", (OUTPUT_DIR / "Retail_Dashboard_Monthly.xlsx").read_bytes(), file_name="Retail_Dashboard_Monthly.xlsx")

    st.caption(f"Data processed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    df.columns