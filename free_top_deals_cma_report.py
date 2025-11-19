#!/usr/bin/env python3

# Example Usage (CLI):
#   py one_time_cma.py --id 5678 --csv properties.csv --out cma.pdf --prepare-export
#
import argparse
import math
import os
import re
import sys
import tempfile
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont
from staticmap import StaticMap, CircleMarker
import requests
from datetime import datetime

# -------------------- Config / Brand --------------------
LEFT_MARGIN = 10.0
RIGHT_MARGIN = 10.0
PAGE_WIDTH = 210.0
CONTENT_W = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN

BRAND_RED      = (229, 12, 20)
BRAND_BLACK    = (0, 0, 0)
BRAND_WHITE    = (255, 255, 255)
BRAND_AQUA     = (0, 210, 232)
BRAND_DIVIDER  = (60, 60, 60)

# Optional logo (safe if missing)
#APA_LOGO_PATH = r"C:\Users\GM\Desktop\Top Deals Aruba\img\aruba_property_appraisals_logo.png"
APA_LOGO_PATH = "logo_transparent_background.png"

# -------------------- Utilities --------------------
def _extract_id(raw_id: str) -> str:
    if not isinstance(raw_id, str):
        return str(raw_id)
    m = re.search(r"-(\d+)-", raw_id)
    if m:
        return m.group(1)
    nums = re.findall(r"\d+", raw_id)
    return nums[-1] if nums else raw_id

def _to_float(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("$", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan

def _to_int(x):
    if pd.isna(x) or str(x).strip() == "":
        return np.nan
    try:
        return int(float(str(x).strip()))
    except Exception:
        return np.nan

def _fix_longitude(lon):
    if pd.isna(lon):
        return np.nan
    try:
        v = float(lon)
        return -abs(v)
    except Exception:
        return np.nan



def haversine_km(lat1, lon1, lat2, lon2) -> float:
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return np.nan
    R = 6371.0088
    p = math.pi / 180
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (math.sin(dlat/2)**2
         + math.cos(lat1*p)*math.cos(lat2*p)*math.sin(dlon/2)**2)
    return 2 * R * math.asin(math.sqrt(a))


def extract_number_before_m(val):
    if pd.isna(val):
        return pd.NA
    # If already numeric, just return it
    if isinstance(val, (int, float, np.number)):
        try:
            return float(val)
        except Exception:
            return pd.NA
    s = str(val).strip().lower()
    m = re.search(r"([\d.,]+)\s*m", s)
    if not m:
        # If no 'm' but the whole thing looks numeric, return that too
        try:
            return float(s.replace(",", ""))
        except Exception:
            return pd.NA
    num = m.group(1).replace(",", "")
    try:
        return float(num)
    except ValueError:
        return pd.NA

        
def dedupe_properties(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Latitude", "Longitude", "Lot size (M^2)", "Built up size (M^2)"}
    if not required.issubset(df.columns):
        return df
    def _id_rank(val, fallback):
        if pd.isna(val):
            return fallback
        m = re.search(r"(\d+)\s*$", str(val))
        return int(m.group(1)) if m else fallback
    df = df.copy()
    df["_rank"] = [_id_rank(val, idx) for idx, val in zip(df.index, df.get("ID", pd.Series(index=df.index)))]
    df["_lat_r"]   = df["Latitude"]
    df["_lon_r"]   = df["Longitude"]
    df["_lot_r"]   = df["Lot size (M^2)"]
    df["_built_r"] = df["Built up size (M^2)"]
    keep_idx = df.groupby(["_lat_r","_lon_r","_lot_r","_built_r"])["_rank"].idxmax()
    out = df.loc[keep_idx].sort_index()
    return out.drop(columns=["_rank","_lat_r","_lon_r","_lot_r","_built_r"], errors="ignore")


def _subject_from_csv_like(row_dict: Dict[str, Any]) -> pd.Series:
    """
    Accepts a dict shaped like one CSV row using your CSV column names,
    normalizes it with `normalize_df_for_comps`, and returns a 1-row Series
    compatible with build_comps/generate_cma_report.

    Expected keys (case-sensitive, same as CSV):
      'ID' (optional), 'Link', 'Price ($)', 'Lot size (M^2)', 'Built up size (M^2)',
      'Bedrooms', 'Baths', 'Latitude', 'Longitude', 'Status' (optional), 'Image URL' (optional)
    """
    # Create a one-row DataFrame with **exact** CSV headers you use
    df_one = pd.DataFrame([{
        "ID": row_dict.get("ID", row_dict.get("id", "")),
        "Link": row_dict.get("Link", row_dict.get("link", "")),
        "Price ($)": row_dict.get("Price ($)", row_dict.get("price", "")),
        "Lot size (M^2)": row_dict.get("Lot size (M^2)", row_dict.get("land_m2", "")),
        "Built up size (M^2)": row_dict.get("Built up size (M^2)", row_dict.get("built_m2", "")),
        "Bedrooms": row_dict.get("Bedrooms", row_dict.get("beds", "")),
        "Baths": row_dict.get("Baths", row_dict.get("baths", "")),
        "Latitude": row_dict.get("Latitude", row_dict.get("lat", row_dict.get("latitude", ""))),
        "Longitude": row_dict.get("Longitude", row_dict.get("lon", row_dict.get("longitude", ""))),
        "Status": row_dict.get("Status", row_dict.get("status", "")),
        "Image URL": row_dict.get("Image URL", row_dict.get("image_url", "")),
    }])

    # Apply the *same* cleaning your loader would
    for col in ["Lot size (M^2)", "Built up size (M^2)"]:
        if col in df_one.columns:
            df_one[col] = df_one[col].map(extract_number_before_m)

    # Force numerics where applicable
    for col in ["Lot size (M^2)", "Built up size (M^2)", "Bedrooms", "Baths", "Latitude", "Longitude", "Price ($)"]:
        if col in df_one.columns:
            df_one[col] = pd.to_numeric(df_one[col], errors="coerce")

    # Ensure Aruba longitude negative
    if "Longitude" in df_one.columns:
        df_one["Longitude"] = pd.to_numeric(df_one["Longitude"], errors="coerce").apply(
            lambda v: (-abs(v)) if pd.notna(v) else np.nan
        )

    # Reuse your normalizer to get comps schema
    df_norm = normalize_df_for_comps(df_one)
    if df_norm.empty:
        raise ValueError("Subject dict could not be normalized (missing critical fields?).")
    return df_norm.iloc[0]


# -------------------- Area assignment (geo-only) --------------------
# Aruba admin-ish centroids (approx)
_AREA_CENTROIDS = {
    "Noord":       (12.5782, -70.0421),
    "Oranjestad":  (12.5246, -70.0270),
    "Paradera":    (12.5461, -69.9748),
    "Santa Cruz":  (12.5098, -69.9672),
    "Savaneta":    (12.4387, -69.9328),
    "San Nicolas": (12.4347, -69.9000),
}
def assign_area_geo(lat, lon, max_km=None):
    if pd.isna(lat) or pd.isna(lon):
        return np.nan
    best = None
    best_d = float("inf")
    for name, (alat, alon) in _AREA_CENTROIDS.items():
        d = haversine_km(lat, lon, alat, alon)
        if d < best_d:
            best_d, best = d, name
    if max_km is not None and best_d > max_km:
        return np.nan
    return best

def bin_budget_with_ranges(price_series: pd.Series) -> pd.Series:
    return pd.cut(
        price_series,
        bins=[0, 250_000, 500_000, 750_000, 1_000_000, 2_000_000, np.inf],
        labels=["≤250k","250-500k","500-750k","750k-1M","1-2M",">2M"],
        include_lowest=True
    )

# -------------------- Loader / Cleaner (your version) --------------------
def read_property_data(file_path: str) -> pd.DataFrame:
    print("read_property_data ->", file_path)
    df = pd.read_csv(file_path, sep=None, engine="python", on_bad_lines='skip')
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
    for col in ["Lot size (M^2)", "Built up size (M^2)"]:
        if col in df.columns:
            df[col] = df[col].map(extract_number_before_m)
    print(df[["Lot size (M^2)", "Built up size (M^2)"]].head())
    print("[load] shape:", df.shape)
    df = df.replace('Not Found', 0)
    df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)

    # Price clean -> int
    df["Price ($)"] = (
        df["Price ($)"].astype(str).str.replace(r"[^\d.]", "", regex=True).astype(float).astype(int)
    )

    # ID from Link
    link_col = "link" if "link" in df.columns else ("Link" if "Link" in df.columns else None)
    if link_col:
        s = df[link_col].astype(str)
        extracted = s.str.extract(r'(?i)arubalistings\.com/(?:sale|rent)/([^/]+)/([a-z0-9-]+-\d+)')
        df["ID"] = np.where(
            extracted[0].notna() & extracted[1].notna(),
            extracted[0].str.lower().str.strip() + "/" + extracted[1].str.lower().str.strip(),
            np.nan
        )

    # Numerics
    for col in ["Lot size (M^2)", "Built up size (M^2)", "Bedrooms", "Baths", "Latitude", "Longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Lenient keep
    keep_mask = pd.Series(True, index=df.index)
    keep_mask &= df["Price ($)"].notna()
    has_any_size = df["Lot size (M^2)"].notna() | df["Built up size (M^2)"].notna()
    has_coords   = df["Latitude"].notna() & df["Longitude"].notna()
    keep_mask &= (has_any_size | has_coords)
    df = df[keep_mask]
    print("[after lenient keep] shape:", df.shape)

    # 95th clip per numeric col
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0 and len(df) > 0:
        thresholds = df[numeric_cols].quantile(0.95)
        for col in numeric_cols:
            thr = thresholds.get(col, np.nan)
            if pd.notna(thr):
                df = df[df[col].le(thr) | df[col].isna()]
    print("[after 95th filter] shape:", df.shape)

    # Filters
    df = df[df["Price ($)"] >= 100000]
    df = df.dropna(subset=["Latitude","Longitude"])
    df = df[(df["Latitude"] != 0) & (df["Longitude"] != 0)]
    df["Longitude"] = -df["Longitude"].abs()
    df = df[df["Lot size (M^2)"] > 100]

    # not logical combos
    df = df.drop(df[(df["Built up size (M^2)"] > 0) & ((df["Bedrooms"] == 0) | (df["Baths"] == 0))].index)
    df = df.drop(df[(df["Built up size (M^2)"] == 0) & ((df["Bedrooms"] > 0) | (df["Baths"] > 0))].index)
    df = df.drop(df[(df["Lot size (M^2)"] == 0) & ((df["Bedrooms"] > 0) | (df["Baths"] > 0))].index)
    # drop condos by size equality later after classify

    df = dedupe_properties(df)
    df = df.dropna(subset=["Latitude", "Longitude", "Lot size (M^2)", "Built up size (M^2)"])
    print("[final] shape (after dedupe):", df.shape)
    return df

# -------------------- Classification / Normalization --------------------
def classify_property(row):
    link = str(row.get("Link", "")).lower()
    if "residential" in link:
        return "Residential"
    if "commercial" in link:
        return "Commercial"
    if "condo" in link:
        return "Condo"
    if "land" in link:
        return "Land"
    return "Residential"  # default

def normalize_df_for_comps(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df["ID"] = df["ID"].astype(str).apply(_extract_id)
    df["Property Type"] = df.apply(classify_property, axis=1)

    # drop condos by URL hint
    df = df[~df["Link"].str.contains("condo", case=False, na=False)].copy()

    # price floor when not Land
    df = df[~((df["Property Type"] != "Land") & (df["Price ($)"] < 200000))].copy()

    # area assignment (geo only)
    df["Area"] = df.apply(lambda r: assign_area_geo(r.get("Latitude"), r.get("Longitude"), max_km=30), axis=1)

    # normalized schema for comps
    out = pd.DataFrame({
        "id":       df["ID"].astype(str),
        "type":     df["Property Type"].astype(str),
        "area":     df["Area"].astype(str),
        "price":    pd.to_numeric(df["Price ($)"], errors="coerce"),
        "land_m2":  pd.to_numeric(df["Lot size (M^2)"], errors="coerce"),
        "built_m2": pd.to_numeric(df["Built up size (M^2)"], errors="coerce"),
        "lat":      pd.to_numeric(df["Latitude"], errors="coerce"),
        "lon":      pd.to_numeric(df["Longitude"], errors="coerce").apply(lambda v: -abs(v) if pd.notna(v) else np.nan),
        "beds":     pd.to_numeric(df["Bedrooms"], errors="coerce"),
        "baths":    pd.to_numeric(df["Baths"], errors="coerce"),
        "status":   df.get("Status", "").astype(str),
        "link":     df["Link"].astype(str),
        "image_url": df.get("Image URL", ""),
    })
    return out

# -------------------- Comps + Stats --------------------
def build_comps(
    df: pd.DataFrame,
    subject: pd.Series,
    top_n: int = 5,
    start_radius_km: float = 1.0,
    step_km: float = 1.0,
    max_radius_km: float = 2.5,
    size_tol_land: float = 0.20,   # interpreted as 20% band (see below)
    size_tol_built: float = 0.20,  # interpreted as 20% band (see below)
    min_required: int = 3,
    price_iqr_k: float = 0.5,
) -> pd.DataFrame:
    """
    Build comparable set around `subject`:
      - same type as subject
      - within a growing radius (start_radius_km → max_radius_km)
      - lot and built-up area within ±size tolerance (default ±20%)
      - returns up to `top_n` closest by distance

    size_tol_land / size_tol_built:
      - If value <= 1, treated as a FRACTION (0.20 = 20%)
      - If value > 1, treated as PERCENT (20 = 20%)
    """
    out_cols = [
        "similar_id", "price", "beds", "baths",
        "land_m2", "built_m2", "distance_km",
        "status", "lat", "lon", "link"
    ]

    # --- extract subject core values ---
    try:
        s_land  = float(subject.get("land_m2"))
        s_built = float(subject.get("built_m2"))
        slat    = float(subject.get("lat"))
        slon    = float(subject.get("lon"))
    except Exception:
        return pd.DataFrame(columns=out_cols)

    if not (
        np.isfinite(s_land) and np.isfinite(s_built) and
        np.isfinite(slat)   and np.isfinite(slon)   and
        s_land > 0 and s_built > 0
    ):
        # if we do not have valid size + coords, we cannot build proper comps
        return pd.DataFrame(columns=out_cols)

    subject_type = str(subject.get("type", "")).strip().lower()
    s_id = subject.get("id")

    comps = df.copy()
    comps["similar_id"] = comps["id"].astype(str)

    # normalize numeric types
    comps["_type_norm"] = comps["type"].astype(str).str.strip().str.lower()
    comps = comps.dropna(subset=["land_m2", "built_m2", "lat", "lon", "price"])
    for c in ["land_m2", "built_m2", "lat", "lon", "price", "beds", "baths"]:
        if c in comps.columns:
            comps[c] = pd.to_numeric(comps[c], errors="coerce")

    # same type, not the subject itself
    comps = comps[(comps["_type_norm"] == subject_type) & (comps["id"] != s_id)]
    if comps.empty:
        return pd.DataFrame(columns=out_cols)

    # --- distance calculation ---
    comps["distance_km"] = comps.apply(
        lambda r: haversine_km(slat, slon, float(r["lat"]), float(r["lon"])),
        axis=1
    )

    # --- STRICT percentage tolerance on size ---
    # Treat values >1 as whole percentages (e.g. 20 -> 0.20)
    land_tol_frac  = size_tol_land / 100.0 if size_tol_land  > 1 else size_tol_land
    built_tol_frac = size_tol_built / 100.0 if size_tol_built > 1 else size_tol_built

    # safety: if someone passes negative, clamp to zero
    land_tol_frac  = max(0.0, land_tol_frac)
    built_tol_frac = max(0.0, built_tol_frac)

    def _within_tol(r):
        try:
            land  = float(r["land_m2"])
            built = float(r["built_m2"])
        except Exception:
            return False

        if not (np.isfinite(land) and np.isfinite(built)):
            return False

        # relative differences vs subject
        land_rel_diff  = abs(land  - s_land)  / s_land
        built_rel_diff = abs(built - s_built) / s_built

        # only keep comps within ±tolerance (default ±20%)
        return (land_rel_diff <= land_tol_frac) and (built_rel_diff <= built_tol_frac)

    # --- radius growth loop ---
    radius = start_radius_km
    selected = pd.DataFrame()
    pool_final_radius = None

    while radius <= max_radius_km:
        pool = comps[
            (comps["distance_km"] <= radius) &
            comps.apply(_within_tol, axis=1)
        ].copy()

        if len(pool) >= top_n:
            selected = pool.nsmallest(top_n, "distance_km")
            pool_final_radius = radius
            break
        elif len(pool) > 0:
            # keep the best we have so far, keep expanding radius
            selected = pool.nsmallest(len(pool), "distance_km")
            pool_final_radius = radius

        radius += step_km

    # not enough comps within tolerance / radius
    if selected.empty or len(selected) < min_required:
        return pd.DataFrame(columns=out_cols)

    # ensure all required columns exist
    for c in out_cols:
        if c not in selected.columns:
            selected[c] = np.nan

    # final: n closest within tolerance
    return selected.loc[:, out_cols].nsmallest(
        min(top_n, len(selected)), "distance_km"
    ).copy()



# -------------------- PDF bits --------------------
def _currency(x) -> str:
    try:
        return "${:,.0f}".format(float(x))
    except Exception:
        return "-"

def _num(x, digits=0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    try:
        v = float(x)
    except Exception:
        return "-"
    return f"{int(round(v)):,}" if digits == 0 else f"{v:,.{digits}f}"

def _safe_mean(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce").dropna()
    return float(v.mean()) if not v.empty else np.nan

def _safe_minmax(s: pd.Series):
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return (np.nan, np.nan)
    return (float(v.min()), float(v.max()))

def _plain_verdict(disc_pct: float) -> str:
    if not np.isfinite(disc_pct):
        return "The price position compared to the market could not be determined."
    if disc_pct >= 20:  return "The price is very low compared to the market."
    if disc_pct >= 10:  return "The price is low compared to the market."
    if disc_pct >= 5:   return "The price is slightly below the market."
    if disc_pct > -5:   return "The price is in line with the market."
    if disc_pct > -10:  return "The price is slightly above the market."
    if disc_pct > -20:  return "The price is high compared to the market."
    return "The price is very high compared to the market."

def _build_summary_text(subject: Dict[str, Any], comps: pd.DataFrame,
                        ask: float, avg_price: float, disc_abs: float, disc_pct: float) -> str:
    n = len(comps)
    nearest_km = pd.to_numeric(comps.get("distance_km"), errors="coerce").min()
    p_lo, p_hi = _safe_minmax(comps.get("price"))
    beds_avg   = _safe_mean(comps.get("beds"))
    baths_avg  = _safe_mean(comps.get("baths"))
    land_avg   = _safe_mean(comps.get("land_m2"))
    built_avg  = _safe_mean(comps.get("built_m2"))

    direction = "lower" if (np.isfinite(disc_pct) and disc_pct >= 0) else ("higher" if np.isfinite(disc_pct) else "")
    delta_money = f"about {_currency(abs(disc_abs))} " if np.isfinite(disc_abs) else ""
    sign = "-" if (np.isfinite(disc_pct) and disc_pct >= 0) else ("+" if np.isfinite(disc_pct) else "")
    pct = f" ({sign}{_num(abs(disc_pct), 1)}%)" if np.isfinite(disc_pct) else ""


    lines = [
        f"We compared the subject property to {n} similar listings nearby (nearest at {_num(nearest_km, 1)} km).",
        "When selecting comparables, we evaluate lot size, built-up area, and distance from the subject property.",
        f"The typical comparable size: {_num(land_avg)} m² lot and {_num(built_avg)} m² built, with {_num(beds_avg,1)} bedrooms and {_num(baths_avg,1)} bathrooms.",
        f"The average asking price of these comparables is {_currency(avg_price)} (range {_currency(p_lo)}-{_currency(p_hi)}).",
    ]

    return " ".join([l.strip() for l in lines if l.strip()])

class CMAReport(FPDF):
    def header(self):
        bar_top, bar_h = 10, 26
        self.set_fill_color(*BRAND_BLACK)
        self.rect(LEFT_MARGIN, bar_top, CONTENT_W, bar_h, "F")

        # --- Logo in center ---
        if os.path.exists(APA_LOGO_PATH):
            try:
                with Image.open(APA_LOGO_PATH) as im:
                    w_px, h_px = im.size
                aspect = w_px / h_px
                logo_h_mm = 18.0
                logo_w_mm = logo_h_mm * aspect
                x_logo = LEFT_MARGIN + (CONTENT_W - logo_w_mm) / 2.0
                y_logo = bar_top + (bar_h - logo_h_mm) * 0.25
                self.image(APA_LOGO_PATH, x=x_logo, y=y_logo, h=logo_h_mm)
            except Exception:
                pass

        # tagline
        self.set_text_color(*BRAND_WHITE)
        self.set_font("Helvetica", "", 9)
        self.set_xy(LEFT_MARGIN, bar_top + bar_h - 6)
        self.cell(CONTENT_W, 5, "Independent, data-backed valuation reports.", align="C")

        # reset text color
        self.set_text_color(0, 0, 0)

        # ================= DISCLAIMER BANNER =================
        disclaimer_text = (
            "This report was automatically generated by Aruba Property Appraisals (APA) using client-provided data to present an indicative price range based on comparable properties. To determine the exact market value, a formal appraisal is required. APA cannot verify the accuracy of the client-provided information; please contact APA for verification or to request an appraisal."
        )

        banner_y = bar_top + bar_h + 2
        banner_x = LEFT_MARGIN
        banner_w = CONTENT_W

        # text settings
        self.set_font("Helvetica", "", 8)
        self.set_text_color(40, 40, 40)

        # 1) remember start position
        self.set_xy(banner_x + 2, banner_y + 2)
        start_y = self.get_y()

        # 2) write the text (this will advance Y depending on wrapping)
        self.multi_cell(banner_w - 4, 4, disclaimer_text, align="L")

        # 3) measure how tall it became
        end_y = self.get_y()
        text_height = end_y - start_y

        # 4) draw the background *behind* it with proper height
        # add 4px padding (2 top, 2 bottom)
        box_height = text_height + 4
        self.set_fill_color(245, 245, 245)
        self.set_draw_color(210, 210, 210)
        self.set_line_width(0.2)
        # draw the rect
        self.rect(banner_x, banner_y, banner_w, box_height, style="DF")

        # 5) reprint the text on top OR move the text drawing before rect?
        # easiest is: redraw text
        self.set_xy(banner_x + 2, banner_y + 2)
        self.set_text_color(40, 40, 40)
        self.multi_cell(banner_w - 4, 4, disclaimer_text, align="L")

        # 6) move cursor below banner
        self.set_y(banner_y + box_height + 2)
        
    # -----------------------------------------------------------
    def footer(self):
        self.set_y(-10)
        self.set_font("Helvetica", "", 8)
        now = datetime.now()
        self.cell(0, 6, f"CMA Report created on {now.strftime('%B')} {now.day}, {now.year}.", align="C", ln=1)
        self.set_y(-12)
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.2)
        self.line(10, self.get_y(), 200, self.get_y())
        
def _table_fullwidth(pdf: CMAReport, headers, rows,
                     alignments=None, header_fill=BRAND_BLACK,
                     min_col_w: float = 15.0,
                     avg_row_index: int = None,
                     avg_fill=(235, 235, 235),
                     avg_text_color=(0, 0, 0),
                     divider_color=BRAND_AQUA,
                     link_col_idx: int = None,
                     link_targets: Optional[List[Optional[str]]] = None):
    n = len(headers)
    alignments = alignments or ["L"] * n
    pdf.set_font("Helvetica", "B", 9)
    col_widths = [pdf.get_string_width(str(h)) + 4 for h in headers]
    pdf.set_font("Helvetica", "", 9)
    for r in rows:
        for i, val in enumerate(r):
            w = pdf.get_string_width(str(val)) + 4
            col_widths[i] = max(col_widths[i], w)
    total_w = sum(col_widths)
    if total_w > 0:
        scale = CONTENT_W / total_w
        col_widths = [max(min_col_w, w * scale) for w in col_widths]

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(*header_fill)
    pdf.set_text_color(*BRAND_WHITE)
    for h, w, a in zip(headers, col_widths, alignments):
        pdf.cell(w, 8, str(h), border=1, align=a, fill=True)
    pdf.ln(8)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 9)

    for idx, r in enumerate(rows):
        if avg_row_index is not None and idx == avg_row_index:
            y = pdf.get_y()
            pdf.set_draw_color(*divider_color)
            pdf.line(LEFT_MARGIN, y, LEFT_MARGIN + CONTENT_W, y)
            pdf.ln(0.5)
            pdf.set_fill_color(*avg_fill)
            pdf.set_text_color(*avg_text_color)
            pdf.set_font("Helvetica", "B", 9)
            for val, w, a in zip(r, col_widths, alignments):
                pdf.cell(w, 7, str(val), border=1, align=a, fill=True)
            pdf.ln(7)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 9)
            continue

        for col_i, (val, w, a) in enumerate(zip(r, col_widths, alignments)):
            link = None
            make_link_style = False
            if link_col_idx is not None and col_i == link_col_idx and link_targets:
                link = link_targets[idx] or None
                make_link_style = link is not None
            if make_link_style:
                pdf.set_font(pdf.font_family or "Helvetica", "U", 9)
                pdf.set_text_color(0, 0, 255)
                pdf.cell(w, 7, str(val), border=1, align=a, link=link)
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(0, 0, 0)
            else:
                pdf.cell(w, 7, str(val), border=1, align=a, link=link)
        pdf.ln(7)

def _kpi_triptych(pdf: FPDF, selling_price: float, market_avg: float, discount_abs: float, discount_pct: float):
    col_w = CONTENT_W / 3.0
    x0 = LEFT_MARGIN
    y0 = pdf.get_y() + 2
    header_bg = BRAND_BLACK
    header_fg = BRAND_WHITE
    value_bg  = (245, 245, 245)
    border    = BRAND_DIVIDER
    is_discount = np.isfinite(discount_abs) and (discount_abs >= 0)
    sp  = _currency(selling_price) if np.isfinite(selling_price) and selling_price > 0 else "-"
    avg = _currency(market_avg)    if np.isfinite(market_avg)    and market_avg    > 0 else "-"
    disc_text = "-"
    if np.isfinite(discount_abs) and np.isfinite(discount_pct):
        sign = "-" if is_discount else "+"
        disc_text = f"{sign}{_currency(abs(discount_abs))} ({sign}{abs(discount_pct):.1f}%)"

    pdf.set_draw_color(*border)
    pdf.set_fill_color(*header_bg)
    pdf.set_text_color(*header_fg)
    pdf.set_font("Helvetica", "B", 11)
    for i, h in enumerate(["Subject Price", "Average on Market", "Difference"]):
        pdf.set_xy(x0 + i*col_w, y0)
        pdf.cell(col_w, 10, h, border=1, align="C", fill=True)
    y0 += 10

    pdf.set_fill_color(*value_bg)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_xy(x0, y0);       pdf.cell(col_w, 14, sp,  border=1, align="C", fill=True)
    pdf.set_xy(x0+col_w, y0); pdf.cell(col_w, 14, avg, border=1, align="C", fill=True)
    pdf.set_xy(x0 + 2*col_w, y0)
    if is_discount and disc_text != "-":
        pdf.set_fill_color(*BRAND_RED)
        pdf.set_text_color(*BRAND_WHITE)
        pdf.set_font("Helvetica", "B", 17)
        pdf.cell(col_w, 14, disc_text, border=1, align="C", fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 10)
    else:
        pdf.set_fill_color(*value_bg)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(col_w, 14, disc_text, border=1, align="C", fill=True)
    pdf.ln(6)


def _kpi_triptych_range(pdf: FPDF, low_val: float, avg_val: float, high_val: float):
    col_w = CONTENT_W / 3.0
    x0 = LEFT_MARGIN
    y0 = pdf.get_y() + 2

    header_bg = BRAND_BLACK
    header_fg = BRAND_WHITE
    value_bg  = (245, 245, 245)
    border    = BRAND_DIVIDER

    low_txt = _currency(low_val)  if np.isfinite(low_val)  else "-"
    avg_txt = _currency(avg_val)  if np.isfinite(avg_val)  else "-"
    high_txt= _currency(high_val) if np.isfinite(high_val) else "-"

    # headers
    pdf.set_draw_color(*border)
    pdf.set_fill_color(*header_bg)
    pdf.set_text_color(*header_fg)
    pdf.set_font("Helvetica", "B", 11)
    for i, h in enumerate(["Low", "Average", "High"]):
        pdf.set_xy(x0 + i*col_w, y0)
        pdf.cell(col_w, 10, h, border=1, align="C", fill=True)
    y0 += 10

    # values
    pdf.set_fill_color(*value_bg)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_xy(x0, y0);            pdf.cell(col_w, 14, low_txt,  border=1, align="C", fill=True)
    pdf.set_xy(x0 + col_w, y0);    pdf.cell(col_w, 14, avg_txt,  border=1, align="C", fill=True)
    pdf.set_xy(x0 + 2*col_w, y0);  pdf.cell(col_w, 14, high_txt, border=1, align="C", fill=True)
    pdf.ln(6)



def _download_image_from_url(url: str, side_px: int = 1200) -> str:
    import hashlib
    from io import BytesIO
    from urllib.parse import urlparse
    h = hashlib.md5((url or "").encode("utf-8")).hexdigest()[:12]
    out_path = os.path.join(tempfile.gettempdir(), f"subject_img_{h}.png")
    placeholder_path = os.path.join(tempfile.gettempdir(), "subject_img_placeholder.png")
    def _placeholder() -> str:
        try:
            if not os.path.exists(placeholder_path):
                img = Image.new("RGB", (side_px, side_px), (235, 238, 243))
                d = ImageDraw.Draw(img)
                d.rectangle([40, 40, side_px-40, side_px-40], outline=(120, 120, 120), width=6)
                d.text((side_px//2 - 60, side_px//2 - 10), "No Photo", fill=(80, 80, 80))
                img.save(placeholder_path, format="PNG")
        except Exception:
            pass
        return placeholder_path
    if os.path.exists(out_path):
        return out_path
    if not url:
        return _placeholder()
    session = requests.Session()
    ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
          "Chrome/120.0.0.0 Safari/537.36")
    try:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else None
    except Exception:
        origin = None
    referers = [origin, "https://arubalistings.com/", None]
    for ref in referers:
        try:
            headers = {
                "User-Agent": ua,
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
            }
            if ref:
                headers["Referer"] = ref
            r = session.get(url, headers=headers, timeout=20)
            if r.status_code == 403:
                alt_headers = dict(headers)
                alt_headers.pop("Referer", None)
                if not ref and origin:
                    alt_headers["Referer"] = origin
                r = session.get(url, headers=alt_headers, timeout=20)
            ct = (r.headers.get("Content-Type") or "").lower()
            if "text/html" in ct:
                continue
            r.raise_for_status()
            with Image.open(BytesIO(r.content)) as im:
                if im.mode in ("RGBA", "P"):
                    im = im.convert("RGB")
                w, h = im.size
                m = min(w, h)
                left = (w - m) // 2
                top = (h - m) // 2
                im = im.crop((left, top, left + m, top + m)).resize((side_px, side_px))
                im.save(out_path, format="PNG")
            return out_path
        except Exception:
            continue
    return _placeholder()

def plot_subject_comps_map(subject: Dict[str, Any], comps: pd.DataFrame, w: int = 600, h: int = 600) -> str:
    BRAND_BLACK_RGBA = (0, 0, 0, 255)
    BRAND_AQUA_RGBA  = (0, 210, 232, 255)
    lat_s, lon_s = subject.get("lat"), subject.get("lon")
    if pd.isna(lat_s) or pd.isna(lon_s):
        raise ValueError("Subject missing latitude/longitude")
    d = comps.dropna(subset=["lat","lon"]).copy()
    if d.empty:
        raise ValueError("No comps with coordinates")
    m = StaticMap(w, h, url_template="https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png")
    for _, r in d.iterrows():
        m.add_marker(CircleMarker((float(r["lon"]), float(r["lat"])), BRAND_AQUA_RGBA, 10))
    m.add_marker(CircleMarker((float(lon_s), float(lat_s)), BRAND_BLACK_RGBA, 14))
    image = m.render()
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    legend_x, legend_y, spacing = 20, 20, 28
    draw.ellipse((legend_x, legend_y, legend_x+15, legend_y+15), fill=BRAND_AQUA, outline=(0,0,0))
    draw.text((legend_x+25, legend_y-2), "Comparable", fill=(0,0,0), font=font)
    draw.ellipse((legend_x, legend_y+spacing, legend_x+15, legend_y+spacing+15), fill=(0,0,0), outline=(255,255,255))
    draw.text((legend_x+25, legend_y+spacing-2), "Subject", fill=(0,0,0), font=font)
    out_path = os.path.join(tempfile.gettempdir(), "subject_comps_map.png")
    image.save(out_path, "PNG")
    return out_path

def generate_cma_report(subject: Dict[str, Any], comps: pd.DataFrame, output_path: str, pdf: bool) -> str:
    ask = float(subject.get("price") or 0)
    comp_prices = pd.to_numeric(comps.get("price"), errors="coerce").dropna()
    avg_price = float(comp_prices.mean()) if not comp_prices.empty else np.nan
    low_price = float(comp_prices.min())  if not comp_prices.empty else np.nan
    high_price = float(comp_prices.max()) if not comp_prices.empty else np.nan

    # keep these if you still use them in text/verdicts
    disc_abs = (avg_price - ask) if (np.isfinite(avg_price) and ask > 0) else np.nan
    disc_pct = (disc_abs / avg_price * 100.0) if (np.isfinite(avg_price) and avg_price > 0) else np.nan

    if pdf:
        doc = CMAReport(orientation="P", unit="mm", format="A4")
        doc.set_auto_page_break(auto=True, margin=12)
        doc.add_page()

        doc.set_font("Helvetica", "B", 12)
        doc.cell(0, 8, "Subject Property", ln=1)
        doc.set_font("Helvetica", "", 10)
        doc.multi_cell(
            0, 6,
            f"ID: {_extract_id(subject.get('id','-'))}   "
            #f"Price: {_currency(ask)}   "
            f"Beds/Baths: {subject.get('beds','-')}/{subject.get('baths','-')}   "
            f"Land: {_num(subject.get('land_m2'),0)} m²   "
            f"Built: {_num(subject.get('built_m2'),0)} m²"
        )

        size = CONTENT_W / 2.0
        x0, y0 = LEFT_MARGIN, doc.get_y() + 3
        left_img = None
        if subject.get("image_url"):
            try:
                left_img = _download_image_from_url(subject.get("image_url"))
            except Exception:
                left_img = None
        try:
            map_img = plot_subject_comps_map(subject, comps)
        except Exception:
            map_img = None

        doc.set_draw_color(200, 200, 200)
        for i, p in enumerate([left_img, map_img]):
            doc.rect(x0 + i*size, y0, size, size)
            if p:
                try:
                    doc.image(p, x=x0 + i*size + 1, y=y0 + 1, w=size - 2, h=size - 2)
                except Exception:
                    pass
        doc.ln(size + 6)

        headers = ["Comparable", "Price", "Beds", "Baths", "Land (m²)", "Built (m²)", "Distance (km)", "Status"]
        aligns  = ["L","R","C","C","R","R","R","C"]
        comps_sorted = comps.sort_values(by="distance_km", ascending=True)
        rows, status_links = [], []
        for _, r in comps_sorted.iterrows():
            rows.append([
                r.get("similar_id"),
                _currency(r.get("price")),
                _num(r.get("beds")),
                _num(r.get("baths")),
                _num(r.get("land_m2")),
                _num(r.get("built_m2")),
                _num(r.get("distance_km"), 2),
                str(r.get("status")),
            ])
            status_links.append(None)
        avg_row = [
            "Averages",
            _currency(pd.to_numeric(comps_sorted["price"], errors="coerce").mean()),
            _num(pd.to_numeric(comps_sorted["beds"], errors="coerce").mean(), 1),
            _num(pd.to_numeric(comps_sorted["baths"], errors="coerce").mean(), 1),
            _num(pd.to_numeric(comps_sorted["land_m2"], errors="coerce").mean()),
            _num(pd.to_numeric(comps_sorted["built_m2"], errors="coerce").mean()),
            _num(pd.to_numeric(comps_sorted["distance_km"], errors="coerce").mean(), 2),
            "",
        ]
        rows.append(avg_row); status_links.append(None)
        _table_fullwidth(doc, headers, rows, alignments=aligns, avg_row_index=len(rows)-1, link_col_idx=7, link_targets=status_links)

        doc.ln(2)
        _kpi_triptych_range(doc, low_val=low_price, avg_val=avg_price, high_val=high_price)

        doc.ln(11)
        summary_text = _build_summary_text(subject, comps_sorted, ask, avg_price, disc_abs, disc_pct)
        doc.set_font("Helvetica", "B", 12); doc.set_text_color(*BRAND_BLACK); doc.cell(0, 8, "Market Analysis", ln=1)
        doc.set_font("Helvetica", "", 10);  doc.set_text_color(0, 0, 0);     doc.multi_cell(0, 6, summary_text)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        doc.output(output_path)
    return ask, avg_price, disc_pct, disc_abs, low_price, high_price

# -------------------- Export builder --------------------
def build_export(df_norm: pd.DataFrame) -> pd.DataFrame:
    # compute discounts for ranking (without PDFs) using comps per row
    results = []
    for idx, row in df_norm.iterrows():
        comps = build_comps(df_norm.drop(index=row.name), row)
        if comps.empty:
            results.append({"id": row["id"], "avg_price": np.nan, "disc_pct": np.nan, "disc_abs": np.nan})
            continue
        ask = float(row.get("price") or 0)
        avg_price = pd.to_numeric(comps.get("price"), errors="coerce").mean()
        disc_abs = (avg_price - ask) if (np.isfinite(avg_price) and ask > 0) else np.nan
        disc_pct = (disc_abs / avg_price * 100.0) if (np.isfinite(avg_price) and avg_price > 0) else np.nan
        results.append({"id": row["id"], "avg_price": avg_price, "disc_pct": disc_pct, "disc_abs": disc_abs})
    stats = pd.DataFrame(results)
    df = df_norm.merge(stats, on="id", how="left")

    # budget group + coordinates text
    price_num = pd.to_numeric(df["price"], errors="coerce")
    df["Budget Group"] = bin_budget_with_ranges(price_num)

    def _fmt_coord(lat, lon):
        try:
            if pd.notna(lat) and pd.notna(lon):
                return f"{float(lat):.6f}, {float(lon):.6f}"
        except Exception:
            pass
        return ""
    df["coordinates"] = df.apply(lambda r: _fmt_coord(r.get("lat"), r.get("lon")), axis=1)

    # rank within (area, budget)
    df = df.sort_values(by=["area","Budget Group","disc_pct"], ascending=[True, True, False], na_position="last")
    rank_series = df.groupby(["area","Budget Group"], dropna=False, observed=False)["disc_pct"].rank(method="first", ascending=False)
    rank_series[df["disc_pct"].isna()] = np.nan
    df["Rank (Area+Budget)"] = rank_series.astype("Int64")

    export = pd.DataFrame({
        "ID": df["id"],
        "Type": df["type"],
        "Area": df["area"],
        "Price ($)": price_num,
        "Built-up area (squared meters)": pd.to_numeric(df["built_m2"], errors="coerce"),
        "Bedrooms": pd.to_numeric(df["beds"], errors="coerce"),
        "Bathrooms": pd.to_numeric(df["baths"], errors="coerce"),
        "Lot size (squared meters)": pd.to_numeric(df["land_m2"], errors="coerce"),
        "Built Year": np.nan,
        "About": df["link"],
        "Thumbnail Image": np.nan,
        "Images": np.nan,
        "Agent": np.nan,
        "Property Name": np.nan,
        "Properties (Property Name)": np.nan,
        "Thumbnail image alt text": np.nan,
        "Featured, Oceanfront, New Listing": np.nan,
        "Coordinates": df["coordinates"],
        "Similar Properties ($)": pd.to_numeric(df["avg_price"], errors="coerce").round(0).astype("Int64"),
        "Your Discount (%)": df["disc_pct"].round(2),
        "Status": df["status"],
        "Rank (Area+Budget)": df["Rank (Area+Budget)"],
        "Budget Group": df["Budget Group"],
        "Latitude": pd.to_numeric(df["lat"], errors="coerce"),
        "Longitude": pd.to_numeric(df["lon"], errors="coerce"),
        "Image URL": df["image_url"].fillna(""),
    })

    # keep only rows that have a rank (like your note)
    export = export[export["Rank (Area+Budget)"].notna() & (export["Rank (Area+Budget)"].astype(str).str.strip() != "")]
    return export

# -------------------- Dict-driven runner --------------------
def run_cma_from_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use CSV as the comps pool, but allow passing a subject that is NOT in the CSV.
    Options:
      - params['subject_csv']: dict with CSV-like keys (preferred). See _subject_from_csv_like docstring.
      - OR fallback to params['id'] (selects subject from CSV as before).

    Also supported (unchanged):
      out='cma.pdf', prepare_export=False, export_path='rankings.csv', generate_all_pdfs=False,
      top_n=5, start_radius_km=1.0, step_km=1.0, max_radius_km=25.0,
      size_tol_land=200.0, size_tol_built=100.0, min_required=3, price_iqr_k=0.5
    """
    p = {
        "csv": params.get("csv", "properties.csv"),
        "id": params.get("id"),
        "subject_csv": params.get("subject_csv"),
        "out": params.get("out", "cma.pdf"),
        "prepare_export": bool(params.get("prepare_export", False)),
        "export_path": params.get("export_path", "rankings.csv"),
        "generate_all_pdfs": bool(params.get("generate_all_pdfs", False)),
        "top_n": int(params.get("top_n", 3)),
        "start_radius_km": float(params.get("start_radius_km", 1.0)),
        "step_km": float(params.get("step_km", 1.0)),
        "max_radius_km": float(params.get("max_radius_km", 10)),
        "size_tol_land": float(params.get("size_tol_land", 0.20)),
        "size_tol_built": float(params.get("size_tol_built", 0.20)),
        "min_required": int(params.get("min_required", 3))
    }
    if not p["csv"]:
        raise ValueError("csv path is required for comps pool.")

    # 1) load CSV as comps pool
    df_raw = read_property_data(p["csv"])
    df_norm = normalize_df_for_comps(df_raw)

    # 2) choose subject:
    #    - if subject_csv is provided, build subject from the dict (NOT in CSV)
    #    - else, pick by id from df_norm (original behavior)
    if isinstance(p["subject_csv"], dict):
        subject = _subject_from_csv_like(p["subject_csv"])
        # ensure we do not match the subject itself from pool (in case same ID appears)
        df_norm = df_norm[df_norm["id"].astype(str) != str(subject.get("id", ""))].copy()
    else:
        if p["id"] is None:
            raise ValueError("Provide either subject_csv (dict) or id (to select from CSV).")
        target_id = _extract_id(str(p["id"]))
        hit = df_norm[df_norm["id"].astype(str) == target_id]
        if hit.empty:
            raise RuntimeError(f"ID '{target_id}' not found after normalization.")
        subject = hit.iloc[0]
        df_norm = df_norm.drop(index=subject.name)

    # 3) (optional) export with subject included for ranking context
    export_rows = None
    if p["prepare_export"]:
        export_df = build_export(pd.concat([df_norm, subject.to_frame().T], ignore_index=True))
        export_df.to_csv(p["export_path"], index=False)
        export_rows = int(len(export_df))

    # 4) comps for subject
    comps = build_comps(
        df_norm, 
        subject,
        top_n=p["top_n"],
        start_radius_km=p["start_radius_km"],
        step_km=p["step_km"],
        max_radius_km=p["max_radius_km"],
        size_tol_land=p["size_tol_land"],
        size_tol_built=p["size_tol_built"],
        min_required=p["min_required"]
    )

    if comps.empty:
        raise RuntimeError("Not enough comparable properties for the subject; no PDF generated.")

    ask, avg_price, disc_pct, disc_abs, low_price, high_price = generate_cma_report(subject.to_dict(), comps, p["out"], pdf=True)

    return {
        "out_pdf": p["out"],
        "target_id": str(subject.get("id", "")),
        "ask": ask,
        "avg_price": avg_price,
        "discount_pct": disc_pct,
        "discount_abs": disc_abs,
        "comp_low": low_price,
        "comp_high": high_price,
        "export_path": p["export_path"] if p["prepare_export"] else None,
        "export_rows": export_rows,
        "all_pdfs": [],
    }

# -------------------- MAIN --------------------
if __name__ == "__main__":

    '''    
    form_submit = {
        "csv": "properties.csv",
        "out": "reports/cma_new_subject.pdf",
        "subject_csv": {
            "ID": "form address",
            "Lot size (M^2)": 1000,
            "Built up size (M^2)": 300,
            "Bedrooms": 6,
            "Baths": 5,
            "Latitude": 12.581671,
            "Longitude": -70.0424,
            "Image URL": "https://static.wixstatic.com/media/5711f6_ec3d3ddc05f541a983bf2084cdfb594c~mv2.png"  
        },
    }
    '''
    form_submit = {
        "csv": "properties.csv",
        "out": "reports/cma_new_subject.pdf",
        "subject_csv": {
            "ID": "form address",
            "Lot size (M^2)": 1000,
            "Built up size (M^2)": 598,
            "Bedrooms": 6,
            "Baths": 5,
            "Latitude": 12.5506,
            "Longitude": -70.0502,
            "Image URL": "https://static.wixstatic.com/media/5711f6_ec3d3ddc05f541a983bf2084cdfb594c~mv2.png"  
        },
    }



    

    
    res = run_cma_from_params(form_submit)
    print(res)
    
    