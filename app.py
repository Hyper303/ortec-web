from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import folium
import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


TARGET_COL = "HistoricalBookedNights"
DATE_COL = "WeekStartDate"

LOCATION_COLS = [
    "SeasonalCluster",
    "CampsiteCluster",
    "BrandGroupCode",
    "CampsiteCode",
    "latitude",
    "longitude",
]

FIXED_LOCATION_PROFILE_COLS = [
    "AccoTypeRangeCode",
    "AccoKindCode",
    "AccommodationType",
    "AccommodationRange",
    "CampsiteCountry",
    "CampsiteRegion",
    "CampsiteType",
]

ACCOMMODATION_FEATURE_COLS = [
    "Airco",
    "Bedrooms",
    "DeckingType",
    "HotTub",
    "Tropical",
    "Roof",
    "Kitchen",
    "DeckingExtras",
    "Bathrooms",
    "Sleeps",
    "TV",
]

CATEGORICAL_COLS = [
    "MarketGroupCode",
    "CampsiteCode",
    "AccoKindCode",
    "AccoTypeRangeCode",
    "SpecialPeriodCode",
    "CampsiteCountry",
    "CampsiteRegion",
    "CampsiteType",
    "AccommodationType",
    "AccommodationRange",
    "Airco",
    "DeckingType",
    "HotTub",
    "Tropical",
    "Roof",
    "Kitchen",
    "DeckingExtras",
    "TV",
    "ArrivalMonth",
]

NUMERIC_COLS = [
    "WeekBeforeArrival",
    "Bedrooms",
    "Bathrooms",
    "Sleeps",
    "DiscountedPrice",
    "Capacity",
    "latitude",
    "longitude",
    "AvgTemperature",
    "stay_week_index",
    "stay_week_of_year",
    "stay_year",
]

MODEL_FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

DEFAULT_GUARDRAILS = {
    "price_floor_multiplier": 0.92,
    "price_ceiling_multiplier": 1.08,
    "low_occupancy_threshold": 0.55,
    "high_occupancy_threshold": 0.85,
}

CANDIDATE_MULTIPLIERS = (0.88, 0.92, 0.96, 1.0, 1.04, 1.08, 1.12)
HIST_START_DATE = "2025-10-06"
HIST_END_DATE = "2025-12-22"
CUSTOM_START_DATE = "2025-12-29"
CUSTOM_END_DATE = "2026-12-28"


def cast_lgb_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CATEGORICAL_COLS:
        out[col] = out[col].astype("category")
    return out


@st.cache_resource
def load_model() -> lgb.Booster:
    model_path = Path("outputs/modeling/lightgbm_poisson.txt")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return lgb.Booster(model_file=str(model_path))


@st.cache_data(show_spinner=False)
def load_guardrails() -> dict[str, float]:
    summary_path = Path("outputs/modeling/price_recommendation_summary.json")
    if not summary_path.exists():
        return DEFAULT_GUARDRAILS.copy()
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    settings = data.get("guardrail_settings", {})
    merged = DEFAULT_GUARDRAILS.copy()
    merged.update({k: float(v) for k, v in settings.items() if k in merged})
    return merged


@st.cache_data(show_spinner=True)
def load_assets(base: Path) -> dict[str, Any]:
    web = base / "web_assets"
    assets = {
        "historical_rows": pd.read_csv(web / "historical_rows.csv.gz", low_memory=False),
        "campsite_master": pd.read_csv(web / "campsite_master.csv", low_memory=False),
        "market_campsite": pd.read_csv(web / "market_campsite.csv", low_memory=False),
        "calendar_2025": pd.read_csv(web / "calendar_2025.csv", low_memory=False),
        "template_rows": pd.read_csv(web / "template_rows.csv.gz", low_memory=False),
        "monthly_defaults_site": pd.read_csv(web / "monthly_defaults_site.csv", low_memory=False),
        "monthly_defaults_global": pd.read_csv(web / "monthly_defaults_global.csv", low_memory=False),
        "value_catalog": json.loads((web / "value_catalog.json").read_text(encoding="utf-8")),
        "date_meta": json.loads((web / "date_index_meta.json").read_text(encoding="utf-8")),
    }
    for key in ["historical_rows", "calendar_2025", "template_rows"]:
        assets[key][DATE_COL] = pd.to_datetime(assets[key][DATE_COL], errors="coerce")
    return assets


def get_special_period_by_mode(calendar_2025: pd.DataFrame, week_start: pd.Timestamp, app_mode: str) -> str:
    if app_mode == "Historical Row Mode":
        s = calendar_2025.loc[calendar_2025[DATE_COL] == week_start, "SpecialPeriodCode"]
        return "Unknown" if s.empty else str(s.iloc[0])
    if calendar_2025.empty:
        return "Unknown"
    deltas = (calendar_2025[DATE_COL] - week_start).abs()
    return str(calendar_2025.loc[deltas.idxmin(), "SpecialPeriodCode"])


def prepare_model_row(selected_row: pd.Series, edits: dict[str, Any]) -> pd.DataFrame:
    row = selected_row.copy()
    for key, value in edits.items():
        row[key] = value
    model_df = pd.DataFrame([row[MODEL_FEATURE_COLS]])
    return cast_lgb_categories(model_df)


def score_candidates(
    model: lgb.Booster,
    selected_row: pd.Series,
    base_edits: dict[str, Any],
    guardrails: dict[str, float],
) -> pd.DataFrame:
    rows = []
    base_price = float(base_edits["DiscountedPrice"])
    capacity = float(base_edits["Capacity"])
    for m in CANDIDATE_MULTIPLIERS:
        edits = dict(base_edits)
        edits["DiscountedPrice"] = base_price * m
        x = prepare_model_row(selected_row, edits)
        pred_bookings = float(np.clip(model.predict(x, num_iteration=model.best_iteration)[0], 0, None))
        pred_bookings_capped = min(pred_bookings, capacity)
        pred_occ = pred_bookings_capped / max(capacity, 1e-9)
        pred_rev = edits["DiscountedPrice"] * pred_bookings_capped
        allowed = (
            (m >= guardrails["price_floor_multiplier"])
            and (m <= guardrails["price_ceiling_multiplier"])
            and ((pred_occ >= guardrails["low_occupancy_threshold"]) or (m <= 1.0))
            and ((pred_occ <= guardrails["high_occupancy_threshold"]) or (m >= 1.0))
        )
        rows.append(
            {
                "candidate_multiplier": m,
                "candidate_price": edits["DiscountedPrice"],
                "pred_bookings_capped": pred_bookings_capped,
                "pred_occupancy": pred_occ,
                "pred_revenue": pred_rev,
                "guardrail_allowed": allowed,
            }
        )
    return pd.DataFrame(rows)


def render_location_map(site_df: pd.DataFrame, selected_campsite: str) -> str | None:
    if site_df.empty:
        st.warning("No location points available for the current filters.")
        return None
    selected_site = site_df[site_df["CampsiteCode"].astype(str) == str(selected_campsite)]
    if not selected_site.empty:
        center = [float(selected_site["latitude"].iloc[0]), float(selected_site["longitude"].iloc[0])]
        zoom = 7
    else:
        center = [float(site_df["latitude"].mean()), float(site_df["longitude"].mean())]
        zoom = 5
    fmap = folium.Map(location=center, zoom_start=zoom, control_scale=True, tiles="CartoDB positron")
    for row in site_df.itertuples(index=False):
        color = "red" if str(row.CampsiteCode) == selected_campsite else "blue"
        folium.CircleMarker(
            location=[float(row.latitude), float(row.longitude)],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.8,
            tooltip=str(row.CampsiteCode),
        ).add_to(fmap)
    map_data = st_folium(fmap, height=500, width=None, returned_objects=["last_object_clicked_tooltip"])
    clicked = None if map_data is None else map_data.get("last_object_clicked_tooltip")
    return str(clicked) if clicked else None


def main() -> None:
    st.set_page_config(page_title="ORTEC Dynamic Pricing Workbench", layout="wide")
    st.title("ORTEC Dynamic Pricing Workbench")
    st.caption("Lightweight deploy build (cloud-ready): no large raw CSV is required at runtime.")

    base = Path(".")
    model = load_model()
    guardrails = load_guardrails()
    assets = load_assets(base)
    hist = assets["historical_rows"]
    campsite_master = assets["campsite_master"]
    market_campsite = assets["market_campsite"]
    calendar_2025 = assets["calendar_2025"]
    template_rows = assets["template_rows"]
    monthly_site = assets["monthly_defaults_site"]
    monthly_global = assets["monthly_defaults_global"]
    value_catalog = assets["value_catalog"]
    date_meta = assets["date_meta"]

    with st.sidebar:
        st.header("Scenario Controls")
        app_mode = st.radio("Mode", ["Historical Row Mode", "Custom Scenario Mode"], index=0)
        market_options = sorted(hist["MarketGroupCode"].dropna().astype(str).unique().tolist())
        market = st.selectbox("Market Segment", market_options, key="market_group")

        if app_mode == "Historical Row Mode":
            week_options = sorted(
                hist[(hist[DATE_COL] >= pd.Timestamp(HIST_START_DATE)) & (hist[DATE_COL] <= pd.Timestamp(HIST_END_DATE))][
                    DATE_COL
                ]
                .dropna()
                .dt.strftime("%Y-%m-%d")
                .unique()
                .tolist()
            )
        else:
            week_options = [d.strftime("%Y-%m-%d") for d in pd.date_range(CUSTOM_START_DATE, CUSTOM_END_DATE, freq="W-MON")]
        week_str = st.selectbox("WeekStartDate", week_options, key="week_start")
        week_start = pd.Timestamp(week_str)
        special_period = get_special_period_by_mode(calendar_2025, week_start, app_mode)
        st.text_input("Special Period", value=special_period, disabled=True)

    if app_mode == "Historical Row Mode":
        site_base = (
            hist[(hist["MarketGroupCode"].astype(str) == str(market)) & (hist[DATE_COL] == week_start)][["CampsiteCode"]]
            .drop_duplicates()
            .copy()
        )
    else:
        site_base = market_campsite[market_campsite["MarketGroupCode"].astype(str) == str(market)][["CampsiteCode"]].copy()

    eligible_sites = site_base.merge(
        campsite_master[["CampsiteCode", "latitude", "longitude", "CampsiteCountry", "CampsiteRegion", "CampsiteType"]],
        on="CampsiteCode",
        how="left",
    )
    if eligible_sites.empty:
        st.error("No campsite available for the current selection.")
        return

    st.subheader("Select Location on Map")
    current_campsite = st.session_state.get("campsite_code")
    if current_campsite not in eligible_sites["CampsiteCode"].astype(str).tolist():
        st.session_state["campsite_code"] = str(eligible_sites["CampsiteCode"].iloc[0])
    clicked = render_location_map(eligible_sites, st.session_state["campsite_code"])
    if clicked is not None and clicked in eligible_sites["CampsiteCode"].astype(str).tolist() and clicked != st.session_state.get(
        "campsite_code"
    ):
        st.session_state["campsite_code"] = clicked
        st.rerun()

    with st.sidebar:
        st.selectbox(
            "Campsite",
            options=eligible_sites["CampsiteCode"].astype(str).sort_values().unique().tolist(),
            key="campsite_code",
        )
        campsite_code = st.session_state["campsite_code"]

    if app_mode == "Historical Row Mode":
        filtered = hist[
            (hist["MarketGroupCode"].astype(str) == str(market))
            & (hist["CampsiteCode"].astype(str) == str(campsite_code))
            & (hist[DATE_COL] == week_start)
        ].copy()
        available_wba = sorted(filtered["WeekBeforeArrival"].dropna().astype(int).unique().tolist())
    else:
        filtered = template_rows[
            (template_rows["MarketGroupCode"].astype(str) == str(market))
            & (template_rows["CampsiteCode"].astype(str) == str(campsite_code))
        ].copy()
        available_wba = list(range(0, 53))

    if filtered.empty:
        st.error("No rows available for the current market/campsite selection.")
        return

    with st.sidebar:
        lead_time = st.select_slider("Weeks Before Arrival", options=available_wba, key="week_before_arrival")

    selected_rows = filtered[filtered["WeekBeforeArrival"].astype(int) == int(lead_time)]
    if not selected_rows.empty:
        selected_row = selected_rows.sort_values(DATE_COL, ascending=False).iloc[0].copy()
    else:
        selected_row = filtered.sort_values(DATE_COL, ascending=False).iloc[0].copy()

    selected_row[DATE_COL] = week_start
    selected_row["WeekBeforeArrival"] = int(lead_time)
    # derived_arrival_date = week_start + pd.to_timedelta(int(lead_time) * 7, unit="D")
    # arrival_month = str(int(derived_arrival_date.month))
    selected_row["ArrivalMonth"] = arrival_month
    selected_row["stay_week_of_year"] = int(week_start.isocalendar().week)
    selected_row["stay_year"] = int(week_start.year)
    if week_start.strftime("%Y-%m-%d") in date_meta["date_to_index"]:
        selected_row["stay_week_index"] = int(date_meta["date_to_index"][week_start.strftime("%Y-%m-%d")])
    else:
        max_date = pd.Timestamp(date_meta["max_known_date"])
        max_idx = int(date_meta["max_known_index"])
        delta = int((week_start - max_date).days // 7)
        selected_row["stay_week_index"] = max_idx + max(delta, 0)

    with st.sidebar:
        # st.text_input("Derived Arrival Date", value=derived_arrival_date.strftime("%Y-%m-%d"), disabled=True)
        # st.text_input("Arrival Month", value=arrival_month, disabled=True)
        st.subheader("Operational Inputs")
        if app_mode == "Historical Row Mode":
            capacity_default = float(selected_row["Capacity"])
            avg_temp_default = float(selected_row["AvgTemperature"])
        else:
            m = int(week_start.month)
            row = monthly_site[
                (monthly_site["CampsiteCode"].astype(str) == str(campsite_code)) & (monthly_site["month"] == m)
            ]
            if row.empty:
                row = monthly_global[monthly_global["month"] == m]
            if row.empty:
                capacity_default = float(selected_row["Capacity"])
                avg_temp_default = float(selected_row["AvgTemperature"])
            else:
                capacity_default = float(row["capacity_month_mean"].iloc[0])
                avg_temp_default = float(row["avgtemp_month_mean"].iloc[0])

        ctx = f"{app_mode}|{week_start.strftime('%Y-%m-%d')}|{campsite_code}|{lead_time}"
        if st.session_state.get("_op_inputs_context") != ctx:
            st.session_state["capacity_input"] = capacity_default
            st.session_state["avg_temp_input"] = avg_temp_default
            st.session_state["_op_inputs_context"] = ctx
        if st.button("Reset to Defaults", key="reset_op_inputs"):
            st.session_state["capacity_input"] = capacity_default
            st.session_state["avg_temp_input"] = avg_temp_default

        capacity = st.number_input("Capacity", min_value=0.0, value=float(st.session_state["capacity_input"]), step=1.0)
        avg_temp = st.number_input(
            "Average Temperature",
            value=float(st.session_state["avg_temp_input"]),
            step=0.1,
            format="%.1f",
        )

    tab_profile, tab_acco, tab_pricing = st.tabs(
        ["Location & Scenario Profile", "Accommodation Features", "Pricing Simulator"]
    )

    with tab_profile:
        st.subheader("Auto-filled Location Profile")
        p = campsite_master[campsite_master["CampsiteCode"].astype(str) == str(campsite_code)].copy()
        st.dataframe(p[LOCATION_COLS + FIXED_LOCATION_PROFILE_COLS], hide_index=True, width="stretch")

    with tab_acco:
        edited_accommodation: dict[str, Any] = {}
        cols = st.columns(2)
        for i, col in enumerate(ACCOMMODATION_FEATURE_COLS):
            allowed = value_catalog.get(col, [])
            cur = str(selected_row[col])
            if cur not in allowed:
                allowed = sorted(list(set(allowed + [cur])))
            choice = cols[i % 2].selectbox(col, options=allowed, index=allowed.index(cur) if cur in allowed else 0)
            edited_accommodation[col] = float(choice) if col in {"Bedrooms", "Bathrooms", "Sleeps"} else str(choice)

    with tab_pricing:
        st.subheader("Price Recommendation & What-if")
        if app_mode == "Historical Row Mode":
            base_price = float(selected_row["DiscountedPrice"])
        else:
            base_price = st.number_input(
                "Reference Price",
                min_value=0.0,
                value=float(selected_row["DiscountedPrice"]),
                step=1.0,
                format="%.2f",
            )
        custom_multiplier = st.slider("Custom Price Multiplier", 0.85, 1.15, 1.00, 0.01)

        base_edits = {
            "Capacity": float(capacity),
            "AvgTemperature": float(avg_temp),
            "DiscountedPrice": base_price,
            "ArrivalMonth": arrival_month,
            "SpecialPeriodCode": special_period,
            **edited_accommodation,
        }
        candidates = score_candidates(model, selected_row, base_edits, guardrails)
        allowed = candidates[candidates["guardrail_allowed"]]
        best = allowed.sort_values("pred_revenue", ascending=False).iloc[0] if not allowed.empty else candidates.sort_values(
            "pred_revenue", ascending=False
        ).iloc[0]
        current = candidates[np.isclose(candidates["candidate_multiplier"], 1.0)].iloc[0]

        k1, k2, k3, k4 = st.columns(4)
        if app_mode == "Historical Row Mode":
            k1.metric("Current Price", f"{current['candidate_price']:.2f}")
            k2.metric("Recommended Price", f"{best['candidate_price']:.2f}", f"{(best['candidate_multiplier']-1)*100:.2f}%")
            k3.metric("Expected Booking Uplift", f"{best['pred_bookings_capped']-current['pred_bookings_capped']:.3f}")
            k4.metric("Expected Revenue Uplift", f"{best['pred_revenue']-current['pred_revenue']:.2f}")
        else:
            k1.metric("Reference Price", f"{current['candidate_price']:.2f}")
            k2.metric("Recommended Price", f"{best['candidate_price']:.2f}", f"{(best['candidate_multiplier']-1)*100:.2f}%")
            k3.metric("Predicted Incremental Bookings @ Recommended", f"{best['pred_bookings_capped']:.3f}")
            k4.metric("Predicted Incremental Revenue @ Recommended", f"{best['pred_revenue']:.2f}")

        rev = candidates["pred_revenue"].sort_values(ascending=False).to_numpy()
        top2_gap = float(rev[0] - rev[1]) if len(rev) >= 2 else 0.0
        e1, e2 = st.columns(2)
        e1.metric("Top-2 Revenue Gap", f"{top2_gap:.2f}")
        e2.metric("Recommended Occupancy", f"{best['pred_occupancy']:.2%}")

        custom_edits = dict(base_edits)
        custom_edits["DiscountedPrice"] = base_price * custom_multiplier
        x = prepare_model_row(selected_row, custom_edits)
        pred_bookings = float(np.clip(model.predict(x, num_iteration=model.best_iteration)[0], 0, None))
        pred_bookings_capped = min(pred_bookings, float(capacity))
        pred_revenue = float(custom_edits["DiscountedPrice"]) * pred_bookings_capped
        pred_occ = pred_bookings_capped / max(float(capacity), 1e-9)
        w1, w2, w3 = st.columns(3)
        w1.metric("What-if Predicted Incremental Bookings", f"{pred_bookings_capped:.3f}")
        w2.metric("What-if Predicted Occupancy", f"{pred_occ:.2%}")
        w3.metric("What-if Predicted Incremental Revenue", f"{pred_revenue:.2f}")

        st.dataframe(candidates.sort_values("candidate_multiplier"), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
