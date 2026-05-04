from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COL = "HistoricalBookedNights"
DATE_COL = "WeekStartDate"

HIST_START = pd.Timestamp("2025-10-06")
HIST_END = pd.Timestamp("2025-12-22")

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
]

BASE_COLS = list(
    dict.fromkeys(
        [DATE_COL, TARGET_COL, "WeekBeforeArrival", "ArrivalMonth", "MarketGroupCode", "SpecialPeriodCode"]
        + LOCATION_COLS
        + FIXED_LOCATION_PROFILE_COLS
        + ACCOMMODATION_FEATURE_COLS
        + ["DiscountedPrice", "Capacity", "AvgTemperature"]
    )
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_py(v):
    if pd.isna(v):
        return None
    return v.item() if hasattr(v, "item") else v


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lightweight web assets from simulation_output.csv")
    parser.add_argument("--data", type=Path, default=Path("simulation_output.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("ortec_web/web_assets"))
    args = parser.parse_args()

    ensure_dir(args.outdir)

    date_set: set[pd.Timestamp] = set()
    historical_chunks: list[pd.DataFrame] = []
    campsite_master_rows: dict[str, dict] = {}
    market_campsite_pairs: set[tuple[str, str]] = set()
    calendar_2025: dict[str, str] = {}
    value_catalog_sets: dict[str, set[str]] = {c: set() for c in ACCOMMODATION_FEATURE_COLS}
    template_rows: dict[tuple[str, str, int], dict] = {}
    monthly_site_sums: dict[tuple[str, int], list[float]] = {}
    monthly_global_sums: dict[int, list[float]] = {}

    for chunk in pd.read_csv(args.data, usecols=BASE_COLS, chunksize=250_000, low_memory=False):
        chunk[DATE_COL] = pd.to_datetime(chunk[DATE_COL], errors="coerce")
        chunk = chunk.dropna(subset=[DATE_COL])
        if chunk.empty:
            continue

        # Track unique dates.
        date_set.update(pd.Timestamp(d) for d in chunk[DATE_COL].dropna().unique())

        # Categorical cleanup.
        for col in CATEGORICAL_COLS:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype("string").fillna("Missing")
        for col in NUMERIC_COLS + [TARGET_COL]:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        # Historical rows (for historical mode runtime selection).
        hist_mask = (chunk[DATE_COL] >= HIST_START) & (chunk[DATE_COL] <= HIST_END)
        hist = chunk.loc[hist_mask].copy()
        if not hist.empty:
            historical_chunks.append(hist)

        # Campsite master + market-campsite map + value catalog.
        for row in chunk.itertuples(index=False):
            campsite = str(getattr(row, "CampsiteCode"))
            market = str(getattr(row, "MarketGroupCode"))
            market_campsite_pairs.add((market, campsite))

            if campsite not in campsite_master_rows:
                rec = {}
                for c in LOCATION_COLS + FIXED_LOCATION_PROFILE_COLS + ACCOMMODATION_FEATURE_COLS:
                    rec[c] = to_py(getattr(row, c))
                campsite_master_rows[campsite] = rec

            for c in ACCOMMODATION_FEATURE_COLS:
                value_catalog_sets[c].add(str(getattr(row, c)))

            # Template row for custom mode (latest date per market+campsite+wba).
            wba = int(getattr(row, "WeekBeforeArrival"))
            key = (market, campsite, wba)
            cur_date = pd.Timestamp(getattr(row, DATE_COL))
            prev = template_rows.get(key)
            if prev is None or cur_date > prev["_date"]:
                rec = {c: to_py(getattr(row, c)) for c in BASE_COLS}
                rec["_date"] = cur_date
                template_rows[key] = rec

            # 2025 calendar mapping.
            dt = pd.Timestamp(getattr(row, DATE_COL))
            if dt.year == 2025:
                calendar_2025[dt.strftime("%Y-%m-%d")] = str(getattr(row, "SpecialPeriodCode"))

            # Monthly defaults from 2024+2025.
            if dt.year in {2024, 2025}:
                m = int(dt.month)
                cap = float(getattr(row, "Capacity")) if pd.notna(getattr(row, "Capacity")) else np.nan
                tmp = float(getattr(row, "AvgTemperature")) if pd.notna(getattr(row, "AvgTemperature")) else np.nan

                if np.isfinite(cap) and np.isfinite(tmp):
                    k_site = (campsite, m)
                    if k_site not in monthly_site_sums:
                        monthly_site_sums[k_site] = [0.0, 0.0, 0]
                    monthly_site_sums[k_site][0] += cap
                    monthly_site_sums[k_site][1] += tmp
                    monthly_site_sums[k_site][2] += 1

                    if m not in monthly_global_sums:
                        monthly_global_sums[m] = [0.0, 0.0, 0]
                    monthly_global_sums[m][0] += cap
                    monthly_global_sums[m][1] += tmp
                    monthly_global_sums[m][2] += 1

    # date meta
    unique_dates_sorted = sorted(date_set)
    date_mapping = {d.strftime("%Y-%m-%d"): i + 1 for i, d in enumerate(unique_dates_sorted)}
    date_meta = {
        "max_known_date": unique_dates_sorted[-1].strftime("%Y-%m-%d"),
        "max_known_index": len(unique_dates_sorted),
        "date_to_index": date_mapping,
    }

    # write historical rows
    historical_df = pd.concat(historical_chunks, ignore_index=True) if historical_chunks else pd.DataFrame(columns=BASE_COLS)
    historical_df.to_csv(args.outdir / "historical_rows.csv.gz", index=False, compression="gzip")

    # write campsite master
    campsite_master = pd.DataFrame(campsite_master_rows.values())
    campsite_master.to_csv(args.outdir / "campsite_master.csv", index=False)

    # market-campsite
    market_campsite = pd.DataFrame(sorted(list(market_campsite_pairs)), columns=["MarketGroupCode", "CampsiteCode"])
    market_campsite.to_csv(args.outdir / "market_campsite.csv", index=False)

    # calendar
    cal = pd.DataFrame(
        [{"WeekStartDate": k, "SpecialPeriodCode": v} for k, v in sorted(calendar_2025.items(), key=lambda x: x[0])]
    )
    cal.to_csv(args.outdir / "calendar_2025.csv", index=False)

    # value catalog
    value_catalog = {k: sorted(list(v)) for k, v in value_catalog_sets.items()}
    (args.outdir / "value_catalog.json").write_text(json.dumps(value_catalog, indent=2), encoding="utf-8")

    # template rows
    template_df = pd.DataFrame([{k: v for k, v in rec.items() if k != "_date"} for rec in template_rows.values()])
    template_df.to_csv(args.outdir / "template_rows.csv.gz", index=False, compression="gzip")

    # monthly defaults site/global
    rows_site = []
    for (site, m), (cap_sum, temp_sum, n) in monthly_site_sums.items():
        rows_site.append(
            {
                "CampsiteCode": site,
                "month": m,
                "capacity_month_mean": cap_sum / max(n, 1),
                "avgtemp_month_mean": temp_sum / max(n, 1),
            }
        )
    pd.DataFrame(rows_site).to_csv(args.outdir / "monthly_defaults_site.csv", index=False)

    rows_global = []
    for m, (cap_sum, temp_sum, n) in monthly_global_sums.items():
        rows_global.append(
            {
                "month": m,
                "capacity_month_mean": cap_sum / max(n, 1),
                "avgtemp_month_mean": temp_sum / max(n, 1),
            }
        )
    pd.DataFrame(rows_global).to_csv(args.outdir / "monthly_defaults_global.csv", index=False)

    (args.outdir / "date_index_meta.json").write_text(json.dumps(date_meta, indent=2), encoding="utf-8")
    print(f"Saved web assets to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
