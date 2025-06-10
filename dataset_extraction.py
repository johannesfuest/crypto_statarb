from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from dateutil.relativedelta import relativedelta

# --------------------------------------------------------------------------- #
# constants & helpers
# --------------------------------------------------------------------------- #
SRC_ROOT = Path("/share/data/jfuest_crypto/all")  # read‑only source dir

# file‑name patterns -------------------------------------------------------- #
PRICE_ZIP_RE = re.compile(
    r"^(?P<filebase>(?P<coin>[A-Z]+)USDT)-1m-(?P<ym>\d{4}-\d{2})\.zip$"
)
FUND_ZIP_RE = re.compile(
    r"^(?P<filebase>(?P<coin>[A-Z]+)USDT)-fundingRate-(?P<ym>\d{4}-\d{2})\.zip$"
)

# identical, but for the **extracted CSVs**
PRICE_CSV_RE = re.compile(
    r"^(?P<filebase>(?P<coin>[A-Z]+)USDT)-1m-(?P<ym>\d{4}-\d{2})\.csv$"
)
FUND_CSV_RE = re.compile(
    r"^(?P<filebase>(?P<coin>[A-Z]+)USDT)-fundingRate-(?P<ym>\d{4}-\d{2})\.csv$"
)

def has_header(first_line: str) -> bool:
    """
    Very quick heuristic:
        * if the first token contains anything other than digits and dot,
          assume it's a textual column name  → header present
        * otherwise treat it as data         → no header
    """
    token = first_line.split(",", 1)[0]
    return not token.replace(".", "").isdigit()

# --------------------------------------------------------------------------- #
def month_range(start: dt.date, end: dt.date) -> list[str]:
    """Return an *inclusive* list of YYYY-MM strings."""
    out: list[str] = []
    current = start.replace(day=1)
    while current <= end:
        out.append(current.strftime("%Y-%m"))
        current += relativedelta(months=1)
    return out


def scan_archives(pattern: re.Pattern) -> dict[str, dict[str, Path]]:
    """
    Return a nested mapping of every archive under ``SRC_ROOT`` that matches
    *pattern*.

        {coin: {YYYY-MM: Path, ...}, ...}
    """
    table: dict[str, dict[str, Path]] = {}
    for p in SRC_ROOT.iterdir():
        m = pattern.match(p.name)
        if m:
            coin, ym = m["coin"], m["ym"]
            table.setdefault(coin, {})[ym] = p
    return table


# --------------------------------------------------------------------------- #
# main workflow
# --------------------------------------------------------------------------- #
def main(start_ym: str, end_ym: str, coins: list[str], dest: Path) -> None:
    # -------- date range --------------------------------------------------- #
    start_date = dt.datetime.strptime(start_ym, "%Y-%m").date().replace(day=1)
    end_date = dt.datetime.strptime(end_ym, "%Y-%m").date().replace(day=1)
    months = set(month_range(start_date, end_date))

    # -------- look‑up tables (built once) ---------------------------------- #
    price_map = scan_archives(PRICE_ZIP_RE)
    fund_map = scan_archives(FUND_ZIP_RE)

    # universe of coins
    if not coins:
        coins = sorted(set(price_map) | set(fund_map))

    # -------- copy + unzip -------------------------------------------------- #
    dest.mkdir(parents=True, exist_ok=True)
    raw_dir = dest / "_raw"           # temporary extraction directory
    raw_dir.mkdir(exist_ok=True)

    for coin in coins:
        for ym in months:
            for mapping in (price_map, fund_map):
                src = mapping.get(coin, {}).get(ym)
                if not src:
                    continue
                dst_zip = dest / src.name
                shutil.copy2(src, dst_zip)
                with ZipFile(dst_zip) as zf:
                    zf.extractall(raw_dir)
                dst_zip.unlink()      # remove zip to save space

    # -------- build tidy frames -------------------------------------------- #
    price_frames: list[pd.DataFrame] = []
    fund_frames: list[pd.DataFrame] = []
    PRICE_COLS = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
       'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume',
       'ignore']
    FUND_COLS = ['calc_time', 'funding_interval_hours', 'last_funding_rate']
    for csv_path in raw_dir.glob("*.csv"):
        name = csv_path.name
        is_price = "-1m-" in name
        is_fund  = "-fundingRate-" in name

        m = (PRICE_CSV_RE if is_price else FUND_CSV_RE).match(name)
        if not m:
            raise ValueError(f"Un-recognised file name: {name}")
        coin = m["coin"]

        with csv_path.open("r") as fh:
            hdr = has_header(fh.readline().rstrip("\n"))

        if is_price:
            df = pd.read_csv(
                csv_path,
                header=0 if hdr else None,
                names=None if hdr else PRICE_COLS,
            )
        else:  # funding
            df = pd.read_csv(
                csv_path,
                header=0 if hdr else None,
                names=None if hdr else FUND_COLS,
                usecols=[0, 1, 2],      # ignore the trailing “symbol” column if present
            )

        df["coin"] = coin
        (price_frames if is_price else fund_frames).append(df)

    price_df = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame()
    fund_df  = pd.concat(fund_frames,  ignore_index=True) if fund_frames  else pd.DataFrame()

    # -------- persist and clean -------------------------------------------- #
    print("Saving prices frames...")
    price_df.to_csv(dest / "prices.csv", index=False)
    print("Saving funding rates frames...")
    fund_df.to_csv(dest / "funding_rates.csv", index=False)

    shutil.rmtree(raw_dir)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge crypto archives.")
    parser.add_argument("start", help="start YYYY-MM (inclusive)")
    parser.add_argument("end", help="end   YYYY-MM (inclusive)")
    parser.add_argument(
        "--coins",
        nargs="*",
        default=[],
        help="list of coin tickers (e.g. AR MATIC); empty = all",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="destination directory (created if absent)",
    )

    args = parser.parse_args()
    dest_dir = args.dest or Path.cwd() / f"crypto_{dt.datetime.now():%Y%m%d_%H%M%S}"

    try:
        main(args.start, args.end, args.coins, dest_dir)
        print(f"✅  Finished. Output saved in: {dest_dir}")
    except Exception as exc:
        print(f"❌  Error: {exc}", file=sys.stderr)
        sys.exit(1)