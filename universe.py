import pandas as pd
from config import UNIVERSE_SIZE, MIN_PRICE, MIN_ADV, ADV_LOOKBACK_DAYS


class UniverseBuilder:

    def __init__(self, price_path="data/prices.parquet"):
        self.prices = pd.read_parquet(price_path)

        # Enforce types
        self.prices["date"] = pd.to_datetime(self.prices["date"])
        self.prices["close"] = pd.to_numeric(self.prices["close"], errors="coerce")
        self.prices["volume"] = pd.to_numeric(self.prices["volume"], errors="coerce")

        # Precompute dollar volume
        self.prices["dollar_volume"] = (
            self.prices["close"] * self.prices["volume"]
        )

        # Sort once for rolling ops
        self.prices = self.prices.sort_values(["ticker", "date"])


    # ------------------------------------------------
    # Compute Rolling ADV
    # ------------------------------------------------
    def compute_adv(self):

        self.prices["adv"] = (
            self.prices
            .groupby("ticker")["dollar_volume"]
            .rolling(ADV_LOOKBACK_DAYS, min_periods=ADV_LOOKBACK_DAYS)
            .mean()
            .reset_index(level=0, drop=True)
        )


    # ------------------------------------------------
    # Get Tradable Universe
    # ------------------------------------------------
    def get_universe(self, as_of_date):
        as_of_date = pd.to_datetime(as_of_date)

        if "adv" not in self.prices.columns:
            self.compute_adv()

        # Only consider data up to rebalance date
        df = self.prices[self.prices["date"] <= as_of_date].copy()

        # For each ticker, get last available row
        df = (
            df.sort_values("date")
            .groupby("ticker")
            .tail(1)
        )

        # Drop rows where ADV not ready
        df = df[df["adv"].notna()]

        # Price filter
        df = df[df["close"] > MIN_PRICE]

        # Rank by liquidity
        df = df.sort_values("adv", ascending=False)

        universe = df.head(UNIVERSE_SIZE)["ticker"].tolist()

        return universe


    # ------------------------------------------------
    # Monthly Rebalance Dates
    # ------------------------------------------------
    def get_month_end_dates(self):
        # Ensure ADV exists
        if "adv" not in self.prices.columns:
            self.compute_adv()

        # First date where ANY stock has valid ADV
        first_valid_date = (
            self.prices
            .loc[self.prices["adv"].notna(), "date"]
            .min()
        )

        # Compute month-end trading dates
        month_ends = (
            self.prices
            .groupby(self.prices["date"].dt.to_period("M"))["date"]
            .max()
            .sort_values()
            .tolist()
        )

        # Only keep months AFTER ADV warmup
        month_ends = [d for d in month_ends if d >= first_valid_date]

        return month_ends


