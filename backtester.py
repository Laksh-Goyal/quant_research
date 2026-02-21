import pandas as pd
import numpy as np
from universe import UniverseBuilder

class BacktestEngine:
    def __init__(self, universe_builder: UniverseBuilder):
        self.ub = universe_builder
        self.prices = self.ub.prices.copy()
        spy = pd.read_parquet("data/spy.parquet")
        self.prices = pd.concat([self.prices, spy], ignore_index=True)

        self.results = None
        self.equity_curve = None

    def run(self):
        rebalance_dates = self.ub.get_month_end_dates()
        all_period_returns = []

        for i in range(len(rebalance_dates) - 1):

            start_date = rebalance_dates[i]
            end_date = rebalance_dates[i + 1]

            universe = self.ub.get_universe(start_date)
            if not universe:
                continue

            # Filter relevant tickers
            df = self.prices[self.prices['ticker'].isin(universe)].copy()

            # Get last price <= start_date
            start_prices = (
                df[df['date'] <= start_date]
                .sort_values('date')
                .groupby('ticker')
                .tail(1)
                [['ticker', 'close']]
                .rename(columns={'close': 'start_close'})
            )

            # Get last price <= end_date
            end_prices = (
                df[df['date'] <= end_date]
                .sort_values('date')
                .groupby('ticker')
                .tail(1)
                [['ticker', 'close']]
                .rename(columns={'close': 'end_close'})
            )

            merged = start_prices.merge(end_prices, on='ticker')

            if merged.empty:
                continue

            merged['return'] = (
                merged['end_close'] / merged['start_close'] - 1
            )

            portfolio_return = merged['return'].mean()

            all_period_returns.append({
                'date': end_date,
                'return': portfolio_return,
                'n_tickers': len(merged)
            })

        self.results = pd.DataFrame(all_period_returns)

        if not self.results.empty:
            self.results['cumulative_return'] = (
                (1 + self.results['return']).cumprod()
            )

        return self.results

    def compute_metrics(self):
        if self.results is None or self.results.empty:
            return "No results to compute metrics."

        returns = self.results['return']
        total_return = (1 + returns).prod() - 1
        
        # Annualized volatility (assuming monthly rebalance)
        vol = returns.std() * np.sqrt(12)
        
        # Annualized return
        n_years = (self.results['date'].max() - self.results['date'].min()).days / 365.25
        if n_years > 0:
            ann_return = (1 + total_return) ** (1/n_years) - 1
        else:
            ann_return = total_return
            
        sharpe = ann_return / vol if vol != 0 else 0
        
        # Drawdown
        cum_ret = self.results['cumulative_return']
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        max_dd = drawdown.min()

        return {
            'Total Return': f"{total_return:.2%}",
            'Annualized Return': f"{ann_return:.2%}",
            'Annualized Vol': f"{vol:.2%}",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Max Drawdown': f"{max_dd:.2%}",
            'Count': len(self.results)
        }
    
    def compute_spy_benchmark(self, ticker="SPY"):
        dates = self.ub.get_month_end_dates()
        
        spy = self.prices[self.prices["ticker"] == ticker].copy()
        spy = spy.sort_values("date")
        
        returns = []
        
        for i in range(len(dates) - 1):
            start = dates[i]
            end = dates[i + 1]
            
            start_price = (
                spy[spy["date"] <= start]
                .sort_values("date")
                .tail(1)["close"]
            )
            
            end_price = (
                spy[spy["date"] <= end]
                .sort_values("date")
                .tail(1)["close"]
            )
            
            if start_price.empty or end_price.empty:
                continue
            
            r = end_price.iloc[0] / start_price.iloc[0] - 1
            
            returns.append({
                "date": end,
                "spy_return": r
            })
        
        spy_df = pd.DataFrame(returns)
        spy_df["spy_nav"] = (1 + spy_df["spy_return"]).cumprod()
        
        return spy_df

    def run_momentum_strategy(self, top_pct=0.1):

        if "momentum" not in self.ub.prices.columns:
            self.ub.compute_monthly_momentum()

        # Refresh prices
        self.prices = self.ub.prices.copy()

        # Compute daily returns
        self.prices = self.prices.sort_values(["ticker", "date"])
        self.prices["return"] = (
            self.prices
                .groupby("ticker")["close"]
                .pct_change()
        )

        rebalance_dates = self.ub.get_month_end_dates()
        results = []


        for i in range(len(rebalance_dates) - 1):

            start = rebalance_dates[i]
            end = rebalance_dates[i + 1]

            universe = self.ub.get_universe(start)
            if not universe:
                continue

            # Get latest available momentum per ticker
            df = self.prices[
                (self.prices["ticker"].isin(universe)) &
                (self.prices["date"] <= start)
            ]

            latest = (
                df.sort_values("date")
                .groupby("ticker")
                .tail(1)
            )

            latest = latest.dropna(subset=["momentum"])

            if latest.empty:
                continue

            # Rank by momentum
            latest = latest.sort_values("momentum", ascending=False)

            n_select = int(len(latest) * top_pct)
            selected = latest.head(n_select)["ticker"]

            # Compute next-period returns
            df_period = self.prices[
                (self.prices["ticker"].isin(selected)) &
                (self.prices["date"] > start) &
                (self.prices["date"] <= end)
            ]

            returns = (
                df_period.groupby("ticker")["return"]
                        .apply(lambda x: (1 + x).prod() - 1)
            )

            if len(returns) == 0:
                continue

            portfolio_return = returns.mean()

            results.append({
                "date": end,
                "return": portfolio_return,
                "n_stocks": len(returns)
            })

        df_res = pd.DataFrame(results)
        if not df_res.empty:
            df_res["nav"] = (1 + df_res["return"]).cumprod()

        return df_res

    def run_momentum_base(
        self,
        top_pct=0.1,
        cost_per_trade=0.001  # 10 bps per turnover
    ):

        # --------------------------------------------------
        # Ensure momentum exists
        # --------------------------------------------------
        if "momentum" not in self.ub.prices.columns:
            self.ub.compute_monthly_momentum()

        # Refresh prices
        self.prices = self.ub.prices.copy()

        # Compute daily returns
        self.prices = self.prices.sort_values(["ticker", "date"])
        self.prices["return"] = (
            self.prices
                .groupby("ticker")["close"]
                .pct_change()
        )

        rebalance_dates = self.ub.get_month_end_dates()

        results = []

        prev_longs = set()
        prev_shorts = set()

        # --------------------------------------------------
        # Main rebalance loop
        # --------------------------------------------------
        for i in range(len(rebalance_dates) - 1):

            start = rebalance_dates[i]
            end = rebalance_dates[i + 1]

            universe = self.ub.get_universe(start)
            if not universe:
                continue

            # ----------------------------------------------
            # Get latest signal values
            # ----------------------------------------------
            df = self.prices[
                (self.prices["ticker"].isin(universe)) &
                (self.prices["date"] <= start)
            ]

            latest = (
                df.sort_values("date")
                .groupby("ticker")
                .tail(1)
            )

            latest = latest.dropna(subset=["momentum"])

            if latest.empty:
                continue

            # ----------------------------------------------
            # Cross-sectional z-score normalization
            # ----------------------------------------------
            mean = latest["momentum"].mean()
            std = latest["momentum"].std()

            if std == 0 or np.isnan(std):
                continue

            latest["zscore"] = (latest["momentum"] - mean) / std
            latest = latest.sort_values("zscore", ascending=False)

            n_select = int(len(latest) * top_pct)

            if n_select == 0:
                continue

            longs = set(latest.head(n_select)["ticker"])
            shorts = set(latest.tail(n_select)["ticker"])

            # ----------------------------------------------
            # Compute next-period returns
            # ----------------------------------------------
            df_period = self.prices[
                (self.prices["ticker"].isin(longs.union(shorts))) &
                (self.prices["date"] > start) &
                (self.prices["date"] <= end)
            ]

            returns = (
                df_period.groupby("ticker")["return"]
                        .apply(lambda x: (1 + x).prod() - 1)
            )

            if returns.empty:
                continue

            long_ret = returns.loc[returns.index.isin(longs)].mean()
            short_ret = returns.loc[returns.index.isin(shorts)].mean()

            if np.isnan(long_ret) or np.isnan(short_ret):
                continue

            portfolio_return = long_ret - short_ret

            # ----------------------------------------------
            # Transaction cost model (turnover-based)
            # ----------------------------------------------
            if i > 0:
                long_turnover = len(longs - prev_longs)
                short_turnover = len(shorts - prev_shorts)

                total_positions = 2 * n_select

                turnover_ratio = (long_turnover + short_turnover) / total_positions

                transaction_cost = turnover_ratio * cost_per_trade

                portfolio_return -= transaction_cost

            prev_longs = longs
            prev_shorts = shorts

            # ----------------------------------------------
            # Store results
            # ----------------------------------------------
            results.append({
                "date": end,
                "return": portfolio_return,
                "n_longs": len(longs),
                "n_shorts": len(shorts)
            })

        df_res = pd.DataFrame(results)

        if not df_res.empty:
            df_res["nav"] = (1 + df_res["return"]).cumprod()

        return df_res

    def run_momentum_vol_scaled(self, top_pct=0.1, cost_per_trade=0.001):

        if "momentum" not in self.ub.prices.columns:
            self.ub.compute_monthly_momentum()

        self.prices = self.ub.prices.copy()
        self.prices = self.prices.sort_values(["ticker", "date"])

        self.prices["return"] = (
            self.prices.groupby("ticker")["close"].pct_change()
        )

        # 63-day rolling volatility
        self.prices["vol_63"] = (
            self.prices.groupby("ticker")["return"]
            .rolling(63).std()
            .reset_index(level=0, drop=True)
        )

        rebalance_dates = self.ub.get_month_end_dates()
        results = []

        prev_longs, prev_shorts = set(), set()

        for i in range(len(rebalance_dates) - 1):

            start, end = rebalance_dates[i], rebalance_dates[i + 1]

            universe = self.ub.get_universe(start)
            if not universe:
                continue

            df = self.prices[
                (self.prices["ticker"].isin(universe)) &
                (self.prices["date"] <= start)
            ]

            latest = (
                df.sort_values("date")
                .groupby("ticker")
                .tail(1)
                .dropna(subset=["momentum", "vol_63"])
            )

            if latest.empty:
                continue

            mean = latest["momentum"].mean()
            std = latest["momentum"].std()
            if std == 0:
                continue

            latest["z"] = (latest["momentum"] - mean) / std

            latest = latest.sort_values("z", ascending=False)

            n = int(len(latest) * top_pct)
            if n == 0:
                continue

            longs = latest.head(n)
            shorts = latest.tail(n)

            # Volatility-scaled weights
            # For longs
            longs["w"] = 1 / longs["vol_63"]
            longs["w"] /= longs["w"].sum()

            # For shorts
            shorts["w"] = 1 / shorts["vol_63"]
            shorts["w"] /= shorts["w"].sum()


            selected = set(longs["ticker"]).union(set(shorts["ticker"]))

            df_period = self.prices[
                (self.prices["ticker"].isin(selected)) &
                (self.prices["date"] > start) &
                (self.prices["date"] <= end)
            ]

            period_ret = (
                df_period.groupby("ticker")["return"]
                .apply(lambda x: (1 + x).prod() - 1)
            )

            long_ret = (period_ret.loc[longs["ticker"]] * longs.set_index("ticker")["w"]).sum()
            short_ret = (period_ret.loc[shorts["ticker"]] * shorts.set_index("ticker")["w"]).sum()

            portfolio_return = (
                (longs.set_index("ticker")["w"] * period_ret.loc[longs["ticker"]]).sum() -
                (shorts.set_index("ticker")["w"] * period_ret.loc[shorts["ticker"]]).sum()
            )

            results.append({"date": end, "return": portfolio_return})

        df_res = pd.DataFrame(results)
        df_res["nav"] = (1 + df_res["return"]).cumprod()
        return df_res

    def apply_overlapping(self, df, window=12):
        df = df.copy()
        df["overlap_return"] = (
            df["return"]
            .rolling(window)
            .mean()
        )

        df["nav"] = (1 + df["overlap_return"]).cumprod()
        return df

    def beta_neutralize(self, strategy_df, spy_df):

        merged = strategy_df.merge(spy_df, on="date", how="inner")

        X = merged["spy_return"].values
        Y = merged["return"].values

        beta = np.cov(Y, X)[0,1] / np.var(X)

        merged["beta_adj_return"] = merged["return"] - beta * merged["spy_return"]

        merged["nav"] = (1 + merged["beta_adj_return"]).cumprod()

        return merged, beta
    

