import pandas as pd
import numpy as np
from universe import UniverseBuilder

class BacktestEngine:
    def __init__(self, universe_builder: UniverseBuilder):
        self.ub = universe_builder
        self.prices = self.ub.prices.copy()
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