# region imports
from AlgorithmImports import *
# endregion

class MultiAssetHMMLSTMAlgorithm(QCAlgorithm):

    def Initialize(self):
        # 1. Backtest Settings
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2025, 12, 31)
        self.SetCash(1000000) # One Million Dollars

        # 2. Strategy Weights
        self.weights = {
            "VTI": 0.50,
            "XLP": 0.125,
            "GLD": 0.125,
            "BND": 0.125,
            "REMX": 0.125
        }

        # 3. Setup Framework for Alpha Streams
        self.SetBrokerageModel(BrokerageName.AlphaStreams)
        
        # Add symbols
        self.symbols = [self.AddEquity(ticker, Resolution.Daily).Symbol for ticker in self.weights.keys()]
        self.SetUniverseSelection(ManualUniverseSelectionModel(self.symbols))

        # 4. Alpha Model with Trade Frequency Limit (5 times a month)
        self.AddAlpha(LimitedTradeAlphaModel(self.weights, trades_per_month=5))

        # 5. Portfolio Construction
        self.SetPortfolioConstruction(InsightWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())

class LimitedTradeAlphaModel(AlphaModel):
    def __init__(self, weights, trades_per_month=5):
        self.weights = weights
        self.trades_per_month = trades_per_month
        self.monthly_trade_count = 0
        self.last_month = -1

    def Update(self, algorithm, data):
        # Reset counter at the start of a new month
        current_month = algorithm.Time.month
        if current_month != self.last_month:
            self.monthly_trade_count = 0
            self.last_month = current_month

        insights = []
        
        # Only emit insights if we haven't reached our monthly limit
        if self.monthly_trade_count < self.trades_per_month:
            for ticker, weight in self.weights.items():
                symbol = [s for s in algorithm.ActiveSecurities.Keys if s.Value == ticker][0]
                
                # We consider one "rebalance" of the portfolio as one trade event
                # Since we are emitting insights for the whole portfolio, 
                # we increment the count once per successful Update call.
                insights.append(Insight.Price(
                    symbol, 
                    timedelta(days=1), 
                    InsightDirection.Up, 
                    None, None, None, 
                    weight
                ))
            
            # Increment the monthly trade/rebalance count
            if len(insights) > 0:
                self.monthly_trade_count += 1
                algorithm.Debug(f"[{algorithm.Time}] Monthly Trade {self.monthly_trade_count}/{self.trades_per_month} emitted.")

        return insights
