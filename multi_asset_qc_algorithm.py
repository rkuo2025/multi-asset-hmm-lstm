# region imports
from AlgorithmImports import *
# endregion

class MultiAssetHMMLSTMAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2025, 12, 31)
        self.SetCash(1000000) # $1 Million Dollars

        # 1. Allocation Logic
        # 50% VTI, remaining 50% split equally among others (12.5% each)
        self.weights = {
            "VTI": 0.50,
            "XLP": 0.125,
            "GLD": 0.125,
            "BND": 0.125,
            "REMX": 0.125
        }

        # 2. Setup Brokerage & Universe
        self.SetBrokerageModel(BrokerageName.AlphaStreams)
        self.symbols = [self.AddEquity(ticker, Resolution.Daily).Symbol for ticker in self.weights.keys()]
        self.SetUniverseSelection(ManualUniverseSelectionModel(self.symbols))

        # 3. Add Alpha Model
        # Note: In a real QC environment, training a full LSTM inside Update() is too slow.
        # This framework sets up the structure for Insight-based trading.
        self.AddAlpha(HMMLSTMAlphaModel(self.weights))

        # 4. Portfolio Construction
        self.SetPortfolioConstruction(InsightWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())

class HMMLSTMAlphaModel(AlphaModel):
    def __init__(self, weights):
        self.weights = weights
        self.lookback = 60 # Matches our research script

    def Update(self, algorithm, data):
        insights = []
        
        # We emit insights for each asset based on our weighted strategy.
        # In a full QC implementation, you would use a pre-trained model 
        # or a simpler statistical proxy here because LSTM training is 
        # computationally expensive for backtests.
        for ticker, weight in self.weights.items():
            symbol = [s for s in algorithm.ActiveSecurities.Keys if s.Value == ticker][0]
            
            # For this multi-asset allocation, we maintain the Master's 
            # requested 50% / 12.5% split.
            insights.append(Insight.Price(
                symbol, 
                timedelta(days=1), 
                InsightDirection.Up, 
                None, None, None, 
                weight
            ))

        return insights
