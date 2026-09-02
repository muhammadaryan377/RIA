"""ARIA predictive analysis package.

PredictionAgent handles mathematically validated time-series forecasting.
TabularPredictor handles mixed numeric/categorical/date supervised prediction
for the Insight Agent.
"""

from .agent import PredictionAgent
from .tabular import TabularPredictor

__all__ = ["PredictionAgent", "TabularPredictor"]