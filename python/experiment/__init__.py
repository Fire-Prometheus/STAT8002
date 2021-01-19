import pandas as pd

from python.preprocessing import DataPreprocessor, PICKLE, AdvancedNewsDataPreprocessor

DELAY = [0, 1, 2, 3, 5, 7, 14, 30]


class Experiment:
    news: pd.DataFrame
    price: pd.DataFrame
    combined_df: pd.DataFrame

    def __init__(self, grain: str, all_data: bool = None) -> None:
        self.news = AdvancedNewsDataPreprocessor.load_pickle_with_filter(
            '../' + PICKLE['NEWS']['ALL'], grain) if all_data else DataPreprocessor.load_pickle(
            '../' + PICKLE['NEWS'][grain.upper()])
        self.price = DataPreprocessor.load_pickle('../' + PICKLE['PRICE'][grain.upper()])
        self._process_news()
        self._combine()
        self._handle_holiday()

    def _process_news(self):
        pass

    def _combine(self):
        pass

    def _handle_holiday(self):
        pass

    def apply_delay(self, trade_day_delay: int) -> pd.DataFrame:
        pass

    def test(self, trade_day_delay: int) -> None:
        pass
