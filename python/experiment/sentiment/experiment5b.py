from typing import List, Dict
import nltk
import numpy as np
import pandas as pd

from python.experiment import Experiment
from collections import Counter
from sklearn.model_selection import train_test_split


class MyExperiment(Experiment):
    POLARITY = {
        'negative': -1,
        'neutral': 0,
        'positive': 1
    }
    THRESHOLD = 0.05

    def __init__(self, grain: str, all_data: bool = None) -> None:
        super().__init__(grain, all_data)

    @staticmethod
    def __convert_polarity(sign: int) -> str:
        return next(p for p, s in MyExperiment.POLARITY.items() if s == sign)

    def _combine(self):
        self.combined_df = pd.merge(self.news, self.price, how='left', on='Date')

    def _handle_holiday(self):
        result: pd.DataFrame = self.combined_df.copy()
        holidays = result[result['direction'].isna()]
        for index, row in holidays.iterrows():
            next_trade_day_direction = self.price[self.price['Date'] > row['Date']].sort_values('Date').at[
                0, 'direction']
            if next_trade_day_direction is not np.nan:
                result.at[index, 'direction'] = next_trade_day_direction
        self.combined_df = result[result['direction'].notna()]

    def test(self, trade_day_delay: int) -> None:
        self.price['direction'] = self.price['direction'].shift(-trade_day_delay)
        self.price = self.price[self.price['direction'].notna()]
        self._combine()
        self._handle_holiday()
        self.combined_df['counter'] = self.combined_df['headline'].swifter.apply(lambda c: Counter(nltk.bigrams(c)))
        count = lambda c: sum(c, Counter())
        counts = self.combined_df.groupby(by=['direction']).aggregate(counter=('counter', count)).reset_index()
        count_df_list = [
            pd.DataFrame.from_dict(row['counter'], orient='index', columns=[self.__convert_polarity(row['direction'])])
            for index, row in counts.iterrows()]
        self.frequency_table: pd.DataFrame = pd.concat(count_df_list, axis='columns')
        self.frequency_table = self.frequency_table.fillna(0)
        self.percentage_table: pd.DataFrame = self.frequency_table.swifter.apply(lambda r: r / r.sum(), axis='columns')
        self.total_table: pd.DataFrame = self.frequency_table.swifter.apply(lambda r: r.sum(), axis='columns')
        self.table: pd.DataFrame = pd.concat([self.total_table, self.percentage_table], axis='columns')
        self.table = self.table.rename(columns={0: 'total'})
        self.table = self.table[self.table['total'] > len(self.news) * self.THRESHOLD]
        self.table_pos = self.table.sort_values('positive', ascending=False)
        self.table_neg = self.table.sort_values('negative', ascending=False)
        self.table_neutral = self.table.sort_values('neutral', ascending=False)


grain = 'WHEAT'
print(grain)
experiment = MyExperiment(grain)
experiment.test(1)
