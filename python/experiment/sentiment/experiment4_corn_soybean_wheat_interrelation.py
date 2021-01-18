from typing import List

import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split

from python.experiment import DELAY
from python.experiment.sentiment import SentimentAnalysis


class AdvancedSentimentAnalysis:
    sentiment_analysis_list: List[SentimentAnalysis]

    def __init__(self) -> None:
        self.sentiment_analysis_list = [
            SentimentAnalysis('CORN'),
            SentimentAnalysis('SOYBEAN'),
            SentimentAnalysis('WHEAT')
        ]
        for sa in self.sentiment_analysis_list:
            self.__process_sentiment_analysis(sa)
        self._combine()

    @staticmethod
    def __process_sentiment_analysis(sa: SentimentAnalysis) -> None:
        sa.combined_df = sa.combined_df[['Date', 'negative', 'positive', 'direction']]
        sa.combined_df = sa.combined_df.rename(
            columns={
                'positive': ('positive_' + sa.grain),
                'negative': ('negative_' + sa.grain),
                'direction': ('direction_' + sa.grain)
            }
        )

    def _combine(self):
        self.combined_df = None
        for sa in self.sentiment_analysis_list:
            if self.combined_df is None:
                self.combined_df = sa.combined_df
            else:
                self.combined_df = pd.merge(self.combined_df, sa.combined_df, on=['Date'])


analysis = AdvancedSentimentAnalysis()

sentiment_columns = [
    'negative_CORN', 'positive_CORN',
    'negative_SOYBEAN', 'positive_SOYBEAN',
    'negative_WHEAT', 'positive_WHEAT'
]


def apply_delay(dataframe: pd.DataFrame, trade_day_delay: int) -> pd.DataFrame:
    result = dataframe.copy()
    result['Date'] = result['Date'].shift(trade_day_delay)
    result = result[result['Date'].notna()]
    return result


for grain in ['CORN', 'SOYBEAN', 'WHEAT']:
    for delay in DELAY:
        print('====================')
        print(grain)
        data = apply_delay(analysis.combined_df, delay)
        x = data[sentiment_columns]
        y = data['direction_' + grain]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        svc = svm.SVC()
        svc.fit(x_train, y_train)
        score = svc.score(x_test, y_test)
        print(str(delay) + " day(s) delay")
        print(score)
    print()
