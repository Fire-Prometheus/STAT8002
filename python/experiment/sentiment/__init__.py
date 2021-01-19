from typing import Dict, List

import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from sklearn import svm
from sklearn.model_selection import train_test_split

from python.experiment import Experiment, DELAY
from python.preprocessing import GRAINS

SCORE_COLUMNS = ['negative', 'neutral', 'positive', 'compound']

analyzer = SentimentIntensityAnalyzer()
mean = np.mean
lemmatizer = WordNetLemmatizer()


def sentiment_analyse(text: str) -> Dict[str, float]:
    global analyzer
    return analyzer.polarity_scores(text)


class SentimentAnalysis(Experiment):
    grain: str

    def __init__(self, grain: str, all_data: bool = False) -> None:
        super().__init__(grain, all_data)
        self.grain = grain

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

    @staticmethod
    def __preprocess(words: List[str]) -> List[str]:
        words = [lemmatizer.lemmatize(word.strip(), pos='v') for word in words]
        words = [lemmatizer.lemmatize(word.strip(), pos='a') for word in words]
        words = [lemmatizer.lemmatize(word.strip(), pos='n') for word in words]
        return words

    def _process_news(self):
        self.news['new content'] = self.news['new content'].apply(lambda c: " ".join(self.__preprocess(c)))
        self.news[SCORE_COLUMNS] = self.news.apply(
            lambda row: pd.Series(list(sentiment_analyse(row['new content']).values())), axis='columns')

    def _combine(self):

        self.combined_df = self.news.groupby(['Date']).agg(
            {'negative': mean, 'neutral': mean, 'positive': mean, 'compound': mean}).reset_index()
        self.combined_df = pd.merge(self.combined_df, self.price, on=['Date'], how='left')
        self.combined_df = self.combined_df[self.combined_df['direction'].notna()]

    def _handle_holiday(self):
        result = self.combined_df.copy()
        last_aggregated_trade_day = None
        holidays = result[result['Price'].isnull()]
        for index, row in holidays.iterrows():
            current_date = row['Date']
            next_trade_day = result[result['Date'] > current_date].Date.min()
            if (last_aggregated_trade_day is not None
                    and next_trade_day is not np.nan
                    and next_trade_day < last_aggregated_trade_day):
                continue
            else:
                news_to_be_aggregated_again = self.news[
                    self.news.Date.between(current_date, next_trade_day, inclusive=True)]
                global SCORE_COLUMNS
                for c in SCORE_COLUMNS:
                    result.loc[result['Date'] == next_trade_day, c] = mean(news_to_be_aggregated_again[c])
                last_aggregated_trade_day = next_trade_day
        result = result.dropna()
        return result

    def apply_delay(self, trade_day_delay: int) -> pd.DataFrame:
        result = self.combined_df.copy()
        global SCORE_COLUMNS
        columns = SCORE_COLUMNS + ['direction']
        result[columns] = result[columns].shift(-trade_day_delay)
        result = result[result['direction'].notna()]
        return result

    def test(self, columns: List[str] = None, trade_day_delay: int = 0) -> None:
        if columns is None:
            columns = ['negative', 'positive']
        data = self.apply_delay(trade_day_delay)
        x = data[columns]
        y = data['direction']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        svc = svm.SVC()
        svc.fit(x_train, y_train)
        score = svc.score(x_test, y_test)
        print(str(trade_day_delay) + " day(s) delay")
        print(score)


class ExtendedSentimentAnalysis(SentimentAnalysis):
    def __init__(self, grain: str) -> None:
        super().__init__(grain, True)


def default_test(cls: type, columns: List[str] = None):
    for grain in GRAINS:
        print('====================')
        print(grain)
        sentiment_analysis = cls(grain)
        for delay in DELAY:
            sentiment_analysis.test(columns=columns, trade_day_delay=delay)
        print()
