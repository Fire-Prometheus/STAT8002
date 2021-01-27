from typing import Dict, List, Union, Iterable

import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt

from python.experiment import Experiment, DELAY
from python.preprocessing import GRAINS

SCORE_COLUMNS = ['positive', 'negative', 'neutral', 'compound']

analyzer = SentimentIntensityAnalyzer()
mean = np.mean
lemmatizer = WordNetLemmatizer()


def sentiment_analyse(text: str) -> Dict[str, float]:
    global analyzer
    return analyzer.polarity_scores(text)


class SentimentScoring:
    def compute_scores(self, text: str) -> pd.Series:
        """
        This function computes the scores, at least scores of its positivity and negativity.
        It's ab abstraction of data type returned by NLTK Vader.
        :param text: a word/phrase
        :return: a dict with keys: 'positive', 'negative', 'neutral' and 'compound'
        """
        return pd.Series(
            {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 0.0,
                'compound': 0.0
            }
        )

    def aggregate_scores(self, scores: Union[Iterable, float]) -> np.ndarray:
        return np.mean(scores)


class VaderSentimentScoring(SentimentScoring):
    analyzer = SentimentIntensityAnalyzer()

    def compute_scores(self, text: str) -> pd.Series:
        scores = analyzer.polarity_scores(text)
        compute_scores = super().compute_scores(text)
        compute_scores.at['positive'] = scores['pos']
        compute_scores.at['negative'] = scores['neg']
        compute_scores.at['neutral'] = scores['neu']
        compute_scores.at['compound'] = scores['compound']
        return compute_scores


class SentimentAnalysis(Experiment):
    grain: str
    sentiment_scoring: SentimentScoring

    def __init__(self, sentiment_scoring: SentimentScoring, grain: str, all_data: bool = False) -> None:
        self.sentiment_scoring = sentiment_scoring
        self.grain = grain
        super().__init__(grain, all_data)

    @staticmethod
    def __preprocess(words: List[str]) -> List[str]:
        words = [lemmatizer.lemmatize(word.strip(), pos='v') for word in words]
        words = [lemmatizer.lemmatize(word.strip(), pos='a') for word in words]
        words = [lemmatizer.lemmatize(word.strip(), pos='n') for word in words]
        return words

    def _process_news(self):
        self.news['new content'] = self.news['new content'].apply(lambda c: " ".join(self.__preprocess(c)))
        self.news = pd.merge(self.news,
                             self.news['new content'].apply(lambda c: self.sentiment_scoring.compute_scores(c)),
                             left_index=True, right_index=True)

    def _combine(self):
        aggregate_scores = self.sentiment_scoring.aggregate_scores
        self.combined_df = self.news.groupby(['Date']).agg(
            {
                'negative': aggregate_scores,
                'neutral': aggregate_scores,
                'positive': aggregate_scores,
                'compound': aggregate_scores
            }
        ).reset_index()
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

    def test(self, columns: List[str] = None, trade_day_delay: int = 0, to_pickle: bool = False) -> None:
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
        if to_pickle:
            data.to_pickle('./data_' + self.grain + '_delay' + str(trade_day_delay) + '.pickle')

    def fit_linear(self, columns: List[str] = None, trade_day_delay: int = 0) -> None:
        if columns is None:
            columns = ['negative', 'positive']
        data = self.apply_delay(trade_day_delay)
        x = data[columns]
        y = data['Change %']
        reg = LinearRegression()
        reg = reg.fit(x, y)
        score = reg.score(x, y)
        print(str(trade_day_delay) + " day(s) delay")
        print(score)

    def svr(self, columns: List[str] = None, trade_day_delay: int = 0) -> None:
        if columns is None:
            columns = ['negative', 'positive']
        data = self.apply_delay(trade_day_delay)
        x = data[columns]
        y = data['Change %']
        svr = SVR(kernel='poly', C=1e3, degree=3)
        svr.fit(x, y)
        score = svr.score(x, y)
        print(str(trade_day_delay) + " day(s) delay")
        print(score)


class ExtendedSentimentAnalysis(SentimentAnalysis):
    def __init__(self, sentiment_scoring: SentimentScoring, grain: str) -> None:
        super().__init__(sentiment_scoring, grain, True)


def default_test(cls: type, sentiment_scoring_cls: type, columns: List[str] = None):
    for grain in GRAINS:
        print('====================')
        print(grain)
        sentiment_analysis = cls(sentiment_scoring_cls(), grain)
        for delay in DELAY:
            sentiment_analysis.test(columns=columns, trade_day_delay=delay)
        print()
