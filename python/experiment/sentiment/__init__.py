import sys
import math
from typing import Dict, List, Union, Iterable, Tuple, Optional

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import sentiwordnet, SentiSynset
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from python.experiment import Experiment, DELAY
from python.preprocessing import GRAINS
import random

SCORE_COLUMNS = ['positive', 'negative', 'neutral', 'compound']
RANDOM_STATES = [16462643,
                 16420919,
                 12815786,
                 807484,
                 13344684,
                 14712439,
                 10114233,
                 11900286,
                 18321740,
                 1363,
                 2435817,
                 13546790,
                 8046657,
                 6378799,
                 15903029,
                 1224393,
                 9197436,
                 4207558,
                 5568659,
                 7313801,
                 1661057,
                 7175454,
                 5958119,
                 149707,
                 5082008,
                 8883810,
                 9972230,
                 15722841,
                 12603123,
                 7245555,
                 3707589,
                 4854218,
                 13559325,
                 6081105,
                 6101872,
                 13186329,
                 15956992,
                 16804426,
                 3087363,
                 5971115,
                 1513588,
                 4136885,
                 957889,
                 3022850,
                 12737466,
                 1708995,
                 7619632,
                 16307329,
                 3942301,
                 3516928
                 ]

analyzer = SentimentIntensityAnalyzer()
mean = np.mean
lemmatizer = WordNetLemmatizer()


def sentiment_analyse(text: str) -> Dict[str, float]:
    global analyzer
    return analyzer.polarity_scores(text)


class SentimentScoring:
    def compute_scores(self, text: List[str]) -> pd.Series:
        """
        This function computes the scores, at least scores of its positivity and negativity.
        It's ab abstraction of data type returned by NLTK Vader.
        :param text: a word/phrase
        :return: a dict with keys: 'positive', 'negative', 'neutral' and 'compound'
        """
        return pd.Series(
            {
                'positive': np.nan,
                'negative': np.nan,
                'neutral': np.nan,
                'compound': np.nan
            }
        )

    def aggregate_scores(self, scores: Union[Iterable, float]) -> np.ndarray:
        return np.nan if scores is None or len(scores) == 0 else np.mean(scores)


class VaderSentimentScoring(SentimentScoring):
    analyzer: SentimentIntensityAnalyzer = SentimentIntensityAnalyzer()

    def compute_scores(self, text: List[str]) -> pd.Series:
        scores = analyzer.polarity_scores(" ".join(text))
        compute_scores = super().compute_scores(text)
        compute_scores.at['positive'] = scores['pos']
        compute_scores.at['negative'] = scores['neg']
        compute_scores.at['neutral'] = scores['neu']
        compute_scores.at['compound'] = scores['compound']
        return compute_scores


class SentiWordNetScoring(SentimentScoring):
    def compute_scores(self, text: List[str]) -> pd.Series:
        pos_tags = nltk.pos_tag(text)
        polarity_scores = {
            'positive': [],
            'negative': [],
            'objective': []
        }
        filtered_words: List[str] = []
        for pos_tag in pos_tags:
            senti = self.__find_senti(pos_tag)
            if self.__is_senti_suitable(senti):
                polarity_scores['positive'].append(senti.pos_score())
                polarity_scores['negative'].append(senti.neg_score())
                polarity_scores['objective'].append(senti.obj_score())
                filtered_words.append(pos_tag[0])
        compute_scores = super().compute_scores(text)
        if len(polarity_scores['positive']) == 0:
            return compute_scores.append(pd.Series({'objective': np.nan, 'filtered words': np.nan}))
        compute_scores.at['positive'] = self._aggregate_polarities(polarity_scores['positive'])
        compute_scores.at['negative'] = self._aggregate_polarities(polarity_scores['negative'])
        compute_scores.append(pd.Series({'objective': self._aggregate_polarities(polarity_scores['objective'])}))
        compute_scores.append(pd.Series({'filtered words': filtered_words}))
        return compute_scores

    def __find_senti(self, pos_tuple: Tuple[str, str]) -> Optional[SentiSynset]:
        try:
            convert_pos = self.__convert_pos(pos_tuple[1])
            syn_sets = sentiwordnet.senti_synsets(pos_tuple[0], convert_pos)
            return next(syn_sets)
        except (StopIteration, NotImplementedError):
            return None

    @staticmethod
    def __is_senti_suitable(senti: SentiSynset):
        return senti is not None and senti.neg_score() > 0.0 and senti.pos_score() > 0.0

    @staticmethod
    def __convert_pos(text: str):
        if text.startswith('J'):
            return 'a'
        elif text.startswith('N'):
            return 'n'
        elif text.startswith('R'):
            return 'r'
        elif text.startswith('V'):
            return 'v'
        raise NotImplementedError

    @staticmethod
    def _aggregate_polarities(polarities: List[float]):
        return np.mean(polarities)


class SentimentAnalysis(Experiment):
    grain: str
    sentiment_scoring: SentimentScoring

    def __init__(self, sentiment_scoring: SentimentScoring, grain: str, is_content: bool,
                 all_data: bool = False) -> None:
        self.sentiment_scoring = sentiment_scoring
        self.grain = grain
        self.is_content = is_content
        super().__init__(grain=grain, all_data=all_data, is_content=is_content)

    @staticmethod
    def __preprocess(words: List[str]) -> List[str]:
        words = [lemmatizer.lemmatize(word.strip(), pos='v') for word in words]
        words = [lemmatizer.lemmatize(word.strip(), pos='a') for word in words]
        words = [lemmatizer.lemmatize(word.strip(), pos='n') for word in words]
        return words

    def _process_news(self):
        self.news = pd.merge(
            self.news,
            (self.news['new content'] if self.is_content else self.news['new headline']).apply(
                lambda c: self.sentiment_scoring.compute_scores(c)),
            left_index=True,
            right_index=True
        )

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

    def test(self, columns: List[str] = None, trade_day_delay: int = 0, to_pickle: bool = False,
             random_state=42) -> None:
        data, scores = self.core_test(columns, trade_day_delay, random_state)
        # data['predicted_next_trade_day_direction_by_headline'] = svc.predict(x)
        print(str(trade_day_delay) + " day(s) delay")
        print(scores[0])
        # data.to_pickle('../../data/data_'+self.grain+'_headline_output.pickle')
        if to_pickle:
            data.to_pickle('./data_' + self.grain + '_delay' + str(trade_day_delay) + '.pickle')

    def core_test(self, columns, trade_day_delay: int, random_state: int = None, test_size: int = 1) -> (
            pd.DataFrame, List[float]):
        if columns is None:
            columns = ['negative', 'positive']
        data = self.apply_delay(trade_day_delay)
        data = data[data['positive'].swifter.apply(lambda e:not(math.isnan(e)))]
        x = data[columns]
        y = data['direction']
        scores: List[float] = []
        for i in RANDOM_STATES:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=i)
            svc = svm.SVC()
            svc.fit(x_train, y_train)
            scores.append(svc.score(x_test, y_test))
        return data, scores

    def generate_samples(self, columns: List[str] = None, trade_day_delay: int = 1,
                         sample_size: int = 50) -> List[float]:
        data, scores = self.core_test(columns, trade_day_delay, test_size=sample_size)
        return scores

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
    def __init__(self, sentiment_scoring: SentimentScoring, grain: str, is_content: bool) -> None:
        super().__init__(sentiment_scoring, grain, is_content=is_content, all_data=True)


def default_test(cls: type, sentiment_scoring_cls: type, columns: List[str] = None, is_content: bool = True):
    for grain in GRAINS:
        print('====================')
        print(grain)
        sentiment_analysis = cls(sentiment_scoring=sentiment_scoring_cls(), grain=grain, is_content=is_content)
        for delay in DELAY:
            sentiment_analysis.test(columns=columns, trade_day_delay=delay, to_pickle=True)
        print()


def generate_samples(cls: type, sentiment_scoring_cls: type, columns: List[str] = None,
                     is_content: bool = True, delays=None, sample_size: int = 50,
                     file_path: str = None) -> pd.DataFrame:
    df: pd.DataFrame = pd.DataFrame()
    if delays is None:
        delays = [0, 1, 2, 3, 5, 7, 14, 30]
    sub_result = dict()
    for grain in GRAINS:
        print('====================')
        print(grain)
        sentiment_analysis = cls(sentiment_scoring=sentiment_scoring_cls(), grain=grain, is_content=is_content)
        for delay in delays:
            temp_result = sentiment_analysis.generate_samples(columns=columns, trade_day_delay=delay,
                                                              sample_size=sample_size)
            temp_df = pd.DataFrame({'ACCURACY': temp_result})
            temp_df['GRAIN'] = grain
            temp_df['DELAY'] = delay
            temp_df['TEXT'] = 'Content' if is_content else 'Headline'
            df = pd.concat([df, temp_df])
            sub_result[grain] = temp_result
    if file_path is not None:
        df_by_grain = pd.DataFrame(sub_result)
        df_by_grain.to_csv(file_path, index=False)
    return df
