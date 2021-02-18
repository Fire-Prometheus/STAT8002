from typing import List

import pandas as pd
from gensim.models import KeyedVectors
from sklearn import svm
from sklearn.model_selection import train_test_split

from python.experiment import Experiment


class Word2VecModel(Experiment):
    word_vectors = KeyedVectors.load_word2vec_format('/media/alextwy/Plextor/Documents/Academic/#STAT#8002/src/python/experiment/vectorization/GoogleNews-vectors-negative300.bin', binary=True)
    variable_names = ['x' + str(i) for i in range(300)]

    def __init__(self, grain: str) -> None:
        super().__init__(grain)
        self.combined_df = pd.DataFrame(self.news[['Date', 'new content']])
        self.combined_df[self.variable_names] = self.combined_df['new content'].apply(
            lambda c: pd.Series(self.__sum_of_vectors_from_words(c)))
        self.combined_df = self.combined_df.groupby('Date').sum().reset_index()
        self.combined_df = pd.merge(self.combined_df, self.price, on=['Date'], how='left')
        self.__handle_holidays()

    def __sum_of_vectors_from_words(self, words_array: List[str]):
        sum = None
        for index in range(0, len(words_array)):
            try:
                vector = self.word_vectors[words_array[index]]
                # vector = vector / linalg.norm(vector)
                if sum is None:
                    sum = vector
                else:
                    sum = sum + vector
            except KeyError:
                continue
        return sum

    def __handle_holidays(self):
        last_aggregated_trade_day = None
        holidays = self.combined_df[self.combined_df['Price'].isnull()]
        for index, row in holidays.iterrows():
            current_date = row['Date']
            next_trade_day = self.combined_df[self.combined_df['Date'] > current_date].Date.min()
            if last_aggregated_trade_day is not None and next_trade_day < last_aggregated_trade_day:
                continue
            else:
                to_be_aggregated_again = self.combined_df[
                    self.combined_df.Date.between(current_date, next_trade_day, inclusive=True)]
                sum = to_be_aggregated_again.sum()
                self.combined_df[self.combined_df['Date'] == next_trade_day].replace(sum)
                # global variable_names
                # for x in variable_names:
                #     result.loc[result['Date'] == next_trade_day, x] = mean(news_to_be_aggregated_again[c])
                last_aggregated_trade_day = next_trade_day
        self.combined_df = self.combined_df.dropna()

    def __apply_day_delay(self, trade_day_delay):
        result = self.combined_df.copy()
        columns = self.variable_names + ['direction']
        result[columns] = result[columns].shift(-trade_day_delay)
        result = result.dropna()
        return result

    def test(self, trade_day_delay: int) -> None:
        data = self.__apply_day_delay(trade_day_delay)
        x = data[self.variable_names]
        y = data['direction']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        svc = svm.SVC()
        svc.fit(x_train, y_train)
        score = svc.score(x_test, y_test)
        print(str(trade_day_delay) + " day(s) delay")
        print(score)

    def test_default(self) -> None:
        DAY_DELAY = [0, 1, 2, 5, 7, 14, 30]
        for delay in DAY_DELAY:
            self.test(delay)


model = Word2VecModel('CORN')
model.test(1)
