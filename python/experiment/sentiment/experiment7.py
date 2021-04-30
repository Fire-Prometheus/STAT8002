import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from python.experiment.sentiment import RANDOM_STATES
import numpy as np
from scipy.special import softmax


def test1(random_state: int):
    df = pd.read_csv('../../data/news_sentiment_score_by_day_CORN.csv')
    min_max_scaler = preprocessing.MinMaxScaler()
    df['headline_negativeness'] = min_max_scaler.fit_transform(df['headline_negativeness'].values.reshape(-1, 1))
    df['headline_negativeness'] = min_max_scaler.fit_transform(df['headline_positiveness'].values.reshape(-1, 1))
    df['content_negativeness'] = min_max_scaler.fit_transform(df['content_negativeness'].values.reshape(-1, 1))
    df['content_positiveness'] = min_max_scaler.fit_transform(df['content_positiveness'].values.reshape(-1, 1))
    # df[['headline_negativeness', 'headline_negativeness', 'content_negativeness',
    #     'content_positiveness']] = min_max_scaler.fit_transform(
    #     df[['headline_negativeness', 'headline_negativeness', 'content_negativeness', 'content_positiveness']])
    df['next_trade_day_direction'] = df['current_trade_day_direction'].shift(-1)
    df = df[0:-1]

    x = df[['headline_negativeness', 'headline_positiveness']]
    y = df['next_trade_day_direction'].values.astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=random_state)
    svc = svm.SVC()
    svc.fit(x_train, y_train)
    return df, svc.score(x_test, y_test)


def test2(random_state: int):
    df = pd.read_csv('../../data/news_sentiment_score_by_day_CORN.csv')
    min_max_scaler = preprocessing.MinMaxScaler()
    df['headline_negativeness'] = softmax(df['headline_negativeness'].values.reshape(-1, 1))
    df['headline_positiveness'] = softmax(df['headline_positiveness'].values.reshape(-1, 1))
    df['content_negativeness'] = softmax(df['content_negativeness'].values.reshape(-1, 1))
    df['content_positiveness'] = softmax(df['content_positiveness'].values.reshape(-1, 1))
    df['next_trade_day_direction'] = df['current_trade_day_direction'].shift(-1)
    df = df[0:-1]

    x = df[['current_trade_day_direction', 'headline_negativeness', 'headline_positiveness']]
    y = df['next_trade_day_direction'].values.astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=random_state)
    svc = svm.SVC()
    svc.fit(x_train, y_train)
    return svc.score(x_test, y_test)


# print(np.mean(list(map(lambda state: test1(state), RANDOM_STATES))))
df, score = test1(RANDOM_STATES[0])
