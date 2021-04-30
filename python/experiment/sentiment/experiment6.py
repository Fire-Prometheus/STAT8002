from typing import Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, SVC
from random import randint

from python.preprocessing import GRAINS
from sklearn.linear_model import LinearRegression


def convert_date_column(dataframe: pd.DataFrame, column_name='Date') -> None:
    dataframe[column_name] = pd.to_datetime(dataframe[column_name]).dt.date


def convert_float(string: Any) -> float:
    return string if type(string) == float else float(string.replace(',', ''))


def generate_column_name(column_name: str) -> str:
    return column_name + ' T-' + str(trade_day_delay)


df = dict()
trade_day_delay = 1

for grain in GRAINS:
    # sentiment scores
    df_sentiment: pd.DataFrame = pd.read_csv('../../data/news_sentiment_score_by_day_' + grain + '.csv')
    convert_date_column(df_sentiment)
    # price table
    df_price: pd.DataFrame = pd.read_pickle('../../data/preprocessed_price_' + grain.lower() + '.pickle')
    df_price['Price'] = df_price['Price'].swifter.apply(lambda x: convert_float(x))
    df_price['Open'] = df_price['Price'].swifter.apply(lambda x: convert_float(x))
    df_price['High'] = df_price['Price'].swifter.apply(lambda x: convert_float(x))
    df_price['Low'] = df_price['Price'].swifter.apply(lambda x: convert_float(x))
    convert_date_column(df_price)
    # shifted price table
    df_price_delay = df_price.copy()
    columns = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %', 'direction']
    column_mapping = dict(zip(columns, map(lambda c: generate_column_name(c), columns)))
    df_price_delay[columns] = df_price_delay[columns].shift(-trade_day_delay)
    df_price_delay = df_price_delay.rename(columns=column_mapping)
    result = pd.merge(df_sentiment, df_price_delay, on=['Date'])
    result = result[trade_day_delay:]
    # join price table again
    result = pd.merge(result, df_price, on=['Date'])
    # store the results
    df[grain] = result


def fit_price(dataframe: pd.DataFrame, is_content: bool, is_null_model: bool) -> pd.DataFrame:
    x_columns = [generate_column_name('Price')]
    if not is_null_model:
        x_columns.extend(['content_positiveness', 'content_negativeness'] if is_content else ['headline_positiveness',
                                                                                              'headline_negativeness'])
    x = dataframe[x_columns]
    y = dataframe['Price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=20210403)
    regression = LinearRegression()
    regression.fit(x_train, y_train)
    print(regression.score(x_test, y_test))
    print(regression.intercept_)
    print(regression.coef_)
    svr_lin = SVR(kernel='rbf', C=100, epsilon=.1)
    svr_lin.fit(x_train, y_train)
    print(svr_lin.score(x_test, y_test))
    return x.join(y)


is_content = True

for grain in GRAINS:
    print(grain)
    data_frame = fit_price(df[grain], is_null_model=False, is_content=is_content)
    data_frame.to_csv('../../../r/data/' + grain + '.csv', index=False)
