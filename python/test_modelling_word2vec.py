from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import numpy.linalg as linalg
from sklearn import svm

from sklearn.model_selection import train_test_split

# read news and price data
news = pd.read_pickle('preprocessed_news.pickle')
price = pd.read_csv('US Corn Futures Historical Data.csv')
price['Vol.'] = price['Vol.'].apply(lambda v: float(v[0:-1]) * 1000 if len(v[0:-1]) >= 1 else np.NaN)
price['Change %'] = price['Change %'].apply(lambda p: float(p[0:-1]) / 100)
price['Date'] = price['Date'].apply(lambda d: pd.to_datetime(d, format='%b %d, %Y').date())
price['direction'] = price['Change %'].apply(lambda change: 0 if change == 0 else (1 if change > 0 else -1))
# read google news vectors
word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# modelling
combined_df = pd.DataFrame(news[['Date', 'new content']])
variable_names = ['x' + str(i) for i in range(300)]


def sum_of_vectors_from_words(words_array):
    sum = None
    for index in range(0, len(words_array)):
        try:
            vector = word_vectors[words_array[index]]
            # vector = vector / linalg.norm(vector)
            if sum is None:
                sum = vector
            else:
                sum = sum + vector
        except KeyError:
            continue
    return sum


combined_df[variable_names] = combined_df['new content'].apply(lambda c: pd.Series(sum_of_vectors_from_words(c)))
combined_df = combined_df.groupby('Date').sum().reset_index()
combined_df = pd.merge(combined_df, price, on=['Date'], how='left')


def handle_holidays(dataframe):
    result = dataframe.copy()
    last_aggregated_trade_day = None
    holidays = result[result['Price'].isnull()]
    for index, row in holidays.iterrows():
        current_date = row['Date']
        next_trade_day = result[result['Date'] > current_date].Date.min()
        if last_aggregated_trade_day is not None and next_trade_day < last_aggregated_trade_day:
            continue
        else:
            to_be_aggregated_again = result[result.Date.between(current_date, next_trade_day, inclusive=True)]
            sum = to_be_aggregated_again.sum()
            result[result['Date'] == next_trade_day].replace(sum)
            # global variable_names
            # for x in variable_names:
            #     result.loc[result['Date'] == next_trade_day, x] = mean(news_to_be_aggregated_again[c])
            last_aggregated_trade_day = next_trade_day
    result = result.dropna()
    return result


combined_df = handle_holidays(combined_df)


def apply_day_delay(dataframe, trade_day_delay):
    result = dataframe.copy()
    global variable_names
    columns = variable_names + ['direction']
    result[columns] = result[columns].shift(-trade_day_delay)
    result = result.dropna()
    return result


DAY_DELAY = [0, 1, 2, 5, 7, 14, 30]
for delay in DAY_DELAY:
    data = apply_day_delay(combined_df, delay)
    X = data[variable_names]
    Y = data['direction']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    svc = svm.SVC()
    svc.fit(X_train, Y_train)
    score = svc.score(X_test, Y_test)
    print(str(delay) + " day(s) delay")
    print(score)
