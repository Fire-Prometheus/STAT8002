from python.experiment.sentiment import default_test, SentimentAnalysis, VaderSentimentScoring
from os import path
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

grain = 'WHEAT'
trade_day_delay = 1

filename = 'data_' + grain + '_delay' + str(trade_day_delay) + '.pickle'
if not path.exists(filename):
    analysis = SentimentAnalysis(VaderSentimentScoring(), grain)
    analysis.test(trade_day_delay=trade_day_delay, to_pickle=True)

data1 = pd.read_pickle(filename)

factor = 'precipitation'
data2 = pd.read_csv('PCA_' + factor + '_' + grain + '.csv')
data2['Date'] = data2['Date'].apply(lambda d: pd.to_datetime(d, format='%Y-%m-%d').date())
data = pd.merge(data1, data2, on='Date')
data = data[data['direction'] != 0]
# CORN
# x = data[['Algeria', 'France', 'Egypt', 'Japan', 'Malaysia', 'Brazil', 'South Africa', 'EU-28',
#           'United States of America', 'Ukraine', 'Spain', 'Rep. of Korea', 'Argentina', 'Colombia',
#           'Russian Federation', 'Romania', 'Serbia', 'Netherlands', 'Italy', 'positive', 'negative']]
# SOYBEAN
# x = data[['Paraguay', 'Indonesia', 'EU-28', 'United States of America', 'Thailand', 'Mexico', 'Brazil', 'Argentina',
#           'Netherlands', 'Spain', 'Germany', 'Ukraine', 'China', 'Canada', 'Japan', 'Russian Federation', 'positive',
#           'negative']]
# WHEAT
x = data[
    ['United States of America', 'Philippines', 'Germany', 'France', 'Spain', 'Indonesia', 'EU-28', 'Brazil', 'Japan',
     'Canada', 'Romania', 'Egypt', 'Russian Federation', 'Ukraine', 'Argentina', 'Algeria', 'Italy', 'Turkey',
     'Australia', 'positive',
     'negative']]
y = data['direction']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
svc = svm.SVC()
svc.fit(x_train, y_train)
score = svc.score(x_test, y_test)
print(str(trade_day_delay) + " day(s) delay")
print(score)

# reg = LinearRegression()
# reg = reg.fit(x, y)
# score = reg.score(x, y)
# print(str(trade_day_delay) + " day(s) delay")
# print(score)
