from typing import List, Dict
from ast import literal_eval
from python.preprocessing import PICKLE, GRAINS, DataPreprocessor
from python.experiment.sentiment import VaderSentimentScoring
import pandas as pd

# df_list: List[pd.DataFrame] = []
# for grain in GRAINS:
#     pickle = DataPreprocessor.load_pickle(PICKLE['NEWS'][grain])
#     df_list.append(pickle.sample(n=20))
# df = pd.concat(df_list)
df = pd.read_csv('../data/pick_predicted.csv')
# df['new content'] = df['new content'].apply(lambda c: literal_eval(c))
# scoring = VaderSentimentScoring()
#
#
# def process(content: List[str]) -> Dict[str, float]:
#     scores = scoring.compute_scores(content)
#     del scores['compound']
#     del scores['neutral']
#     return scores

#
# df[['positivity', 'negativity']] = df['new content'].apply(lambda c: pd.Series(process(c)))
df['Date'] = pd.to_datetime(df['timestamp']).dt.date
price = pd.read_pickle('../data/preprocessed_price_corn.pickle')
temp = pd.merge(df, price, how='left', on='Date')
temp = temp[temp['direction'].notna()]
temp = temp[temp['user_sentiment'] != 0]
temp['equal'] = temp.apply(lambda row: row['direction'] == row['user_sentiment'], axis=1)
count_true = len(temp[temp['equal']])
print(count_true / len(temp))

