from python.experiment.sentiment import VaderSentimentScoring
from python.preprocessing import GRAINS, PICKLE
import pandas as pd

for grain in GRAINS:
    pickle1 = pd.read_pickle('../data/data_' + grain + '_content_output.pickle')
    pickle1 = pickle1[['Date', 'direction', 'negative', 'positive', 'predicted_next_trade_day_direction_by_content']]
    pickle1 = pickle1.rename(columns={
        'direction': 'current_trade_day_direction',
        'negative': 'content_negativeness',
        'positive': 'content_positiveness'
    })
    pickle2 = pd.read_pickle('../data/data_' + grain + '_headline_output.pickle')
    pickle2 = pickle2[['Date', 'negative', 'positive', 'predicted_next_trade_day_direction_by_headline']]
    pickle2 = pickle2.rename(columns={'negative': 'headline_negativeness', 'positive': 'headline_positiveness'})
    merge = pd.merge(pickle1, pickle2, how='inner', on='Date')
    merge.to_csv('./news_sentiment_score_by_day_' + grain + '.csv')

scoring = VaderSentimentScoring()
for grain in GRAINS:
    pickle = pd.read_pickle(PICKLE['NEWS'][grain])
    scores: pd.DataFrame = pickle['new content'].swifter.apply(lambda c: scoring.compute_scores(c))
    scores = scores[['negative', 'positive']]
    scores = scores.rename(columns={
        'negative': 'content_negativeness',
        'positive': 'content_positiveness'
    })
    merge = pd.merge(pickle, scores, left_index=True, right_index=True)
    scores = pickle['new headline'].swifter.apply(lambda c: scoring.compute_scores(c))
    scores = scores[['negative', 'positive']]
    scores = scores.rename(columns={
        'negative': 'headline_negativeness',
        'positive': 'headline_positiveness'
    })
    merge = pd.merge(merge, scores, left_index=True, right_index=True)
    merge.to_csv('./news_sentiment_score_by_each_' + grain + '.csv')

pickle = pd.read_pickle('../data/preprocessed_news_all.pickle')
