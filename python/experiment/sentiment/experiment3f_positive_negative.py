from python.experiment.sentiment import default_test, SentimentAnalysis, SentiWordNetScoring, VaderSentimentScoring
from python.preprocessing import GRAINS

for grain in GRAINS:
    print(grain)
    analysis = SentimentAnalysis(VaderSentimentScoring(), grain, is_content=True)
    analysis.test(trade_day_delay=1, to_pickle=True)
