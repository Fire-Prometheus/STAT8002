from python.experiment.sentiment import default_test, SentimentAnalysis, SentiWordNetScoring
from python.preprocessing import GRAINS

for grain in GRAINS:
    print(grain)
    analysis = SentimentAnalysis(SentiWordNetScoring(), grain)
    analysis.test(trade_day_delay=1, to_pickle=True)
