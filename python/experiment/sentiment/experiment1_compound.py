from python.experiment import DELAY
from python.experiment.sentiment import SentimentAnalysis
from python.preprocessing import GRAINS

for grain in GRAINS:
    print('====================')
    print(grain)
    sentiment_analysis = SentimentAnalysis(grain)
    for delay in DELAY:
        sentiment_analysis.test(columns=['compound'], trade_day_delay=delay)
    print()
