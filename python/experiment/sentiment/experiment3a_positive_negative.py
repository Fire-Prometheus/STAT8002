from python.experiment.sentiment import default_test, SentimentAnalysis, VaderSentimentScoring, SentiWordNetScoring

default_test(SentimentAnalysis, SentiWordNetScoring, is_content=True)
