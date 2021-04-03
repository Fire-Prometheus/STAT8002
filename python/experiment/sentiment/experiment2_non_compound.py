from python.experiment.sentiment import default_test, SentimentAnalysis, VaderSentimentScoring

default_test(SentimentAnalysis, VaderSentimentScoring, ['negative', 'positive', 'neutral'], is_content=False)
