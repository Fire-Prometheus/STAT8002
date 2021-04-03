from python.experiment.sentiment import default_test, ExtendedSentimentAnalysis, VaderSentimentScoring, \
    SentiWordNetScoring

default_test(ExtendedSentimentAnalysis, SentiWordNetScoring, columns=['positive', 'negative'], is_content=True)
