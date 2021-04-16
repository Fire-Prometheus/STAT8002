from python.experiment.sentiment import default_test, ExtendedSentimentAnalysis, VaderSentimentScoring, \
    SentiWordNetScoring, generate_samples
import pandas as pd

# default_test(ExtendedSentimentAnalysis, SentiWordNetScoring, columns=['positive', 'negative'], is_content=True)

samples1 = generate_samples(ExtendedSentimentAnalysis, VaderSentimentScoring, is_content=True)
samples1['TOOL'] = 'VADER'
samples2 = generate_samples(ExtendedSentimentAnalysis, VaderSentimentScoring, is_content=False)
samples2['TOOL'] = 'VADER'

samples3 = generate_samples(ExtendedSentimentAnalysis, SentiWordNetScoring, is_content=True)
samples3['TOOL'] = 'SENTIWORDNET'
samples4 = generate_samples(ExtendedSentimentAnalysis, SentiWordNetScoring, is_content=False)
samples4['TOOL'] = 'SENTIWORDNET'

df = pd.concat([samples1, samples2, samples3, samples4])
df['EXPERIMENT'] = '3b'
df.to_csv('../../data/Experiment3b_samples.csv', index=False)
