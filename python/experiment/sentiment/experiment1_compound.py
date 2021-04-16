import pandas as pd

from python.experiment.sentiment import default_test, SentimentAnalysis, VaderSentimentScoring, generate_samples

# default_test(SentimentAnalysis, VaderSentimentScoring, ['compound'], is_content=False)

samples1 = generate_samples(SentimentAnalysis, VaderSentimentScoring, ['compound'], is_content=True,
                            file_path='../../data/Experiment1_samples_content.csv')
samples2 = generate_samples(SentimentAnalysis, VaderSentimentScoring, ['compound'], is_content=False,
                            file_path='../../data/Experiment1_samples_headline.csv')
df = pd.concat([samples1, samples2])
df['EXPERIMENT'] = '1'
df['TOOL'] = 'VADER'
df.to_csv('../../data/Experiment1_samples.csv', index=False)
