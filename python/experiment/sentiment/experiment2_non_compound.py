from python.experiment.sentiment import default_test, SentimentAnalysis, VaderSentimentScoring, generate_samples
import pandas as pd

columns = ['negative', 'positive', 'neutral']
# default_test(SentimentAnalysis, VaderSentimentScoring, columns, is_content=False)

samples1 = generate_samples(SentimentAnalysis, VaderSentimentScoring, columns, is_content=True,
                            file_path='../../data/Experiment2_samples_content.csv')
samples2 = generate_samples(SentimentAnalysis, VaderSentimentScoring, columns, is_content=False,
                            file_path='../../data/Experiment2_samples_headline.csv')
df = pd.concat([samples1, samples2])
df['EXPERIMENT'] = '2'
df['TOOL'] = 'VADER'
df.to_csv('../../data/Experiment2_samples.csv', index=False)
