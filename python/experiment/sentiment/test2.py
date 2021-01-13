import re

import nltk
import numpy
import pandas
from nltk.corpus import wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from numpy import mean
from sklearn import svm

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
from sklearn.model_selection import train_test_split

all_stopwords = nltk.corpus.stopwords.words()
all_stopwords.extend(
    ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November',
     'December'])
all_stopwords.extend(['A', 'An', 'The'])
all_stopwords.extend(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
all_stopwords.extend(['day', 'Day', 'Month', 'month', 'Year', 'year'])
all_stopwords.extend(['may', 'might', 'can', 'could'])


def filter_part_of_speech(words, filters):
    return [tag[0] for tag in nltk.pos_tag(words) if tag[1] in filters]


def preprocess(text):
    words = nltk.word_tokenize(text)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word.strip(), pos='v') for word in words]
    words = [lemmatizer.lemmatize(word.strip(), pos='a') for word in words]
    words = [lemmatizer.lemmatize(word.strip(), pos='n') for word in words]
    words = [word.strip() for word in words if (len(word.strip()) > 1 and word.strip() not in all_stopwords)]
    words = [word for word in words if re.search("\\d", word) is None]
    processed_index = []
    words_length = len(words)
    for i in range(words_length):
        if i not in processed_index:
            synonyms = set([w for ln in wordnet.synsets(words[i]) for w in ln.lemma_names()])
            for j in range(i + 1, words_length):
                if j not in processed_index and words[j] in synonyms:
                    words[j] = words[i]
                    processed_index.append(j)
            processed_index.append(i)
    words = filter_part_of_speech(words, ['JJ', 'JJR', 'JJS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
    return words


def sentiment_analyse(dataframe):
    dataframe['new content'] = dataframe['content'].apply(lambda c: " ".join(preprocess(c)))
    analyzer = SentimentIntensityAnalyzer()
    score_map = {'negative': [], 'neutral': [], 'positive': [], 'compound': []}
    for text in dataframe['new content']:
        polarity_scores = analyzer.polarity_scores(text)
        score_map['negative'].append(polarity_scores['neg'])
        score_map['neutral'].append(polarity_scores['neu'])
        score_map['positive'].append(polarity_scores['pos'])
        score_map['compound'].append(polarity_scores['compound'])
    score_map['negative'] = mean(score_map['negative'])
    score_map['neutral'] = mean(score_map['neutral'])
    score_map['positive'] = mean(score_map['positive'])
    score_map['compound'] = mean(score_map['compound'])
    return score_map


csv = pandas.read_csv('../../data/agricultural news from 2017.csv')
# historical prices
prices = pandas.read_csv('../../US Corn Futures Historical Data.csv')
prices['Vol.'] = prices['Vol.'].apply(
    lambda v: float(v[0:-1]) * 1000 if len(v[0:-1]) >= 1 else numpy.NaN)
prices['Change %'] = prices['Change %'].apply(lambda p: float(p[0:-1]) / 100)
prices['Date'] = prices['Date'].apply(lambda d: pandas.to_datetime(d, format='%b %d, %Y').date())
# timezone is not yet counted
news = csv[csv["timestamp"].between(1546300800000, 1577836799000)]
news = news[news.tags.str.contains('corn') == True]
news['timestamp'] = news['timestamp'].apply(lambda t: pandas.Timestamp(t, unit='ms'))
news['Date'] = news['timestamp'].apply(lambda t: pandas.to_datetime(t, format='%b %d, %Y').date())
l = len(prices)
result = pandas.DataFrame(columns=['Date', 'negative', 'neutral', 'positive', 'compound'], index=None)

for index, row in prices.iterrows():
    tmp_df = news[(news['Date'] > prices.loc[index + 1]['Date']) & (
            news['Date'] <= row['Date'])] if index < l - 1 else news[
        news['Date'] <= row['Date']]
    t = {'negative': None, 'neutral': None, 'positive': None, 'compound': None} if len(
        tmp_df) == 0 else sentiment_analyse(tmp_df)
    s = pandas.Series({
        'Date': row['Date'],
        'negative': t['negative'],
        'neutral': t['neutral'],
        'positive': t['positive'],
        'compound': t['compound']
    })
    result = result.append(s, ignore_index=True)

result = pandas.merge(result, prices, on='Date')[['Date', 'Change %', 'negative', 'neutral', 'positive', 'compound']]

X = result[['negative', 'neutral', 'positive']]
X = X.fillna(0)
Y = result['Change %'].apply(lambda y: 0 if y == 0 else (1 if y > 0 else -1))

print(result)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
svc = svm.SVC()
svc.fit(X_train, Y_train)
score = svc.score(X_test, Y_test)
print(score)
