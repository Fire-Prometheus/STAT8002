import re

import nltk
import numpy
import pandas
from nltk.corpus import wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn import svm

##################################################
# Preparation
##################################################
# download modules
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
from sklearn.model_selection import train_test_split

# define better stopwords
all_stopwords = nltk.corpus.stopwords.words()
all_stopwords.extend(
    ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November',
     'December'])
all_stopwords.extend(['A', 'An', 'The'])
all_stopwords.extend(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
all_stopwords.extend(['day', 'Day', 'Month', 'month', 'Year', 'year'])
all_stopwords.extend(['may', 'might', 'can', 'could'])


##################################################
# Functions
##################################################
def filter_part_of_speech(words, filters):
    return [tag[0] for tag in nltk.pos_tag(words) if tag[1] in filters]


def preprocess(text):
    # tokenization
    words = nltk.word_tokenize(text)
    # lemmatization
    lemmatizer = nltk.stem.WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word.strip(), pos='v') for word in words]
    words = [lemmatizer.lemmatize(word.strip(), pos='a') for word in words]
    words = [lemmatizer.lemmatize(word.strip(), pos='n') for word in words]
    # stopwords removal
    words = [word.strip() for word in words if (len(word.strip()) > 1 and word.strip() not in all_stopwords)]
    # numeric strings removal
    words = [word for word in words if re.search("\\d", word) is None]
    # synonym replacement
    # processed_index = []
    # words_length = len(words)
    # for i in range(words_length):
    #     if i not in processed_index:
    #         synonyms = set([w for ln in wordnet.synsets(words[i]) for w in ln.lemma_names()])
    #         for j in range(i + 1, words_length):
    #             if j not in processed_index and words[j] in synonyms:
    #                 words[j] = words[i]
    #                 processed_index.append(j)
    #         processed_index.append(i)
    # part of speech filtering
    words = filter_part_of_speech(words, ['JJ', 'JJR', 'JJS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
    return words


def sentiment_analyse(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)


##################################################
# Load data
##################################################
csv = pandas.read_csv('../../data/agricultural news from 2017.csv')
# news
# news = csv[csv["timestamp"].between(1546300800000, 1577836799000)]  # timezone is not yet counted
news = csv
news = news[news.tags.str.contains('corn') == True]
# news = news[news['content'].str.contains('midday') == False]
news['timestamp'] = news['timestamp'].apply(lambda t: pandas.Timestamp(t, unit='ms'))
news['Date'] = news['timestamp'].apply(lambda t: pandas.to_datetime(t, format='%b %d, %Y').date())
# historical prices
prices = pandas.read_csv('../../US Corn Futures Historical Data.csv')
prices['Vol.'] = prices['Vol.'].apply(lambda v: float(v[0:-1]) * 1000 if len(v[0:-1]) >= 1 else numpy.NaN)
prices['Change %'] = prices['Change %'].apply(lambda p: float(p[0:-1]) / 100)
prices['Date'] = prices['Date'].apply(lambda d: pandas.to_datetime(d, format='%b %d, %Y').date())

##################################################
# Data processing and aggregation
##################################################
# preprocess
SCORE_COLUMNS = ['negative', 'neutral', 'positive', 'compound']
news['new content'] = news['new content'].apply(lambda c: " ".join(preprocess(c)))
prices['direction'] = prices['Change %'].apply(lambda change: 0 if change == 0 else (1 if change > 0 else -1))
# analyze
news[SCORE_COLUMNS] = news.apply(
    lambda row: pandas.Series(list(sentiment_analyse(row['new content']).values())), axis='columns')
# aggregate
mean = numpy.mean
result = news.groupby(['Date']).agg(
    {'negative': mean, 'neutral': mean, 'positive': mean, 'compound': mean}).reset_index()
result = pandas.merge(result, prices, on=['Date'], how='left')

temp = {}


def handle_holidays(dataframe):
    result = dataframe.copy()
    last_aggregated_trade_day = None
    holidays = result[result['Price'].isnull()]
    for index, row in holidays.iterrows():
        current_date = row['Date']
        next_trade_day = result[result['Date'] > current_date].Date.min()
        if (last_aggregated_trade_day is not None
                and next_trade_day is not numpy.nan
                and next_trade_day < last_aggregated_trade_day):
            continue
        else:
            global news
            news_to_be_aggregated_again = news[news.Date.between(current_date, next_trade_day, inclusive=True)]
            global SCORE_COLUMNS
            for c in SCORE_COLUMNS:
                result.loc[result['Date'] == next_trade_day, c] = mean(news_to_be_aggregated_again[c])
            last_aggregated_trade_day = next_trade_day
    result = result.dropna()
    return result


result = handle_holidays(result)


##################################################
# Model fitting
##################################################
def apply_day_delay(dataframe, trade_day_delay):
    result = dataframe.copy()
    global SCORE_COLUMNS
    columns = SCORE_COLUMNS + ['direction']
    result[columns] = result[columns].shift(-trade_day_delay)
    result = result.dropna()
    return result


DAY_DELAY = [0, 1, 2, 5, 7, 14, 30]
for delay in DAY_DELAY:
    data = apply_day_delay(result, delay)
    X = data[['negative', 'positive']]
    Y = data['direction']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    svc = svm.SVC()
    svc.fit(X_train, Y_train)
    score = svc.score(X_test, Y_test)
    print(str(delay) + " day(s) delay")
    print(score)
# data = data[['Date', 'negative', 'neutral', 'positive', 'compound', 'direction']]
# data.to_pickle('corn_2019.pickle')
