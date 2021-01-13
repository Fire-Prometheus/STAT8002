import nltk
import pandas
import sklearn
import numpy
import re
from sklearn import svm
from nltk.corpus import wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
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


def nlp(dataframe, top_n):
    dataframe['new content'] = dataframe['content'].apply(lambda c: " ".join(preprocess(c)))
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
    fit_transform = vectorizer.fit_transform(dataframe['new content'])
    names = vectorizer.get_feature_names()
    array = numpy.array(names)
    flatten_ = numpy.argsort(fit_transform.toarray()).flatten()[::-1][:top_n]
    return array[flatten_]


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
result = pandas.DataFrame(columns=['Date', 'Hot words'], index=None)
hot_words = set()
hot_words_dict = {}

for index, row in prices.iterrows():
    tmp_df = news[(news['Date'] > prices.loc[index + 1]['Date']) & (
            news['Date'] <= row['Date'])] if index < l - 1 else news[
        news['Date'] <= row['Date']]
    t = None if len(tmp_df) == 0 else nlp(tmp_df, 5)
    s = pandas.Series({
        'Date': row['Date'],
        'Hot words': t
    })
    if t is not None:
        for i in t:
            hot_words.add(i)
    hot_words_dict[row['Date']] = t
    result = result.append(s, ignore_index=True)

result = pandas.merge(result, prices, on='Date')[['Date', 'Change %', 'Hot words']]

X = pandas.DataFrame(columns=list(hot_words), index=result['Date'])
for index, row in X.iterrows():
    hot_words_dict_get = hot_words_dict.get(index)
    if hot_words_dict_get is not None:
        for e in hot_words_dict_get:
            row[e] = 1

X = X.fillna(0)
Y = result['Change %'].apply(lambda y: 0 if y == 0 else (1 if y > 0 else -1))

print(result)
print(hot_words)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
svc = svm.SVC()
svc.fit(X_train, Y_train)
score = svc.score(X_test, Y_test)
print(score)
