import re

import nltk
import numpy
import pandas
from nltk.corpus import wordnet


class DataPreprocessor:
    data_frame: pandas.DataFrame

    def __init__(self, csv_path: str) -> None:
        self.data_frame = pandas.read_csv(csv_path)

    def save_as_pickle(self, filename: str) -> None:
        self.data_frame.to_pickle(filename)

    @staticmethod
    def load_pickle(pickle_path: str) -> pandas.DataFrame:
        return pandas.read_pickle(pickle_path)


class NewsDataPreprocessor(DataPreprocessor):
    def __init__(self, csv_path: str, tag: str = None) -> None:
        super().__init__(csv_path)
        self.__download_modules()
        self.stopwords = self.__get_stopwords()
        self.undesired_words = self.__get_undesired_words()
        if tag is not None:
            self.data_frame = self.data_frame[self.data_frame.tags.str.contains(tag) == True]
        self.data_frame['timestamp'] = self.data_frame['timestamp'].apply(lambda t: pandas.Timestamp(t, unit='ms'))
        self.data_frame = self.data_frame[
            self.data_frame["timestamp"] >= pandas.to_datetime('20170101', format='%Y%m%d')]
        self.data_frame['Date'] = self.data_frame['timestamp'].apply(
            lambda t: pandas.to_datetime(t, format='%b %d, %Y').date())
        self.data_frame['new content'] = self.data_frame['content'].apply(lambda c: self.__preprocess(c))
        self.data_frame['new headline'] = self.data_frame['headline'].apply(lambda c: self.__preprocess(c))

    @staticmethod
    def __download_modules() -> None:
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('vader_lexicon')

    @staticmethod
    def __get_stopwords() -> list[str]:
        return nltk.corpus.stopwords.words()

    @staticmethod
    def __get_undesired_words() -> list[str]:
        undesired_words = []
        undesired_words_time = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september',
                                'october',
                                'november', 'december']
        undesired_words_time.extend(['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'])
        undesired_words_time.extend(['day', 'week', 'month', 'quarter', 'half-year', 'year', 'annual'])
        undesired_words_time.extend(['daily', 'weekly', 'monthly', 'quarterly', 'half-yearly', 'yearly', 'annually'])
        undesired_words_time.extend(['today', 'tomorrow', 'tonight'])
        undesired_words.extend(undesired_words_time)

        undesired_words_unit = {'length': ['meter', 'm', 'foot', 'ft', 'yard', 'yd', 'centimeter', 'cm', 'inch', 'in',
                                           'kilometer', 'km', 'mile', 'mi'],
                                'area': ['acre', 'hectare'],
                                'volume': ['liter', 'litre', 'gallon', 'bushel', 'bsh', 'bu', 'bale'],
                                'temperature': ['°C', '°F'],
                                'mass': ['gram', 'kilogram', 'kg', 'ton', 'tonne', 'quintal', 'pound', 'lb', 'ounce',
                                         'oz',
                                         'hundredweight', 'cwt']}
        undesired_words.extend(list(undesired_words_unit.values()))
        return undesired_words

    def __preprocess(self, text) -> list[str]:
        # cast to lower cases first
        text = text.lower()
        # numeric strings removal
        text = re.sub(r"\d", "", text)
        # tokenization
        words = nltk.word_tokenize(text)
        # stopwords removal
        words = [word for word in words if re.fullmatch(r"^\w[\w-]+$", word) and word not in self.stopwords]
        # lemmatization
        lemmatizer = nltk.stem.WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        # part of speech identification and lemmatization
        filtered_words = []
        for word in words:
            pos_tag_word_ = nltk.pos_tag([word])[0][1]
            if re.search(r"^(NN|JJ|VB|RB|CC)", pos_tag_word_):
                def match(pos):
                    nonlocal pos_tag_word_
                    return re.search(r"^" + pos, pos_tag_word_)

                def get_pos_parameter():
                    return wordnet.ADJ if match("JJ") else (
                        wordnet.VERB if match("VB") else wordnet.ADV if match("RB") else wordnet.NOUN)

                filtered_words.append(lemmatizer.lemmatize(word, pos=get_pos_parameter()))
        # remove undesired words
        # words = [word for word in filtered_words if len(word) > 1 and word not in undesired_words]
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
        # remove undesired words
        words = [word for word in filtered_words if len(word) > 1 and word not in self.undesired_words]
        return words


class PriceDataPreprocessor(DataPreprocessor):

    def __init__(self, csv_path: str) -> None:
        super().__init__(csv_path)
        self.data_frame['Vol.'] = self.data_frame['Vol.'].apply(
            lambda v: float(v[0:-1]) * 1000 if len(v[0:-1]) >= 1 else numpy.NaN)
        self.data_frame['Change %'] = self.data_frame['Change %'].apply(lambda p: float(p[0:-1]) / 100)
        self.data_frame['Date'] = self.data_frame['Date'].apply(
            lambda d: pandas.to_datetime(d, format='%b %d, %Y').date())
        self.data_frame['direction'] = self.data_frame['Change %'].apply(
            lambda change: 0 if change == 0 else (1 if change > 0 else -1))


grains = ['corn', 'soybean', 'wheat', 'rice', 'oat']
for g in grains:
    news_data_preprocessor = NewsDataPreprocessor('agricultural news from 2017.csv', g)
    news_data_preprocessor.save_as_pickle('preprocessed_news_' + g + '.pickle')

price_data_preprocessor = PriceDataPreprocessor('US Corn Futures Historical Data.csv')
price_data_preprocessor.save_as_pickle('preprocessed_price.pickle')
