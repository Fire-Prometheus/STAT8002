import re
from typing import List

import nltk
import numpy
import pandas
import swifter
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    data_frame: pandas.DataFrame

    def __init__(self, csv_path: str) -> None:
        self.data_frame = pandas.read_csv(csv_path)

    def save_as_pickle(self, filename: str) -> None:
        self.data_frame.to_pickle(filename)

    @staticmethod
    def load_pickle(pickle_path: str) -> pandas.DataFrame:
        return pandas.read_pickle(pickle_path)


def download_modules() -> None:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('vader_lexicon')
    nltk.download('sentiwordnet')


class NewsDataPreprocessor(DataPreprocessor):
    def __init__(self, csv_path: str, tag: str = None) -> None:
        super().__init__(csv_path)
        download_modules()
        self.stopwords = self.__get_stopwords()
        self.undesired_words = self.__get_undesired_words()
        if tag is not None:
            self.data_frame = self.data_frame[self.data_frame.tags.str.contains(tag) == True]
        self.data_frame['timestamp'] = self.data_frame['timestamp'].apply(lambda t: pandas.Timestamp(t, unit='ms'))
        self.data_frame = self.data_frame[
            self.data_frame["timestamp"] >= pandas.to_datetime('20170101', format='%Y%m%d')]
        self.data_frame['Date'] = self.data_frame['timestamp'].apply(
            lambda t: pandas.to_datetime(t, format='%b %d, %Y').date())
        self.data_frame['new content'] = self.data_frame['content'].swifter.apply(lambda c: self.__preprocess(c))
        self.data_frame['new headline'] = self.data_frame['headline'].swifter.apply(lambda c: self.__preprocess(c))

    @staticmethod
    def __get_stopwords() -> List[str]:
        return nltk.corpus.stopwords.words()

    @staticmethod
    def __get_undesired_words() -> List[str]:
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

    def __preprocess(self, text) -> List[str]:
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


GRAINS = ['CORN', 'SOYBEAN', 'WHEAT', 'RICE', 'OAT']
# GRAINS = ['OAT']
PICKLE = {
    'NEWS': {},
    'PRICE': {}
}
for key, value in PICKLE.items():
    for grain in GRAINS:
        value[grain] = '../data/preprocessed_' + key.lower() + '_' + grain.lower() + '.pickle'
PICKLE['NEWS']['ALL'] = '../data/preprocessed_news_all_new_tags.pickle'


class AdvancedNewsDataPreprocessor(NewsDataPreprocessor):
    def __init__(self, csv_path: str) -> None:
        super().__init__(csv_path)
        # self.data_frame['new tags'] = self.data_frame['tags'].apply(lambda t: self.__transform_tags(t))
        # self.data_frame = self.data_frame[self.data_frame['new tags'].str.len() > 0]
        # documents = [TaggedDocument(words=word_tokenize(row['content']), tags=row['new tags']) for index, row in
        #              self.data_frame.iterrows()]
        # self.train_doc, self.test_doc = train_test_split(documents, test_size=0.33, random_state=42)
        # self.model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
        # self.model.build_vocab(self.train_doc)
        # self.model.train(documents, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    @staticmethod
    def __transform_tags(tags_list_str: str) -> List[str]:
        result: List[str] = []
        if pandas.notna(tags_list_str):
            for G in GRAINS:
                g = G.lower()
                if g in tags_list_str:
                    result.append(g)
        return result

    def evaluate(self):
        total = 0
        success = 0
        # for doc in self.test_doc:
        #     self.model.

    @staticmethod
    def load_pickle_with_filter(pickle_path: str, grain: str) -> pandas.DataFrame:
        pickle = DataPreprocessor.load_pickle(pickle_path)
        if grain is not None:
            pickle = pickle[pickle['new tags'].str.contains(grain) == True]
        return pickle
