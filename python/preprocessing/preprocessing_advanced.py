from python.preprocessing import AdvancedNewsDataPreprocessor
from gensim.models import Word2Vec, Phrases

# advanced_data_preprocessor = AdvancedNewsDataPreprocessor('../data/agricultural news from 2017.csv')
d = AdvancedNewsDataPreprocessor.load_pickle_with_filter('../data/preprocessed_news_all.pickle', None)
new_content = d['new headline']
phrases = Phrases(new_content)
w2v = Word2Vec(sentences=phrases[new_content], size=10, window=5)
w2v.wv.save('../data/word2vec_headline.model')
