from python.preprocessing import AdvancedNewsDataPreprocessor

# advanced_data_preprocessor = AdvancedNewsDataPreprocessor('../data/agricultural news from 2017.csv')
d = AdvancedNewsDataPreprocessor.load_pickle_with_filter('../data/preprocessed_news_all.pickle', None)
