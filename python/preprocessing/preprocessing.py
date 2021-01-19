from python.preprocessing import NewsDataPreprocessor, PriceDataPreprocessor, PICKLE

news_data_preprocessor = NewsDataPreprocessor('../data/agricultural news from 2017.csv')
news_data_preprocessor.save_as_pickle('../data/preprocessed_news_all.pickle')

GRAINS = ['corn', 'soybean', 'wheat', 'rice', 'oat']
for g in GRAINS:
    news_data_preprocessor = NewsDataPreprocessor('../data/agricultural news from 2017.csv', g)
    news_data_preprocessor.save_as_pickle(PICKLE['NEWS'][g.upper()])

price_data_preprocessor = PriceDataPreprocessor('../data/US Corn Futures Historical Data.csv')
price_data_preprocessor.save_as_pickle(PICKLE['PRICE']['CORN'])

price_data_preprocessor = PriceDataPreprocessor('../data/US Soybeans Futures Historical Data.csv')
price_data_preprocessor.save_as_pickle(PICKLE['PRICE']['SOYBEAN'])

price_data_preprocessor = PriceDataPreprocessor('../data/US Wheat Futures Historical Data.csv')
price_data_preprocessor.save_as_pickle(PICKLE['PRICE']['WHEAT'])

price_data_preprocessor = PriceDataPreprocessor('../data/Rough Rice Futures Historical Data.csv')
price_data_preprocessor.save_as_pickle(PICKLE['PRICE']['RICE'])

price_data_preprocessor = PriceDataPreprocessor('../data/Oats Futures Historical Data.csv')
price_data_preprocessor.save_as_pickle(PICKLE['PRICE']['OAT'])
