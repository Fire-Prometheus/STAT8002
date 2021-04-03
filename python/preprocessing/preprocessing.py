from python.preprocessing import NewsDataPreprocessor, PriceDataPreprocessor, PICKLE, GRAINS, \
    AdvancedNewsDataPreprocessor

news_data_preprocessor = NewsDataPreprocessor('../data/news_2017_2020.csv')
news_data_preprocessor.save_as_pickle(PICKLE['NEWS']['ALL'])

# for g in GRAINS:
#     news_data_preprocessor = NewsDataPreprocessor('../data/news_2017_2020.csv', g.lower())
#     news_data_preprocessor.save_as_pickle(PICKLE['NEWS'][g.upper()])
#
# price_data_preprocessor = PriceDataPreprocessor('../data/US Corn Futures Historical Data.csv')
# price_data_preprocessor.save_as_pickle(PICKLE['PRICE']['CORN'])
#
# price_data_preprocessor = PriceDataPreprocessor('../data/US Soybeans Futures Historical Data.csv')
# price_data_preprocessor.save_as_pickle(PICKLE['PRICE']['SOYBEAN'])
#
# price_data_preprocessor = PriceDataPreprocessor('../data/US Wheat Futures Historical Data.csv')
# price_data_preprocessor.save_as_pickle(PICKLE['PRICE']['WHEAT'])
#
# price_data_preprocessor = PriceDataPreprocessor('../data/Rough Rice Futures Historical Data.csv')
# price_data_preprocessor.save_as_pickle(PICKLE['PRICE']['RICE'])
#
# price_data_preprocessor = PriceDataPreprocessor('../data/Oats Futures Historical Data.csv')
# price_data_preprocessor.save_as_pickle(PICKLE['PRICE']['OAT'])
