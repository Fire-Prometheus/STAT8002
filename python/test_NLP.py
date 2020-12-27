import re

import nltk
import numpy
import pandas
from nltk.corpus import wordnet

##################################################
# Preparation
##################################################
# download modules
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

stopwords = nltk.corpus.stopwords.words()
# define lists of undesired words
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
                        'mass': ['gram', 'kilogram', 'kg', 'ton', 'tonne', 'quintal', 'pound', 'lb', 'ounce', 'oz',
                                 'hundredweight', 'cwt']}
undesired_words.extend(list(undesired_words_unit.values()))


##################################################
# Functions
##################################################
def preprocess(text):
    # cast to lower cases first
    text = text.lower()
    # numeric strings removal
    text = re.sub(r"\d", "", text)
    # tokenization
    words = nltk.word_tokenize(text)
    # stopwords removal
    words = [word for word in words if re.fullmatch(r"^\w[\w-]+$", word) and word not in stopwords]
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
    words = [word for word in filtered_words if len(word) > 1 and word not in undesired_words]
    # synonym replacement
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
    # remove undesired words
    words = [word for word in filtered_words if len(word) > 1 and word not in undesired_words]
    return words


##################################################
# Experiment
##################################################
# test_cases = [
#     'After suffering steep losses since last Friday, the cotton market is anticipating Thursday’s weekly sales and exports from USDA. After showing positive numbers across the Jan-Feb-March time-frame, the recent data has been anything but positive.\nIn fact, the report has spanned the depths of net cancellations to the heights of massively high sales. So, Thursday’s will be critical to the psyche of the market.\nWednesday is the last trading session for May 2020 cotton. Over its lifetime, it posted a high of 83.50 cents in June of 2018 to a contract low of 48.35 cents just last month. There were no notices Wednesday, and for the course of its delivery period, only 10 were issued.\nBeyond exports-sales, the market is anticipating Thursday’s jobless claims data, and then Friday’s all-important monthly jobs report.\nTo the latter, April’s non-farm jobs loss is expected to show 21-plus million. This compares to March’s loss of 700,000 jobs. The unemployment rate has moved from 4.4% to its present 16% unemployed. That translates into some thirty million people out of work.',
#     'Corn is 3 to 4 cents lower, soybeans are 5 to 6 cents lower, and wheat was 3 to 7 cents lower.\nCORN\nCorn trade was 3 to 4 cents lower at midday with trade fading from early strength again with little fresh news and negative spillover from other commodities. The ethanol report showed production 67,000 barrels higher, with stocks down 725,000 barrels as gasoline demand improves.\nWeather should remain a non-issue for now with planters rolling in many areas still, with many done with corn. Basis faded to end last week with sideways action likely to continue.\nOn the July contract support is continuous chart low at 3.00 1/2, and resistance the 20-day at 3.23.\nSOYBEANS\nSoybean trade is 4 to 6 cents lower with trade working lower as the dollar gains vs. the ral again overnight, along with little other fresh news. Meal is mixed and oil is 40 to 50 points lower. South America continues to move along harvest wise with strong shipments out of Brazil likely to continue after a record April.\nContinued planting progress is likely in most areas. The July soybean chart support is the lower Bollinger Band at $8.25, with resistance the 20-day at $8.46 which we faded below yesterday.\nWHEAT\nWheat trade is 3 to 8 cents lower at midday with the recent pattern of weaker overnight continuing into the day session. The western plains will continue to look to see how the crop recovers from the April freezes, along with Europe and Russia progress with the near term showers.\nKansas City is at a 41-cent discount to Chicago on the July with wider action at midday, while Minneapolis has narrowed back to 8 cents. The dollar is above 100 on the index with the strong start to the week.\nThe July Kansas City chart support is the lower Bollinger Band at $4.71, with resistance the 20-day at $4.88.\nGeneral Comments\nThe U.S. stock market is mixed with the Dow 5 points lower as active trade continues. The dollar index is 40 higher. Interest rate products are lower. Energies are weaker with crude $1.20 lower. Livestock trade is weaker. Precious metals are mostly lower with gold down $20.00.',
#     'During the 2010s, northeastern South Dakota was indicative of South Dakota’s slogan — the Land of Infinite Variety. Prolific rains during 2010 and 2011 drenched fields and formed swamps out of formerly productive farmland. The 2012 drought started cleaning that up, and those formerly flooded fields started snapping back into shape. Then came the fall of 2018, when October snowfall and prolific rainfall started cascading into a wet 2019 that sent many of the region’s farmers into their crop insurance agent’s office to pore over prevented planting plans. So far, 2020 is shaping up to be another battle year in some areas. Fieldwork today in the region is at a standstill, as 1.5 to 2.5-inch rainfalls peppered the region yesterday and last night, pouring water on top of already flooded fields. On the other hand, fieldwork has proceeded ahead of schedule in other parts of the region. In 2019, prolific precipitation on the fields of Jason Frerichs and his family near Wilmot in northeastern South Dakota pushed back planting well into late spring. This year, though, they have been able to plant spring wheat, corn, and soybeans so far on a timely basis. “We started around April 23,” he says. “We’ve been planting corn and soybeans at the same time. It’s been fairly steady.” So far, they’ve finished planting about one half of their crop. They have adjusted their rotation from original plans at season’s start. “We backed off corn acres due to market conditions and also from a logistics standpoint,” he says. Had they planted all the corn acres they had planned on, they would have had to add another 12-row combine head to the 12-row unit they already have to handle additional corn volume. Instead, they increased the amount of spring wheat and soybeans. In the midst of plunging grain prices partially keyed by COVID-19, bright spots include low fertilizer and fuel prices. “The fuel price comes at a significant burden in terms of the ethanol industry, though,” he says. COVID-19 has spurred some ethanol plants to shutter. So far, though, ethanol plants in Frerichs’ area have been able to keep operating. Calving season, too, is a bright spot compared with the last two years. “We started backing off a couple years ago to start calving in late April,” he says. “Last year, I thought it wasn’t late enough, but now late April is the sweet spot.',
#     'As the old saying goes, “you need to make hay while the sun is shining,” and U.S. farmers took that to heart last week with another solid gain in planting progress. No doubt, it was right they did, as much of the Midwest has now returned to wet and unseasonably cool temperatures, not to mention the threat of a freeze. Nevertheless, corn planting increased 24% to reach 51% complete, and bean planting advanced 15% and stood at 23% complete. Compared with the disastrous spring of 2019, we are now 30% and 18% ahead for these two crops, respectively. To round out a few of the key figures in the update, cotton planting moved forward by 5% and stood at 18% complete, and winter wheat conditions improved 1%. In the novel Tale of Two Cities, Charles Dickens provided us with the immortal lines, “it was the best the times, it was the worst of times,” and it would seem nothing could better describe the contrast between Brazilian and U.S. agriculture at this moment. This was again highlighted via recent export data. During the month of April, Brazil set a new monthly record for bean exports of 16.3 MMT. This was nearly 7 million above the same month a year ago and almost 4 million above the previous record of 12.35 MMT set in May of 2018. Comparatively, last week bean exports from the U.S. set a marketing-year low of just 318k MT (11.7 MB) and for the month as a whole totaled less than 2 MMT. Needless to say, a record low currency valuation against the dollar goes a long way towards boosting exports. Dry weather continues to be a concern in Southern Brazil as the safrinha corn is moving through the pollination stage. There is currently rain in the forecast for some of the more parched regions, but if it will be too little too late is yet to be determined. Further south in Argentina, rains slowed the pace of harvest somewhat, and it is estimated that beans are now 68% complete and corn 37%. While one could argue as to which market first felt the effects of the COVID-19 inspired contraction in demand, crude would have to be towards the top of the list. If that were the case, you could then possibly make the argument that it would be one of the first to recover when business reopens, and it would appear that may be the case as well. Since posting the extreme low one week ago today, crude has closed higher each day since and has now risen back above old support. While hardly what you would label a bull market, it does offer a hopeful sign for the rest of the commodity sector that the worst may finally be behind us. Analysis Markets (General) News and Analysis',
#     "SAO PAULO, May 4 (Reuters) - Brazilian soybean exports in April reached 16.3 million tonnes, an all-time record for a single month and an increase from 9.4 million tonnes in same month last year, according to average daily export data released on Monday by the government. The previous record was 12.35 million tonnes, set in May 2018. Brazil, the world's largest exporter of soybeans, had shipped 11.64 million tonnes of soybeans in March, according to government data, as local farmers finish collecting yet another bumper crop. (Reporting by Roberto Samora Ana Mano) © Copyright Thomson Reuters 2020. Click For Restrictions - http://about.reuters.com/fulllegal.asp"
# ]
# for test_case in test_cases:
#     print(preprocess(test_case))

csv = pandas.read_csv('agricultural news from 2017.csv')
news = csv[csv["timestamp"].between(1546300800000, 1577836799000)]
news = news[news.tags.str.contains('corn') == True]
news['timestamp'] = news['timestamp'].apply(lambda t: pandas.Timestamp(t, unit='ms'))
news['Date'] = news['timestamp'].apply(lambda t: pandas.to_datetime(t, format='%b %d, %Y').date())
news['new content'] = news['content'].apply(lambda c: preprocess(c))
news['new headline'] = news['headline'].apply(lambda c: preprocess(c))
news.to_pickle('preprocessed_news.pickle')

price = pandas.read_csv('US Corn Futures Historical Data.csv')
price['Vol.'] = price['Vol.'].apply(lambda v: float(v[0:-1]) * 1000 if len(v[0:-1]) >= 1 else numpy.NaN)
price['Change %'] = price['Change %'].apply(lambda p: float(p[0:-1]) / 100)
price['Date'] = price['Date'].apply(lambda d: pandas.to_datetime(d, format='%b %d, %Y').date())
price['direction'] = price['Change %'].apply(lambda change: 0 if change == 0 else (1 if change > 0 else -1))
price.to_pickle('preprocessed_price.pickle')
