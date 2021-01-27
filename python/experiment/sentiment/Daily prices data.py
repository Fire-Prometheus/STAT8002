import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats
import os


def PCA_transformation(predictor):
    x_std = StandardScaler().fit_transform(predictor)
    pca = PCA(n_components=predictor.shape[1], svd_solver='full')
    x = pca.fit_transform(x_std)
    return x, pca


def get_daily_price_data(file_path):
    g = os.walk(file_path)

    i = 0
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if 'Futures' in file_name:
                if i == 0:
                    price_data = pd.read_csv(file_path + file_name)
                    price_data = price_data[['Date', 'Price', 'Change %']]
                    price_data.columns = ['Date', 'Price_' + file_name.replace(' Futures daily Data.csv', ''),
                                          'Change %_' + file_name.replace(' Futures daily Data.csv', '')]
                    price_data['Date'] = price_data['Date'].apply(lambda d: pd.to_datetime(d).date())
                    i = i + 1
                else:
                    read_file = pd.read_csv(file_path + file_name)
                    read_file = read_file[['Date', 'Price', 'Change %']]
                    read_file.columns = ['Date', 'Price_' + file_name.replace(' Futures daily Data.csv', ''),
                                         'Change %_' + file_name.replace(' Futures daily Data.csv', '')]
                    read_file['Date'] = read_file['Date'].apply(lambda d: pd.to_datetime(d).date())
                    price_data = pd.merge(price_data, read_file,
                                          on='Date',
                                          how='outer')
    price_data.sort_values('Date', ascending=False, inplace=True, ignore_index=True)
    price_data = price_data[price_data['Date'] <= pd.to_datetime('2020-12-31').date()]
    price_data = price_data.dropna()
    price_data['Price_US Soybeans'] = price_data['Price_US Soybeans'].apply(lambda x: float(x.replace(',', '')))
    # price.to_csv(filepath_1+'aggregated.csv')
    price_data = price_data.set_index('Date')
    return price_data


# data
filepath_1 = './'
price = get_daily_price_data(filepath_1)
Wheat = np.array(price['Price_US Wheat'])
Oats = np.array(price['Price_Oats'])
Rice = np.array(price['Price_Rough Rice'])
Corn = np.array(price['Price_US Corn'])
Soybean = np.array(price['Price_US Soybeans'])
Cattle = np.array(price['Price_Live Cattle'])
Oil = np.array(price['Price_Crude Oil WTI'])
Pig = np.array(price['Price_Lean Hogs'])
test_name = ['Price_US Wheat', 'Price_Oats', 'Price_Rough Rice', 'Price_US Corn',
             'Price_US Soybeans', 'Price_Live Cattle', 'Price_Crude Oil WTI', 'Price_Lean Hogs']

X = price[test_name].shift(-1).dropna().to_numpy()
Y = Rice[0:-1]

predict_data, X_test, target, y_test = train_test_split(X, Y, test_size=0.4, random_state=540)

predict_data_pca, pc_result = PCA_transformation(predict_data)
rs_pred = sm.OLS(target, sm.add_constant(predict_data_pca)).fit()
print(rs_pred.summary2(yname='Wheat price',
                 xname=['constant', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'],
                 alpha=0.05,
                 title='regression with Lag1 Prices(with PCA)'))

print(np.around(np.dot(rs_pred.params[1:9], pc_result.components_), decimals=4))

# testing the model.

test = StandardScaler().fit_transform(X_test)
coef = np.transpose(np.dot(rs_pred.params[1:9], pc_result.components_))
y_pred = np.dot(test, coef) + rs_pred.params[0]

# plt.plot(y_pred)
# plt.plot(y_test)
sse = np.sum((y_pred - y_test) ** 2)
sst = np.sum((np.mean(y_test) - y_test) ** 2)
test_size = np.shape(y_test)[0]
r = 1 - sse / sst
adj_r = 1 - (1 - r) * 597 / 589
std = np.std(y_pred - y_test)
mean = np.mean(y_pred - y_test)
reg = LinearRegression().fit(predict_data_pca, target)
R = reg.score(test, y_test)
print(' r ', adj_r, ' std ', std, ' mean ', mean, ' sse ', sse)
