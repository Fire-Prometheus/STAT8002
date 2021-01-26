import pandas
import numpy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


file_path = './'

# =====================================================================================================================

tempe = pandas.read_csv(file_path+'average temperature.csv').drop('Unnamed: 0', axis=1)
tempe = tempe[tempe['Period'].isin(range(20190102,20200101))].rename(columns={'Period':'Date'})
tempe['Date'] = tempe['Date'].apply(lambda d: pandas.to_datetime(d, format='%Y%m%d').date().strftime('%Y-%m-%d'))

prec = pandas.read_csv(file_path+'precipitation.csv').drop('Unnamed: 0', axis=1)
prec = prec[prec['Period'].isin(range(20190102,20200101))].rename(columns={'Period':'Date'})
prec['Date'] = prec['Date'].apply(lambda d: pandas.to_datetime(d, format='%Y%m%d').date().strftime('%Y-%m-%d'))

# weather = pandas.merge(prec,tempe, on=['Reporter','Date','Code'],how='left',suffixes=['_rainfall','_temperature'])
# test = pandas.pivot_table(weather,index='Date',columns='Reporter',values=['Value_temperature','Value_rainfall'])

exp_country = pandas.read_excel(file_path+'export_country.xlsx', engine='openpyxl').drop('Unnamed: 0', axis=1)
imp_country = pandas.read_excel(file_path+'import_country.xlsx', engine='openpyxl').drop('Unnamed: 0', axis=1)

# =====================================================================================================================
country = set()
country.update(set(exp_country['Corn']))
country.update(set(imp_country['Corn']))
print(country)
EU_28 = ['Austria','Belgium','Bulgaria','Croatia','Cyprus','Denmark','Estonia','Finland',
         'France','Germany','Greece','Hungary','Ireland','Italy','Latvia','Lithuania',
         'Luxembourg','Malta','Netherlands','Poland','Portugal','Romania','Slovakia',
         'Slovenia','Spain','Sweden','United Kingdom']
# country.update(EU_28)

# =====================================================================================================================

test1 = pandas.pivot(tempe,index='Date',columns='Reporter',values='Value')
test1.fillna(test1.mean(), inplace=True)
test1.to_csv(file_path+'PCA_average_temperature.csv')

X = test1.to_numpy()
pca = PCA(n_components=20,svd_solver='full')
Y = pca.fit_transform(X)

print(pca.explained_variance_ratio_)
numpy.cumsum(pca.explained_variance_ratio_)
plt.plot(pca.explained_variance_ratio_,marker = 'o',label='temperature')
# pc1 = pca.components_[0]
test1['PC1'] = Y[:,0]



# =====================================================================================================================

test2 = pandas.pivot(prec,index='Date',columns='Reporter',values='Value')
test2['EU-28'] = test2[EU_28].apply(lambda x: x.sum() / 28, axis=1)
test2 = test2[country]
test2.fillna(test2.mean(), inplace=True)
test2.to_csv(file_path+'PCA_precipitation.csv')

# R2 = test2.corr()
X2 = test2.to_numpy()
# X2_std = StandardScaler().fit_transform(X2)
# numpy.where(numpy.isnan(X2))
pca2 = PCA(n_components=18,svd_solver='full')
Y2 = pca2.fit_transform(X2)

print(pca.score(X,Y))
print(pca2.explained_variance_ratio_)
numpy.cumsum(pca2.explained_variance_ratio_)
plt.plot(pca2.explained_variance_ratio_,marker = 'o',label='precipitation')
plt.legend()
test2['PC1']=Y2[:,0]
test2['PC2']=Y2[:,1]
test2['PC3']=Y2[:,2]
test2['PC4']=Y2[:,3]
test2['PC5']=Y2[:,4]
test2['PC6']=Y2[:,5]
test2['PC7']=Y2[:,6]

test1['PC1'].to_csv(file_path+'temperature_principle_components')