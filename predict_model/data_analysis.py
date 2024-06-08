import sys
sys.path.append('/Users/kamile/Desktop/Bachelor-Project/bachelor-project')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data.scopus_data_retrieval import fetch_db

# read data
data = fetch_db()

data = data.drop(["doi", "title", "h_index"], axis=1)

print(data.describe().T)
print(data.describe(include="all").T)

# separate categorical and numerical data
cat_cols = data.select_dtypes(include=['object']).columns
num_cols = data.select_dtypes(include=np.number).columns.tolist()
print(cat_cols)
print(num_cols)

# for col in num_cols:
#     print(col)
#     print('Skew :', round(data[col].skew(), 2))
#     plt.figure(figsize = (15, 4))
#     plt.subplot(1, 2, 1)
#     data[col].hist(grid=False)
#     plt.ylabel('count')
#     plt.subplot(1, 2, 2)
#     sns.boxplot(x=data[col])
#     plt.show()

# fig, axes = plt.subplots(2, 2, figsize = (18, 18))
# fig.suptitle('Bar plot for all categorical variables in the dataset')
# sns.countplot(ax = axes[0, 0], x = 'query', data = data, color = 'blue', 
#               order = data['query'].value_counts().index);
# sns.countplot(ax = axes[0, 1], x = 'impact', data = data, color = 'blue', 
#               order = data['impact'].value_counts().index);

# plt.show()

# data transformation
# Function for log transformation of the column
def log_transform(data,col):
    for colname in col:
        if (data[colname] == 1.0).all():
            data[colname + '_log'] = np.log(data[colname]+1)
        else:
            data[colname + '_log'] = np.log(data[colname])
    

# dont use 
# log_transform(data, num_cols)
# sns.distplot(data["year_log"], axlabel="year_log")
# data['total_citations_log'] = data['total_citations_log'].replace([np.inf, -np.inf], np.nan)
# data = data.dropna(subset=['total_citations_log'])
# sns.distplot(data["total_citations_log"], axlabel="total_citations_log")
# data['limited_citations_log'] = data['limited_citations_log'].replace([np.inf, -np.inf], np.nan)
# data = data.dropna(subset=['limited_citations_log'])
# sns.distplot(data["limited_citations_log"], axlabel="limited_citations_log")

# data['log_citations'] = np.log1p(data['total_citations'])
# sns.histplot(data['log_citations'], kde=True, stat='density')
# plt.xlabel('Log-transformed Citations')
# plt.title('Histogram of Log-transformed Citations')
# plt.show()

# adjust plot size
plt.figure(figsize=(13,17))
sns.pairplot(data=data)
plt.show()

# fig, axarr = plt.subplots(4, 2, figsize=(12, 18))
# data.groupby('Location')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axarr[0][0], fontsize=12)
# axarr[0][0].set_title("Location Vs Price", fontsize=18)
# data.groupby('Transmission')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axarr[0][1], fontsize=12)
# axarr[0][1].set_title("Transmission Vs Price", fontsize=18)
# data.groupby('Fuel_Type')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axarr[1][0], fontsize=12)
# axarr[1][0].set_title("Fuel_Type Vs Price", fontsize=18)
# data.groupby('Owner_Type')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axarr[1][1], fontsize=12)
# axarr[1][1].set_title("Owner_Type Vs Price", fontsize=18)
# data.groupby('Brand')['Price_log'].mean().sort_values(ascending=False).head(10).plot.bar(ax=axarr[2][0], fontsize=12)
# axarr[2][0].set_title("Brand Vs Price", fontsize=18)
# data.groupby('Model')['Price_log'].mean().sort_values(ascending=False).head(10).plot.bar(ax=axarr[2][1], fontsize=12)
# axarr[2][1].set_title("Model Vs Price", fontsize=18)
# data.groupby('Seats')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axarr[3][0], fontsize=12)
# axarr[3][0].set_title("Seats Vs Price", fontsize=18)
# data.groupby('Car_Age')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axarr[3][1], fontsize=12)
# axarr[3][1].set_title("Car_Age Vs Price", fontsize=18)
# plt.subplots_adjust(hspace=1.0)
# plt.subplots_adjust(wspace=.5)
# sns.despine()

# plt.figure(figsize=(12, 7))
# sns.heatmap(data.drop(['Kilometers_Driven','Price'],axis=1).corr(), annot = True, vmin = -1, vmax = 1)
# plt.show()





#print(data.head())