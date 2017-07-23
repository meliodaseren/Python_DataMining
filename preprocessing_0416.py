# coding: utf-8
import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# 處理遺漏值
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
df


# 檢查是否為空值
df.isnull()

# 欄位的空值數量
df.isnull().sum()

# 刪除遺漏值
# 將後兩筆 1 2 刪除了
df.dropna()
df.dropna(axis=0)
df.dropna(axis=1)

# only drop rows where all columns are NaN
df.dropna(how='all')

# drop rows that have not at least 4 non-NaN values
df.dropna(thresh=4)
# only drop rows where NaN appear in specific columns (here: 'C')
df.dropna(subset=['C'])


# Impot (補值)
# 補行的平均; strategy='mean'or'median'or'most_frequen'
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
imputed_data

df.values

# 處理分類數據
df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']
df

# 有序特徵 ex:size , XL＞L＞M
# 名目特徵 ex:color

# 對應有序特徵
size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}

df['size'] = df['size'].map(size_mapping)
df

# 反向對應
inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'].map(inv_size_mapping)

# 將類別標籤轉成整數值
df['classlabel']

np.unique(df['classlabel'])

# scikit-learn 大多會自動轉

class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
class_mapping

# 內建函數 enumerate()，回傳以參數(parameter) iterable 與連續整數配對的 enumerate 物件， start 為整數的起始值，預設為 0
d = ['Spring', 'Summer', 'Fall', 'Winter']
for i, j in enumerate(d, 1):
    print(i, j)

df['classlabel'] = df['classlabel'].map(class_mapping)
df

inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
df

# 直接用 scikit LabelEncoder 比較快
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
y

class_le.inverse_transform(y)

# 對名目特徵做 one-hot encoding
X = df[['color', 'size', 'price']].values
print(X)

color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)

ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(X).toarray()

pd.get_dummies(df[['price', 'color', 'size']])


# 將數據集分成訓練集和測試集
df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

print('Class labels', np.unique(df_wine['Class label']))
df_wine.head()


X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test =     train_test_split(X, y, test_size=0.3, random_state=0)


# 縮放特徵
# 最大最小縮放　normalized

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

X_test_norm


# 標準化縮放 standardized
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

ex = pd.DataFrame([0, 1, 2, 3, 4, 5])

# standardize
ex[1] = (ex[0] - ex[0].mean()) / ex[0].std(ddof=0)

# normalize
ex[2] = (ex[0] - ex[0].min()) / (ex[0].max() - ex[0].min())
ex.columns = ['input', 'standardized', 'normalized']
ex
