# coding: utf-8
# (1) SciPy
# (發音為 “Sigh Pie”) 是一套為數學，科學，以及工程而特別開發的開放原始碼 (open source)的軟體。請參考官網 http://www.scipy.org/ 。
# 
# (2) NumPy
# 對於科學計算使用者，NumPy特別加強了多維陣列(muli-dimensional arrays)的運算效能。 請參考官網 http://www.numpy.org/ 。
# 
# (3) matplotlib
# matplotlib 是一套Python語言的畫圖資料庫，對於有使用過Matlab經驗的讀者，對於介面會相當熟悉。 請參考官網 http://matplotlib.org/ 。

import pandas as pd
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# input iris dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)

iris = datasets.load_iris()
# 透過 data 屬性觀察預測變數
X = iris.data[:, [2, 3]]  #我們只用後面兩個特徵方便繪圖 
# 花瓣長 / 花萼長
# 透過 target 屬性觀察目標變數
y = iris.target

print('Class labels:', np.unique(y))
# 有三種花

print(X)
print(y)

# 將以上150筆訓練資料隨機分成訓練數據集和測試數據集
# 其中70%為訓練數據集30%為測試數據集

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# random_state=0 確保每次切分資料的結果都相同
# test_size 參數設定切分比例
# random_state 參數設定隨機種子

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=46)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 特徵縮放Standardizing the features:
# 前處理的一種
sc = StandardScaler() 
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
print(X_test_std)

# 以上處理好了我們的資料
# 使用scikit-learn中的PLA做機器學習
# 因為有三類別
# 預設會使用one vs rest 方法

# ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=46)
# 設定最多迭代40次,學習速率0.1
ppn.fit(X_train_std, y_train)
# 用fit方法來訓練模型

print(y_test.shape)

y_pred = ppn.predict(X_test_std)
# 用predict方法來預測測試集
print('Misclassified samples: %d' % (y_test != y_pred).sum())
# 計算預測出來的類別和事實不符的數量

print(4 / 45)

from sklearn.metrics import accuracy_score

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# 作圖
def versiontuple(v):
    return tuple(map(int, (v.split("."))))

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    edgecolor='black',
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()


np.hstack((y_train, y_test))


np.vstack((X_train_std, X_test_std))


# 黑色的o是測試數據集
# 事實上訓練數據集畫出的決策邊界沒有很成功
# 正確率只用測試數據集計算 45個數據只有4個錯誤分類

# 邏輯斯迴歸

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)

plt.plot(z, phi_z)
# 畫圖
plt.axvline(0.0, color='k')
# 畫縱軸
plt.ylim(-0.1, 1.1)
# 縱軸範圍
plt.xlabel('z')
plt.ylabel('$\phi (z)$')

# y axis ticks and gridline
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)

plt.show()


# sigmoid function

def cost_1(z):
    return - np.log(sigmoid(z))

def cost_0(z):
    return - np.log(1 - sigmoid(z))

z = np.arange(-10, 10, 0.1)
phi_z = sigmoid(z)

c1 = [cost_1(x) for x in z]
plt.plot(phi_z, c1, label='J(w) if y=1')

c0 = [cost_0(x) for x in z]
plt.plot(phi_z, c0, linestyle='--', label='J(w) if y=0')

plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.xlabel('$\phi$(z)')
plt.ylabel('J(w)')
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('./figures/log_cost.png', dpi=300)
plt.show()


# cost function

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
# 實例化後使用fit方法一行就好了
# C正規化的參數

plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.show()

lr.predict_proba(X_test_std[0, :].reshape(1, -1))

X_test_std[0, :]

X_test_std[0, :].reshape(1, -1)

lr.predict_proba(X_test_std[0, :])

y_pred = lr.predict(X_test_std)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


# SVM

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)

# 實例化後使用fit方法一行就好了
# C正規化的參數

plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.show()

y_pred = svm.predict(X_test_std)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


# LR 和 SVM 結果很像
# 但是LR比較容易被離群值影響
# LR 優點是比較快  處理串流數據的時候比較好用

# SVM 解非線性問題
# 製造XOR亂數集

np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='b', marker='x',
            label='1')
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c='r',
            marker='s',
            label='-1')

plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')

plt.show()

print(X_xor)
print(y_xor)

svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
# 使用rbf kernel gamma是參數
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor,classifier=svm)

plt.legend(loc='upper left')

plt.show()

y_pred = svm.predict(X_xor)
print('Accuracy: %.2f' % accuracy_score(y_xor, y_pred))


# 實驗gamma

svm = SVC(kernel='rbf', random_state=0, gamma=0.5, C=1.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.show()

y_pred = svm.predict(X_test_std)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

svm = SVC(kernel='rbf', random_state=0, gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, 
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.show()


# KNN

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()

plt.show()
