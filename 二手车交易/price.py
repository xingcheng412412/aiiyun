import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from IPython.display import Image
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold

# 根据最常出现填充，都是零，也可以根据其他来填充，不是瞎填的。。
def fill_missing(df):
    df['fuelType'] = df['fuelType'].fillna(0)
    df['gearbox'] = df['gearbox'].fillna(0)
    df['bodyType'] = df['bodyType'].fillna(0)
    df['model'] = df['model'].fillna(0)
    return df

# 处理完
def data_astype(df):
    # string
    df['SaleID'] = df['SaleID'].astype(int).astype(str)
    df['name'] = df['name'].astype(int).astype(str)
    df['model'] = df['model'].astype(str)
    df['brand'] = df['brand'].astype(str)
    df['bodyType'] = df['bodyType'].astype(str)
    df['fuelType'] = df['fuelType'].astype(str)
    df['gearbox'] = df['gearbox'].astype(str)
    df['notRepairedDamage'] = df['notRepairedDamage'].astype(str)
    df['regionCode'] = df['regionCode'].astype(int).astype(str)
    df['seller'] = df['seller'].astype(int).astype(str)
    df['offerType'] = df['offerType'].astype(int).astype(str)
    df['regDate'] = df['regDate'].astype(str)
    df['creatDate'] = df['creatDate'].astype(str)
    
    # date
    df['creatDate'] = pd.to_datetime(df['creatDate'])
    return df

# 定义评测指标
def cv_mae(model, X):
    mae = -cross_val_score(model, X, train_labels, scoring='neg_mean_absolute_error')
    return mae

train = pd.read_csv('train.csv', sep=' ')
test = pd.read_csv('test.csv', sep=' ')

# 目标值做log处理
train['price'] = np.log1p(train['price'])
# 查看转化后的分布，有点正态的感觉了
fig, ax = plt.subplots(figsize=(8, 7))
sns.displot(train['price'], kde=True)

# 可以根据计算结果或者其他特征进行移除
train.drop(train[train['price'] < 4].index, inplace=True)
# 整合训练集测试集以便后续特征工程
train_labels = train['price'].reset_index(drop=True)
train_features = train.drop(['price'], axis=1)
test_features = test
all_features = pd.concat([train_features, test_features]).reset_index(drop=True)

all_features = fill_missing(all_features)
all_features.isnull().sum().head()
all_features = data_astype(all_features)

# 先删除掉一些不要的特征
all_features = all_features.drop(['SaleID', 'name', 'regDate', 'model', 'seller',
                                  'offerType', 'creatDate', 'regionCode'], axis=1)
all_features = pd.get_dummies(all_features).reset_index(drop=True)

X = all_features.iloc[:len(train_labels), :]
X_test = all_features.iloc[len(train_labels):, :]

# K折交叉验证
kf = KFold(n_splits=5, random_state=15, shuffle=True)
ridge_alphas = [0.1, 1, 3, 5, 10]
ridge = RidgeCV(alphas=ridge_alphas, cv=kf)

# 查看交叉验证分数
score = cv_mae(ridge, X)

score.mean()

ridge.fit(X, train_labels)

# 查看R的平方
ridge.score(X, train_labels)

# 查看预测结果
fig, ax = plt.subplots(figsize=(8, 6))
sns.displot(ridge.predict(X_test))