import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
## 数据标准化
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

def count_out_of_range(series):
    q1, q3 = np.percentile(series, [25, 75])
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    # 计算大于上限的个数
    count_above = len(series[series > upper_limit])

    # 计算小于下限的个数
    count_below = len(series[series < lower_limit])

    # 返回结果
    return count_above, count_below

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams["axes.unicode_minus"] = False

train_data = pd.read_csv(r'train.csv', sep=' ')
test_data = pd.read_csv(r'test.csv', sep=' ')

train_data['model'].fillna(-1, inplace=True)
mode_value = train_data['bodyType'].mode()[0]
train_data['bodyType'].fillna(mode_value, inplace=True)
mode_value1 = test_data['bodyType'].mode()[0]
test_data['bodyType'].fillna(mode_value1, inplace=True)
mode_value = train_data['fuelType'].mode()[0]
train_data['fuelType'].fillna(mode_value, inplace=True)
mode_value1 = test_data['fuelType'].mode()[0]
test_data['fuelType'].fillna(mode_value1, inplace=True)
mode_value = train_data['gearbox'].mode()[0]
train_data['gearbox'].fillna(mode_value, inplace=True)
mode_value1 = test_data['gearbox'].mode()[0]
test_data['gearbox'].fillna(mode_value1, inplace=True)

date_cols = ['regDate', 'creatDate']
cate_cols = ['name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode', 'seller', 'offerType']
num_cols = ['power', 'kilometer'] + ['v_{}'.format(i) for i in range(15)]
data = pd.concat([train_data, test_data])
cols = date_cols + cate_cols + num_cols

tmp = pd.DataFrame()
tmp['count'] = data[cols].count().values
tmp['missing_rate'] = (data.shape[0] - tmp['count']) / data.shape[0]
tmp['nunique'] = data[cols].nunique().values
tmp.index = cols
tmp

## 删去无效特征字段SaleID
train_data.drop('SaleID', axis=1, inplace=True)
SaleID = test_data['SaleID']
test_data.drop('SaleID', axis=1, inplace=True)

#对regDate和creatDate拆分，年月日
train_data['regDate_y'] = (train_data['regDate']/10000).astype('int64')
train_data['regDate_m'] = (train_data['regDate']/100-train_data['regDate_y']*100).astype('int64')
train_data['regDate_d'] = (train_data['regDate']-train_data['regDate_m']*100-train_data['regDate_y']*10000).astype('int64')
train_data['creatDate_y'] = (train_data['creatDate']/10000).astype('int64')
train_data['creatDate_m'] = (train_data['creatDate']/100-train_data['creatDate_y']*100).astype('int64')
train_data['creatDate_d'] = (train_data['creatDate']-train_data['creatDate_m']*100-train_data['creatDate_y']*10000).astype('int64')
test_data['regDate_y'] = (test_data['regDate']/10000).astype('int64')
test_data['regDate_m'] = (test_data['regDate']/100-test_data['regDate_y']*100).astype('int64')
test_data['regDate_d'] = (test_data['regDate']-test_data['regDate_m']*100-test_data['regDate_y']*10000).astype('int64')
test_data['creatDate_y'] = (test_data['creatDate']/10000).astype('int64')
test_data['creatDate_m'] = (test_data['creatDate']/100-test_data['creatDate_y']*100).astype('int64')
test_data['creatDate_d'] = (test_data['creatDate']-test_data['creatDate_m']*100-test_data['creatDate_y']*10000).astype('int64')
train_data.drop(['regDate','creatDate'], axis=1, inplace=True)
test_data.drop(['regDate','creatDate'], axis=1, inplace=True)

print(train_data['offerType'].unique())
train_data['offerType'].isna().any()
train_data.drop('offerType', axis=1, inplace=True)
test_data.drop('offerType', axis=1, inplace=True)

print(train_data['seller'].unique())
train_data['seller'].isna().any()

#notRepairedDamage	汽车有尚未修复的损坏：是：0，否：1
mapping = {'0.0': 0, '-': -1, '1.0': 1}
train_data['notRepairedDamage'] = train_data['notRepairedDamage'].map(mapping).astype(int)
test_data['notRepairedDamage'] = test_data['notRepairedDamage'].map(mapping).astype(int)

outlier_counts = train_data.apply(count_out_of_range)

#model	车型编码，已脱敏
train_data['model'].value_counts()

#gearbox	变速箱：手动：0，自动：1
train_data['gearbox'].value_counts()

#异常值截断
train_data.loc[train_data['power']>600, 'power'] = 600
train_data.loc[train_data['power']<1, 'power'] = 1
test_data.loc[test_data['power']>600, 'power'] = 600
test_data.loc[test_data['power']<0, 'power'] = 0
train_data.loc[train_data['v_13']>6, 'v_13'] = 6
test_data.loc[test_data['v_13']>6, 'v_13'] = 6
train_data.loc[train_data['v_14']>4, 'v_14'] = 4
test_data.loc[test_data['v_14']>4, 'v_14'] = 4

# 使用 fillna() 方法填充缺失值将含有空值和“-”的值全部替换为
train_data['notRepairedDamage'] = train_data['notRepairedDamage'].replace('-', np.nan)
train_data['notRepairedDamage'].fillna(0, inplace=True)

test_data['notRepairedDamage'] = test_data['notRepairedDamage'].replace('-', np.nan)
test_data['notRepairedDamage'].fillna(0, inplace=True)

tmp = pd.DataFrame(index = num_cols)
for col in num_cols:
    tmp.loc[col, 'train_Skewness'] = train_data[col].skew()
    tmp.loc[col, 'test_Skewness'] = test_data[col].skew()
    tmp.loc[col, 'train_Kurtosis'] = train_data[col].kurt()
    tmp.loc[col, 'test_Kurtosis'] = test_data[col].kurt()
tmp

# #设置画布大小
# plt.figure(figsize=(20,15))
# # 通过热力图查看特征之间的相关性
# sns.heatmap(train_data.corr().round(2), annot=True, cmap='coolwarm')
# plt.show()
print(1)
##对特征值排序可视化
f, ax = plt.subplots(figsize=(10, 6))
train_data.corr()['price'].drop('price').sort_values().plot.barh()
print(1)
#绘制特征与车价的散点图
#设置画布大小
plt.figure(figsize=(10,6))
#绘制散点图
plt.scatter(train_data['v_0'],train_data['price'])
#显示图形
plt.show()

#绘制特征与车价的散点图
#设置画布大小
plt.figure(figsize=(10,6))
#绘制散点图
plt.scatter(train_data['v_3'],train_data['price'])
#显示图形
plt.show()

#绘制特征与车价的散点图
#设置画布大小
plt.figure(figsize=(10,6))
#绘制散点图
plt.scatter(train_data['v_8'],train_data['price'])
#显示图形
plt.show()

#绘制特征与车价的散点图
#设置画布大小
plt.figure(figsize=(10,6))
#绘制散点图
plt.scatter(train_data['v_12'],train_data['price'])
#显示图形
plt.show()

sns.histplot(train_data['price'], kde=True)
plt.show()

#分布转换
train_data['price'] = np.log1p(train_data['price'])
plt.figure(figsize=(15,5))
sns.histplot(train_data['price'], kde=True)

train_data['price']

x = train_data.drop('price', axis=1)
y = train_data['price']

scaler = StandardScaler()
X_stand = scaler.fit_transform(x)
print(x, X_stand)

#抽取最相关的几个特征进行训练
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(f_regression, k=17)
x_best = selector.fit_transform(X_stand,y)
x_best

from sklearn.model_selection import train_test_split
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x_best, y, test_size=0.2)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
# 训练模型
model = LinearRegression()
model = model.fit(X_train, y_train)

# 预测价格
y_pred = model.predict(X_test)
#均方误差
mse = mean_squared_error(y_test,y_pred)
print("测试数据的误差：",mse)
train_y_pred = model.predict(X_train)
print('MAE of Stacking-LR for training set:',mean_absolute_error(y_train,train_y_pred))
print('MAE of Stacking-LR for testing set:',mean_absolute_error(y_test,y_pred))

# plt.plot(y_test.values,c="r",label="y_test")
# plt.plot(y_pred,c="b",label="y_pred")
# plt.legend()
# plt.show()
print(1)
# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X_stand, y, test_size=0.2, random_state=42)
print(1)
# 建立随机森林回归模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
print(1)
# 在测试集上进行预测
y_pred = rf_model.predict(x_test)

# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("均方误差（MSE）：", mse)
mae = mean_absolute_error(y_test, y_pred)
print("MAE of Random Forest Regression: ", mae)

# 将测试级数据标准化
t_X_stand = scaler.fit_transform(test_data)
# 选择k个最相关的特征进行训练
best_x = t_X_stand[:,selector.get_support(indices=True)]

# 预测二手车交易价格
t_y_pred = rf_model.predict(t_X_stand)
t_y_pred

prices = np.exp(t_y_pred) - 1
prices = prices

series = pd.Series(range(150000, 200000))
result = pd.DataFrame(data={
    'SaleID':series,
    'price':prices
})
result.to_csv('predictions.csv', index=False)