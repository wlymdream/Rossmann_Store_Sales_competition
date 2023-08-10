
import pandas as pd
import datetime
import csv
import numpy as np
import os
import xgboost as xgb
import itertools
import operator
import warnings

from DateFrameImputerUtils import DataFrameImputerUtils

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import TransformerMixin
from sklearn import model_selection
from matplotlib import pylab as plt
import matplotlib.pyplot as pltt
# % matplotlib inline

plot = True

# 我们要预测的目标
goal = 'Sales'
# 某天的某个商店 为了后续方便取数据，我们先这么定义
my_id = 'Id'

# 定义一些变换和评判准则
def toWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y!=0
    w[ind] = 1./(y[ind]**2)
    return w

def rmspe(yhat, y):
    w = toWeight(y)
    rmspe = np.sqrt(np.mean(w*(y-yhat)**2))
    return rmspe

'''
    自定义损失函数
'''
def rmspe_xg(yhat, y):
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = toWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return 'rmspe', rmspe

# 分析目标变量 最后发现，其实我们需要的是营业的数据，未营业的数据我们可以给过滤掉
# store = pd.read_csv('./data/store.csv')
# # print(store.head())
# train_df = pd.read_csv('./data/train.csv')
# print(train_df)
# 5674        215  卖出去5674的有215个店铺
# sales_count = train_df['Sales'].value_counts()
# print(sales_count)
# # pltt.hist(sales_count, bins=100)
# pltt.hist(np.log(train_df[train_df['Sales']>0]['Sales']), bins=100)
# pltt.show()
# train_df_opens = train_df['Sales', 'Open']

# 加载数据以及对数据的预处理
def load_data():
    store = pd.read_csv('./data/store.csv')
    train_org = pd.read_csv('./data/train.csv', dtype={'StateHoliday':pd.np.string_})
    test_org = pd.read_csv('./data/test.csv', dtype={'StateHoliday':pd.np.string_})
    # 相当于左链接
    train_data = pd.merge(train_org, store, on='Store', how='left')
    test_data = pd.merge(test_org, store, on='Store', how='left')
    features = test_data.columns.tolist()
    # print(features)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    # 只保留数据类型是以上几种的列
    features_numeric = test_data.select_dtypes(include=numerics).columns.tolist()
    # 保留非数字类型的数据
    features_non_numeric = [f for f in features if f not in features_numeric]
    # print(features_numeric)
    return (train_data, test_data, features, features_non_numeric)

# 数据预处理
def preprocess_data(train_data, test_data, features, features_non_numeric):
    # 我们取销量大于0的数据作为训练集
    train_data = train_data[train_data['Sales'] > 0]

    for data in [train_data, test_data]:
        # 新增3列，把年月日变成数据类型

        data['year'] = data.Date.apply(lambda x: x.split('-')[0])
        data['year'] = data['year'].astype(float)
        data['month'] = data.Date.apply(lambda x: x.split('-')[1])
        data['month'] = data['month'].astype(float)
        data['day'] = data.Date.apply(lambda x: x.split('-')[2])
        data['day'] = data['day'].astype(float)

        # Jan,Apr,Jul,Oct 变成1000100这种编码 每一条数据新增12列
        month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for single_month in month :
            data['promo' + single_month] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if single_month in x else 0)

        # data['promoJan'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Jan' in x else 0)

    noisy_features = [my_id, 'Date']
    # 去掉噪声字段
    features = [c for c in features if c not in noisy_features]
    features_non_numeric = [c for c in features_non_numeric if c not in noisy_features]
    features.extend(['year', 'month', 'day'])
    # 填充空值
    data_util = DataFrameImputerUtils()
    train_data = data_util.fit_transform(train_data)
    test_data = data_util.fit_transform(test_data)


    #处理非数字类型的数据
    le_obj = LabelEncoder()
    for col in features_non_numeric:
        le_obj.fit(list(train_data[col]) + list(test_data[col]))
        train_data[col] = le_obj.transform(train_data[col])
        test_data[col] = le_obj.transform(test_data[col])

    # 对数据进行归一化操作
    scaler_obj = StandardScaler()
    for col in set(features) - set(features_non_numeric):
        #set([]):
        # train_data[col]是Serires 此对象没有reshape方法 可以用.values.reshape
        scaler_obj.fit(train_data[col].values.reshape((-1, 1)))
        train_data[col] = scaler_obj.transform(train_data[col].values.reshape(-1, 1))
        test_data[col] = scaler_obj.transform(test_data[col].values.reshape(-1, 1))
    return (train_data, test_data, features, features_non_numeric)

# 利用XGBoost建模
def XGB_native(train_data, test_data, features, features_non_numeric):
    tree_max_depth = 13
    eta = 0.01
    ntrees = 8000
    mcw = 3
    # booster: 基础学习器 梯度提升树
    params = {
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'eta': eta,
        'max_depth': tree_max_depth,
        'min_child_weight': mcw,
        'subsample': 0.9,
        'colsample_bytree': 0.7,
        'silent': 1
    }

    tsize = 0.05
    x_train, x_test = model_selection.train_test_split(train_data, test_size=tsize)
    # +1 y数据大于等于0
    print(features)
    dtrain = xgb.DMatrix(x_train[features], np.log(x_train[goal]) + 1)
    dvaild = xgb.DMatrix(x_test[features], np.log(x_test[goal]) + 1)
    # 用于后续打印日志，评估测试集和训练集(损失) early_stopping_rounds 如果连续100轮停止下降，那么8000没用，后面不会继续迭代 feval loss是啥，以啥为标准评估
    watchlist = [(dvaild, 'eval'), (dtrain, 'train')]
    gbm = xgb.train(params, dtrain, ntrees, evals=watchlist, early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=True)
    train_probs = gbm.predict(xgb.DMatirx(x_test[features]))
    indices = train_probs < 0
    train_probs[indices] = 0
    error = rmspe(np.exp(train_probs) - 1, x_test[goal].values)
    print(error)

    test_probs = gbm.predict(xgb.DMatrix(test_data[features]))
    indices = test_probs < 0
    test_probs[indices] = 0
    submission = pd.DataFrame('./result/dat-xgb_d%s_eta%s_ntree%s_mcw%s_tsize%s.csv'
                              % (str(tree_max_depth), str(eta), str(ntrees), str(mcw), str(tsize)), index=False)






train_data, test_data, features, features_non_numeric = load_data()
train_data, test_data, features, features_non_numeric = preprocess_data(train_data, test_data, features, features_non_numeric)
XGB_native(train_data, test_data, features, features_non_numeric)





