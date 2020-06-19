import os
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 1、加载数据，包括训练集和测试集
cur_path = os.getcwd() + os.sep
train_path = cur_path + "train.csv"
test_path = cur_path + "test.csv"
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# 2、简单看看数据，展示不全，可以上excel看
print(train_data.head(5))


# 3、简单分析数据缺失情况
def get_na_rate(data_df):
    column_list = data_df.columns
    na_rate_dict = dict()
    for column in column_list:
        na_rate_dict.update({column: data_df[column].isna().sum()})
    return na_rate_dict


print(get_na_rate(train_data))


# 4、看看哪些数据要做数值化处理
# print(train_data.dtypes) # 看完以后简单归归类
def get_obj_rate(data_df):
    obj_rate_dict = dict()
    useless_columns = list()
    data_length = data_df.shape[0]
    for column in data_df.columns:
        unique_count = len(data_df[column].unique())
        if unique_count == 1 or unique_count == data_length:
            useless_columns.append(column)  # 找到明显无效的特征列
            continue
        if isinstance(data_df[column][0], str):
            obj_rate_dict.update({column: (unique_count, unique_count / (1.0 * data_length))})
    return obj_rate_dict, useless_columns


obj_rate_dict, useless_columns = get_obj_rate(train_data)
print(obj_rate_dict)
print(useless_columns)


# 5、对那些完全无效信息，先去除掉
def remove_useless_data(data_df_train, data_df_test, useless_columns):
    for column in useless_columns:
        data_df_train.drop(column, axis=1, inplace=True)
        data_df_test.drop(column, axis=1, inplace=True)


remove_useless_data(train_data, test_data, useless_columns)


# 6、数值化一下，因为我们要使用的都是树模型，这里我们都用label encoding，具体原因可以看LightGBM这篇订阅号
def to_numeric(data_df_train, data_df_test, columns_dict, label_column):
    for column in columns_dict.keys():
        if column == label_column:
            data_df_train[column] = data_df_train[column].map(lambda x: 1 if x == 'Yes' else 0)
            continue
        lbe = LabelEncoder()
        data_df_train[column] = lbe.fit_transform(data_df_train[column])
        data_df_test[column] = lbe.transform(data_df_test[column])


to_numeric(train_data, test_data, obj_rate_dict, label_column="Attrition")
train_data.head(5)


# 7、数据不做深度剖析，这里只做到这里，然后分一下训练集和验证集
def get_train_valid_data(features, labels, test_size=0.2, random_state=666):
    x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size=test_size,
                                                          random_state=random_state)
    return x_train, x_valid, y_train, y_valid


label_column = "Attrition"
x_train, x_valid, y_train, y_valid = get_train_valid_data(train_data.drop(label_column, axis=1),
                                                          train_data[label_column])

# 分别对训练集，验证集，以及测试集进行一式五份的拷贝，因为我们下面要用五个模型分别跑一下
x_train_gbdt, x_train_xgb, x_train_lgb, x_train_catb, x_train_ngb = \
    x_train.copy(), x_train.copy(), x_train.copy(), x_train.copy(), x_train.copy()
x_valid_gbdt, x_valid_xgb, x_valid_lgb, x_valid_catb, x_valid_ngb = \
    x_valid.copy(), x_valid.copy(), x_valid.copy(), x_valid.copy(), x_valid.copy()
y_train_gbdt, y_train_xgb, y_train_lgb, y_train_catb, y_train_ngb = \
    y_train.copy(), y_train.copy(), y_train.copy(), y_train.copy(), y_train.copy()
y_valid_gbdt, y_valid_xgb, y_valid_lgb, y_valid_catb, y_valid_ngb = \
    y_valid.copy(), y_valid.copy(), y_valid.copy(), y_valid.copy(), y_valid.copy()

test_gbdt, test_xgb, test_lgb, test_catb, test_ngb = \
    test_data.copy(), test_data.copy(), test_data.copy(), test_data.copy(), test_data.copy()

# 准备好打分函数，这里我们只是统计准确率
from sklearn.metrics import accuracy_score


def get_scores(y_true, y_predict):
    accuracy = '%.4f' % accuracy_score(y_true, y_predict)
    return accuracy


# GBDT
from sklearn.ensemble import GradientBoostingRegressor  # 这里为了能获得更好的AUC，使用回归模型

start_time = time.time()
model = GradientBoostingRegressor(random_state=10)
model.fit(x_train_gbdt, y_train_gbdt)
y_valid_predict_gbdt = model.predict(x_valid_gbdt)
y_valid_predict_gbdt = pd.Series(y_valid_predict_gbdt)
y_valid_predict_gbdt = y_valid_predict_gbdt.map(lambda x: 1 if x >= 0.5 else 0)

# 验证集看一下得分效果
print("auc: {}".format(get_scores(y_valid_gbdt, y_valid_predict_gbdt)))

predict = model.predict(test_gbdt)

test_gbdt[label_column] = predict
# 转化为二分类输出
test_gbdt[label_column] = test_gbdt[label_column].map(lambda x: 1 if x >= 0.5 else 0)
test_gbdt[label_column].to_csv('submit_gbdt.csv')
print("total costs {} seconds".format(time.time() - start_time))

# XGBoost
import xgboost as xgb

start_time = time.time()
x_train_xgb = xgb.DMatrix(x_train_xgb, label=y_train_xgb)
x_valid_xgb = xgb.DMatrix(x_valid_xgb, label=y_valid_xgb)
test_xgb_transformed = xgb.DMatrix(test_xgb)

param = {'boosting_type': 'gbdt',
         'objective': 'binary:logistic',
         'eval_metric': 'auc',
         'eta': 0.01,
         'max_depth': 15,
         'colsample_bytree': 0.8,
         'subsample': 0.9,
         'subsample_freq': 8,
         'alpha': 0.6,
         'lambda': 0,
         }
model = xgb.train(param, x_train_xgb, evals=[(x_train_xgb, 'train'), (x_valid_xgb, 'valid')], num_boost_round=10000,
                  early_stopping_rounds=200, verbose_eval=100)
y_valid_predict_xgb = model.predict(x_valid_xgb)
y_valid_predict_xgb = pd.Series(y_valid_predict_xgb)
y_valid_predict_xgb = y_valid_predict_xgb.map(lambda x: 1 if x >= 0.5 else 0)

# 验证集看一下得分效果
print("auc: {}".format(get_scores(y_valid_xgb, y_valid_predict_xgb)))

predict = model.predict(test_xgb_transformed)
test_xgb[label_column] = predict
test_xgb[label_column] = test_xgb[label_column].map(lambda x: 1 if x >= 0.5 else 0)
test_xgb[label_column].to_csv('submit_xgb.csv')
print("total costs {} seconds".format(time.time() - start_time))

# lightGBM
import lightgbm as lgb

start_time = time.time()
x_train_lgb = lgb.Dataset(x_train_lgb, label=y_train_lgb)
x_valid_lgb_transformed = lgb.Dataset(x_valid_lgb, label=y_valid_lgb)

param = {'boosting_type': 'gbdt',
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.01,
         'max_depth': 15,
         'feature_fraction': 0.8,
         'bagging_fraction': 0.9,
         'bagging_freq': 8,
         'lambda_l1': 0.6,
         'lambda_l2': 0,
         # 'scale_pos_weight':k,
         # 'is_unbalance':True
         }
category = list(obj_rate_dict.keys())
category.remove(label_column)

model = lgb.train(param, x_train_lgb, valid_sets=[x_train_lgb, x_valid_lgb_transformed], num_boost_round=10000,
                  early_stopping_rounds=200, verbose_eval=200, categorical_feature=category)

y_valid_predict_lgb = model.predict(x_valid_lgb)
y_valid_predict_lgb = pd.Series(y_valid_predict_lgb)
y_valid_predict_lgb = y_valid_predict_lgb.map(lambda x: 1 if x >= 0.5 else 0)

# 验证集看一下得分效果
print("auc: {}".format(get_scores(y_valid_lgb, y_valid_predict_lgb)))

predict = model.predict(test_lgb)
test_lgb[label_column] = predict
test_lgb[label_column] = test_lgb[label_column].map(lambda x: 1 if x >= 0.5 else 0)
test_lgb[label_column].to_csv('submit_lgb.csv')
print("total costs {} seconds".format(time.time() - start_time))

# catBoost
import catboost as cb

model = cb.CatBoostClassifier(iterations=1000,
                              depth=7,
                              learning_rate=0.01,
                              loss_function='Logloss',
                              eval_metric='AUC',
                              logging_level='Verbose',
                              metric_period=200
                              )
# 得到分类特征的列号
categorical_features_indices = []
for i in range(len(x_train_catb.columns)):
    if x_train_catb.columns.values[i] in category:
        categorical_features_indices.append(i)
print(categorical_features_indices)

model.fit(x_train_catb, y_train_catb, eval_set=(x_valid_catb, y_valid_catb), cat_features=categorical_features_indices)

y_valid_predict_catb = model.predict(x_valid_catb)
y_valid_predict_catb = pd.Series(y_valid_predict_catb)
y_valid_predict_catb = y_valid_predict_catb.map(lambda x: 1 if x >= 0.5 else 0)

# 验证集看一下得分效果
print("auc: {}".format(get_scores(y_valid_catb, y_valid_predict_catb)))

predict = model.predict(test_catb)
test_catb[label_column] = predict
test_catb[label_column] = test_catb[label_column].map(lambda x: 1 if x >= 0.5 else 0)
test_catb[label_column].to_csv('submit_catb.csv')
print("total costs {} seconds".format(time.time() - start_time))

# ngBoost
import ngboost as ng

model = ng.NGBClassifier(n_estimators=1000,
                         learning_rate=0.01,
                         verbose=True,
                         verbose_eval=200
                         )

model.fit(x_train_ngb, y_train_ngb)

y_valid_predict_ngb = model.predict_proba(x_valid_ngb)[:, 1]
y_valid_predict_ngb = pd.Series(y_valid_predict_ngb)
y_valid_predict_ngb = y_valid_predict_ngb.map(lambda x: 1 if x >= 0.5 else 0)

# 验证集看一下得分效果
print("auc: {}".format(get_scores(y_valid_ngb, y_valid_predict_ngb)))

# predict = model.predict(test_ngb)
predict = model.predict_proba(test_ngb)[:, 1]  # 这里可以直接获取概率值，我们取其为1对应的概率
test_ngb[label_column] = predict
test_ngb[label_column] = test_ngb[label_column].map(lambda x: 1 if x >= 0.5 else 0)
test_ngb[label_column].to_csv('submit_ngb.csv')
print("total costs {} seconds".format(time.time() - start_time))
