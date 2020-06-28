import pandas as pd
from pandas.api.types import is_string_dtype, is_float_dtype
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from tpot import TPOTClassifier

# ####################################  数据加载和探索
# 1、加载数据
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

# 2、数据探索
print(train_data.info())  # 直观看出每列数据的条数及类型，可以先简单确认数据缺失情况


# print("=" * 10)
# print(train_data.describe())  # 查看所有连续型特征的基本信息，包括总条数，均值，标准差，最大值，最小值和四分位数
# print("=" * 10)
# print(train_data.describe(include=["O"]))  # 查看所有离散型变量的基本信息，包括总条数，取值个数，最多取值，以及最多取值个数
# print("=" * 10)
# print(train_data.head())  # 查看前五条数据，感觉一下
# print("=" * 10)
# print(train_data.tail())  # 查看后五条数据，感觉一下
# print("=" * 10)
# print(train_data.columns) # 查看一下列名
# print("=" * 10)
# print(train_data.select_dtypes(['float']).columns) # 查看一下float类型的列有哪些
# print("=" * 10)
# print(train_data.select_dtypes(['object']).columns.values) # 查看一下object类型的列有哪些，这些列往往是必须数值化的列
# print("=" * 10)
# print(train_data.select_dtypes(['int64']).columns) # 查看一下int64类型的列有哪些
# print("=" * 10)

# ####################################  数据清洗
# 1、分析数据缺失量
def get_na_count(data_df):
    column_list = data_df.columns
    na_dict = dict()
    for column in column_list:
        na_dict.update({column: data_df[column].isna().sum()})
    return na_dict


na_count_dict = get_na_count(train_data)
print(na_count_dict)
print(get_na_count(test_data))


# 2、根据缺失量情况进行数据填充
# 整行缺失数据进行删除，重复行数据删除，年龄不合理数据删除。
# 部分缺失数据按类型填充，这里对连续型特征用均值填充，离散型特征值用众数填充
def judge_variable_type(data_df):
    # 这里我们通过特征变量类型来区分出连续型和离散型。主要是为了解决object类离散型变量。
    # 也可以通过对特征取值个数做一个阈值限定，小于多少个，则认为是离散型，否则则认为是连续型。
    column_list = data_df.columns
    column_type_dict = dict()
    for column in column_list:
        if is_string_dtype(data_df[column]):
            column_type_dict.update({column: "discrete"})  # object归到离散型变量
        if is_float_dtype(data_df[column]):
            column_type_dict.update({column: "continuous"})  # float归到连续型变量
        else:
            column_type_dict.update({column: "discrete"})  # 其他也归到离散型变量，这个看情况调整
    return column_type_dict


def fill_data(data_df, na_count_dict=None):
    if na_count_dict is None:
        na_count_dict = get_na_count(data_df)

    # 填充之前，先适当删除一些不需要的数据
    data_df.dropna(how="all", inplace=True)  # 删除全行都是空的
    data_df.drop_duplicates(keep="first", inplace=True)  # 删除重复的行，只保留第一处
    data_df.drop(data_df[(data_df["Age"] < 0) | (data_df["Age"] > 100)].index, inplace=True)  # 删除年龄中不合适的
    data_df.reset_index(drop=True, inplace=True)  # 删除了数据，重置一下索引

    column_type_dict = judge_variable_type(data_df)
    print(column_type_dict)
    for column, na_count in na_count_dict.items():
        if na_count == 0:
            continue
        if column_type_dict[column] == "continuous":
            target_num = data_df[column].mean()  # 获得均值
        else:
            target_num = data_df[column].value_counts().head(1).index[0]  # 获得众数
        data_df[column].fillna(target_num, inplace=True)


fill_data(train_data)
fill_data(test_data)
print(get_na_count(train_data))
print(get_na_count(test_data))


# 3、看看哪些数据要做数值化处理，同时找出信息量为0或信息量太大的特征列，进行删除操作
def get_obj_column(data_df):
    obj_column_list = list()
    useless_columns = list()
    data_length = data_df.shape[0]
    column_list = data_df.columns
    for column in column_list:
        unique_count = len(data_df[column].unique())
        if unique_count == 1 or unique_count == data_length:
            useless_columns.append(column)  # 找到明显无效的特征列
            continue
        if isinstance(data_df[column][0], str):
            # obj_column_dict.update({column:len(data_df[column].unique())})
            obj_column_list.append(column)
    return obj_column_list, useless_columns


obj_column_list, useless_columns = get_obj_column(train_data)
print("obj_column_list: {}".format(obj_column_list))
print("useless_columns: {}".format(useless_columns))


def remove_useless_data(data_df_train, data_df_test, useless_columns=None):
    if useless_columns is None:
        _, useless_columns = get_obj_column(data_df_train)
    for column in useless_columns:
        data_df_train.drop(column, axis=1, inplace=True)
        data_df_test.drop(column, axis=1, inplace=True)


remove_useless_data(train_data, test_data, useless_columns)
print("useless column already removed")


# 4、对字符串特征进行数值化，这里提供两种方式：OneHotEncoder和DictVectorizer，但我们使用DictVectorizer
# 离散无序变量，变量取值为整数：OneHotEncoder
# 当想要看到特征对应某个取值的重要性时，适合使用OneHot。
# 当特征数过多时，树模型不宜使用OneHot，防止树过深过拟合，参考LightGBM
# 离散无序变量，变量取值为字符串：DictVectorizer
# 能自动处理新出现的特征取值，会默认归置为0。
def dict_encoder(data_df_train, data_df_test, obj_column_list=None):
    if not obj_column_list:
        obj_column_list, _ = get_obj_column(data_df_train)
    print("obj_column_list: ", obj_column_list)

    data_obj_train = data_df_train[obj_column_list]
    data_obj_test = data_df_test[obj_column_list]

    encoder = DictVectorizer(sparse=False)
    obj_train_encode = encoder.fit_transform(data_obj_train.to_dict(orient='record'))
    obj_test_encode = encoder.transform(data_obj_test.to_dict(orient='record'))

    data_df_train = pd.concat([data_df_train, pd.DataFrame(obj_train_encode)], axis=1).drop(obj_column_list, axis=1)
    data_df_test = pd.concat([data_df_test, pd.DataFrame(obj_test_encode)], axis=1).drop(obj_column_list, axis=1)
    print("obj encode finished")

    return data_df_train, data_df_test


def one_hot_encoder(data_df_train, data_df_test, obj_column_list=None, concat_first=True):
    if not obj_column_list:
        obj_column_list, _ = get_obj_column(data_df_train)

    print("obj_column_list: ", obj_column_list)

    data_obj_train = data_df_train[obj_column_list]
    data_obj_test = data_df_test[obj_column_list]

    if concat_first:  # 将训练集和测试集进行拼接，来解决测试集中出现新特征的问题，适用于小数据量
        len_train = data_df_train.shape[0]  # 先记录训练集数量，以备后续还原
        data_df_obj = pd.concat([data_obj_train, data_obj_test], join="inner").reset_index(drop=True)

        encoder = OneHotEncoder(sparse=False)
        total_result = encoder.fit_transform(data_df_obj)

        total_result = pd.DataFrame(total_result)
        obj_train_encode = total_result.iloc[:len_train].reset_index(drop=True)
        obj_test_encode = total_result.iloc[len_train:].reset_index(drop=True)
    else:  # 直接对新特征进行忽略，减少内存开销
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        obj_train_encode = encoder.fit_transform(data_obj_train)
        obj_test_encode = encoder.transform(data_obj_test)

    data_df_train = pd.concat([data_df_train, obj_train_encode], axis=1).drop(obj_column_list, axis=1)
    data_df_test = pd.concat([data_df_test, obj_test_encode], axis=1).drop(obj_column_list, axis=1)
    print("obj encode finished")

    return data_df_train, data_df_test


data_df_train_encode, data_df_test_encode = dict_encoder(train_data, test_data)
# data_df_train_encode, data_df_test_encode = one_hot_encoder(train_data, test_data)


# 5、对年龄进行简单分箱操作
# 示例：pd.cut(data_aa["Age"],[0,20,30,40,70,100],labels=["少年","青年","中年","老年","暮年"],right=False)
data_df_train_encode["Age"] = pd.Series(
    pd.cut(data_df_train_encode["Age"], [0, 20, 30, 40, 70, 100], labels=False, right=False))
data_df_test_encode["Age"] = pd.Series(
    pd.cut(data_df_test_encode["Age"], [0, 20, 30, 40, 70, 100], labels=False, right=False))

# 6、因为还存在类如Fare这种取值过大的特征列，故而做个去量纲操作，比如标准化、归一化、区间放缩法或者小数定标规范化
# 这里选择区间放缩，即x=(x-x.min)/(x.max-x.min)。目的是为了减少数值太大导致的易过拟合情况
min_max_scaler = MinMaxScaler()

data_df_train_encode["Fare"] = min_max_scaler.fit_transform(data_df_train_encode[["Fare"]])  # 可以考虑只对某些列进行放缩
data_df_test_encode["Fare"] = min_max_scaler.fit_transform(data_df_test_encode[["Fare"]])


# label_column = "Survived"
# data_df_train_encode = min_max_scaler.fit_transform(data_df_train_encode.drop(label_column,axis=1))  # 也可以考虑对全部放缩
# data_df_test_encode = min_max_scaler.transform(data_df_test_encode)

# #################################### 上模型
# 1、定义得分函数，分类问题四个常见指标可以简单进行计算
# 同时也可以配置KS值与KS曲线来辅助一同进行结果阐述
def get_scores(y_true, y_predict):
    label_num = max(len(set(y_predict)), len(set(y_true)))
    average = "macro" if label_num > 2 else "binary"  # 二分类用binary， 多分类用macro
    precise = '%.4f' % precision_score(y_true, y_predict, average=average)
    accuracy = '%.4f' % accuracy_score(y_true, y_predict)
    recall = '%.4f' % recall_score(y_true, y_predict, average=average)
    f1 = '%.4f' % f1_score(y_true, y_predict, average=average)
    return [precise, accuracy, recall, f1]


# 2、进行数据拆分
label_column = "Survived"

features, labels = data_df_train_encode.drop(label_column, axis=1), data_df_train_encode[label_column]
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=0, stratify=labels)
print("data split finished")

# 3.1、简简单单使用逻辑回归
logistic_ins = LogisticRegression(random_state=0, solver="sag", multi_class="multinomial")
logistic_ins.fit(x_train, y_train)
y_predict_3 = logistic_ins.predict(x_test)
print(get_scores(y_test, y_predict_3))

result = pd.DataFrame(logistic_ins.predict(data_df_test_encode))
result.to_csv('lr_titanic.csv')

# 3.2、使用CART决策树并通过网格搜索的方式进行调参
cart_ins = DecisionTreeClassifier(criterion="gini", random_state=0)
cart_ins.fit(x_train, y_train)
y_predict_1 = cart_ins.predict(x_test)
print(get_scores(y_test, y_predict_1))

# 使用网格搜索看看最佳参数设置
cart_params = [{"max_depth": [None, 2, 5, 10],
                "min_samples_split": [2, 3, 5],
                "min_samples_leaf": [1, 2, 3]}]

grid_ins = GridSearchCV(cart_ins, cart_params, cv=5)
grid_ins.fit(x_train, y_train)
print("best param and best score are: {}, {}".format(grid_ins.best_params_, grid_ins.best_score_))

y_predict_2 = grid_ins.predict(x_test)
print(get_scores(y_test, y_predict_2))

result = pd.DataFrame(grid_ins.predict(data_df_test_encode))
result.to_csv('cart_grid_search_titanic.csv')

# 3.3、必杀器：使用自动学习来寻找最厉害的模型及其超参
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
tpot.fit(x_train, y_train)
print(tpot.score(x_test, y_test))
tpot.export('titanic_tpot_pipeline.py')  # 导出模型以备下次直接使用
result = pd.DataFrame(tpot.predict(data_df_test_encode))
result.to_csv("automl_titanic.csv")
