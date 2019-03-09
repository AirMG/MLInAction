from sklearn import tree
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pydotplus
from sklearn.externals.six import StringIO


'''
对字符数据进行序列化
'''


def serialization(data):
    # 将数据集中的数据按行读取,并且转换为数组形式
    lenses = [line.strip().split('\t') for line in data.readlines()]
    # print(lenses[0])
    # 提取每组数据的类别
    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1])
    # 特征标签
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # 存储每一个属性的值的数组
    lenses_list = []
    # 保存lenses数据的字典
    lenses_dict = {}
    # 对每一个属性
    for each_label in lensesLabels:
        # 对数据中的每一行(一组数据)
        for each in lenses:
            # 存储lenses的数据
            lenses_list.append(each[lensesLabels.index(each_label)])
        # 将属性和每个对象该属性的值存到字典中
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    # print(lenses_dict)
    # 使用pandas的数据形式表示数据
    lenses_pd = pd.DataFrame(lenses_dict)
    # 用于序列化的encoder()对象
    le = LabelEncoder()
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    return lenses_pd, lenses_target


if __name__ == '__main__':
    data = open('lenses.txt')
    serializedData, serializedTarget = serialization(data)
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf = clf.fit(serializedData.values.tolist(), serializedTarget)
    print(clf.predict([[1, 1, 1, 0]]))
