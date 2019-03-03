import numpy as np
import operator
import time

'''
函数
@描述:创建数据集
@return:group-数据集,labels-分类标签
'''


def createDataSet():
    # 给出数据的四组二维特征
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])

    # 给出四组特征的标签
    labels = ['爱情片', '爱情片', '动作片', '动作片']

    return group, labels


'''
函数
@描述:KNN算法,分类器
@参数:inX-还未分类的数据,dataSet-用于训练的数据,labels-分类标签,k-KNN算法参数
@returns:sortResult[0][0],分类结果
'''


def classify(inX, dataSet, labels, k):
    # 返回dataSet的行数
    dataSetSize = dataSet.shape[0]

    # 列向量方向上重复inX1次,行向量方向上重复inX功dataSetSize次
    # np.tile(mat,(n,m)),在横向上将mat复制m次,在纵向上将mat复制n次,构成一个矩阵

    # L2距离计算
    # 这一步实现了(x1-x2),(y1-y2)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 二维特征相减后平方
    # 这一步实现了(x1-x2)^2,(y1-y2)^2
    sqDiffMat = diffMat ** 2
    # sum()所有元素相加,sum(0)列相加,sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方计算距离
    distances = sqDistances ** 0.5
    # 返回distances中元素从小到大排序后的索引值
    # x.argsort(),x中的元素从小到大排序后,输出x中对应位置的元素的index(索引)
    # argsort(x),x默认大于0,x取负值时,输出是大于0的结果的反向的索引
    sortedDistIndices = distances.argsort()
    # print(sortedDistIndices)
    # 选择过程
    # 定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        # 获得前k个元素的类别
        voteLabel = labels[sortedDistIndices[i]]
        # print(voteLabel)
        # dict.get(key,default=None),字典的get方法,返回指定键的值,如果值不再字典中返回默认值
        # 计算类别的次数
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    # key=operator.itemgetter(1),根据字典的值进行排序
    # key=operator.itemgetter(0),根据字典的键进行排序
    # reverse降序排列字典
    # print(classCount)
    sortResult = sorted(classCount.items(),
                        key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别
    # print(sortResult)
    return sortResult[0][0]


if __name__ == '__main__':
    # 创建数据集
    group, labels = createDataSet()
    # 测试数据
    test = [101, 20]
    start = time.clock()
    # 获得KNN分类结果
    test_class = classify(test, group, labels, 3)
    end = time.clock()
    print(test_class)
    print(end - start)
