from math import log
import operator

'''
函数
给定数据集的熵
参数-dataset-数据集
returns-shnnonEnt-香农熵(经验熵)
'''


def clacShannonEnt(dataset):
    # 返回数据集的行数
    numEntires = len(dataset)
    # 存储类别出现次数的字典
    labelCounts = {}
    # 对每组特征向量进行统计
    for featVec in dataset:
        # 提取类别信息
        currentLabel = featVec[-1]
        # 如果类别没有被放入到字典中,则加入字典中
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # 该类别存在,则出现次数加一
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # 计算熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntires
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


'''
函数
创建测试数据集
returns=dataSet-数据集,label-特征标签
'''


def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    # 返回数据集和分类属性
    return dataSet, labels


'''
函数
按照给定特征划分数据集
参数-dataSet-待划分的数据集,axis-划分数据集的特征,value-需要返回的特征的值
'''


def splitDataSet(dataSet, axis, value):
    # 创建返回的数据集列表
    retDataSet = []
    # 遍历数据集
    for featVec in dataSet:
        if featVec[axis] == value:
            # 去掉axis特征
            reducedFeatVec = featVec[:axis]
            # 将符合条件的添加到返回的数据集
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    # 返回划分后的数据集
    return retDataSet


'''
函数
选择最优特征
参数-dataSet-数据集
returns-bestFeature-信息增益最大(最优)特征的索引值
'''


def chooseBestFeatureToSplit(dataSet):
    # 计算特征数量
    numFeatures = len(dataSet[0]) - 1
    # 计算数据集的香农熵
    baseEntropy = clacShannonEnt(dataSet)
    # 信息增益
    bestInfoGain = 0.0
    # 最优特征的索引值
    bestFeature = -1
    # 遍历所有的特征
    for i in range(numFeatures):
        # 获取dataSet的第i个所有特征
        featlist = [example[i] for example in dataSet]
        # 创建set集合{},元素不可重复
        uniqueVals = set(featlist)
        # 经验条件熵
        newEntropy = 0.0
        # 计算信息增益
        for value in uniqueVals:
            # subDataSet划分后的子集
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算子集的概率
            prob = len(subDataSet)/float(len(dataSet))
            # 根据公式计算经验条件熵
            newEntropy += prob*clacShannonEnt(subDataSet)
        # 信息增益
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            # 更新信息增益,找到最大的信息增益
            bestInfoGain = infoGain
            # 记录信息增益最大的特征的索引值
            bestFeature = i
    # 返回信息增益最大的特征的索引值
    return bestFeature


'''
函数
统计classList中出现次数最多的元素(类标签)
参数-classList-类标签列表
returns-sortedClassCount[0][0]-出现次数最多的元素(类标签)
'''


def majorityCnt(classList):
    # 用于存放最多类别的字典
    classCount = {}
    # 遍历类别集合中的类别
    for vote in classList:
        # 如果不在计数集中,属性作为键加入,值加一
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote] += 1
    # 对计数集按升序排序
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回出现次数最多的类别及其数量
    return sortedClassCount[0][0]


'''
函数
创建决策树
参数-dataSet-训练数据集,labels-分类属性标签,featLabels-存储选择的最优特征标签
returns-决策树
'''


def createTree(dataSet, labels, featLabels):
    # 取分类标签
    # 即是否还清了贷款这一项
    classList = [example[-1] for example in dataSet]
    # 如果类别完全相同则停止继续划分
    # # count对对象中的和classList[0]类别相同的属性的个数进行计数,如果和属性对象的长度一致,说明所有的数据都被分到了同一个类别当中,直接返回该类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有的特征后返回出现次数最多的类别
    if len(dataSet[0]) == 1 or len(labels) == 0:
        return majorityCnt(classList)
    # 选择最优的特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 选择最优特征的类别
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)
    # 根据最优的类别生成决策树
    # 决策树是一个字典,对应的键值对是最佳特征的类别和分支
    Tree = {bestFeatLabel: {}}
    # 删除已经使用的特征类别
    del (labels[bestFeat])
    # 得到训练集中所有最优特征的类别
    featValues = [example[bestFeat] for example in dataSet]
    # 删除重复的特征类别
    uniqueVals = set(featValues)
    # 遍历特征,创建决策树
    for value in uniqueVals:
        Tree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), labels, featLabels)
    return Tree


'''
函数
使用决策树分类
参数-inputTree-已经生成的决策树,featLabels-存储选择的最佳特征的类别,testVec-测试数据列表,顺序对应最优特征类别
returns-分类结果
'''


def classify(inputTree, featLabels, testVec):
    # 获取决策树结点
    # next()函数返回迭代器中下一个对象的值
    firstStr = next(iter(inputTree))
    # 下一个字典,即根节点之后的子树
    # 通过给出键值对中的键,获得对应的值,即是从父节点到了子结点
    secondDict = inputTree[firstStr]
    # 寻找给定数据在最优特征类别中的index
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        # 如果给出的测试数据的值和键相对应,说明满足该结点的判断条件
        if testVec[featIndex] == key:
            # 如果
            if type(secondDict[key]).__name__ == 'dict':
                # 递归到子树中
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    # 选择的最优特征类
    featLabels = []
    Tree = createTree(dataSet, labels, featLabels)
    testVec=[0,1]
    result=classify(Tree,featLabels,testVec)
    # if result == 'yes':
    #     print('放贷')
    # if result == 'no':
    #     print('不放贷')
