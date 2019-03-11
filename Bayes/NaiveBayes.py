import numpy as np

'''
函数
创建样本
returns-postingList-实验样本切分的词条
classVec-类别标签向量
'''


def loadDataSet():
    # 切分的词条
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                                     ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak',
                       'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 词条的分类,0代表非侮辱,1代表侮辱
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


'''
函数
根据词汇表,将inputSet向量化,向量的每个元素为1或0
参数-vocabList-createVocabList返回的列表,inputSet-切分的词条列表
returns-returnVec-文档向量,词集模型
'''


def setOfWords2Vec(vocabList, inputSet):
    # 创建一个其中所含元素都为0的向量
    returnVec = [0] * len(vocabList)
    # 遍历每个词条
    for word in inputSet:
        if word in vocabList:
            # 如果词条存在于词汇表中,则置1
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word:%s is not in my Vocabulary" % word)
    return returnVec


'''
函数
将切分的样本整理为不重复的词条列表,即词汇表
参数-dataSet-整理的样本数据集
returns-VocabSet-返回不重复的词条列表,即词汇表
'''


def createVocabList(dataSet):
    # 创建空的不重复列表
    vocabSet = set([])
    # 除去重复的单词,放到词汇表中
    for document in dataSet:
        # 取并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


'''
函数
朴素贝叶斯分类器训练函数
参数-trainMatrix-训练文档矩阵,即SetOfWords2Vec返回的returnVec构成的矩阵
trainCategory-训练类别标签向量,即loadDatSet返回的classVec
returns-p0Vect-非侮辱类的条件概率数组,p1Vect-侮辱类条件概率数组
pAbusive-文档属于侮辱类的概率
'''


def trainNB(trainMatrix, trainCategory):
    # 计算训练的数据条目数量
    numTrainDocs = len(trainMatrix)
    # 计算每个条目的词条数
    numWords = len(trainMatrix[0])
    # 文档属于侮辱类的概率
    # 统计分类标签数组的和(即是其中1的数目),计算侮辱类在所有词条中的占比
    pAbusive = sum(trainCategory) / float(numTrainDocs)

    '''
    此处将数组初始化为1数组,分母初始化为2,避免了一个分词为0,整个词条属于某个类别的概率也为0的问题,这种方法是拉普拉斯平滑
    为了解决计算结果下溢出为0的情况,对结果取自然对数
    '''
    # 创建numpy.zeros数组,词条出现数初始化为0
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    # 分母初始化为0
    p0Denom = 2.0
    p1Denom = 2.0
    # 统计属于侮辱类的条件概率所需要的数据,即P(w0|1),P(w1|1)....
    for i in range(numTrainDocs):
        # if成立表明该词条的分类为侮辱类
        if trainCategory[i] == 1:
            # 将侮辱性词条相加
            p1Num += trainMatrix[i]
            # print(trainMatrix[i])
            # print('p1Num:\n', p1Num)
            # 统计词条中出现的词汇的数目
            p1Denom += sum(trainMatrix[i])
            # print('p1Denom:\n', p1Denom)
        else:
            # 统计属于非侮辱类的条件概率所需的数据,即P(w0|0),P(w1|0)...
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # print(p1Denom) #取对数，防止下溢出
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive


'''
函数
朴素贝叶斯分类函数
参数-vec2Classify-待分类的词条数组,p0Vec-非侮辱类的条件概率数组,p1Vec-侮辱类的条件概率数组,pClass1-文档属于侮辱类的概率
returns-0-属于非侮辱类,1-属于侮辱类
'''


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 自然对数,logab=loga+logb
    p1 = sum(vec2Classify*p1Vec)+np.log(pClass1)
    p0 = sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    postingList, classVec = loadDataSet()
    # print('postingList:\n', postingList)
    vocabList = createVocabList(postingList)
    # print('VocabList:\n', vocabList)
    trainMat = []
    for postinDoc in postingList:
        trainMat.append(setOfWords2Vec(vocabList, postinDoc))
    print('trainMat:\n', trainMat)
    # p0V,每个单词属于类别0(非侮辱词汇)的概率
    # p1V,每个单词属于类别1(侮辱词汇)的概率
    # pAv,侮辱类词条占所有样本的概率
    p0V, p1V, pAb = trainNB(trainMat, classVec)
    print('p0V:\n', p0V)
    print('p1V:\n', p1V)
    print('classVec:\n', classVec)
    print('pAv:\n', pAb)
