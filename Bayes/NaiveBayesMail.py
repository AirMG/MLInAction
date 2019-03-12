import re
import numpy as np
import random

'''
函数
接受一个大字符串,将其解析为字符串列表
'''

# 将字符串转换为字符列表


def textPrase(bigString):
    # 将非字母,非数字的特殊符号作为切分词汇的标志
    listOfTokens = re.split(r'\W+', bigString)
    # 除了单个的字母以外,其他词汇全部变成小写
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


'''
函数
将切分的样本的词条整理成不重复的词条列表,即词汇表
参数-dataSet-整理的样本数据集
returns-vocabSet-返回不重复的词条列表,即词汇表
'''


def createVocabList(dataSet):
    # 设置词汇表为一个集合
    vocabSet = set([])
    # 对传入的分割好的词汇表,进行求并集的操作,除去其中的重复词汇
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


'''
函数
根据vocabList词汇表,将inputSet向量化,向量的每个元素为1或者0
参数-vocabList-createVocabList返回的列表,inputSet-切分的词条列表
returns-returnVec-文档向量,词集模型
'''


def setOfWords2Vec(vocabList, inputSet):
    # 设置返回向量为空,长度和词汇表相同
    returnVec = [0] * len(vocabList)
    # 遍历词条集中的词条
    for word in inputSet:
        # 如果词条集中的词在词汇表中
        if word in vocabList:
            # 将该
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in vocabULARY" % word)
        # 返回对应的文档向量
        return returnVec


'''
函数
根据vocabList词汇表,创建词袋模型
参数-vocabList-createVocabList返回的列表,inputSet-切分的词条列表
returns-returnVec-文档向量,词袋模型
'''


def bagOfWords2VecMn(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    # 遍历每一个词条
    for word in inputSet:
        # 如果词条存在在词汇表中
        if word in vocabList:
            # 计数加1
            returnVec[vocabList.index(word)] += 1
    return returnVec


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


'''
测试朴素贝叶斯分类器
'''


def spamTest():
    docList = []
    classList = []
    fullText = []
    # 标记25个txt文件
    for i in range(1, 26):
        # 读取每个垃圾邮件,1表示垃圾邮件
        wordList = textPrase(
            open('G:/CODE/ML/Algorithm/Bayes/email/spam/%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        # 读取非垃圾邮件
        wordList = textPrase(
            open('G:/CODE/ML/Algorithm/Bayes/email/ham/%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    # 创建不重复的词汇表
    vocabList = createVocabList(docList)
    # 创建存储训练集的索引值
    trainingSet = list(range(50))
    testSet = []
    # 从50个邮件中,随机挑选40个作为训练集,10个做测试集
    for i in range(10):
        # 随机选取索引值
        randIndex = int(random.uniform(0, len(trainingSet)))
        # 添加测试集的索引值
        testSet.append(trainingSet[randIndex])
        # 在训练集列表中删除添加到测试集的索引值
        del (trainingSet[randIndex])
    # 创建训练集矩阵和训练集类别标号
    trainMat = []
    trainClasses = []
    # 遍历训练集
    for docIndex in trainingSet:
        # 将生成的词集模型添加到训练矩阵中
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        # 将类别添加到训练集类别标签向量中
        trainClasses.append(classList[docIndex])
    # 训练朴素贝叶斯模型
    p0V, p1V, pSpam = trainNB(np.array(trainMat), np.array(trainClasses))
    # 错误分类计数
    errorCount = 0
    # 遍历测试集
    for docIndex in testSet:
        # 测试集的词集模型
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        # 分类错误,则错误计数加1
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("分类错误集", docList[docIndex])
    print("错误率:%.2f%%" % (float(errorCount) / len(testSet) * 100))


if __name__ == '__main__':
    # docList = []
    # classList = []
    # for i in range(1, 26):
    #     # 读取每个垃圾邮件文本,将字符串转换为字符列表
    #     wordList = textPrase(
    #         open('G:/Code/ML/Algorithm/Bayes/email/spam/%d.txt' % i, 'r').read())
    #     docList.append(wordList)
    #     # 垃圾邮件的类别标记为1
    #     classList.append(1)
    #     # 读取非垃圾邮件文本,将字符串转换为字符列表
    #     wordList = textPrase(
    #         open('G:/Code/ML/Algorithm/Bayes/email/ham/%d.txt' % i, 'r').read())
    #     docList.append(wordList)
    #     # 非垃圾邮件的类别标记为0
    #     classList.append(0)
    # # 创建从垃圾邮件和非垃圾邮件中的得到的词汇表的去重表
    # vocabList = createVocabList(docList)
    # print(vocabList)
    spamTest()
