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
if __name__ == '__main__':
    docList = []
    classList = []
    for i in range(1, 26):
        # 读取每个垃圾邮件文本,将字符串转换为字符列表
        wordList = textPrase(
            open('G:/Code/ML/Algorithm/Bayes/email/spam/%d.txt' % i, 'r').read())
        docList.append(wordList)
        # 垃圾邮件的类别标记为1
        classList.append(1)
        # 读取非垃圾邮件文本,将字符串转换为字符列表
        wordList = textPrase(
            open('G:/Code/ML/Algorithm/Bayes/email/ham/%d.txt' % i, 'r').read())
        docList.append(wordList)
        # 非垃圾邮件的类别标记为0
        classList.append(0)
    # 创建从垃圾邮件和非垃圾邮件中的得到的词汇表的去重表
    vocabList = createVocabList(docList)
    print(vocabList)
