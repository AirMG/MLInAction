import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import operator


'''
函数
KNN算法,分类器
参数:inX-还未分类的数据(测试集),dataSet-用于训练的数据(训练集),labels-分类标签,k-KNN算法参数
returns:sortResult[0][0]-分类结果
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


'''
函数
打开并解析文件,对数据进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
参数:filename,文件名
returns:returnMat,特征矩阵,classLabelVector-分类Label向量
'''


def file2mat(filename):
    fn = open(filename)
    arrayLines = fn.readlines()
    numberOfLines = len(arrayLines)
    # 返回的numpy矩阵,解析完成的数据:numberOfLines行,3列(取决于属性的个数)
    returnMat = np.zeros((numberOfLines, 3))
    # 返回的分类标签向量
    classLabelVector = []
    # 行的索引值
    index = 0
    for line in arrayLines:
        #s.strip(rm),当rm空时,默认删除空白符('\n','\r','\t',' ')
        line = line.strip()
        # s.split(str="",num=string.cout(str))将字符串根据分隔符进行num次切片.默认num=-1,分割所有
        listFromLine = line.split('\t')
        # 提取数据的前三列,存到returnMat中
        returnMat[index, :] = listFromLine[0:3]
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector


'''
函数
数据可视化
参数:datingDataMat-约会数据,datingLabels-分类Label
'''


def showdatas(datingDataMat, datingLabels):
     # 设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False,
                            sharey=False, figsize=(13, 8))

    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:, 0], y=datingDataMat[:,
                                                             1], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(
        u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(
        u'每年获得的飞行常客里程数', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(
        u'玩视频游戏所消耗时间占', FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:,
                                                             2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(
        u'每年获得的飞行常客里程数与每周消费的冰激淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(
        u'每年获得的飞行常客里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(
        u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:,
                                                             2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(
        u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(
        u'玩视频游戏所消耗时间占比', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(
        u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    # 设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                              markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=6, label='largeDoses')
    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # 显示图片
    plt.show()


'''
函数
对数据归一化
参数:dataSet-特征矩阵
returns:normDataSet-归一化之后的特征矩阵,ranges-数据范围,minVals-数据最小值
'''


def autoNorm(dataSet):
    # 获得数据的最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 最大值和最小值的范围
    ranges = maxVals - minVals
    # shape(dataSet)返回dataSet矩阵行列数
    normDataSet = np.zeros(np.shape(dataSet))
    # 返回DataSet的行数
    m = dataSet.shape[0]
    # 原始值减去最小值
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    # 除以最大值和最小值的差,得到归一化数据
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    # 返回归一化数据结果,数据范围,最小值
    return normDataSet, ranges, minVals


'''
函数
分类器测试函数
returns:normDataSet-归一化之后的特征矩阵,ranges-数据范围,minVals-数据最小值
'''


def datingClassTest():
    filename = r"G:\Code\ML\算法\KNN\datingTestSet.txt"
    datingDataMat, datingLabels = file2mat(filename)

    # 抽取数据的比例
    hoRatio = 0.1

    # 所有数据归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 获得normMat的行数
    m = normMat.shape[0]

    # 百分之十的测试数据的个数
    numTestVecs = int(m*hoRatio)
    # 分类错误记数
    errorCount = 0.0

    for i in range(numTestVecs):
        # numTextVecs个数据作为测试机,后m-numTestVecs个数据作为训练集
        classifierResult = classify(
            normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 4)
        print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))

        # 分类错误
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率:%f%%" % (errorCount/float(numTestVecs)*100))


'''
函数
算法的使用
输入一个人的三维特征,进行分类
'''


def classifyPerson():
    # 输出结果
    resultList = ['讨厌', '有些喜欢', '非常喜欢']
    # 三维特征用户输入
    precentTats = float(input("玩视频游戏所耗时间百分比:"))
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    # 打开的文件名
    filename = "datingTestSet.txt"
    # 打开并处理数据
    datingDataMat, datingLabels = file2mat(filename)
    # 训练集归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 生成NumPy数组,测试集
    inArr = np.array([ffMiles, precentTats, iceCream])
    # 测试集归一化
    norminArr = (inArr - minVals) / ranges
    # 返回分类结果
    classifierResult = classify(norminArr, normMat, datingLabels, 3)
    # 打印结果
    print("你可能%s这个人" % (resultList[classifierResult-1]))


if __name__ == '__main__':
    # filename = 'G:/Code/ML/算法/KNN/datingTestSet.txt'
    # datingDataMat, datingLabels = file2mat(filename)
    # print(datingDataMat)
    # print(datingLabels)
    #showdatas(datingDataMat, datingLabels)
    # normDataSet, ranges, minVals = autoNorm(datingDataMat)
    # print(normDataSet)
    # print(ranges)
    # print(minVals)
    # datingClassTest()
    classifyPerson()
