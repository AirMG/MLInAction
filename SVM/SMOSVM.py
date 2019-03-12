import matplotlib.pyplot as plt
import numpy as np

'''
函数
读取数据
参数-fileName-文件名
returns-dataMat-数据矩阵,labelMat-数据标签
'''


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    # 逐行读取
    for line in fr.readlines():
        # 删除空格等
        lineArr = line.strip().split('\t')
        # 添加数据
        # 原数据中,line0,line1是值
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        # 添加标签
        # 原数据中,line2是类别标号
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


'''
函数
数据可视化
参数-dataMat-数据矩阵,labelMat-数据标签
'''


def showDataSet(dataMat, labelMat):
    # 正样本
    data_plus = []
    # 负样本
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    # 转换为numpy矩阵
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    # 绘制散点图
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])
    plt.show()


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('testSet.txt')
    showDataSet(dataMat, labelMat)
