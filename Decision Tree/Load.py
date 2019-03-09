import pickle

'''
函数
读取决策树
参数-filename-决策树存储文件名
returns-pickle.load(fr)-存储决策树的字典
'''


def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == '__main__':
    Tree = grabTree('classifierSave.txt')
    print(Tree)
