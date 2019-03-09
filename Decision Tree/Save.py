import pickle

'''
函数
存储决策树
参数-inputTree-已经生成的决策树,filename-决策树的存储文件名
'''


def saveTree(inputTree, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)


if __name__ == '__main__':
    Tree = {'有自己的房子': {0: {'有工作': {0: 'no', 1: 'yes'}}, 1: 'yes'}}
    saveTree(Tree, 'classfierSave.txt')
