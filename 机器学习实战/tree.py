from math import log 
import operator
import treePlotter 
# 计算香农熵
def calcShannonEnt(dataSet):
    # 计算给定数据集的大小
    numEntries = len(dataSet)
    # 定义一个标签字典
    labelsDict = {}
    # 对数据集的每行数据向量进行操作
    for featVec in dataSet:
        # 令每行数据的最后一个数据为特征数据
        featureData = featVec[-1]
        # 如果特征数据不在标签字典中，则为其创建key-value对
        if featureData not in labelsDict.keys():
            labelsDict[featureData] = 0
        # 如果在标签字典中，则统计出现次数
        labelsDict[featureData] += 1
    # 计算香农熵
    shannonEnt = 0.0
    for key in labelsDict.keys():
        prob = labelsDict[key] / float(numEntries)
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

# 按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    '''
    dataSet:待划分的数据集
    axis：划分数据集的特征
    value：需要返回的特征的值
    '''
    # 存放符合特征的数据集
    retDataSet = []
    # 去数据集的每一行数据向量
    for featVec in dataSet:
        # 提取符合给定特征的向量（不含划分点axis）
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
# 选择最好的数据集划分方式
def bestChooseData(dataSet):
    # 判断数据集具有多少特征属性
    numFeatures = len(dataSet[0]) - 1
    # 计算原始数据的香农熵，以便于与新熵值进行比较
    baseEntroy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # 遍历所有特征
    for i in range(numFeatures):
        featureList = [example[i] for example in dataSet]
        uniqueVal = set(featureList)
        newEntroy = 0.0
        # 对每一种特征值划分一次数据集，计算熵值
        for value in uniqueVal:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            # 对所有唯一特征值的熵进行求和
            newEntroy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntroy - newEntroy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature 
# 构建决策树
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
# 创建树
def createTree(dataSet, labels):
    '''
    dataSet:数据集
    labels:标签列表，包含了数据集中所有的特征标签
    '''
    # 创建一个包含数据集所有类别的变量
    classList = [example[-1] for example in dataSet]
    # 第一个终止条件：如果类别完全相同则停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 第二个终止条件：使用完所有的特征值后，仍不能将数据集划分成仅包含唯一类别的分组
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 寻找最好的数据特征划分点
    bestFeature = bestChooseData(dataSet)
    print("bestFeature:",bestFeature)
    # 返回特征划分点对应的值
    bestFeatureLabel = labels[bestFeature]
    print("bestFeatureLabel:",bestFeatureLabel)
    myTree = {bestFeatureLabel:{}}
    del labels[bestFeature]
    print("labels:",labels)
    featureValue = [example[bestFeature] for example in dataSet]
    uniqueVal = set(featureValue)
    for value in uniqueVal:
        subLabels = labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet,bestFeature,value), subLabels)
    return myTree

# 运行
if __name__ == '__main__':
    dataSet = [[1, 1, 'maybe'], [1, 1, 'yes'], [1, 1, 'yes'],
    [1, 0, 'no'], [0, 1, 'no'], [0, 0, 'no']]
    labels = ['no surfacing', 'flippers']
    splitResult = splitDataSet(dataSet, 1, 0)
    print(splitResult)
    best = bestChooseData(dataSet)
    print(best)
    mytree = createTree(dataSet,labels)
    print(mytree)
    print(mytree.keys())
    treePlotter.createPlot(mytree)