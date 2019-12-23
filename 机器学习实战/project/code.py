from cv2 import cv2
from PIL import Image
import numpy as np
import random 
import os
import operator 

# 创建一个垃圾分类的字典
rubbishDict = {'dry':1, 'harmful':2, 'recycle':3, 'residual':4}
# 创建一个存放源数据的数组
data = []
testData = []
# 存放对应标签的数组
label = []
# 测试集标签
testLabel = [1,1,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,1,1,1]
# 图片总文件夹
imageFile = 'E:/python/Project/rubbish/image'
# 对各个子文件夹进行读取
for i in rubbishDict.keys():
    # 图片路径
    imageListDir = imageFile + '/' + i
    # 获取该路径下的文件名目录
    imageList = os.listdir(imageListDir)
    # 对每一张图片进行操作
    # 图片统一尺寸
    row = 100
    col = 100
    for j in imageList:
        # 图片路径
        imageDir = imageListDir + '/' + j
        # 读取图片
        src = cv2.imread(imageDir)
        # 将array类型的输入转换成RGB格式的图片
        arrayToImage = Image.fromarray(src, 'RGB')
        # 将输入数组统一尺寸
        sizeArray = arrayToImage.resize((row, col))
        data.append(np.array(sizeArray))
        label.append(rubbishDict[i])
# 生成训练数据集
x_train = np.array(data)
y_train = np.array(label)
# 测试集图像
testImageDir = "E:/python/Project/rubbish/image/test"
testImageList = os.listdir(testImageDir)
for i in testImageList:
    testImage = testImageDir + '/' + i
    testSrc = cv2.imread(testImage)
    # 将array类型的输入转换成RGB格式的图片
    arrayToImage = Image.fromarray(src, 'RGB')
    # 将输入数组统一尺寸
    testSizeArray = arrayToImage.resize((row, col))
    testData.append(np.array(sizeArray))
# 将测试集数据转换成矩阵数组
testArray = np.array(testData)
# 一幅图像含有的数据量
numData = row*col*3
# 将训练数据转换成一维数组
# 获取训练集数量
trainRow = x_train.shape[0]
listTrain = np.zeros((trainRow, numData))
for i in range(trainRow):
    listTrain[i] = x_train[i].reshape(1,numData)
# 将测试数据换成一维数组
# 测试集数量
testRow = testArray.shape[0]
listTest = np.zeros((testRow, numData))
for j in range(testRow):
    listTest[j] = testArray[j].reshape(1,numData)
print(listTrain)

# K-邻近算法
def classify0(inX, dataSet, labels, k):
    # 距离计算
    dataSetSize = dataSet.shape[0]  # 查看输入数组的数量
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet  # 计算输入与训练集的差分
    sqDiffMat = diffMat**2  #  计算平方
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5  # 计算根方
    sortedDistIndicies = distances.argsort()  #  按行从小到大返回下标
    # 选择距离最小的k个点
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # 排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

errorCount = 0.0
for i in range(testRow):
    classifierResult = classify0(listTest[i],listTrain,y_train,2)
    print("the classifier came back with:%d,the real answer is:%d"%(classifierResult,testLabel[i]))
    if(classifierResult != testLabel[i]):errorCount += 1.0
print("the total number of errors is : %d"%errorCount)
print("the total error rate is : %f"% (errorCount/float(testRow)))