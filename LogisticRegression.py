import numpy as np
import matplotlib.pyplot as plt

#################载入数据#################
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('C:\\Users\Administrator\Desktop\机器学习实战\MLiA_SourceCode\machinelearninginaction\Ch05\\testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #向数据矩阵 dataMat 添加数据，设第 1 列全为 1
        labelMat.append(int(lineArr[2]))                            #向标签矩阵 labelMat 添加数据
    return dataMat, labelMat                                        #此文件中两个矩阵分别为：100*3，100*1

#################激励函数#################
def sigmoid(inX):                                           #记输入 inX 为 z
    z = np.longfloat(1.0/(1 + np.exp(-inX)))                #z = w0*x0 + w1*x1 + w2*x2 + ... + wn*xn
    return z

#################梯度上升法#################
def grandAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)                          #转换成 NumPy 矩阵，mat 相当于 asmatrix，100*3
    labelMat = np.mat(classLabels).transpose()              #转置矩阵，100*1
    m, n = np.shape(dataMatrix)                             #获取数据矩阵的行数 m，列数 n
    alpha = 0.001                                           #设置学习率 0.001
    maxCycles = 500                                         #设置最大循环次数为 500
    weights = np.ones((n,1))                                #初始化一个 n*1 的值全为1的权值矩阵，3*1
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)                   #将数据与权值的乘积放入激励函数，得到预测类别的值，100*1
        error = labelMat - h                                #计算真实类别与预测类别的误差，100*1
        weights += alpha * dataMatrix.transpose() * error   #修改权值，3*1
    return weights                                          #返回权值 w0，w1，w2

#################绘制最佳拟合曲线#################
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]                                            #获得数据的数量 n
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):                                                  #根据标签分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])    #标签为 1
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])    #标签为 0
    fig = plt.figure()                                                  #创建一个 figure 绘图对象 fig
    ax = fig.add_subplot(111)                                           #使用 fig 对象创建子图
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')               #绘制分散点：s 设置点的尺寸；maker 设置点的形状
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)                                       #设置拟合曲线的定义域 x
    y = (-weights[0] - weights[1] * x) / weights[2]                     #设置拟合曲线 y = (-w0 - w1*x) / w2
    ax.plot(x, y)                                                       #绘制拟合曲线，plot(x, y, 'bo') 表示用蓝色小圆来画曲线
    plt.xlabel('X1'); plt.ylabel('X2')                                  #设置 x, y 轴的标题
    plt.show()                                                          #展示图像

#################随机梯度上升法#################
def stochasticGradAscent(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)                                                #初始化权值 [1., 1., 1., ... 1.]，n 个 1
    for i in range(m):                                                  #循环 m=100 次
        h = sigmoid(sum(dataMatrix[i] * weights))                       #每组数据与权值矩阵求内积后再求和
        error = classLabels[i] - h                                      #分别计算每组数据对应的误差
        weights += alpha * error * dataMatrix[i]                        #每组数据分别与相对应的误差修改权值，修改 m 次
    return weights

#################改进的随机梯度上升法#################
def modifiedStochasticGradAscent(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):                                            #循环 numIter 次，默认 150 次
        dataIndex = list(range(m))                                      #创建一个数据的下标的列表
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01                            #动态修改学习率，使其越来越小，当 j<<max(i) 时 alpha 非严格下降
            randIndex = int(np.random.uniform(0, len(dataIndex)))       #uniform(low=0.0, high=1.0, size=None) 用于随机取得一个值，size 用于设置数量
            h = sigmoid(sum(dataMatrix[randIndex] * weights))           #随机取得一组数据来修改权值
            error = classLabels[randIndex] - h                          #分别计算每组数据对应的误差
            weights += alpha * error * dataMatrix[randIndex]            #每组数据分别与相对应的误差修改权值，修改 m 次
            del dataIndex[randIndex]                                    #删除当前数据，防止重复影响权值
    return weights

#################分类器#################
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

#################疝气病症预测函数#################
def colicTest():
    #训练集中的原始数据不完整，缺失值设置为 0 ：1.numpy 矩阵不能有缺失值 2.sigmoid(0) = 0.5，对结果无倾向性
    #测试集中若存在缺失标签的数据，则可以简单的丢弃此数据，至少在 Logistic 回归中可以这样做，其他如 kNN 可能不合适
    frTrain = open('C:\\Users\Administrator\Desktop\机器学习实战\MLiA_SourceCode\machinelearninginaction\Ch05\horseColicTraining.txt')
    frTest = open('C:\\Users\Administrator\Desktop\机器学习实战\MLiA_SourceCode\machinelearninginaction\Ch05\horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):                                         #训练集有 22 列：前 21 列为特征，第 22 列为标签
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    #使用改进的随机梯度上升法来计算回归系数向量，设置迭代次数为 1000 次
    trainWeights = modifiedStochasticGradAscent(np.array(trainingSet), trainingLabels, 1000)

    #以上训练部分完成
    #下面测试部分开始
    errorCount = 0; numTestVec = 0
    for line in frTest.readlines():
        numTestVec += 1                                             #统计测试集的样本数目
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights))\
                != int(currLine[21]):                               #统计预测类别与真实类别的错误预测数目
            errorCount += 1
    errorRate = float(errorCount) / numTestVec                      #计算错误预测率
    print('The error rate of this test is: %f%%' % (errorRate * 100))
    return errorRate

#################平均值函数#################
def multiTest():
    numTests = 10; errorSum = 0.0                                   #运行测试集 10 次
    for i in range(numTests):
        errorSum += colicTest()
    print('After %d iterations, the average error is: %f%%' % (numTests, errorSum / float(numTests) * 100))


dataArr, labelMat = loadDataSet()

weights0 = grandAscent(dataArr, labelMat)
print('梯度上升法学习所得的权值矩阵：\n', weights0)
plotBestFit(weights0)

weights1 = stochasticGradAscent(np.array(dataArr), labelMat)
print('\n随机梯度上升法学习所得的权值矩阵：\n', weights1)
plotBestFit(weights1)

weights2 = modifiedStochasticGradAscent(np.array(dataArr), labelMat, 1000)
print('\n改进的随机梯度上升法学习所得的权值矩阵：\n', weights2)
plotBestFit(weights2)

print('\n下面是从疝气病症样本数据预测病马的死亡率的情况：')
multiTest()