"""
K-近邻算法（KNN: K Nearest Neighbors）
一般流程：
    (1) 收集数据：可以使用任何方法
    (2) 准备数据：距离计算所需要的数值，最好是结构化的数据
    (3) 分析数据：可以使用任何方法。
    (4) 训练算法：此步骤不适用于k-近邻算法。
    (5) 测试算法：计算错误率。
    (6) 使用算法：首先需要输入样本数据和结构化的输出结果，然后运行KNN算法判定输入数据分别属于哪个分类，最后应用对计算出的分类执行后续处理
"""
from numpy import *  # numpy\n
import operator  # 运算符

def createDataSet():
    # 添加已有数据信息
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])  # 数据的坐标信息（数据集）
    labels = ['A', 'A', 'B', 'B']  # 数据点的分类标签信息
    return group, labels # 返回数据集和标签

"""
对未知类别属性的数据集中的每个点依次执行以下操作：
    (1) 计算已知类别数据集中的点与当前点之间的距离; 
    (2) 按照距离递增次序排序;
    (3) 选取与当前点距离最小的K个点; 
    (4) 确定前K个点所在类别的出现频率; 
    (5) 返回前K个点出现频率最高的类别作为当前点的预测分类。
"""
def classify0(inX, dataSet, labels, k):
    """
    K-近邻算法
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labes - 分类标签
    k - kNN算法参数,选择最近邻居的数目（距离最小的k个点）
    """
    # numpy函数shape[0]返回dataSet的行数，shape返回的是一个元组（行，列）
    dataSetSize = dataSet.shape[0]
    # 将inX复制成dataSetSize行数据相同的数组，并将其与测试集中的数据对应位置相减
    # tile: numpy的一个函数，b = tile(a, (m, n)): 即是把a数组里面的元素复制n次放进一个数组c中，然后再把数组c复制m次放进一个数组b中
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 各个元素分别平方
    sqDiffMat = diffMat ** 2
    # 所有元素行相加，得到距离的平方，sum（axis = 0）为列相加，sum（axis = 1）为行相加
    sqDistances = sqDiffMat.sum(axis = 1)
    # 开方，算出距离
    distances = sqDistances ** 0.5
    # 升序排列，返回元素从小到大排序后的索引值
    sortedDistIndicies = distances.argsort()
    # 定义一个记录类别的字典
    classCount = {}
    # 只取前k个数据
    for i in range(k):
        #找到下标对应的标签数据
        voteIlabel = labels[sortedDistIndicies[i]]
        # 计算类别的次数
        # dict.get(key, default = None),字典的get()方法,返回指定键的值,如果值不在字典中则返回默认值，这里为初始化，然后数值加1。
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 按照字典的值的大小进行降序排序
    # key = operator.itemgetter(1)根据字典的值进行排序
    # key = operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    # 返回计数最大的值所对应的标签
    return sortedClassCount[0][0]

"""
在约会网站上使用K-近邻算法：
    (1) 收集数据：提供文本文件; 
    (2) 准备数据：使用 Python 解析文本文件;
    (3) 分析数据：使用 Matplotlib 画二维扩散（散点）图; 
    (4) 训练算法：此步骤不适用于 k-近邻算法; 
    (5) 测试算法：使用海伦提供的部分数据作为测试样本;         
        测试样本和非测试样本的区别在于: 测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误
    (6) 使用算法：产生简单的命令行程序，然后海伦可以输入一些特征数据以判断对方是否为自己喜欢的类型。
"""
def file2matrix(filename):
    """
    将文本数据转换成Numpy的解析程序
    filename - 数据文件路径
    """
    # 打开文件
    fr = open(filename)
    # 逐行读取文件内容
    arrayOLines = fr.readlines()
    # 获得文件的数据行的行数
    numberOfLines = len(arrayOLines)
    # 创建一个numberOfLines行，3列，元素全为0的矩阵
    returnMat = zeros((numberOfLines,3))
    # 创建一个空矩阵，存放标签
    classLabelVector = []
    # 行的索引
    index = 0
    for line in arrayOLines:
        # 默认删除字符串头尾的空白符如: '\n', '\t', '\r' , ' '
        line = line.strip()
        # 将字符串按照'\t'进行分隔
        listFromLine = line.split('\t')
        # 将数据前3列数据取出放在returnMat矩阵中
        returnMat[index,:] = listFromLine[0:3]
        # 将数据最后1列取出放在classLabelVector中
        classLabelVector.append(listFromLine[-1])
        # 索引加1
        index += 1
    # 返回特征矩阵以及对应类别的分类标签
    return returnMat,classLabelVector

def autoNorm(dataSet):
    """
    归一化公式: newValue = (oldValue - min) / (max - min)
    """
    # 每一列最小值
    minVals = dataSet.min(0)
    # 每一列最大值
    maxVals = dataSet.max(0)
    # 每一列最大值和最小值的差
    ranges = maxVals - minVals
    # 创建一个与dataSet行列数相同，元素全为0的矩阵
    normDataSet = zeros(shape(dataSet))
    # 返回行数
    m = dataSet.shape[0]
    # 矩阵相减，用dataSet减去m行1列全为minValue的矩阵
    normDataSet = dataSet - tile(minVals, (m,1))
    # 用normDataSet除以m行1列全为ranges的矩阵（仅为矩阵对应位置的元素值相除）
    normDataSet = normDataSet/tile(ranges, (m,1))   
    # 返回normDataSet, ranges和minVals
    return normDataSet, ranges, minVals

def datingClassTest():
    # 预留出10%的数据作为测试集验证准确率
    hoRatio = 0.50  
    # 读取文件中的数据以及标签
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')    
    # 将数据进行归一化处理
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 得到数据的行数
    m = normMat.shape[0]
    # 得到测试数据的行数
    numTestVecs = int(m * hoRatio)
    # 错误计数
    errorCount = 0.0
    for i in range(numTestVecs):
        # KNN算法（测试集，训练集，标签，k）
        # 第i行的所有数据作为测试集，从0到m所有数据作为训练集，标签数据为从0到m的数据标签，k值选为前3个
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        # 输出测试结果和实际结果
        print("the classifier came back with: %d, the real answer is: %d" %(int(classifierResult), int(datingLabels[i])))
        # 测试结果若和实际结果不同则错误加1
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    # 输出错误率
    print("the total error rate is: %f" %(errorCount/float(numTestVecs)))
    # 输出错误数
    print(errorCount)