"""
K-近邻算法（KNN）
一般流程：
    (1) 收集数据：可以使用任何方法
    (2) 准备数据：距离计算所需要的数值，最好是结构化的数据
    (3) 分析数据：可以使用任何方法。
    (4) 训练算法：此步骤不适用于k-近邻算法。
    (5) 测试算法：计算错误率。
    (6) 使用算法：首先需要输入样本数据和结构化的输出结果，然后运行KNN算法判定输入数据分别属于哪个分类，最后应用对计算出的分类执行后续处理
"""
from numpy import *  # numpy
import operator  # 运算符

def createDataSet():
    # 添加已有数据信息
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])  # 数据的坐标信息（数据集）
    labels = ['A', 'A', 'B', 'B']  # 数据点的分类标签信息
    return group, labels

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
    inX：- 用于分类的数据(测试集)
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
