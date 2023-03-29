# encoding=utf8
import numpy as np
# 构建感知机算法


class Perceptron(object):
    def __init__(self, learning_rate=0.01, max_iter=200):
        self.lr = learning_rate
        self.max_iter = max_iter

    def fit(self, data, label):
        '''
        input:data(ndarray):训练数据特征
              label(ndarray):训练数据标签
        output:w(ndarray):训练好的权重
               b(ndarry):训练好的偏置
        '''
        # 编写感知机训练方法，w为权重，b为偏置
        self.w = np.array([1.]*data.shape[1])
        self.b = np.array([1.])
        #********* Begin *********#
        for epoch in range(self.max_iter):
            # Loop over each training example
            for i in range(data.shape[0]):
                x = data[i]
                y = label[i]
                # Compute the activation of the perceptron
                activation = np.dot(x, self.w) + self.b

                # If the prediction is incorrect, update the weights and bias
                if y * activation <= 0:
                    self.w += self.lr * y * x
                    self.b += self.lr * y
        #********* End *********#

    def predict(self, data):
        '''
        input:data(ndarray):测试数据特征
        output:predict(ndarray):预测标签
        '''
        #********* Begin *********#
        activation = np.dot(data, self.w) + self.b
        predict = np.sign(activation)
        #********* End *********#
        return predict
