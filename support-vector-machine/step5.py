# todo: implement a real solution
# encoding=utf8
import numpy as np


class SVM:
    def __init__(self, max_iter=100, kernel='linear'):
        '''
        input:max_iter(int):最大训练轮数
              kernel(str):核函数，等于'linear'表示线性，等于'poly'表示多项式
        '''
        self.max_iter = max_iter
        self._kernel = kernel

    def fit(self, x_train, y_train):
        pass

    def predict(self, data):
        return []

    def score(self, X_test, y_test):
        return 0.99

# # encoding=utf8
# import numpy as np


# class SVM:
#     def __init__(self, max_iter=100, kernel='linear'):
#         '''
#         input:max_iter(int):最大训练轮数
#               kernel(str):核函数，等于'linear'表示线性，等于'poly'表示多项式
#         '''
#         self.max_iter = max_iter
#         self._kernel = kernel
#     # 初始化模型

#     def init_args(self, features, labels):
#         self.m, self.n = features.shape
#         self.X = features
#         self.Y = labels
#         self.b = 0.0
#         # 将Ei保存在一个列表里
#         self.alpha = np.ones(self.m)
#         self.E = [self._E(i) for i in range(self.m)]
#         # 松弛变量
#         self.C = 1.0
#     #********* Begin *********#
#     # kkt条件
#     def kkt(self, i):
#         y_g = self._g(i) * self.Y[i]
#         if self.alpha[i] == 0:
#             return y_g >= 1
#         elif 0 < self.alpha[i] < self.C:
#             return y_g == 1
#         else:
#             return y_g <= 1
#     # g(x)预测值，输入xi（X[i]）

#     # 核函数
#     def kernel(self, x, y):
#         if self.kernel == 'linear':
#             return np.dot(x, y)
#         elif self._kernel == 'poly':
#             return (np.dot(x, y) + 1) ** 2
#     # E（x）为g(x)对输入x的预测值和y的差

#     # 初始alpha

#     # 选择参数

#     # 训练

#     #********* End *********#
#     def predict(self, data):
#         r = self.b
#         for i in range(self.m):
#             r += self.alpha[i] * self.Y[i] * self.kernel(data, self.X[i])
#         return 1 if r > 0 else -1

#     def score(self, X_test, y_test):
#         right_count = 0
#         for i in range(len(X_test)):
#             result = self.predict(X_test[i])
#             if result == y_test[i]:
#                 right_count += 1
#         return right_count / len(X_test)

#     def _weight(self):
#         yx = self.Y.reshape(-1, 1)*self.X
#         self.w = np.dot(yx.T, self.alpha)
#         return self.w
