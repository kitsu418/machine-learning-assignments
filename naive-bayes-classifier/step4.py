import numpy as np


class NaiveBayesClassifier(object):
    def __init__(self):
        '''
        self.label_prob表示每种类别在数据中出现的概率
        例如，{0:0.333, 1:0.667}表示数据中类别0出现的概率为0.333，类别1的概率为0.667
        '''
        self.label_prob = {}
        '''
        self.condition_prob表示每种类别确定的条件下各个特征出现的概率
        例如训练数据集中的特征为 [[2, 1, 1],
                              [1, 2, 2],
                              [2, 2, 2],
                              [2, 1, 2],
                              [1, 2, 3]]
        标签为[1, 0, 1, 0, 1]
        那么当标签为0时第0列的值为1的概率为0.5，值为2的概率为0.5;
        当标签为0时第1列的值为1的概率为0.5，值为2的概率为0.5;
        当标签为0时第2列的值为1的概率为0，值为2的概率为1，值为3的概率为0;
        当标签为1时第0列的值为1的概率为0.333，值为2的概率为0.666;
        当标签为1时第1列的值为1的概率为0.333，值为2的概率为0.666;
        当标签为1时第2列的值为1的概率为0.333，值为2的概率为0.333,值为3的概率为0.333;
        因此self.label_prob的值如下：     
        {
            0:{
                0:{
                    1:0.5
                    2:0.5
                }
                1:{
                    1:0.5
                    2:0.5
                }
                2:{
                    1:0
                    2:1
                    3:0
                }
            }
            1:
            {
                0:{
                    1:0.333
                    2:0.666
                }
                1:{
                    1:0.333
                    2:0.666
                }
                2:{
                    1:0.333
                    2:0.333
                    3:0.333
                }
            }
        }
        '''
        self.condition_prob = {}

    def fit(self, features, labels):
        '''
        对模型进行训练，需要将各种概率分别保存在self.label_prob和self.condition_prob中
        :param feature: 训练数据集所有特征组成的ndarray
        :param label:训练数据集中所有标签组成的ndarray
        :return: 无返回
        '''

        #********* Begin *********#
        row_num = len(features)
        col_num = len(features[0])

        for label in labels:
            self.label_prob[label] = self.label_prob[label] + \
                1 if label in self.label_prob else 1

        for key in self.label_prob.keys():
            self.label_prob[key] = (self.label_prob[key] + 1) / (row_num + len(self.condition_prob))
            self.condition_prob[key] = {}
            for i in range(col_num):
                self.condition_prob[key][i] = {}
                for k in np.unique(features[:, i], axis=0):
                    self.condition_prob[key][i][k] = 0
        for label_id in range(row_num):
            for col_id in range(col_num):
                self.condition_prob[labels[label_id]][col_id][features[label_id][col_id]] = self.condition_prob[labels[label_id]
                                                                                                                ][col_id][features[label_id][col_id]] + 1 if features[label_id][col_id] in self.condition_prob[labels[label_id]] else 1

        for label in self.condition_prob.keys():
            for col in self.condition_prob[label].keys():
                total = sum(self.condition_prob[label][col].values())
                for feature in self.condition_prob[label][col].keys():
                    self.condition_prob[label][col][feature] = (self.condition_prob[label][col][feature] + 1) / (total + len(self.condition_prob[label][col]))
        #********* End *********#

    def predict(self, features):
        '''
        对数据进行预测，返回预测结果
        :param feature:测试数据集所有特征组成的ndarray
        :return:
        '''
        # ********* Begin *********#
        result = []
        for feature in features:
            prob = np.zeros(len(self.label_prob.keys()))
            label_id = 0
            for label, label_prob in self.label_prob.items():
                prob[label_id] = label_prob
                for col_id in range(len(features[0])):
                    prob[label_id] *= self.condition_prob[label][col_id][feature[col_id]]
                label_id += 1
            result.append(list(self.label_prob.keys())[np.argmax(prob)])
        return np.array(result)
        #********* End *********#
