import numpy as np
from sklearn import tree

class TrAdaBoost:
    def __init__(self, iters=10):
        '''
        Implement TrAdaBoost
        To read more about the TrAdaBoost, check the following paper:
            Dai, Wenyuan, et al. "Boosting for transfer learning." Proceedings of the 24th international conference on Machine learning. ACM, 2007.
        The code is modified according to https://github.com/chenchiwei/tradaboost/blob/master/TrAdaboost.py 
        '''
        self.iters = iters
    def fit_predict(self, trans_S, trans_A, label_S, label_A, test):
        N = self.iters
        trans_data = np.concatenate((trans_A, trans_S), axis=0)
        trans_label = np.concatenate((label_A, label_S), axis=0)

        row_A = trans_A.shape[0]
        row_S = trans_S.shape[0]
        row_T = test.shape[0]

        test_data = np.concatenate((trans_data, test), axis=0)

        # 初始化权重
        weights_A = np.ones([row_A, 1]) / row_A
        weights_S = np.ones([row_S, 1]) / row_S
        weights = np.concatenate((weights_A, weights_S), axis=0)

        bata = 1 / (1 + np.sqrt(2 * np.log(row_A / N)))

        # 存储每次迭代的标签和bata值？
        bata_T = np.zeros([1, N])
        result_label = np.ones([row_A + row_S + row_T, N])

        predict = np.zeros([row_T])

        # print ('params initial finished.')
        trans_data = np.asarray(trans_data, order='C')
        trans_label = np.asarray(trans_label, order='C')
        test_data = np.asarray(test_data, order='C')

        for i in range(N):
            P = self.calculate_P(weights, trans_label)

            result_label[:, i] = self.train_classify(trans_data, trans_label,
                                                test_data, P)
            # print ('result,', result_label[:, i], row_A, row_S, i, result_label.shape)

            error_rate = self.calculate_error_rate(label_S, result_label[row_A:row_A + row_S, i],
                                            weights[row_A:row_A + row_S, :])
            print ('Error rate:', error_rate)
            if error_rate > 0.5:
                error_rate = 0.5
            if error_rate == 0:
                N = i
                break  # 防止过拟合
                # error_rate = 0.001

            bata_T[0, i] = error_rate / (1 - error_rate)

            # 调整源域样本权重
            for j in range(row_S):
                weights[row_A + j] = weights[row_A + j] * np.power(bata_T[0, i],
                                                                (-np.abs(result_label[row_A + j, i] - label_S[j])))

            # 调整辅域样本权重
            for j in range(row_A):
                weights[j] = weights[j] * np.power(bata, np.abs(result_label[j, i] - label_A[j]))
        # print bata_T
        for i in range(row_T):
            # 跳过训练数据的标签
            left = np.sum(
                result_label[row_A + row_S + i, int(np.ceil(N / 2)):N] * np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))
            right = 0.5 * np.sum(np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))

            if left >= right:
                predict[i] = 1
            else:
                predict[i] = 0
                # print left, right, predict[i]

        return predict


    def calculate_P(self, weights, label):
        total = np.sum(weights)
        return np.asarray(weights / total, order='C')


    def train_classify(self, trans_data, trans_label, test_data, P):
        clf = tree.DecisionTreeClassifier(criterion="gini", max_features="log2", splitter="random")
        clf.fit(trans_data, trans_label, sample_weight=P[:, 0])
        return clf.predict(test_data)


    def calculate_error_rate(self, label_R, label_H, weight):
        total = np.sum(weight)

        # print(weight[:, 0] / total)
        # print(np.abs(label_R - label_H))
        return np.sum(weight[:, 0] / total * np.abs(label_R - label_H))