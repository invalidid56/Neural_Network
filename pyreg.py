import numpy as np

import tensorflow as t

p = 0  # 점 개수


class SimpleLinearRegression:  # 단순 선형회귀
    def __init__(self, dataset):
        self.p = len(dataset)  # 점 개수

    def draw(self, graph=0):  # 0 : 점만 표현, 1 : 그래프만 표현 2 : 점과 그래프를 함께 표현
        pass

    def regress(self):  # 훈련
        pass

    def f(self, x):  # 회귀식
        pass
