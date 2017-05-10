import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

p = 0  # 점 개수


class SimpleLinearRegression:  # 단순 선형회귀 클래스
    def __init__(self, dataset):
        self.p = len(dataset)  # 점 개수

        self.x_data = [v[0] for v in dataset]
        self.y_data = [v[1] for v in dataset]

    def draw(self, graph=0):  # 0 : 점만 표현, 1 : 그래프만 표현 2 : 점과 그래프를 함께 표현
        if graph is 0:
            plt.plot(self.x_data, self.y_data, 'ro')
            plt.show()
        elif graph is 1:
            pass

    def regress(self):  # 훈련
        a = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # 기울기
        b = tf.Variable(tf.zeros([1]))  # x 절편
        y = a*self.x_data+b  # 회귀식
        loss = tf.reduce_mean(tf.square(y - self.y_data))  # 최소제곱법-경사하강법

        optimizer = tf.train.GradientDescentOptimizer(0.5)  # 옵티마이저 생성
        train = optimizer.minimize(loss)  # 비용함수 : 실제 값과 함수 사이의 오차의 합계에 대한 평균값

        init = tf.global_variables_initializer()  # 변수 초기화

        sess = tf.Session()  # 세션 생성
        sess.run(init)  # 초기화 실행

        for step in range(8):
            sess.run(train)

        plt.plot(self.x_data, self.y_data, 'ro')
        plt.plot(self.x_data, sess.run(a)*self.x_data+sess.run(b))  # 우히힣 꾸히힣 에헤헿
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def f(self, x):  # 회귀식
        pass

### Test ###
n = 1000
data_set = []
for i in range(n):
    x = np.random.normal(0.0, 0.55)
    y = x*0.1+0.3+np.random.normal((0.0, 0.03))
    data_set.append([x, y])

s = SimpleLinearRegression(data_set)

s.regress()
s.draw()
