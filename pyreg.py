import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sympy import symbols, Limit, Derivative


class SimpleLinearRegression:  # 단순 선형회귀 클래스
    def __init__(self, dataset):
        self.p = len(dataset)  # 데이터 개수(순서쌍 개수)

        self.x_data = [v[0] for v in dataset]  # 데이터세트에서 x, y 분리
        self.y_data = [v[1] for v in dataset]

        self.gradient = 0  # 기울기 (ax+b에서 a)
        self.intercept = 0  # x 절편 (ax+b 에서 b)

    def draw(self, graph=0):  # 0 : 점만 표현, 1 : 그래프만 표현 2 : 점과 그래프를 함께 표현
        if graph is 0:
            plt.plot(self.x_data, self.y_data, 'ro')  # 산포도
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

        elif graph is 1:  # 회귀식 그래프
            plt.plot(self.x_data, self.gradient * self.x_data + self.intercept)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

        elif graph is 2:
            print(self.gradient)
            plt.plot(self.x_data, self.y_data, 'ro')  # 회귀식 그래프와 산포도를 함께 표현
            plt.plot(self.x_data, self.gradient * self.x_data + self.intercept)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

    def regress(self, k):  # 훈련
        a = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # 기울기
        b = tf.Variable(tf.zeros([1]))  # x 절편
        y = a*self.x_data+b  # 회귀식
        loss = tf.reduce_mean(tf.square(y - self.y_data))  # 비용함수 생성

        optimizer = tf.train.GradientDescentOptimizer(0.5)  # 옵티마이저 생성
        train = optimizer.minimize(loss)  # 비용함수를 최소화

        init = tf.global_variables_initializer()  # 변수 초기화

        sess = tf.Session()  # 세션 생성
        sess.run(init)  # 초기화 실행

        for step in range(k):  # k번 반복하여 학습
            sess.run(train)

        self.gradient = sess.run(a)  # 기울기, 절편 값 지정
        self.intercept = sess.run(b)

    def f(self, x):  # 회귀식
        return self.gradient*x + self.intercept

    def reg(self, k, v):
        W, x, h, a = symbols('W, x, h, a')  # 기울기, 절편, 독립변수 x
        y = W*x  # 회귀식

        op = sum([(self.y_data[i]-y.subs({x: i}))**2 for i in range(self.p)])  # 비용함수
        d = Derivative(op, W)

        print(d.doit())
        r = []
        print(np.arange(-1.0, 1.0, 0.25))
        for i in np.arange(-1.0, 1.0, 0.2):
            r.append(d.doit().subs({W: i}))
        self.gradient = min(r)
        self.intercept = 0
        print(self.f(1))

def main():
    p = int(input('Input number of points : '))  # 점 개수
    p_set = []
    for i in range(p):  # 산포도 생성
        x = np.random.normal(0.0, 0.5)
        y = x * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
        p_set.append(([x, y]))

    s = SimpleLinearRegression(p_set)   
    s.draw()  # 그래프 그리기
    s.regress(16)
    s.draw(0)
    s.draw(1)
    s.draw(2)

main()
