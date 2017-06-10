import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

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
            plt.plot(self.x_data, [self.gradient * self.x_data[i] + self.intercept for i in range(self.p)])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

        elif graph is 2:
            print(self.gradient, self.intercept)
            plt.plot(self.x_data, self.y_data, 'ro')  # 회귀식 그래프와 산포도를 함께 표현
            plt.plot(self.x_data, [self.gradient * self.x_data[i] + self.intercept for i in range(self.p)])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

    def regress(self, a=0.005, tolerance = 0.00001):  # 학습 메서드, a : 학습률(Learning Rate), tolerance : 오차범위
        w, b, x = sp.symbols('w b x')  # 심볼릭 변수 정의

        f = w * x + b  # 회귀식 원형 정의
        cost = (1 / (2 * self.p)) * sum([(self.y_data[i] - f.subs({x: self.x_data[i]})) ** 2 for i in range(self.p)])
        # 오차함수 : 보고서 ()번 식 참조

        W, nW, B, nB = 0, -1, 0, -1  # 이전 W, 갱신 W, 이전 B, 갱신 B
        self.gradient, self.intercept = self.argmin(np.array([W, B]).T, cost, (w, b))

    def f(self, x):  # 회귀식
        return self.gradient*x + self.intercept

    def argmin(self, v, func, sym, a=0.005,tolerance =0.00001):  # 경사감소법 옵티마이저
        nv = np.array([-1, -1]).T  # 새로운 v 벡터

        while abs(nv[0] - v[0]) > tolerance:  # v 변화량이 허용범위보다 작을 때 까지
            v = nv
            nc = np.array(
                [sp.diff(func, sym[0]).subs({sym[0]: v[0], sym[1]: v[1]}),
                 sp.diff(func, sym[1]).subs({sym[0]: v[0], sym[1]: v[1]})]).T  # Nabla C : 목표함수의 w, b에 대한 편미분 값 벡터의 전치행렬
            nv = v - a * nc  # 새로운 v 벡터 할당

        return (nv[0], nv[1])

def main():
    p = int(input('Input number of points : '))  # 점 개수
    p_set = []
    for i in range(p):  # 산포도 생성
        x = np.random.normal(0.0, 0.5)
        y = x * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
        p_set.append(([x, y]))

    s = SimpleLinearRegression(p_set)   
    s.draw()  # 그래프 그리기
    s.regress()
    s.draw(0)
    s.draw(1)
    s.draw(2)

main()
