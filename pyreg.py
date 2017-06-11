import sys
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from anal_excel import xlanal

class LinearRegression:  # 선형회귀 클래스
    def __init__(self, dataset):
        self.p = len(dataset)  # 데이터 개수(순서쌍 개수)

        self.dataset = dataset

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

    def regress(self, dimension=2, a=0.005, tolerance = 0.00001):
        var = sp.symbols(' '.join([chr(ord('A')+i) for i in range(dimension)]))
        sym = sp.symbols(' '.join([chr(ord('a')+i) for i in range(dimension+1)]))

        if type(var) is type(sp.symbols('a')):
            var = var,
            sym = sym,

        f = sum([ sym[i] * var[i] for i in range(dimension)]) + sym[-1]
        print(type(f))

        cost = (1 / (2 * self.p))\
               * sum([(self.dataset[i][-1] - f.subs({var[l]:self.dataset[i][l] for l in range(dimension)})) ** 2 for i in range(self.p)])

        print(self.argmin(dimension+1, cost, sym))

    def argmin(self, k, func, sym, a=0.005,tolerance =0.00001):  # 경사감소법 옵티마이저
        v = np.array([0 for _ in range(k)]).T
        nv = np.array([-1 for _ in range(k)]).T  # 새로운 v 벡터

        while abs(nv[0] - v[0]) > tolerance:  # v 변화량이 허용범위보다 작을 때 까지
            v = nv
            nc = np.array(
                [sp.diff(func, sym[i]).subs({sym[l]: v[l] for l in range(k)}) for i in range(k)]
            ).T  # Nabla C : 목표함수의 w, b에 대한 편미분 값 벡터의 전치행렬
            nv = v - a * nc  # 새로운 v 벡터 할당

        return nv

def main():
    p = int(input('Input number of points : '))  # 점 개수
    p_set = []
    for i in range(p):  # 산포도 생성
        x = np.random.normal(0.0, 0.5)
        y = 0.3*x+0.5
        p_set.append([x, y])

    s = LinearRegression(p_set)
    #s.draw()  # 그래프 그리기
    s.regress(1)
    #s.draw(0)
    #s.draw(1)
    #s.draw(2)

main()

args = sys.argv[1:]
if args[1]=='test':
    main()

elif args[1]=='excel':
    s = LinearRegression(xlanal(args[2]))
    s.regress(1)

elif args[2]=='run':
    pass