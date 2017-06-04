import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

p = 100

p_set = []

for i in range(p):  # 산포도 생성
    x = np.random.normal(0.0, 0.5)
    y = x * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    p_set.append(([x, y]))

x_data = [p_set[i][0] for i in range(len(p_set))]
y_data = [p_set[i][1] for i in range(len(p_set))]

def draw(W, b, x=x_data, y=y_data):
    plt.plot(x, y, 'ro')
    plt.plot(x, [W*x[i]+b for i in range(p)])

    plt.show()

w, b, x = sp.symbols('w b x')

f = w*x+b
cost = (1/(2*p)) * sum([(y_data[i]-f.subs({x:x_data[i]}))**2 for i in range(p)])  # 오차함수
W = 0; nW = -1  # w 초기값, newW
B = 0; nB = -1  # b 초기값
a = 0.005  # 학습률, Learning Rate
tolerance = 0.00001  # 오차범위

while abs(nW-W) > tolerance:
    W, B = nW, nB
    nW = W-a*sp.diff(cost, w).subs({w: W, b: B})
    nB = B - a * sp.diff(cost, b).subs({w: W, b: B})

draw(nW, nB)