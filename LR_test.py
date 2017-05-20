import numpy as np
from sympy import symbols, Derivative
import matplotlib.pyplot as plt



p = 100

p_set = []

for i in range(p):  # 산포도 생성
    x = np.random.normal(0.0, 0.5)
    y = x * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    p_set.append(([x, y]))

x_data = [p_set[i][0] for i in range(len(p_set))]
y_data = [p_set[i][1] for i in range(len(p_set))]

f = lambda W, x : W*x

W, x = symbols('W, x')
y = W*x

op = sum([(y_data[i] - y.subs({x: i})) ** 2 for i in range(p)])  # 비용함수

d = Derivative(op, W)

def compare(v1, v2):
    if v1-v2 > 0:
        return 0
    else:
        return 1

k= [1.0]
for i in range(10):
    k.append(k[-1]*0.5)
w = -1
for i in k:
    w=w-d.doit().subs({W:w})*abs(1/d.doit().subs({W:w}))*i
print(w)

w = [w]
plt.plot(x_data, y_data, 'ro')  # 회귀식 그래프와 산포도를 함께 표현
plt.plot(x_data, w *x_data)
plt.xlabel('x')
plt.ylabel('y')
plt.show()