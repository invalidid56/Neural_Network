import numpy as np
from sympy import symbols, Limit, Derivative


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

print(d.doit())
for i in np.arange(-1, 1, 0.0001):
    print(d.doit().subs({W: i}))


def compare(v1, v2):
    if v1-v2 > 0:
        return 0
    else:
        return 1

safe_distance = [100, 1.0, 0.1, 0.01, 0.001, 0.0001]

grad=
for i in range(100):
    k = d.doit().subs({W:grad})
    if p