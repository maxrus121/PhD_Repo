import MathLibrary as Ml
import pandas as pd
import numpy as np
import warnings


def function_for_something(input_df):
    rules = [1, 1, 1, 1, 1, 1]
    new_df = pd.DataFrame()
    for k in range(len(rules)):
        split_df = input_df.loc[(input_df[0].apply(str).apply(lambda x: x.split('.')[0]) == str(k + 1)) & (
                input_df[0].apply(str).apply(lambda x: x.split('.')[1]) == str(rules[k]))]
        new_df = new_df.append(split_df)
    new_df = new_df.reset_index().drop(columns=[0, 'index'])
    return new_df.to_numpy()


def func_x(x, alfa, b):
    return b + np.dot(alfa, x)


def calculate_w(b, pi, qu):
    w = np.zeros([len(pi), len(pi)])
    w[0] = np.dot(pi, qu).T
    w[1] = np.dot(b - pi, qu).T
    for k in range(2, len(pi)):
        w[k] = np.dot(-b, w[k - 1]).T
    return w


def calculate_y(x):
    y = np.zeros([len(x), len(x)])
    for k in range(len(x)):
        y[k] = x[k]
        for n in range(k):
            y[k] -= y[n] * gamma(x[k], y[n])
    return y


def calculate_alfa(x, y):
    alfa = np.eye(len(x))
    for k in range(len(x)):
        for n in range(1 + k, len(x)):
            for l in range(n):
                alfa[n][k] -= gamma(x[n], y[l]) * alfa[l][k]
    return alfa


def gamma(xn, yn):
    return np.dot(xn, yn) / np.dot(yn, yn)


def print_ndarray(m):
    for line in m:
        for element in line:
            #print(f"{element:.8f}", end=' ')
            print(element, end=' ')
        print()


warnings.filterwarnings('ignore')
dfp1 = pd.read_excel('p1.xlsx', sheet_name='Лист1', header=None)
dfp2 = pd.read_excel('p2.xlsx', sheet_name='Лист1', header=None)
dfc = pd.read_excel('c.xlsx', sheet_name='Лист1', header=None)

dfp_count = dfp1[0].apply(str).apply(lambda x: x.split('.')[0])
dfp_count = dfp_count.value_counts().sort_index()

p1 = function_for_something(dfp1)
p2 = function_for_something(dfp2)
c = dfc.to_numpy().transpose()
p = np.zeros([len(p1), len(p1)])
for i in range(len(p1)):
    a = (p1[i], p2[i])
    p_final, F = Ml.solver(c, a, 'max')
    p[i] = p_final
#q = np.dot(p, c)
# Преобразование из двумерного массива в вектор
#q = np.array([a for b in q for a in b])
p = [[0.1, 0.2, 0.3, 0., 0., 0.4],
     [0.3, 0.3, 0., 0.1, 0.3, 0.],
     [0., 0., 0.4, 0.6, 0., 0.],
     [0., 0., 0.5, 0.5, 0., 0.],
     [0., 0., 0., 0., 0.3, 0.7],
     [0., 0., 0., 0., 0.4, 0.6]]
q = [60, 50, 40, 30, 20, 10]
print('q:' + '\n', *q)
print('P:' + '\n', p)
df = pd.DataFrame(p)
# Найдем матрицу ПИ
gates, boxes = Ml.find_boxes(df)
print('Проходные состояния:' + '\n', gates)
print('Ящики:' + '\n', *boxes)
LinResult = Ml.linear_matrix(df, boxes)
Pi_matrix = Ml.pi_matrix(df, gates, boxes, LinResult)
print('Рассчитанная матрица Пи:' + '\n', Pi_matrix)

# Найдем W
B = np.linalg.inv(np.eye(len(p)) - p + Pi_matrix)
print('B:' + '\n', B, '\n')
'''W1 = np.dot(Pi_matrix, q)
W2 = np.dot(B - Pi_matrix, q)
W3 = np.dot(-B, W2)
W4 = np.dot(-B, W3)
W5 = np.dot(-B, W4)
W6 = np.dot(-B, W5)
W_New = [W1, W2, W3, W4, W5, W6]
W_New = np.array(W_New)
W_New = W_New.T
print('W:')
print_ndarray(W_New)
print()'''
X = calculate_w(B, Pi_matrix, q)
X = X.T
print('X:')
for elem in X: print(*elem)
print()
'''X1, X2, X3, X4, X5, X6 = W_New[0], W_New[1], W_New[2], W_New[3], W_New[4], W_New[5]
y1 = X1
y2 = X2 - y1 * gamma(X2, y1)
y3 = X3 - y1 * gamma(X3, y1) - y2 * gamma(X3, y2)
y4 = X4 - y1 * gamma(X4, y1) - y2 * gamma(X4, y2) - y3 * gamma(X4, y3)
y5 = X5 - y1 * gamma(X5, y1) - y2 * gamma(X5, y2) - y3 * gamma(X5, y3) - y4 * gamma(X5, y4)
y6 = X6 - y1 * gamma(X6, y1) - y2 * gamma(X6, y2) - y3 * gamma(X6, y3) - y4 * gamma(X6, y4) - y5 * gamma(X6, y5)
print('Y_manual:' + '\n', *y1)
print(*y2)
print(*y3)
print(*y4)
print(*y5)
print(*y6, '\n')'''
Y = calculate_y(X)
print('Y_calculated')
for elem in Y: print(*elem)
print()
'''a1 = [1, 0, 0, 0, 0, 0]
a2 = [- gamma(X2, y1), 1, 0, 0, 0, 0]
a3 = [- gamma(X3, y1) - a2[0] * gamma(X3, y2), - gamma(X3, y2), 1, 0, 0, 0]
a4 = [- gamma(X4, y1) - a2[0] * gamma(X4, y2) - a3[0] * gamma(X4, y3), - gamma(X4, y2) - a3[1] * gamma(X4, y3),
      - gamma(X4, y3), 1, 0, 0]
a5 = [- gamma(X5, y1) - a2[0] * gamma(X5, y2) - a3[0] * gamma(X5, y3) - a4[0] * gamma(X5, y4), - gamma(X5, y2)
      - a3[1] * gamma(X5, y3) - a4[1] * gamma(X5, y4), - gamma(X5, y3) - a4[2] * gamma(X5, y4), - gamma(X5, y4), 1, 0]
a6 = [- gamma(X6, y1) - a2[0] * gamma(X6, y2) - a3[0] * gamma(X6, y3) - a4[0] * gamma(X6, y4) - a5[0] * gamma(X6, y5),
      - gamma(X6, y2) - a3[1] * gamma(X6, y3) - a4[1] * gamma(X6, y4) - a5[1] * gamma(X6, y5), - gamma(X6, y3)
      - a4[2] * gamma(X6, y4) - a5[2] * gamma(X6, y5), - gamma(X6, y4) - a5[3] * gamma(X6, y5), - gamma(X6, y5), 1]
print('Alfa_manual:' + '\n', *a1)
print(*a2)
print(*a3)
print(*a4)
print(*a5)
print(*a6, '\n')'''
Alfa = calculate_alfa(X, Y)
print('Alfa_calculated:')
print_ndarray(Alfa)

y3 = X[2] + Alfa[2][1] * X[1] + Alfa[2][0] * X[0]
print(y3)
print(Y[2])
