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


def generate_w(b, pi, qu, number):
    w = np.zeros([number, len(qu)])
    w[0] = np.dot(b - pi, qu).T
    for k in range(1, number):
        w[k] = np.dot(-b, w[k - 1]).T
    return w


def calculate_y(x, number):
    y = np.zeros([number, len(x[0])])
    for k in range(number):
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
q = np.dot(p, c)
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

# Найдем W1
B = np.linalg.inv(np.eye(len(p)) - p + Pi_matrix)
print('B:' + '\n', B, '\n')
W1 = np.dot(B - Pi_matrix, q)
W2 = np.dot(-B, W1)
W3 = np.dot(-B, W2)
W4 = np.dot(-B, W3)
X = generate_w(B, Pi_matrix, q, 4)
print('W:' + '\n', *W1)
print(*W2)
print(*W3)
print(*W4, '\n')
print('X:')
for elem in X: print(*elem)
print()
# Так W1 представляет собой двумерный массив, мы преобразовываем его вначале в одномерный и затем в формат array
X1 = np.array([a for b in W1 for a in b])
X2 = np.array([a for b in W2 for a in b])
X3 = np.array([a for b in W3 for a in b])
X4 = np.array([a for b in W4 for a in b])
y1 = X1
y2 = X2 - y1 * gamma(X2, y1)
y3 = X3 - X1 * gamma(X3, y1) - y2 * gamma(X3, y2)
y4 = X4 - y1 * gamma(X4, y1) - y2 * gamma(X4, y2) - y3 * gamma(X4, y3)
print('Y_manual:' + '\n', *y1)
print(*y2)
print(*y3)
print(*y4, '\n')
Y = calculate_y(X, 4)
#print('Y:', *Y, sep='\n')
print('Y_calculated')
for elem in Y: print(*elem)
print()
a1 = [1, 0, 0, 0]
a2 = [- gamma(X2, y1), 1, 0, 0]
a3 = [- gamma(X3, y1) - a2[0] * gamma(X3, y2), - gamma(X3, y2), 1, 0]
a4 = [- gamma(X4, y1) - a2[0] * gamma(X4, y2) - a3[0] * gamma(X4, y3), - gamma(X4, y2) - a3[1] * gamma(X4, y3),
      - gamma(X4, y3), 1]
print('Alfa_manual:' + '\n', *a1)
print(*a2)
print(*a3)
print(*a4, '\n')
Alfa = calculate_alfa(X, Y)
print('Alfa_calculated:' + '\n', Alfa)
