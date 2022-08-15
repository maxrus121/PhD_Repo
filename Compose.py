import MathLibrary as Ml
import pandas as pd
import numpy as np


def function_for_something(input_df):
    rules = [1, 1, 1, 1, 1, 1]
    new_df = pd.DataFrame()
    for k in range(len(rules)):
        split_df = input_df.loc[(input_df[0].apply(str).apply(lambda x: x.split('.')[0]) == str(k + 1)) & (
                input_df[0].apply(str).apply(lambda x: x.split('.')[1]) == str(rules[k]))]
        new_df = new_df.append(split_df)
    new_df = new_df.reset_index().drop(columns=[0, 'index'])
    return new_df.to_numpy()


dfp1 = pd.read_excel('p1.xlsx', sheet_name='Лист1', header=None)
dfp2 = pd.read_excel('p2.xlsx', sheet_name='Лист1', header=None)
dfc = pd.read_excel('c.xlsx', sheet_name='Лист1', header=None)

dfp_count = dfp1[0].apply(str).apply(lambda x: x.split('.')[0])
dfp_count = dfp_count.value_counts().sort_index()

p1 = function_for_something(dfp1)
p2 = function_for_something(dfp2)
c = dfc.to_numpy()
c = c.transpose()
p = np.zeros([len(p1), len(p1)])
for i in range(len(p1)):
    a = (p1[i], p2[i])
    p_itog, F = Ml.solver(c, a, 'max')
    p[i] = p_itog
q = np.dot(p, c)
print('q:' + '\n', *q)
df = pd.DataFrame(p)
# Найдем матрицу ПИ
dataframe, gates, boxes = Ml.find_boxes(df)
print('Проходные состояния:' + '\n', gates)
print('Ящики:' + '\n', *boxes)
LinResult = Ml.linear_matrix(dataframe, boxes)
Pi_matrix = Ml.pi_matrix(dataframe, gates, boxes, LinResult)
print('Рассчитанная матрица Пи:' + '\n', Pi_matrix)

# найдем r(P) формула в задаче 3.2
#r_P = copy.deepcopy(Pi_matrix)
r_P1 = np.dot(Pi_matrix, q)
print('r(P):' + '\n', *r_P1)
# Ml.ergo_solver('PrimerAV.xlsx')
