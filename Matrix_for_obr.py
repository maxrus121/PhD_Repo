import MathLibrary
import MathLibrary as Ml
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
# dfP = pd.read_excel('PrimerKS.xlsx', header=None)
P = [[0.1, 0.2, 0.3, 0., 0., 0.4],
     [0.3, 0.3, 0., 0.1, 0.3, 0.],
     [0., 0., 0.4, 0.6, 0., 0.],
     [0., 0., 0.5, 0.5, 0., 0.],
     [0., 0., 0., 0., 0.3, 0.7],
     [0., 0., 0., 0., 0.4, 0.6]]
q = [60, 50, 40, 30, 20, 10]
a = 6

L = (np.eye(len(P)) - P) * a
# P = dfP.to_numpy()
print('P = ', P)
Pi = Ml.ergo_solver(P)
print('PI = ', Pi)
B1 = np.linalg.inv(np.eye(len(P)) - P + Pi)
print('B = ', B1)
#print('B-1', np.eye(len(P)) - P + Pi)
#print('Матрица B^4', np.linalg.matrix_power(B1, 4))
W = np.zeros([len(P), len(P)])
for i in range(len(P)):
    if i == 0:
        W[i] = np.dot(Pi, q)
    elif i == 1:
        W[i] = np.dot(B1 - Pi, q)
    else:
        W[i] = np.dot(-B1, W[i-1])

W_new = W[:][0:6]
W = W.transpose()
W_new = W_new.transpose()
print('матрица W', W)

detW = np.linalg.det(W)
print(detW)
rankW = np.linalg.matrix_rank(W_new)
print(rankW)
W_preobr = MathLibrary.gauss(W)
W_preobr = np.array(W_preobr)
detW_preobr = np.linalg.det(W_preobr)

a, y, Fi = Ml.Ort(W)
print('a = ', a)
print('y=', y)
print('Fi=', Fi)

print('преобразованная матрица', W_preobr)



A1 = [[1., 1., 1.],
      [0., 0., 1.],
      [0., -3., -3.]]
A1 = np.array(A1)
print('Матрица А = ',A1)
#A_preobr = MathLibrary.Gauss_A(A1)
A_preobr = MathLibrary.gauss(A1)
A_preobr = np.array(A_preobr)



print('Треугольная матрица = ', A_preobr, sep='\n')
detA =np.linalg.det(A_preobr)
detA1 = np.linalg.det(A1)
print('Определитель матрицы A = ', detA)
print('Определитель матрицы A1 = ', detA1)
