import math

import MathLibrary as Ml
import pandas as pd
import numpy as np
import warnings
import Linear
# warnings.filterwarnings('ignore')
# dfP = pd.read_excel('PrimerKS.xlsx', header=None)
# P = [[0.1, 0.2, 0.3, 0., 0., 0.4],
#      [0.5, 0.5, 0., 0., 0., 0.],
#      [0., 0., 0.4, 0.6, 0., 0.],
#      [0., 0., 0.5, 0.5, 0., 0.],
#      [0., 0., 0., 0., 0.3, 0.7],
#      [0., 0., 0., 0., 0.4, 0.6]]
# q = [60, 50, 40, 30, 20, 10]
# a = 10
#
# L = (np.eye(len(P)) - P) * a
# print('L = ', L)
# P = dfP.to_numpy()
# print('P = ', P)
# Pi = Ml.ergo_solver(P)
# print(Pi)
# P_preobr = Ml.gauss_a(P)
# #P_preobr = np.array(P_preobr)
# print(P)
# Pi_preobr = Ml.gauss_a(Pi[4:6,4:6])
# print(Pi_preobr)
# answer = np.eye(len(P)) - Pi
# print('answer = ',answer)
#
# A = (np.eye(len(P)) - P + Pi)
# print(A)
# A_preobr = Ml.gauss_a(A)
# print(A_preobr)
# B = np.linalg.inv(A)
# print('B = ', B)
# W1 = (B-Pi)*q
# print('W1 = ', W1)
#
# W_sob, V_sob =np.linalg.eig(P - Pi)
# B1 = np.linalg.inv(np.eye(len(P)) - P + Pi)
# print('B = ', B1)
# print('Собственный вектор = ', V_sob, sep='\n')
# print('Собственные значения = ', W_sob, sep='\n')
#
# K = [1, 2, 5]
# print(np.linalg.matrix_power(P,1))
# float_formatter = "{:.2f}".format
# np.set_printoptions(formatter={'float_kind':float_formatter})
# for i in K:
#      MAS = np.dot((np.eye(len(P))+Pi-np.linalg.matrix_power(P,i)), W1)
#      print('MAS', i, MAS)
#
# print('B-1', np.eye(len(P)) - P + Pi)
# print('Матрица B^4', np.linalg.matrix_power(B1, 4))
# W = np.zeros([len(P), len(P)])
# for i in range(len(P)):
#     if i == 0:
#         W[i] = np.dot(Pi, q)
#     elif i == 1:
#         W[i] = np.dot(B1 - Pi, q)
#     else:
#         W[i] = np.dot(-B1, W[i - 1])
#
# W_new = W[:][0:6]
# W = W.transpose()
# W_new = W_new.transpose()
# print('матрица W', W)

#detW = np.linalg.det(W)
#print(detW)
#rankW = np.linalg.matrix_rank(W_new)
#print(rankW)
#W_preobr = Ml.gauss(W)
#W_preobr = np.array(W_preobr)
#detW_preobr = np.linalg.det(W_preobr)

#a, y, Fi = Ml.ort(W)
#print('a = ', a)
#print('y=', y)
#print('Fi=', Fi)

#print('преобразованная матрица', W_preobr)

#A1 = [[1., 1., 1.],
#      [0., 0., 1.],
#      [0., -3., -3.]]
#A1 = np.array(A1)
#print('Матрица А = ', A1)
# A_preobr = MathLibrary.Gauss_A(A1)
#A_preobr = Ml.gauss(A1)
#A_preobr = np.array(A_preobr)

#print('Треугольная матрица = ', A_preobr, sep='\n')
#detA = np.linalg.det(A_preobr)
#detA1 = np.linalg.det(A1)
#print('Определитель матрицы A = ', detA)
#print('Определитель матрицы A1 = ', detA1)




# L3 = [[0, 0.008, 0.04, 0.0008],
#       [0.064, 0, 0.008,  0.0008],
#       [0.0008, 0.06, 0, 0.0007],
#       [0.0006, 0.0005, 0.06, 0]]
#
# L4 = [[0, 0.0012, 0.06, 0.0012],
#       [0.096, 0, 0.012, 0.0012],
#       [0.0012, 0.07, 0, 0.001],
#       [0.001, 0.008, 0.08, 0]]

L1 = [[0, 4.3*10**(-6), 2.2*10**(-6), 1.1*10**(-6), 1*10**(-6)],
       [0.0025, 0, 0, 0, 0],
       [0.025, 0, 0, 0, 0],
       [0.125, 0, 0, 0, 0],
       [0.05, 0, 0, 0, 0]]

L2 = [[0, 9*10**(-4), 5*10**(-4), 4*10**(-4), 5*10**(-5)],
       [0.0025, 0, 0, 0, 0],
       [0.025, 0, 0, 0, 0],
       [0.125, 0, 0, 0, 0],
       [0.05, 0, 0, 0, 0]]

C = [[0], [51400*2073.95], [51400*2073.95], [51400*2073.95], [51400*2073.95]]

Pi_1 = Ml.calc_stab_1(L1, L2, C, 1)
Pi_2 = Ml.calc_stab_1(L1, L2, C, 2)
 #Pi_1 = Pi_1.tolist()
 #Pi_2 = Pi_2.tolist()
print('Pi_1 = ', Pi_1)
print('Pi_2 = ', Pi_2)

Data = [C, [Pi_1, Pi_2]]
Q1 = Ml.universal_solver(Data, 'max')
Q2 = Ml.universal_solver(Data, 'min')
print(Q1[1])
print(Q2[1])


#def B_a(a, Pi, L):
#    return np.linalg.inv(Pi - (1/a)*L)
#def V1(a,q,Pi,L):
#    return (1/a)*(np.dot(B_a(a,Pi,L),q)- np.dot(Pi,q))
#a = 10
#t = 2
#O = math.exp(-a*t*(1-max(W_sob))) * math.sqrt(np.dot(V1(a,q,Pi,L),V1(a,q,Pi,L)))
#print('O = ',O)
