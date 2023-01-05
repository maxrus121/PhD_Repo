import MathLibrary as Ml
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
dfL = pd.read_excel('Lambda.xlsx', sheet_name='Лист1', header=None)
a = 5
q = [1, 3, 5]
edin = np.eye(3)
L = dfL.to_numpy()
P = np.eye(len(q)) - 1/a * L
Pi = Ml.ergo_solver(P)
B = np.linalg.inv(Pi - 1/a * L)
print(B, end='\n\n')
W1 = np.dot(B-Pi, q)
print(W1, end='\n\n')
W2 = np.dot(-B, W1)
print(W2, end='\n\n')
R = np.array([np.dot(Pi, q), W1, W2]).transpose()
print(R, end='\n\n')
W = Ml.gauss_solver("Gauss.xlsx")
print('htu')