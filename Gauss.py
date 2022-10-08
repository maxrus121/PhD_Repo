import pandas as pd
import numpy as np


def swap_rows(a, b, row1, row2):
    a[row1], a[row2] = a[row2], a[row1]
    b[row1], b[row2] = b[row2], b[row1]


def divide_row(a, b, row, divider):
    a[row] = [n / divider for n in a[row]]
    b[row] /= divider


def combine_rows(a, b, row, source_row, weight):
    a[row] = [(n + k * weight) for n, k in zip(a[row], a[source_row])]
    b[row] += b[source_row] * weight


def gauss(a, b):
    column = 0
    while column < len(b):
        current_row = None
        for r in range(column, len(a)):
            if current_row is None or abs(a[r][column]) > abs(a[current_row][column]):
                current_row = r
        if current_row is None:
            print("решений нет")
            return None
        if current_row != column:
            swap_rows(a, b, current_row, column)
        divide_row(a, b, column, a[column][column])
        for r in range(column + 1, len(a)):
            combine_rows(a, b, r, column, -a[r][column])
        column += 1
    return [a, b]


df = pd.read_excel('Gauss.xlsx', header=None)
A = df.to_numpy()[:, :len(df)]
B = df.to_numpy()[:, len(df)+1]
M = gauss(A.tolist(), B.tolist())
print(np.array(M[0]), np.array(M[1]), sep='\n\n')
