import numpy as np
import copy
import pandas as pd
import functools


'''
Список основных функций в библиотеке:
universal_solver - Функция для решения задачи линейного программирования
ergo_solver - Функция, решающая задачу нахождения ящиков в матрице состояний и генерирующая Пи матрицу
gauss_solver - Функция, приводящая матрицу к треугольному виду
gauss_right_solver - Функция, приводящая матрицу к треугольному виду с учетом правой части

Список вспомогательных функций в библиотеке:
input_converter - Преобразование Excel файла или чистых данных в Dataframe
solver - Расчет задачи линейного программирования
heapify - Создание дерева индексов для сортировки
heap_sort - Сортировка
prepare_data - Преобразование данных для задачи линейного программирования
linear_matrix - Решение системы линейных уравнений
pi_matrix - Расчет матрицы Пи
swap_rows - Изменение позиций строк 
divide_row - Деление строк 
combine_rows - Соединение строк 
gauss - Создание треугольной матрицы 
find_boxes - Нахождение ящиков и проходов в матрице состояний
swap_rows_right - Изменение позиций строк с учетом правой части
divide_row_right - Деление строк с учетом правой части
combine_rows_right - Соединение строк с учетом правой части
gauss_right - Создание треугольной матрицы с учетом правой части
'''


def universal_solver(input_data, method):
    # Объявляем имя столбцов содержащих решение задачи
    X, F = 'Минимизирующее решение X' if method == 'min' else 'Максимизирующее решение Х', 'Значение  F'
    if '.csv' in input_data:
        # Читаем данные из файла и создаем объект DataFrame
        df = pd.read_csv('in.csv', sep=';')
        # Используем функцию подготовки данных, для получения списков значений
        C, A1, A2 = prepare_data(df)
        # Дополняем DataFrame столбцом со значениями х-сов полученных из функции solver
        df[X], df[F] = solver(C, [A1, A2], method)
        df[F][1:] = None
        # Записываем данные DataFrame в файл csv
        df.to_csv('out.csv', index=False, sep=';')
        print('Решение записано в файл out.csv')
    elif '.xlsx' in input_data:
        # Читаем данные из файла и создаем объект DataFrame
        df = pd.read_excel('Data.xlsx', sheet_name='Лист1', engine='openpyxl')
        # Используем функцию подготовки данных, для получения списков значений
        C, A1, A2 = prepare_data(df)
        # Дополняем DataFrame столбцом со значениями х-сов полученных из функции solver
        df[X], df[F] = solver(C, [A1, A2], method)
        df[F][1:] = None
        # Записываем данные DataFrame в исходный файл Excel
        writer = pd.ExcelWriter('Data.xlsx')
        df.to_excel(writer, 'Test', index=False)
        writer.save()
        print('Решение записано в файл Excel')
    elif len(input_data) == 2 and len(input_data[0]) == len(input_data[1][0]):
        # Входные данные передаются напрямую в функцию
        x_final, f_final = solver(input_data[0], input_data[1], method)
        # Печатаем решение в консоль
        print('X =', x_final, ' F max =' if method == 'max' else ' F min =', f_final)
        return x_final, f_final
    else:
        print('Ошибка! Некорректные входные данные.')


def ergo_solver(input_data):
    df = input_converter(input_data)
    gates, boxes = find_boxes(df)
    print('Проходные состояния:' + '\n', gates)
    print('Ящики:' + '\n', *boxes)
    LinResult = linear_matrix(df, boxes)
    Pi_matrix = pi_matrix(df, gates, boxes, LinResult)
    print('Рассчитанная матрица Пи:' + '\n', Pi_matrix)
    return Pi_matrix


def gauss_solver(input_data):
    # Считывание и распознавание данных
    df = input_converter(input_data)
    # Преобразование из DataFrame в массив
    A = df.to_numpy()[:, :len(df)]
    #B = df.to_numpy()[:, len(df) + 1]
    # Запуск основной функции расчета
    #M = gauss_right(A.tolist(), B.tolist())
    M = gauss(A.tolist())
    #print(np.array(M[0]), np.array(M[1]), sep='\n\n')
    print(np.array(M[0]))
    # Отправляем на выход обработанную матрицу и вектор значений
    #return np.array(M[0]), np.array(M[1])
    return np.array(M[0])


def gauss_right_solver(input_data):
    df = input_converter(input_data)
    A = df.to_numpy()[:, :len(df)]
    B = df.to_numpy()[:, len(df) + 1]
    M = gauss_right(A.tolist(), B.tolist())
    print(np.array(M[0]), np.array(M[1]), sep='\n\n')
    return np.array(M[0]), np.array(M[1])


def input_converter(input_data):
    if '.xlsx' in input_data:
        # Считываем данные из файла Excel
        df = pd.read_excel(input_data, sheet_name='Лист1', header=None)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data
    elif not isinstance(input_data, pd.DataFrame):
        df = pd.DataFrame(input_data)
    else:
        print('Ошибка! Некорректные входные данные.')
        return None
    return df


def solver(c, a, method):
    sum_a1 = 0
    sum_a2 = 0
    k = 0
    for i in range(len(a[0])):
        sum_a1 += a[0][i]
        sum_a2 += a[1][i]
    if sum_a1 > 1 or sum_a2 < 1:
        print("Ошибка ограничений, множество пустое!")
        return
    for i in range(len(a[0])):
        if a[0][i] == a[1][i]:
            k += 1
        elif a[0][i] < 0 or a[1][i] < 0:
            print("Ограничения Отрицательные!")
            return
        elif a[0][i] > 1 or a[1][i] > 1:
            print("Ограничения больше единицы!")
            return
        elif a[0][i] > a[1][i]:
            print("Нижнее ограничение больше верхнего!")
            return
    if k == len(a[0]) or sum_a1 == 1 or sum_a2 == 1:
        print("Ошибка ограничений, множество состоит из одного элемента!")
        return
    k = 0
    for i in range(len(c) - 1):
        if c[i] == c[i + 1]:
            k = k + 1
    if k == len(c) - 1:
        print("Любой вектор из множества является решением задачи!")
        return
    r = [[0] * len(c) for _ in range(4)]
    for i in range(len(c)):
        r[0][i] = c[i]
        r[1][i] = i + 1
        r[2][i] = a[0][i]
        r[3][i] = a[1][i]
    heap_sort(copy.deepcopy(r[0]), r)
    if method == 'max':
        for b in r:
            b.reverse()
    x, y, z, sum_for_alpha, alpha, r2 = [], [], [], 0, 0, copy.deepcopy(r[2])
    if sum_a1 < 1:
        if sum_a2 > 1:
            for _ in range(len(a[0])):
                x.append(r2)
            for i in range(len(a[0])):
                for j in range(i + 1):
                    x[j][i] = r[3][j]
                y.append(0)
                for j in range(len(a[0])):
                    y[i] += x[i][j]
                if y[i] >= 1:
                    for j in range(len(a[0])):
                        if i == j:
                            sum_for_alpha = sum_for_alpha
                        else:
                            sum_for_alpha += x[i][j]
                    alpha = (1 - sum_for_alpha) / r[3][i]
                    for j in range(len(a[0])):
                        z.append(0)
                        z[j] = x[i][j]
                        if i == j:
                            z[j] = x[i][j] * alpha
                    break
            r.append(z)
            # Обратная пирамидальная сортировка
            heap_sort(copy.deepcopy(r[1]), r)
    # Производим округление результата, что бы избежать проблемы цифрового нуля
    x = np.around(r[4], 4)
    # Находим значение целевой функции
    f = (c * x).sum()
    return x, f


def heapify(nums, heap_size, root_index, m):
    # Предположим, что индекс самого большого элемента является корневым индексом
    largest = root_index
    left_child = (2 * root_index) + 1
    right_child = (2 * root_index) + 2
    # Если левый потомок корня является допустимым индексом, а элемент больше
    # чем текущий самый большой элемент, то обновляем самый большой элемент
    if left_child < heap_size and nums[left_child] > nums[largest]:
        largest = left_child
    # Делаем то же самое для right_child
    if right_child < heap_size and nums[right_child] > nums[largest]:
        largest = right_child
    # Если самый большой элемент больше не является корневым элементом, меняем их местами
    if largest != root_index:
        nums[root_index], nums[largest] = nums[largest], nums[root_index]
        for j in range(len(m)):
            m[j][root_index], m[j][largest] = m[j][largest], m[j][root_index]
        # Еще раз проходим функцией, чтобы проверить, что новый узел максимальный по значению
        heapify(nums, heap_size, largest, m)


def heap_sort(nums, m):
    n = len(nums)
    # Создаем Max Heap из списка
    # Второй аргумент означает, что мы останавливаемся на элементе перед -1, то есть на первом элементе списка.
    # Третий аргумент означает, что мы повторяем в обратном направлении, уменьшая количество i на 1
    for i in range(n, -1, -1):
        heapify(nums, n, i, m)
    # Перемещаем корень max heat в конец
    for i in range(n - 1, 0, -1):
        nums[i], nums[0] = nums[0], nums[i]
        for j in range(len(m)):
            m[j][i], m[j][0] = m[j][0], m[j][i]
        heapify(nums, i, 0, m)


def prepare_data(raw_data):
    # Объявляем имена столбцов в файле Excel чтобы по ним разделять данные
    c_title, a1_title, a2_title = 'Cтоимости', 'Нижние границы интервалов вероятности A1', \
                                  'Верхние границы интервалов вероятности A2'
    # Создаем кортежи нужного размера заполненные нулями, затем заполняем их данными из DataFrame
    c_raw, a1_raw, a2_raw = [0] * len(raw_data[c_title]), [0] * len(raw_data[c_title]), [0] * len(raw_data[c_title])
    for I in range(len(raw_data[c_title])):
        c_raw[I] = raw_data[c_title][I]
        a1_raw[I] = raw_data[a1_title][I]
        a2_raw[I] = raw_data[a2_title][I]
    return c_raw, a1_raw, a2_raw


def linear_matrix(dataframe, boxes):
    # Решение системы линейных уравнений
    linResult = []
    for i in range(len(boxes)):
        boxes[i] = [h - 1 for h in boxes[i]]
        linMatrix = dataframe.iloc[boxes[i], boxes[i]]
        box_values = np.array(linMatrix.values.tolist()).transpose()
        for j in range(len(box_values)):
            box_values[j][j] = box_values[j][j] - 1
        box_values[len(box_values) - 1] = [1 for _ in range(len(box_values))]
        V = np.zeros(len(box_values))
        V[len(box_values) - 1] = 1
        answer = np.linalg.solve(box_values, V)
        linResult.append(answer)
    return linResult


def pi_matrix(dataframe, gates, boxes, linear_result):
    # Решение линейной системы и создание матрицы Пи
    matrix_PI = np.zeros([len(dataframe), len(dataframe)])
    right = np.zeros([len(gates), len(dataframe)])
    left = np.zeros([len(gates), len(gates)])
    edin = np.zeros([len(gates), len(gates)])
    for i in range(len(boxes)):
        for j in boxes[i]:
            r = 0
            for k in boxes[i]:
                matrix_PI[j, k] = linear_result[i][r]
                r += 1
    table = np.asarray(dataframe.values.tolist())
    k = 0
    for i in gates:
        right[k] = np.matmul(table[i - 1], matrix_PI)
        k += 1
    for i in range(len(gates)):
        for j in range(len(gates)):
            left[i][j] = table[gates[i] - 1][gates[j] - 1]
            edin[i][i] = 1
    left = edin - left
    answer = np.linalg.solve(left, right)
    for i in range(len(gates)):
        matrix_PI[gates[i] - 1] = answer[i]
    result = (np.around(matrix_PI, 4))
    return result


def find_boxes(dataframe):
    # Инициализируем списки для переходов и ящиков
    step, box, bins = [], [], []
    # Создаем матрицу, заполненную нулями
    R = np.zeros((len(dataframe), len(dataframe)))
    for i in range(len(dataframe)):
        Ri, Rj, R_obj = [], [], []
        step.append(i)
        for j in range(len(dataframe)):
            if dataframe[j][i] > 0:
                Ri.append(j)
        if Ri == [i]:
            # Во всей строке только пересечение с собой получается ящик из одного элемента
            print(i, '= Bad')
            # Одиночный ящик добавляем в финальный ответ
            bins.append([i + 1])
            # Одиночный ящик исключаем из списка проходных состояний
            step.remove(i)
        for j in Ri:
            for k in range(len(dataframe)):
                if dataframe[k][j] > 0:
                    Rj.append(k)
        Ri = Ri + Rj
        for item in Ri:
            if item not in R_obj:
                R_obj.append(item)
        R_obj.sort()
        for j in range(int(len(dataframe) - len(R_obj))):
            R_obj.append(0)
        R[i] = R_obj
    # print('Получившаяся матрица переходов '+'\n', R)
    # Внутри цикла ищем ящики
    g = np.arange(0, len(R))
    for i in range(len(dataframe)):
        case = []
        for j in range(len(dataframe)):
            x, y = R[i], R[j]
            if i != j:
                if functools.reduce(lambda a, b: a and b, map(lambda p, q: p == q, x, y)):
                    if all(R[j] != g.astype(np.float64)):
                        case.append(j)
        if case:
            # Удаляем найденные ящики из списка проходных состояний
            for k in case:
                if k in step:
                    step.remove(k)
    # Вторая итерация поиска ящиков
    for i in range(len(dataframe)):
        case = []
        for j in range(len(dataframe)):
            x, y = R[i], R[j]
            if functools.reduce(lambda a, b: a and b, map(lambda p, q: p == q, x, y), True):
                case.append(j + 1)
        box.append(case)
    ways = [h + 1 for h in step]
    # Исключаем из списка ящиков повторения и проходные состояния
    for z in box:
        if len(z) > 1 and (z not in bins) and (z[0] not in ways):
            bins.append(z)
            bins.sort()
    return ways, bins


def swap_rows_right(a, b, row1, row2):
    a[row1], a[row2] = a[row2], a[row1]
    b[row1], b[row2] = b[row2], b[row1]


def divide_row_right(a, b, row, divider):
    # Деление создающее единицы на главной диагонали отключено
    #a[row] = [n / divider for n in a[row]]
    #b[row] /= divider
    a[row] = [n for n in a[row]]


def combine_rows_right(a, b, row, source_row, weight):
    a[row] = [(n + k * weight) for n, k in zip(a[row], a[source_row])]
    b[row] += b[source_row] * weight


def gauss_right(a, b):
    column = 0
    # Проход по всем строкам
    while column < len(b):
        current_row = None
        for r in range(column, len(a)):
            if current_row is None or abs(a[r][column]) > abs(a[current_row][column]):
                current_row = r
        # Проверка случая когда входные данные пустые
        if current_row is None:
            print("Решений нет")
            return None
        if current_row != column:
            swap_rows_right(a, b, current_row, column)
        divide_row_right(a, b, column, a[column][column])
        for r in range(column + 1, len(a)):
            combine_rows_right(a, b, r, column, -a[r][column])
        column += 1
    return [a, b]


def swap_rows(a, row1, row2):
    a[row1], a[row2] = a[row2], a[row1]


def divide_row(a, row, divider):
    # Деление создающее единицы на главной диагонали отключено
    #a[row] = [n / divider for n in a[row]]
    a[row] = [n for n in a[row]]


def combine_rows(a, row, source_row, weight):
    a[row] = [(n + k * weight) for n, k in zip(a[row], a[source_row])]


def gauss(a):
    column = 0
    while column < len(a):
        current_row = None
        for r in range(column, len(a)):
            if current_row is None or abs(a[r][column]) > abs(a[current_row][column]):
                current_row = r
        if current_row is None:
            print("Решений нет")
            return None
        if current_row != column:
            swap_rows(a, current_row, column)
        divide_row(a, column, a[column][column])
        for r in range(column + 1, len(a)):
            combine_rows(a, r, column, -a[r][column])
        column += 1
    return [a]


def gauss_a(a):
    for k in range(len(a) - 1):
        At = a.copy()
        for i in range(len(a)):
            for j in range(len(a)):
                if i <= k:
                    a[i][j] = At[i][j]
                elif i > k and j > k:
                    a[i][j] = round(At[i][j] - (At[i][k] / At[k][k]) * At[k][j], 4)
                elif i > k >= j:
                    a[i][j] = 0
    return a


def ort(x):
    a = np.zeros([len(x), len(x)])
    Fi = np.zeros([len(x), len(x)])
    y = np.zeros([len(x), len(x)])
    for i in range(len(x)):
        a[i][i] = 1
        for j in range(i):
            Fi[i][j] = -(np.dot(x[i], y[j]) / np.dot(y[j], y[j]))
            for k in range(i):
                a[i][j] += - a[k][j] * Fi[i][k]
            y[i] += np.dot(a[i][j], y[j])
        if i == 0:
            y[i] = x[0]
    return a, y, Fi
