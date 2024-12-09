---
title: numpy
date: 2024.12.2
updated:
tags: 
categories:
keywords:
description:
top_img:
comments:
cover:
toc:
toc_number:
toc_style_simple:
copyright:
copyright_author:
copyright_author_href:
copyright_url:
copyright_info:
mathjax:
katex:
aplayer:
highlight_shrink:
aside:
sticky: 
---

1. **创建数组**
   - **使用`np.array()`创建简单数组**
     - 可以从Python列表或元组创建NumPy数组。例如，创建一个一维数组：
       ```python
       import numpy as np
       one_d_array = np.array([1, 2, 3, 4, 5])
       print(one_d_array)
       ```
       这会输出`[1 2 3 4 5]`。
     - 创建一个二维数组：
       ```python
       two_d_array = np.array([[1, 2, 3], [4, 5, 6]])
       print(two_d_array)
       ```
       输出为：
       ```
       [[1 2 3]
        [4 5 6]]
       ```
   - **使用特定函数创建数组**
     - `np.zeros()`：创建全零数组。例如，创建一个形状为(3, 4)的二维全零数组：
       ```python
       zeros_array = np.zeros((3, 4))
       print(zeros_array)
       ```
       输出为：
       ```
       [[0. 0. 0. 0.]
        [0. 0. 0. 0.]
        [0. 0. 0. 0.]]
       ```
     - `np.ones()`：创建全一数组。创建一个形状为(2, 3)的二维全一数组：
       ```python
       ones_array = np.ones((2, 3))
       print(ones_array)
       ```
       输出为：
       ```
       [[1. 1. 1.]
        [1. 1. 1.]]
       ```
     - `np.arange()`：类似于Python的`range()`函数，用于创建一个等差数组。例如，创建一个从0开始，以2为步长，到10结束（不包括10）的数组：
       ```python
       arange_array = np.arange(0, 10, 2)
       print(arange_array)
       ```
       输出为`[0 2 4 6 8]`。
     - `np.linspace()`：创建一个在指定区间内均匀分布的数组。例如，创建一个在0到1之间，包含5个元素的数组：
       ```python
       linspace_array = np.linspace(0, 1, 5)
       print(linspace_array)
       ```
       输出为`[0.   0.25 0.5  0.75 1.  ]`。
2. **数组的基本属性和操作**
   - **形状（Shape）和维度（Dimension）**
     - 可以通过`shape`属性获取数组的形状。对于二维数组，它返回一个包含行数和列数的元组。例如：
       ```python
       two_d_array = np.array([[1, 2, 3], [4, 5, 6]])
       print(two_d_array.shape)
       ```
       输出为`(2, 3)`，表示这是一个2行3列的二维数组。
     - 可以通过`ndim`属性获取数组的维度。例如：
       ```python
       one_d_array = np.array([1, 2, 3, 4, 5])
       print(one_d_array.ndim)
       ```
       输出为1，表示这是一个一维数组。
   - **数据类型（Data Type）**
     - 可以通过`dtype`属性查看数组的数据类型。例如：
       ```python
       int_array = np.array([1, 2, 3])
       print(int_array.dtype)
       ```
       输出为`int64`（在大多数情况下）。可以在创建数组时指定数据类型，如：
       ```python
       float_array = np.array([1.0, 2.0, 3.0], dtype = np.float32)
       print(float_array.dtype)
       ```
       输出为`float32`。
   - **数组的索引和切片**
     - 对于一维数组，索引和Python列表类似。例如：
       ```python
       one_d_array = np.array([1, 2, 3, 4, 5])
       print(one_d_array[2])
       ```
       输出为3，表示获取索引为2的元素。
     - 切片操作也类似。例如，获取索引从1开始（包括1）到4结束（不包括4）的元素：
       ```python
       print(one_d_array[1:4])
       ```
       输出为`[2 3 4]`。
     - 对于二维数组，使用两个索引来定位元素。例如：
       ```python
       two_d_array = np.array([[1, 2, 3], [4, 5, 6]])
       print(two_d_array[1, 2])
       ```
       输出为6，表示获取第二行（索引为1）第三列（索引为2）的元素。
     - 二维数组的切片操作。例如，获取第一行的所有元素：
       ```python
       print(two_d_array[0, :])
       ```
       输出为`[1 2 3]`，也可以写成`two_d_array[0]`。获取第一列的所有元素：
       ```python
       print(two_d_array[:, 0])
       ```
       输出为`[1 4]`。
3. **数学运算**
   - **元素级运算（Element - wise Operations）**
     - 可以对数组进行加、减、乘、除等运算。例如，对于两个形状相同的数组进行加法运算：
       ```python
       array1 = np.array([1, 2, 3])
       array2 = np.array([4, 5, 6])
       print(array1 + array2)
       ```
       输出为`[5 7 9]`。同样可以进行减法、乘法和除法运算。
     - 还可以对数组和一个标量进行运算。例如，将数组中的每个元素乘以2：
       ```python
       print(array1 * 2)
       ```
       输出为`[2 4 6]`。
   - **矩阵运算（Matrix Operations）**
     - 对于二维数组，可以进行矩阵乘法。使用`np.dot()`函数或`@`运算符。例如：
       ```python
       matrix1 = np.array([[1, 2], [3, 4]])
       matrix2 = np.array([[5, 6], [7, 8]])
       print(np.dot(matrix1, matrix2))
       # 或者
       print(matrix1 @ matrix2)
       ```
       输出为：
       ```
       [[19 22]
        [43 50]]
       ```
     - 计算矩阵的转置。对于二维数组`matrix`，可以使用`matrix.T`来获取其转置。例如：
       ```python
       matrix = np.array([[1, 2], [3, 4]])
       print(matrix.T)
       ```
       输出为：
       ```
       [[1 3]
        [2 4]]
       ```
4. **统计和聚合函数**
   - **基本统计函数**
     - `np.mean()`：计算数组的平均值。例如，对于一维数组：
       ```python
       array = np.array([1, 2, 3, 4, 5])
       print(np.mean(array))
       ```
       输出为3.0。对于二维数组，可以指定轴来计算每行或每列的平均值。例如，计算二维数组每行的平均值：
       ```python
       two_d_array = np.array([[1, 2, 3], [4, 5, 6]])
       print(np.mean(two_d_array, axis = 1))
       ```
       输出为`[2. 5.]`，表示第一行平均值为2，第二行平均值为5。
     - `np.median()`：计算数组的中位数。例如：
       ```python
       print(np.median(array))
       ```
       输出为3。
     - `np.std()`：计算数组的标准差。例如：
       ```python
       print(np.std(array))
       ```
       输出为`1.4142135623730951`（约为$\sqrt{2}$）。
   - **最值和求和函数**
     - `np.max()`和`np.min()`：分别计算数组的最大值和最小值。例如：
       ```python
       print(np.max(array))
       print(np.min(array))
       ```
       输出为5和1。对于二维数组，可以指定轴来获取每行或每列的最值。
     - `np.sum()`：计算数组的总和。例如：
       ```python
       print(np.sum(array))
       ```
       输出为15。对于二维数组，也可以指定轴来计算每行或每列的和。
5. **排序和搜索**
   - **排序函数**
     - `np.sort()`：对数组进行排序。对于一维数组，它会返回一个排序后的新数组。例如：
       ```python
       array = np.array([3, 1, 4, 1, 5])
       print(np.sort(array))
       ```
       输出为`[1 1 3 4 5]`。对于二维数组，默认对每行进行排序。例如：
       ```python
       two_d_array = np.array([[3, 1, 4], [1, 5, 9]])
       print(np.sort(two_d_array))
       ```
       输出为：
       ```
       [[1 3 4]
        [1 5 9]]
       ```
     - 可以使用`np.argsort()`函数获取排序后的索引。例如：
       ```python
       array = np.array([3, 1, 4, 1, 5])
       print(np.argsort(array))
       ```
       输出为`[1 3 0 2 4]`，表示排序后元素原来的索引顺序。
   - **搜索函数**
     - `np.argmax()`和`np.argmin()`：分别返回数组中最大值和最小值的索引。例如：
       ```python
       array = np.array([3, 1, 4, 1, 5])
       print(np.argmax(array))
       print(np.argmin(array))
       ```
       输出为4和1。对于二维数组，可以指定轴来获取每行或每列最值的索引。
6. **广播（Broadcasting）**
   - 广播是NumPy的一个强大功能，它允许不同形状的数组在一定规则下进行运算。例如，一个形状为(3, 1)的数组和一个形状为(1, 3)的数组相加，NumPy会自动将它们的形状进行扩展以进行元素级别的加法运算。
     - 假设我们有一个列向量和一个行向量：
       ```python
       column_vector = np.array([[1], [2], [3]])
       row_vector = np.array([4, 5, 6])
       print(column_vector + row_vector)
       ```
       输出为：
       ```
       [[5 6 7]
        [6 7 8]
        [7 8 9]]
       ```
7. **线性代数操作**
   - **求解线性方程组**
     - 可以使用`np.linalg.solve()`函数来求解线性方程组。例如，对于方程组$Ax = b$，其中$A$是系数矩阵，$x$是未知数向量，$b$是常数项向量。假设$A=\begin{bmatrix}1&2\\3&4\end{bmatrix}$，$b=\begin{bmatrix}5\\6\end{bmatrix}$，则可以这样求解：
       ```python
       A = np.array([[1, 2], [3, 4]])
       b = np.array([5, 6])
       x = np.linalg.solve(A, b)
       print(x)
       ```
       输出为`[-4.  4.5]`，即$x_1=-4$，$x_2 = 4.5$。
   - **计算矩阵的特征值和特征向量**
     - 使用`np.linalg.eig()`函数。例如，对于矩阵$A=\begin{bmatrix}1&2\\2&1\end{bmatrix}$：
       ```python
       A = np.array([[1, 2], [2, 1]])
       eigenvalues, eigenvectors = np.linalg.eig(A)
       print("特征值:", eigenvalues)
       print("特征向量:", eigenvectors)
       ```
       输出为：
       ```
       特征值: [-1.  3.]
       特征向量: [[-0.70710678 -0.70710678]
                    [ 0.70710678 -0.70710678]]
       ```
8. **随机数生成**
   - **均匀分布随机数**
     - 使用`np.random.uniform()`函数生成在指定区间内均匀分布的随机数。例如，生成10个在0到1之间均匀分布的随机数：
       ```python
       uniform_random_numbers = np.random.uniform(0, 1, 10)
       print(uniform_random_numbers)
       ```
       会输出10个介于0和1之间的随机数。
   - **正态分布随机数**
     - 使用`np.random.normal()`函数生成服从正态分布的随机数。例如，生成5个均值为0，标准差为1的正态分布随机数：
       ```python
       normal_random_numbers = np.random.normal(0, 1, 5)
       print(normal_random_numbers)
       ```
       会输出5个近似服从正态分布的随机数。