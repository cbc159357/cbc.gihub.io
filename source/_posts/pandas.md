---
title: matplotlib
date: 2024.12.3
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
1. **`plot`函数所属包**
   - 在Python中，`plot`函数最常见于`matplotlib`库。`matplotlib`是一个广泛使用的绘图库，用于创建各种静态、动态和交互式的可视化图表。在`matplotlib`中，`plot`函数主要用于绘制折线图。例如，绘制一个简单的函数 $y = x^2$ 在区间 $[0, 10]$ 的图像，代码如下：
```python
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 10, 100)
y = x ** 2
plt.plot(x, y)
plt.show()
```
   - 这里先导入了`matplotlib.pyplot`模块并简称为`plt`，以及`numpy`模块。然后生成了`x`轴的数据（从0到10均匀分布的100个点），计算出对应的`y`轴数据（`x`的平方），通过`plt.plot`绘制折线图，最后使用`plt.show`展示图形。

 
   2. - **`scatter`（`matplotlib`库）**：用于绘制散点图，它可以很好地展示两个变量之间的关系或者数据的分布情况。例如，展示学生的身高和体重的关系：
```python
import matplotlib.pyplot as plt
height = [170, 175, 160, 180, 165]
weight = [60, 70, 55, 80, 65]
plt.scatter(height, weight)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()
```
   3. - **`bar`（`matplotlib`库）**：用于绘制柱状图，适合比较不同类别之间的数据大小。比如比较不同水果的销量：
```python
import matplotlib.pyplot as plt
fruits = ["Apple", "Banana", "Cherry"]
sales = [100, 150, 80]
plt.bar(fruits, sales)
plt.xlabel("Fruits")
plt.ylabel("Sales")
plt.show()
```
  4. - **`hist`（`matplotlib`库）**：用于绘制直方图，主要用于展示数据的分布。例如，展示一组学生考试成绩的分布：
```python
import matplotlib.pyplot as plt
import numpy as np
scores = np.random.normal(70, 10, 100)
plt.hist(scores, bins=10)
plt.xlabel("Scores")
plt.ylabel("Frequency")
plt.show()
```
  5. - **`pie`（`matplotlib`库）**：用于绘制饼图，用于展示各部分占总体的比例关系。例如，展示一个班级不同成绩等级的学生比例：
```python
import matplotlib.pyplot as plt
grades = ["A", "B", "C", "D", "F"]
percentages = [20, 30, 25, 15, 10]
plt.pie(percentages, labels=grades)
plt.show()
```
  6. - **`seaborn`库中的画图函数**：
  pairplot：用于绘制数据集中变量两两之间的关系图。如果有一个包含多个变量的数据集，`pairplot`可以同时展示多个散点图和直方图，方便观察变量之间的相关性和分布。例如，对于`iris`数据集（鸢尾花数据集）：
```python
import seaborn as sns
import pandas as pd
iris = pd.read_csv("iris.csv")
sns.pairplot(iris)
```
    7. - **`heatmap`**：用于绘制热力图，通常用于展示二维数据的密度或相关性。例如，展示一个相关系数矩阵的热力图：
   
```python
import seaborn as sns
import numpy as np
import pandas as pd
data = np.random.rand(5, 5)
corr_matrix = pd.DataFrame(data)
sns.heatmap(corr_matrix)
```