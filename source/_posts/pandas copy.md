---
title: pandas
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
1. **Pandas的作用**
   - **数据处理与分析**：Pandas是一个强大的数据分析库，用于处理和分析结构化数据。它提供了高效的数据结构，如`Series`（一维数据）和`DataFrame`（二维数据，类似表格），可以方便地进行数据读取、清洗、转换、筛选等操作。
   - **数据导入和导出**：能够轻松地从各种数据源（如CSV文件、Excel文件、SQL数据库等）导入数据，并将处理后的数据导出为不同的格式。
   - **数据可视化辅助**：虽然不是专门的可视化工具，但可以和可视化库（如Matplotlib、Seaborn）紧密结合，为数据可视化提供良好的数据准备。

2. **代码演示**
   - **数据读取与查看**
     - 假设我们有一个CSV文件`data.csv`，内容如下：
       | Name | Age | Score |
       | ---- | --- | ----- |
       | Tom  | 20  | 80    |
       | Jerry| 22  | 85    |
       - 用Pandas读取这个文件并查看数据：
```python
import pandas as pd
# 读取CSV文件
data = pd.read_csv("data.csv")
# 查看前几行数据
print(data.head())
```
   - **数据筛选与计算**
     - 从数据中筛选出年龄大于20岁的人的信息，并计算他们的平均分数：
```python
# 筛选年龄大于20岁的人
filtered_data = data[data["Age"] > 20]
# 计算平均分数
average_score = filtered_data["Score"].mean()
print("平均分数:", average_score)
```
   - **数据合并与排序**
     - 假设我们还有一个新的数据文件`new_data.csv`，内容如下：
       | Name | Grade |
       | ---- | ----- |
       | Tom  | A     |
       | Jerry| B     |
       - 读取新数据并和原数据合并，然后按照分数进行排序：
```python
new_data = pd.read_csv("new_data.csv")
# 合并数据
merged_data = pd.merge(data, new_data, on="Name")
# 按照分数排序
sorted_data = merged_data.sort_values(by="Score", ascending=False)
print(sorted_data)
```