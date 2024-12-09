---
title: python内置报错
date: 2024.12.4
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

1. **SyntaxError（语法错误）**
   - 描述：当Python解释器遇到不符合Python语法规则的代码时会抛出此错误。例如，遗漏了冒号、括号不匹配、缩进错误等。比如下面的代码会引发SyntaxError：
   ```python
   if True
       print("Hello")
   ```
   - 原因是`if`语句后缺少冒号。正确的应该是`if True:`。
2. **TypeError（类型错误）**
   - 描述：当操作或函数应用于不适当类型的对象时会出现。例如，对字符串和整数进行不恰当的加法运算，像`"abc"+1`就会引发TypeError，因为Python不知道如何将一个字符串和一个整数相加。
3. **NameError（名称错误）**
   - 描述：当尝试访问一个未定义的变量或函数时会抛出。例如：
   ```python
   print(a)
   ```
   - 如果`a`没有在之前定义，就会出现NameError。
4. **IndexError（索引错误）**
   - 描述：当尝试访问序列（如列表、元组、字符串）中不存在的索引时会出现。例如，一个长度为3的列表`my_list = [1, 2, 3]`，如果执行`my_list[3]`，就会引发IndexError，因为索引是从0开始的，这个列表最大索引为2。
5. **KeyError（键错误）**
   - 描述：当在字典中访问不存在的键时会抛出。例如，有一个字典`my_dict = {"a": 1, "b": 2}`，如果执行`my_dict["c"]`，就会引发KeyError，因为字典中没有键`c`。
6. **ValueError（值错误）**
   - 描述：当传入一个函数的参数类型正确，但值不合适时会出现。例如，使用`int()`函数将一个非数字字符串（如`"abc"`）转换为整数时，就会引发ValueError。
7. **ZeroDivisionError（零除错误）**
   - 描述：当尝试将一个数除以零时会抛出。例如，`1/0`会引发ZeroDivisionError。