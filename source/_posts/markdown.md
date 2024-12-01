---
title: Markdown语法与外挂标签写法汇总
description: 🥧本文汇总Markdown格式
mathjax: true
tags:
  - Markdown
categories:
  - 演示
abbrlink: 2013454d
sticky: 2
swiper_index: 2
date: 2024-011-11 18:19:03
updated: 2024-11-11 22:00:00
---

# 1.Markdown语法自带格式
{% note info flat %}参考：[Markdown语法图文全面详解](https://blog.csdn.net/u014061630/article/details/81359144){% endnote %}
{% note warning flat %}注意：此页面偶尔会存在CSS冲突问题!{% endnote %}


## 1.1 代码块

{% tabs 分栏 %}

<!-- tab 示例源码 -->
```shell
\```shell
# VSCode终端
hexo clean; hexo s
hexo clean; hexo g; hexo d
git add .; git commit -m "npm publish"; npm version patch; 
git push

# Cmder终端
hexo clean && hexo s
hexo clean && hexo g && hexo d
git add . && git commit -m "npm publish" && npm version patch
git push
\```
```
<!-- endtab -->

<!-- tab 渲染演示 -->
```shell
# VSCode终端
hexo clean; hexo s
hexo clean; hexo g; hexo d
git add .; git commit -m "npm publish"; npm version patch; 
git push

# Cmder终端
hexo clean && hexo s
hexo clean && hexo g && hexo d
git add . && git commit -m "npm publish" && npm version patch
git push
```


<!-- endtab -->

{% endtabs %}


## 1.2 多级标题
{% tabs 分栏 %}

<!-- tab 示例源码 -->
```Markdown
# H1
## H2
### H3
#### H4
##### H5
###### H6
```
<!-- endtab -->

<!-- tab 渲染演示 -->
见本文章标题!
<!-- endtab -->

{% endtabs %}

## 1.3 文字样式

{% tabs 分栏 %}

<!-- tab 示例源码 -->
```Markdown
<u>下划线演示</u>

文字**加粗**演示

文字*斜体*演示

文本`高亮`演示

文本~~删除~~线演示

<font size = 5 >5号字</font>
<font face="黑体">黑体</font>
<font face="Arial" size="3" color="red">这是一段设置了字体、字号和颜色的文本</font>

此处的<font></font>被css取代
<!-- endtab -->

<!-- tab 渲染演示 -->
<u>下划线演示</u>

文字**加粗**演示

文字*斜体*演示

文本`高亮`演示

文本~~删除~~线演示

<font size = 5>5号字</font>
<font face="黑体">黑体</font>
<font color=blue>蓝色</font>

<table><tr><td bgcolor=MistyRose>这里的背景色是：MistyRosen，此处输入任意想输入的内容</td></tr></table>
<!-- endtab -->

{% endtabs %}

{% note info flat %}上述要点可参考:[【Markdown语法】字体颜色大小及文字底色设置](https://blog.csdn.net/qq_43732429/article/details/108034518)
{% endnote %}


## 1.4 引用

{% tabs 分栏 %}

<!-- tab 示例源码 -->
Markdown
>  Java
> 二级引用演示
> MySQL
> >外键
> >
> >事务
> >
> >**行级锁**(引用内部一样可以用格式)
> 
> ....
```
<!-- endtab -->

<!-- tab 渲染演示 -->
>  Java
> 二级引用演示
> MySQL
> >外键 
> >
> >事务
> >
> >**行级锁**(引用内部一样可以用格式)
> 
> ....
<!-- endtab -->

{% endtabs %}



## 1.5 分割线
{% tabs 分栏 %}

<!-- tab 示例源码 -->
```Markdown
---
***
```
<!-- endtab -->

<!-- tab 渲染演示 -->
---
***
<!-- endtab -->

{% endtabs %}




## 1.6 列表(*,+,-跟空格都可以)
### 1.6.1 无序列表
{% tabs 分栏 %}

<!-- tab 示例源码 -->
```Markdown
* Java
* Python
* ...

+ Java
+ Python
+ ...

- Java
- Python
- ...
```
<!-- endtab -->

<!-- tab 渲染演示 -->
* Java
* Python
* ...

+ Java
+ Python
+ ...

- Java
- Python
- ...
<!-- endtab -->

{% endtabs %}



### 1.6.2 有序列表
{% tabs 分栏 %}

<!-- tab 示例源码 -->
```Markdown
# 注意后面有空格
1. 
2. 
3. 
4. 
```
<!-- endtab -->

<!-- tab 渲染演示 -->
1. 
2. 
3. 
4. 
<!-- endtab -->

{% endtabs %}

## 1.7 图片

{% tabs 分栏 %}

<!-- tab 示例源码 -->
```Markdown
# 本地图片
<img src="/assets/pusheencode.webp" alt="示例图片" style="zoom:50%;" />
# 在线图片
![code](https://cdn.jsdelivr.net/gh/fomalhaut1998/markdown_pic/img/code.png)
```
<!-- endtab -->

<!-- tab 渲染演示 -->
本地图片:
<img src="/assets/pusheencode.webp" alt="示例图片" style="zoom:50%;" />
在线图片:
![code](https://cdn.jsdelivr.net/gh/fomalhaut1998/markdown_pic/img/code.png)
<!-- endtab -->

{% endtabs %}


## 1.8 表格

{% tabs 分栏 %}

<!-- tab 示例源码 -->
```Markdown
| 项目标号 | 资金     | 备注 |
| -------- | -------- | ---- |
| 1        | 100，000 | 无   |
| 2        | 200，000 | 无   |
| 3        | 300,600  | 重要 |
```
<!-- endtab -->

<!-- tab 渲染演示 -->
| 项目标号 | 资金     | 备注 |
| -------- | -------- | ---- |
| 1        | 100，000 | 无   |
| 2        | 200，000 | 无   |
| 3        | 300,600  | 重要 |
<!-- endtab -->

{% endtabs %}




## 1.9 公式

{% tabs 分栏 %}

<!-- tab 示例源码 -->
```Markdown
$$
\Gamma(z)=\int_0^\infty t^{z-1}e^{-t}dt.
$$
```
<!-- endtab -->

<!-- tab 渲染演示 -->
$$
\Gamma(z)=\int_0^\infty t^{z-1}e^{-t}dt.
$$
<!-- endtab -->

{% endtabs %}

