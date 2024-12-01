---
title: css
date: 2024.11.14
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
sticky: 1
---

# CSS样式

**`*`（通配选择器）**：

**`margin`**：
- 设置元素外边距，控制与周围元素间隔，代码中设为0清除默认外边距。

**`padding`**：
- 设置元素内边距，即内容与边框距离，代码中设为0清除默认内边距。

**`background`（`body`元素）**：
- 设置元素背景，`body`中用`url(4.jpeg)`指定图片路径且`background-size: cover`让图片自适应铺满（因`body`宽100%、高100vh，铺满视口）。

**`width`（`body`元素）**：
- 设置元素宽度，`body`中设为100%与浏览器视口宽度相同。

**`height`（`body`元素）**：
- 设置元素高度，`body`中设为100vh即视口高度100%，填满视口。

**`display: flex`（`header`元素）**：
- 把`header`设为弹性容器，内部子元素按弹性布局规则排列，方便水平或垂直方向对齐、分布。

**`align-items`（`header`元素）**：
- 在`header`弹性容器中，控制子元素在交叉轴（垂直主轴）上居中对齐。

**`justify-content`（`header`元素）**：
- 在`header`弹性容器中，控制子元素在主轴（默认水平，可改）上均匀分布且两端留间隔。

**`background-color`（`header`元素）**：
- 设置`header`元素背景色为半透明黑色`rgba(0, 0, 0, 0.2)`。

**`backdrop-filter: blur(10px)`（`header`元素）**：
- 对`header`背后内容应用模糊效果，半径10px，营造层次感和视觉焦点。

**`height`（`header`元素）**：
- 设置`header`元素高度为80px。

**`color`（`loge`、`nav a`、`user-form-groud a`等元素）**：
- 设置元素内部文本颜色为白色。

**`text-decoration`（`nav a`、`user-form-groud a`等元素）**：
- 控制文本无下划线等装饰效果。

**`font-weight`（`nav a`、`user-form-groud a`等元素）**：
- 设置文本字体粗细为600，呈现较粗效果。

**`font-size`（`nav a`、`user-form-groud a`等元素）**：
- 设置文本字体大小为18px。

**`padding`（`nav a`等元素）**：
- 在`nav a`等元素设置内边距，如`padding: 8px 16px`，增加文本与边框间隔、点击区域和视觉效果。

**`transition`（`nav a`、`signup`等元素）**：
- 设置元素属性变化时过渡效果，设过渡时间300ms，属性变化有平滑过渡。

**`border-radius`（`nav a`、`signup`等元素）**：
- 设置元素边框圆角半径为99px，使边框圆润、外观更美观。

**`display: none`（`nav.loge`、`#check`、`user-form-groud label`等元素）**：
- 隐藏元素，页面布局中不占空间。

**`position: absolute`（`nav`元素媒体查询中、`adjust`元素）**：
- 设置元素为绝对定位，脱离文档流，依指定坐标（如`top`、`left`等）定位。

**`flex-direction: column`（`nav`元素媒体查询中）**：
- 在`nav`弹性容器（媒体查询中），改主轴方向为垂直，子元素按弹性布局垂直排列。

**`gap`（`nav`元素媒体查询中）**：
- 在`nav`弹性容器（媒体查询中），设置子元素间隔为16px。

**`box-shadow`（`nav`元素媒体查询中）**：
- 给`nav`元素（媒体查询中）添加阴影效果，如`box-shadow: 10px 0px 10px rgba(0, 0, 0, 0.2)`。

**`z-index`（`adjust`元素）**：