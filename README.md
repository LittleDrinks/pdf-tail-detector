# PDF 短尾段识别工具

`pdf_tail_detector.py` 面向 ICLR 常见的单栏论文 PDF。它按字符坐标重建文本行和段落，检查段落末字的右边界 `x1`：

依赖：`pdfplumber`、`pypdf`、`reportlab`。安装：`python -m pip install pdfplumber pypdf reportlab`。

```text
x1 < 页面宽度 * 2/3
```

命中的段落会直接打印到终端，并在复制出的标记 PDF 中标红；橙色虚线表示阈值位置。源 PDF 不会被修改。

## 使用

```sh
python pdf_tail_detector.py paper.pdf --output paper.tail-marked.pdf
```

也可以指定比例，例如更严格的 70%：

```sh
python pdf_tail_detector.py paper.pdf --ratio 0.70 --output paper.tail-marked.pdf
```

输出：

- 终端：每个命中的页码、栏、段落文本、末行、最后字符及坐标，并显示相邻行。
- 标记 PDF：原页面加上阈值线和红色命中框，默认文件名为 `<输入文件名>.tail-marked.pdf`。

## 规则边界

- 坐标使用 PDF 页面坐标，默认比较最后字符的右边界 `x1`，比用字符左边界更符合“最后一个字到达页面右侧”的视觉含义。
- 页眉、页脚、标题、图表说明、独立公式及常见算法块会被过滤；正文中的行内公式保留。
- 段落由垂直间距和缩进相近的正文行组成。工具只报告几何候选，需要结合标记 PDF 复核。扫描版 PDF 需要先有文字层。
- 当前阈值相对整页宽度定义；NeurIPS/NIPS 等双栏版式需要先定义每栏的正文边界，不能把本工具的结果直接用于右栏。

架构保持单模块：`scan_pdf(input_pdf, ratio)` 依次做字形到行、版面过滤、行到段落、候选判定；标记 PDF 和终端报告只消费检测结果。规则优先依据坐标与字体，少量文本模式识别算法环境和版面标签。

## 回归样本

`tests/fixtures/iclr/` 内置两篇 ICLR 2024 论文，覆盖单栏正文、公式、表格和算法环境：

- [What does automatic differentiation compute for neural networks?](https://proceedings.iclr.cc/paper_files/paper/2024/file/e8711daef520be07cb9852390c673de8-Paper-Conference.pdf)
- [Efficiently Computing Similarities to Private Datasets](https://proceedings.iclr.cc/paper_files/paper/2024/file/6fca3ed3c54ffeae947ae668a0841ab2-Paper-Conference.pdf)

测试只依赖仓库内样本，不在测试期间联网；它检查正文短尾仍能命中，算法步骤不会变成候选。

运行：`python -m pytest -q test_pdf_tail_detector.py`（另需安装 `pytest`）。
