# PDF 短尾段识别工具

`pdf_tail_detector.py` 面向 ICLR、NeurIPS/NIPS 常见的单栏或双栏论文 PDF。它按字符坐标重建文本行和段落，检查每个段落最后一行最后一个可见字符的右边界 `x1`：

依赖：`pdfplumber`、`pypdf`、`reportlab`。建议使用 Codex 随附的 Python，或先执行 `python -m pip install pdfplumber pypdf reportlab`。

```text
x1 < 页面宽度 * 2/3
```

命中的段落会直接打印到终端，并在复制出的标记 PDF 中标红；橙色虚线表示阈值位置。源 PDF 不会被修改。

## 使用

```powershell
$py = "C:\Users\q2635\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py .\pdf_tail_detector.py .\paper.pdf --template iclr --output .\paper.tail-marked.pdf
```

也可以指定比例，例如更严格的 70%：

```powershell
& $py .\pdf_tail_detector.py .\paper.pdf --ratio 0.70 --output .\paper.tail-marked.pdf
```

输出：

- 终端：每个命中的页码、栏、段落文本、末行、最后字符及坐标，并显示相邻行。
- 标记 PDF：原页面加上阈值线和红色命中框，默认文件名为 `<输入文件名>.tail-marked.pdf`。

## 规则边界

- 坐标使用 PDF 页面坐标，默认比较最后字符的右边界 `x1`，比用字符左边界更符合“最后一个字到达页面右侧”的视觉含义。
- 页眉、页脚、页码、标题、节标题、图表标题、独立公式和常见伪代码行会被过滤；正文中的行内公式保留。
- 段落由同一栏中垂直间距较小、缩进相近的正文行组成。模板变体或扫描版 PDF 可能没有可用的文字坐标，此时需要先 OCR，或改用带字符框的文本层。
- 工具只报告几何上的候选，不替代人工判断；公式、引用串、特殊字距和双栏跨栏内容建议在标记 PDF 中复核。
- 标题、图表/表格标题、独立公式、常见伪代码关键词、等宽或明显较小的算法字体会被过滤。模板自定义宏、扫描版 PDF、复杂表格仍可能产生候选，需要看红框和终端上下文。

## 已用示例

可以直接对 OpenReview 的 ICLR 2027 论文运行：

```powershell
& $py .\pdf_tail_detector.py "C:\Users\q2635\Downloads\45321_Certified_End_to_End_Gra.pdf" --template iclr --output .\45321.tail-marked.pdf
```
