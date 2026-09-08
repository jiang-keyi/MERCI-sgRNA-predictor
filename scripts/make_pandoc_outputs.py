# -*- coding: utf-8 -*-
"""从 OUP 投稿版 tex 派生一个 pandoc 友好的纯 LaTeX 版本，并转换 docx/html。
用法: python make_pandoc_outputs.py
输出: paper/MERCI_paper_plain.tex, MERCI_paper.docx, MERCI_paper.html
"""
import os
import re
import subprocess

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PAPER = os.path.join(BASE, "paper")
SRC = os.path.join(PAPER, "MERCI_paper.tex")
PLAIN = os.path.join(PAPER, "MERCI_paper_plain.tex")
PANDOC = r"E:\claude\deepmeans_repo\_dl_tmp\pandoc\pandoc-3.1.11.1\pandoc.exe"

s = open(SRC, encoding="utf-8").read()

# 1) 头部：换成普通 article 类
head = r"""% -*- coding: utf-8 -*-
% MERCI — pandoc 友好的纯 LaTeX 版本（由 make_pandoc_outputs.py 自动生成）
\documentclass[11pt]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage[hidelinks]{hyperref}
\graphicspath{{../figures/}}
\def\FILLMAIN#1{#1}
\def\FILLABL#1{#1}
\def\FILLFAM#1{#1}
\def\FILLFUND{}
\begin{document}
"""
body = s[s.index(r"\begin{document}") + len(r"\begin{document}"):]
# 去掉 OUP 专用 front-matter 命令行
for pat in [r"\\journaltitle\{[^}]*\}",
            r"\\DOI\{[^}]*\}",
            r"\\copyrightyear\{[^}]*\}",
            r"\\pubyear\{[^}]*\}",
            r"\\vol\{[^}]*\}",
            r"\\issue\{[^}]*\}",
            r"\\access\{[^}]*\}",
            r"\\appnotes\{[^}]*\}",
            r"\\firstpage\{[^}]*\}",
            r"\\received\{[^}]*\}\{[^}]*\}\{[^}]*\}",
            r"\\revised\{[^}]*\}\{[^}]*\}\{[^}]*\}",
            r"\\accepted\{[^}]*\}\{[^}]*\}\{[^}]*\}",
            r"\\corresp\[[^\]]*\]\{(?:[^{}]|\{[^{}]*\})*\}",
            r"\\keywords\{[^}]*\}"]:
    body = re.sub(pat, "", body)
# \title[...]{...} -> \title{...}; \author[...]{...} -> \author{...}
body = body.replace(r"\title[Short Article Title]{", r"\title{")
body = body.replace(r"\author[1]{Keyi Jiang}", r"\author{Keyi Jiang\textsuperscript{1}, Dong He\textsuperscript{2,*}}")
body = body.replace(r"\author[2,$\ast$]{Dong He}", "")
# address 块删除（两段连续，止于 \corresp；用区间匹配避免嵌套花括号问题）
body = re.sub(r"\\address\[1\].*?\n\\corresp", lambda m: r"\corresp", body, flags=re.S)
# abstract: 去掉 \abstract{ 的包裹，直接放原文文本；并去掉 \maketitle 后的 \begin{abstract}
body = body.replace(r"\abstract{", r"\begin{abstract}" + "\n")
body = body.replace("online.}", "online.\n\\end{abstract}")
# 删掉 \keywords 段（含嵌套花括号的已经在上面的 pat 循环删掉，这里兜底）
body = re.sub(r"\\keywords\{[^}]*\}", "", body)
# bibitem 自定义标签 -> 标准 \bibitem{key}（pandoc 兼容）
body = re.sub(r"\\bibitem\[[^\]]*\]\{", r"\\bibitem{", body)
out = head + body
open(PLAIN, "w", encoding="utf-8").write(out)
print("wrote", PLAIN)

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        print("STDERR:", r.stderr[:800])
    return r

# 编译验证纯版（可选，用于确认结构有效）
r = run([r"E:\texlive\2026\bin\windows\pdflatex.exe",
         "-interaction=nonstopmode", "-halt-on-error", PLAIN])
print("plain pdflatex rc =", r.returncode)

# pandoc -> docx
r = run([PANDOC, PLAIN, "-o", os.path.join(PAPER, "MERCI_paper.docx"),
         "--resource-path=" + os.path.join(BASE, "figures"),
         "--mathml"])
print("pandoc docx rc =", r.returncode, "size =",
      os.path.getsize(os.path.join(PAPER, "MERCI_paper.docx")))

# pandoc -> html（内嵌图片资源，独立可用）
r = run([PANDOC, PLAIN, "-o", os.path.join(PAPER, "MERCI_paper.html"),
         "--resource-path=" + os.path.join(BASE, "figures"),
         "--mathml", "--standalone", "--embed-resources"])
print("pandoc html rc =", r.returncode, "size =",
      os.path.getsize(os.path.join(PAPER, "MERCI_paper.html")))
