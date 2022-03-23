# -*- coding: UTF-8 -*-
"""
@author: eritpchy
@file  : reformat.py
@time  : 2021/12/17 15:43
@desc  : 将连起来的英文单词切分
"""
import json
import os

import pysrt
import wordsegment
import re


def reformat(path):
    wordsegment.load()
    subs = pysrt.open(path)
    verb_forms = ["I'm", "you're", "he's", "she's", "we're", "it's", "isn't", "aren't", "they're", "there's", "wasn't",
                 "weren't", "I've", "you've", "he's", "she's", "it's", "we've", "they've", "there's", "hasn't",
                 "haven't", "I'd", "you'd", "he'd", "she'd", "it'd", "we'd", "they'd", "doesn't", "don't", "didn't",
                 "I'll", "you'll", "he'll", "she'll", "we'll", "they'll", "there'll", "I'd", "you'd", "he'd", "she'd",
                 "it'd", "we've", "they'd", "there'd", "there'd", "can't", "couldn't", "daren't", "hadn't", "mightn't",
                 "mustn't", "needn't", "oughtn't", "shan't", "shouldn't", "usedn't", "won't", "wouldn't", "that's",
                 "what's", "haven't"]
    verb_form_map = {}

    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'typoMap.json'), 'r') as load_f:
        typo_map = json.load(load_f)

    for verb in verb_forms:
        verb_form_map[verb.replace("'", "").lower()] = verb

    def formatSegList(seg_list):
        new_seg = []
        for seg in seg_list:
            if seg in verb_form_map:
                new_seg.append([seg, verb_form_map[seg]])
            else:
                new_seg.append([seg])
        return new_seg

    def typoFix(text):
        for k, v in typo_map.items():
            text = re.sub(re.compile(k, re.I), v, text)
        return text

    # 逆向过滤seg
    def removeInvalidSegment(seg, text):
        seg_len = len(seg)
        span = None
        new_seg = []
        for i in range(seg_len - 1, -1, -1):
            s = seg[i]
            if len(s) > 1:
                regex = re.compile(f"({s[0]}|{s[1]})", re.I)
            else:
                regex = re.compile(f"({s[0]})", re.I)
            try:
                ss = [(i) for i in re.finditer(regex, text)][-1]
            except IndexError:
                ss = None
            if ss is None:
                continue
            text = text[:ss.span()[0]]
            if span is None:
                span = ss.span()
                new_seg.append(s)
                continue
            if span > ss.span():
                new_seg.append(s)
                span = ss.span()
        return list(reversed(new_seg))

    for sub in subs:
        sub.text = typoFix(sub.text)
        seg = wordsegment.segment(sub.text)
        if len(seg) == 1:
            seg = wordsegment.segment(re.sub(re.compile(f"(\ni)([^\\s])", re.I), "\\1 \\2", sub.text))
        seg = formatSegList(seg)

        # 替换中文前的多个空格成单个空格, 避免中英文分行出错
        sub.text = re.sub(' +([\\u4e00-\\u9fa5])', ' \\1', sub.text)
        # 中英文分行
        sub.text = sub.text.replace("  ", "\n")
        lines = []
        remain = sub.text
        seg = removeInvalidSegment(seg, sub.text)
        seg_len = len(seg)
        for i in range(0, seg_len):
            s = seg[i]
            global regex
            if len(s) > 1:
                regex = re.compile(f"(.*?)({s[0]}|{s[1]})", re.I)
            else:
                regex = re.compile(f"(.*?)({s[0]})", re.I)
            ss = re.search(regex, remain)
            if ss is None:
                if i == seg_len - 1:
                    lines.append(remain.strip())
                continue

            lines.append(remain[:ss.span()[1]].strip())
            remain = remain[ss.span()[1]:].strip()
            if i == seg_len - 1:
                lines.append(remain)
        if seg_len > 0:
            ss = " ".join(lines)
        else:
            ss = remain
        # 非大写字母的大写字母前加空格
        ss = re.sub("([^\\sA-Z])([A-Z])", "\\1 \\2", ss)
        # 删除重复空格
        ss = ss.replace("  ", " ")
        ss = ss.replace("。", ".")
        # 删除,?!,前的多个空格
        ss = re.sub(" *([\\.\\?\\!\\,])", "\\1", ss)
        # 删除'的前后多个空格
        ss = re.sub(" *([\\']) *", "\\1", ss)
        # 删除换行后的多个空格, 通常时第二行的开始的多个空格
        ss = re.sub('\n\\s*', '\n', ss)
        # 删除开始的多个空格
        ss = re.sub('^\\s*', '', ss)
        # 结尾·改成.
        ss = re.sub('·$', '.', ss)
        ss = ss.replace(" Dr. ", " Dr.")
        ss = ss.replace("\n\n", "\n")
        sub.text = ss.strip()
    subs.save(path, encoding='utf-8')


if __name__ == '__main__':
    path = "/home/yao/Videos/null.srt"
    reformat(path)

