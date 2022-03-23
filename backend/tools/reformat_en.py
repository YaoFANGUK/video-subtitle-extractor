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
import sys
import wordsegment
import re
import os
import spacy
import subprocess

nlp = spacy.load("en_core_web_sm")

NLP_MAP_KEY_WORD_SEGMENT = "wordsegment"
NLP_MAP_KEY_SENTENCE = "sentence"


def reformat(path, bd_video_path=None):
    wordsegment.load()
    subs = pysrt.open(path)
    subs.save(f"{path}.bak", encoding='utf-8')
    verb_forms = ["I'm", "you're", "he's", "she's", "we're", "it's", "isn't", "aren't", "they're", "there's", "wasn't",
                  "weren't", "I've", "you've", "we've", "they've", "hasn't", "haven't", "I'd", "you'd", "he'd", "she'd",
                  "it'd", "we'd", "they'd", "doesn't", "don't", "didn't", "I'll", "you'll", "he'll", "she'll", "we'll",
                  "they'll", "there'll", "there'd", "can't", "couldn't", "daren't", "hadn't", "mightn't", "mustn't",
                  "needn't", "oughtn't", "shan't", "shouldn't", "usedn't", "won't", "wouldn't", "that's", "what's", "it'll"]
    verb_form_map = {}

    water_mark_map = {
        "扫码下载  ": "",
    }

    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'typoMap.json'), 'r') as load_f:
        typo_map = json.load(load_f)

    for verb in verb_forms:
        verb_form_map[verb.replace("'", "").lower()] = verb

    def format_seg_list(seg_list):
        new_seg = []
        for seg in seg_list:
            if seg in verb_form_map:
                new_seg.append([seg, verb_form_map[seg]])
            else:
                new_seg.append([seg])
        return new_seg

    def typo_fix(text):
        for k, v in typo_map.items():
            text = re.sub(re.compile(k, re.I), v, text)
        return text

    def water_mark_fix(text):
        for k, v in water_mark_map.items():
            text = text.replace(k, v)
        return text

    # 逆向过滤seg
    def remove_invalid_segment(seg, text):
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
        sub.text = typo_fix(sub.text)
        sub.text = water_mark_fix(sub.text)
        seg = wordsegment.segment(sub.text)
        if len(seg) == 1:
            seg = wordsegment.segment(re.sub(re.compile(f"(\ni)([^\\s])", re.I), "\\1 \\2", sub.text))
        seg = format_seg_list(seg)

        # 替换中文前的多个空格成单个空格, 避免中英文分行出错
        # TODO USE split_ch_and_eng
        sub.text = re.sub(' +([\\u4e00-\\u9fa5])', ' \\1', sub.text)
        # 中英文分行
        sub.text = sub.text.replace("  ", "\n")
        if bool(re.search('[\\u4e00-\\u9fa5]', 'sub.text')):
            ch, eng = split_ch_and_eng(sub.text)
            sub.text = join_ch_and_eng(ch, eng)
        lines = []
        remain = sub.text
        seg = remove_invalid_segment(seg, sub.text)
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
        # again
        ss = typo_fix(ss)
        # 非大写字母的大写字母前加空格
        ss = re.sub("([^\\sA-Z\\-])([A-Z])", "\\1 \\2", ss)
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
        # 删除-左侧空格
        ss = re.sub("([A-Za-z0-9]) (\\-[A-Za-z0-9])", '\\1\\2', ss)
        # 删除%左侧空格
        ss = re.sub("([A-Za-z0-9]) %", '\\1%', ss)
        # 结尾·改成.
        ss = re.sub('·$', '.', ss)
        # 移除Dr.后的空格
        ss = re.sub(r'\bDr\. *\b', "Dr.", ss)
        ss = ss.replace("\n\n", "\n")
        sub.text = ss.strip()
    do_original_eng_sub_match_and_replace(subs, bd_video_path)
    subs.save(path, encoding='utf-8')


# 用mkv中的英语字幕修复subs中的英语字幕错误
def do_original_eng_sub_match_and_replace(subs, bd_video_path):
    srt_path = extract_subtitle_from_video(bd_video_path)
    if srt_path is None:
        srt_path = extract_subtitle_from_video_by_ffprobe(bd_video_path)
    if srt_path is None:
        print(f"Error: No English subtitle from '{bd_video_path}'")
        return
    eng_subs = pysrt.open(srt_path)
    eng_subs_len = len(eng_subs)
    eng_subs_nlp_map = {}
    for i in range(0, eng_subs_len):
        eng_subs[i].text = filter_original_eng_sub_text(eng_subs[i].text)
        nlps = []
        text = eng_subs[i].text.strip()
        text_parts = list(filter(len, re.split(r'\- +| +\-|^- *|[~\!@#$%\^&\*\(\)\+\.]- *', text)))
        if len(text_parts) > 1:
            text_parts.insert(0, text)
        else:
            text_parts = list([text])
        text_parts_len = len(text_parts)

        for idx, text in enumerate(text_parts):
            splitted = idx != 0
            primary = splitted == False or idx == text_parts_len - 1
            nlps.append({
                'nlp':  nlp(' '.join(wordsegment.segment(text)).lower()),
                'text': text.strip(),
                'type': NLP_MAP_KEY_WORD_SEGMENT,
                'primary': primary,
                'splitted': splitted
            })
            nlps.append({
                'nlp':  nlp(nlp_sentence_clean(text).lower()),
                'text': text.strip(),
                'type': NLP_MAP_KEY_SENTENCE,
                'primary': primary,
                'splitted': splitted
            })

        eng_subs_nlp_map[i] = nlps

    eng_subs.save(srt_path, encoding='utf-8')

    for sub in subs:
        ch, eng = split_ch_and_eng(sub.text)
        original_sub_text = sub.text
        sub.text = eng
        similar_sub, similar_sub_score, selected = find_similar_sub(eng_subs_nlp_map, eng_subs, sub)
        if selected:
            sub.text = join_ch_and_eng(ch, similar_sub.text)
        else:
            sub.text = original_sub_text
    # os.remove(srt_path)

def extract_subtitle_from_video_by_ffprobe(bd_video_path):
    if bd_video_path is None:
        return None
    srt_path = os.path.join(os.path.splitext(bd_video_path)[0] + '.eng.srt')
    try:
        output = subprocess.check_output('ffprobe -v error -show_entries stream=index,codec_name,codec_type:stream_tags=language,title -select_streams s  -of json "{0}"'.format(bd_video_path), shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        return None
    info = json.loads(output)
    if 'streams' not in info:
        return None
    subtitle_index = None
    for s in info['streams']:
        if 'tags' not in s:
            continue
        if 'title' in s['tags'] and s['tags']['title'] == 'SDH':
            continue
        subtitle_index = s['index']
    if subtitle_index is None:
        return None
    os.system('ffmpeg -y -i "{0}" -map "0:2" -c:s srt "{1}"'.format(bd_video_path, srt_path))
    if not os.path.isfile(srt_path):
        return None
    if os.path.getsize(srt_path) <= 0:
        return None
    return srt_path

def extract_subtitle_from_video(bd_video_path):
    if bd_video_path is None:
        return None
    srt_path = os.path.join(os.path.splitext(bd_video_path)[0] + '.eng.srt')
    os.system('ffmpeg -y -i "{0}" -map "0:s:m:title:English" -c:s srt "{1}"'.format(bd_video_path, srt_path))
    if not os.path.isfile(srt_path):
        return None
    if os.path.getsize(srt_path) <= 0:
        return None
    return srt_path


def nlp_sentence_clean(text):
    text = text.replace("'", "")
    # 删除非字母数字空格
    text = re.sub('[^a-zA-Z0-9\\s]', ' ', text)
    text = text.replace("  ", " ")
    return text


last_correction_index = -1


def find_similar_sub(eng_subs_nlp_map, eng_subs, sub):
    selected = False
    global last_correction_index
    sub_eng_part = re.sub('.+[\r\n](.+)$', "\\1", sub.text)
    sub_eng_part = re.sub('(.*: )', '', sub_eng_part)
    sub_eng_part = sub_eng_part.strip()
    if len(sub_eng_part) <= 0:
        print("Empty eng line at index: " + str(sub.index))
        return None, 0, False
    sub_eng_part_nlp = {
        NLP_MAP_KEY_WORD_SEGMENT: nlp(' '.join(wordsegment.segment(sub_eng_part)).lower()),
        NLP_MAP_KEY_SENTENCE: nlp(nlp_sentence_clean(sub_eng_part).lower()),
    }
    # print('sub_eng_part_nlp', sub_eng_part_nlp)
    eng_subs_part = eng_subs.slice(starts_after=sub.start + {'minutes': -1}, ends_before=sub.end + {'minutes': 1})

    if len(eng_subs_part) <= 0:
        print("Out of range")
        return None, 0, False

    eng_subs_part_score_list = []
    eng_subs_part_index_list = []
    eng_subs_part_nlp_list = []
    i = 0
    for eng_sub_part in eng_subs_part:
        if eng_sub_part.index not in eng_subs_nlp_map:
            eng_subs_part_score_list.append(0)
            eng_subs_part_index_list.append(eng_sub_part.index)
            eng_subs_part_nlp_list.append({
                'primary': True,
                'splitted': False
            })
            continue
        nlps = eng_subs_nlp_map[eng_sub_part.index]
        scores = []
        for n in nlps:
            scores.append(n['nlp'].similarity(sub_eng_part_nlp[n['type']]))
        max_score = max(scores)
        eng_subs_part_score_list.append(max_score)
        eng_subs_part_index_list.append(eng_sub_part.index)
        eng_subs_part_nlp_list.append(nlps[scores.index(max_score)])
        i = i + 1

    eng_subs_part_score_list_max = max(eng_subs_part_score_list)
    eng_subs_part_score_list_max_index = eng_subs_part_score_list.index(eng_subs_part_score_list_max)

    # print("max score", eng_subs_part_score_list_max, "index", eng_subs_part_score_list_max_index)

    if eng_subs_part_score_list_max >= 0.9:
        # 如果权重>=0.9, 将他作为下一行字幕的基准
        if eng_subs_part_nlp_list[eng_subs_part_score_list_max_index]['primary']: # 如果是primary才能作为基准, 拆分的不算
            last_correction_index = eng_subs_part_index_list[eng_subs_part_score_list_max_index]
        else:
            last_correction_index = -1
        selected = True
    else:
        # 如果权重<0.9, 用上一行作为基准+1行测算权重
        # TODO 实际上可以更复杂, 比如+2+3+4+5+...
        if last_correction_index != -1:
            # 看看有没有按顺序
            diff_index = eng_subs_part_index_list[eng_subs_part_score_list_max_index] - last_correction_index
            if diff_index not in [1, 2] and last_correction_index + 1 in eng_subs_part_index_list:
                # 没有按顺序...
                eng_subs_part_score_list_correction_index = eng_subs_part_index_list.index(last_correction_index + 1)
                eng_subs_part_score_list_max_correction = eng_subs_part_score_list[
                    eng_subs_part_score_list_correction_index]
                # 如果测算的权重变化在合理的范围, 比如0.1
                diff_score = eng_subs_part_score_list_max_correction - eng_subs_part_score_list_max
                if abs(diff_score) < 0.1:
                    selected = True
                    eng_subs_part_score_list_max_index = eng_subs_part_score_list_correction_index
                    eng_subs_part_score_list_max = eng_subs_part_score_list[eng_subs_part_score_list_max_index]
                    # print("fix score", eng_subs_part_score_list_max, "index", eng_subs_part_score_list_max_index,
                    #       'diff_score', diff_score, 'diff_index', diff_index)
    try:
        similar_sub = eng_subs[eng_subs_part_index_list[eng_subs_part_score_list_max_index]]
        if eng_subs_part_nlp_list[eng_subs_part_score_list_max_index]['splitted']: # 如果是分割的字幕
            similar_sub.text = eng_subs_part_nlp_list[eng_subs_part_score_list_max_index]['text']
        return similar_sub, eng_subs_part_score_list_max, selected
    except IndexError:
        print("Index out of range")
        return None, 0, False


def split_ch_and_eng(text):
    chs = []
    eng = ''
    while len(text) > 0:
        ch, text = re.search('(.*[\\u4e00-\\u9fa5]+.*[\r\n]*)?([\\S\\s]*)', text).groups()
        if ch is not None:
            chs.append(ch.strip())
        if ch is None:
            eng = text.strip()
            break
    eng = re.sub('[\r\n]', ' ', eng)
    return ' '.join(chs), eng


def join_ch_and_eng(ch, eng):
    if len(ch) <= 0:
        return eng
    if len(eng) <= 0:
        return ch
    return f"{ch}\n{eng}"


def filter_original_eng_sub_text(text):
    # 删掉xml标签
    text = re.sub("</?[^>]+/?>", '', text)
    # 删掉结尾(语气词)
    text = re.sub(" ?\\([^\\(]+?\\)$", '', text)
    # 删掉(声音等):
    text = re.sub("^\\-?\\(.+?\\):? ?[\r\n]*", '', text)
    # 删掉(一些神奇的内容)
    text = re.sub(" *\\(.+?\\)", '', text)
    # 删掉人名:
    text = re.sub("^[A-Za-z0-9\']+:[\r\n]+", '', text)
    # 清理多余字符
    text = text.strip()
    # 所有字幕整理成一行
    text = re.sub('[\r\n]', ' ', text)
    # 清理多余空格
    text = re.sub(' +', ' ', text)
    # 再清理一遍多余字符
    text = text.strip()
    return text


if __name__ == '__main__':
    reformat(sys.argv[1], sys.argv[2])
