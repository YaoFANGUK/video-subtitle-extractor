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
nlp = spacy.load("en_core_web_sm")

NLP_MAP_KEY_WORD_SEGMENT = "wordsegment"
NLP_MAP_KEY_SENTENCE = "sentence"

def reformat(path, bd_video_path=None):
    wordsegment.load()
    subs = pysrt.open(path)
    subs.save(f"{path}.bak", encoding='utf-8')
    verb_forms = ["I'm", "you're", "he's", "she's", "we're", "it's", "isn't", "aren't", "they're", "there's", "wasn't",
                 "weren't", "I've", "you've", "he's", "she's", "it's", "we've", "they've", "there's", "hasn't",
                 "haven't", "I'd", "you'd", "he'd", "she'd", "it'd", "we'd", "they'd", "doesn't", "don't", "didn't",
                 "I'll", "you'll", "he'll", "she'll", "we'll", "they'll", "there'll", "I'd", "you'd", "he'd", "she'd",
                 "it'd", "we've", "they'd", "there'd", "there'd", "can't", "couldn't", "daren't", "hadn't", "mightn't",
                 "mustn't", "needn't", "oughtn't", "shan't", "shouldn't", "usedn't", "won't", "wouldn't", "that's",
                 "what's", "haven't"]
    verb_form_map = {}

    waterMarkMap = {
        "扫码下载  ": "",
    }

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
            text = text.replace(k, v)
        return text

    def waterMarkFix(text):
        for k,v in waterMarkMap.items():
            text = text.replace(k, v)
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
        sub.text = waterMarkFix(sub.text)
        seg = wordsegment.segment(sub.text)
        if len(seg) == 1:
            seg = wordsegment.segment(re.sub(re.compile(f"(\ni)([^\\s])", re.I), "\\1 \\2", sub.text))
        seg = formatSegList(seg)

        sub.text = re.sub(' +([\\u4e00-\\u9fa5])', ' \\1', sub.text)
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
        ss = re.sub("([^\\sA-Z])([A-Z])", "\\1 \\2", ss)
        ss = ss.replace("  ", " ")
        ss = ss.replace("。", ".")
        ss = re.sub(" *([\\.\\?\\!\\,])", "\\1", ss)
        ss = re.sub(" *([\\']) *", "\\1", ss)
        ss = re.sub('\n\\s*', '\n', ss)
        ss = re.sub('^\\s*', '', ss)
        ss = re.sub('·$', '.', ss)
        ss = ss.replace(" Dr. ", " Dr.")
        ss = ss.replace("\n\n", "\n")
        sub.text = ss.strip()
    doOriginalEngSubMatch(subs, bd_video_path)
    subs.save(path, encoding='utf-8')

def doOriginalEngSubMatch(subs, bd_video_path):
    srtPath = extractSubtitleFromVideo(bd_video_path)
    if srtPath is None:
        print(f"Error: No English subtitle from '{bd_video_path}'")
        return
    engSubs = pysrt.open(srtPath)
    engSubsLen = len(engSubs)
    engSubsNlpMap = {}
    for i in range(0, engSubsLen):
        engSubs[i].text = filterOriginalEngSubText(engSubs[i].text)
        engSubsNlpMap[i] = {
            NLP_MAP_KEY_WORD_SEGMENT: nlp(' '.join(wordsegment.segment(engSubs[i].text)).lower()),
            NLP_MAP_KEY_SENTENCE: nlp(nlpSentenceClean(engSubs[i].text).lower()),
        }

    for sub in subs:
        ch, eng = splitChAndEng(sub.text)
        originalSubText = sub.text
        sub.text = eng
        similarSub, similarSubScore, selected = findSimilarSub(engSubsNlpMap, engSubs, sub)
        if selected:
            sub.text = joinChAndEng(ch, similarSub.text)
        else:
            sub.text = originalSubText
    os.remove(srtPath)

def extractSubtitleFromVideo(bd_video_path):
    if bd_video_path is None:
        return None
    srtPath = f"{bd_video_path}.eng.srt"
    os.system('ffmpeg -y -i "{0}" -map "0:s:m:title:English" -c:s srt "{1}"'.format(bd_video_path, srtPath))
    if os.path.isfile(srtPath) == False:
        return None
    if os.path.getsize(srtPath) <= 0:
        return None
    return srtPath


def nlpSentenceClean(text):
    text = text.replace("'", "")
    text = re.sub('[^a-zA-Z0-9\s]', ' ', text)
    text = text.replace("  ", " ")
    return text

lastCorrectionIndex = -1
def findSimilarSub(engSubsNlpMap, engSubs, sub):
    selected = False
    global lastCorrectionIndex
    subEngPart = re.sub('.+[\r\n](.+)$', "\\1", sub.text)
    subEngPart = re.sub('(.*: )', '', subEngPart)
    subEngPartNlp = {
        NLP_MAP_KEY_WORD_SEGMENT: nlp(' '.join(wordsegment.segment(subEngPart)).lower()),
        NLP_MAP_KEY_SENTENCE: nlp(nlpSentenceClean(subEngPart).lower()),
    }
    # print('subEngPartNlp', subEngPartNlp)
    engSubsPart = engSubs.slice(starts_after=sub.start + {'minutes': -1}, ends_before=sub.end + {'minutes': 1})

    if len(engSubsPart) <= 0:
        print("Out of range")
        return None, 0, False

    engSubsPartScoreList = []
    engSubsPartIndexList = []
    i = 0
    for engSubPart in engSubsPart:
        if engSubPart.index not in engSubsNlpMap:
            engSubsPartScoreList.append(0)
            engSubsPartIndexList.append(engSubPart.index)
            continue
        engSubPartTextWordSegmentNlp = engSubsNlpMap[engSubPart.index][NLP_MAP_KEY_WORD_SEGMENT]
        wordSegmentScore = engSubPartTextWordSegmentNlp.similarity(subEngPartNlp[NLP_MAP_KEY_WORD_SEGMENT])

        engSubPartTextSentenceNlp = engSubsNlpMap[engSubPart.index][NLP_MAP_KEY_SENTENCE]
        sentenceScore = engSubPartTextSentenceNlp.similarity(subEngPartNlp[NLP_MAP_KEY_SENTENCE])

        # print('index', engSubPart.index, i, "engSubPartTextWordSegmentNlp", engSubPartTextWordSegmentNlp, "wordSegmentScore", wordSegmentScore, "sentenceScore", sentenceScore)

        engSubsPartScoreList.append(max(wordSegmentScore, sentenceScore))
        engSubsPartIndexList.append(engSubPart.index)
        i = i + 1

    engSubsPartScoreListMax = max(engSubsPartScoreList)
    engSubsPartScoreListMaxIndex = engSubsPartScoreList.index(engSubsPartScoreListMax)

    # print("max score", engSubsPartScoreListMax, "index", engSubsPartScoreListMaxIndex)

    if (engSubsPartScoreListMax >= 0.9):
        lastCorrectionIndex = engSubsPartIndexList[engSubsPartScoreListMaxIndex]
        selected = True
    else:
        if lastCorrectionIndex != -1:
            diffIndex = engSubsPartIndexList[engSubsPartScoreListMaxIndex] - lastCorrectionIndex
            if diffIndex not in [1,2] and lastCorrectionIndex + 1 in engSubsPartIndexList:
                engSubsPartScoreListCorrectionIndex = engSubsPartIndexList.index(lastCorrectionIndex + 1)
                engSubsPartScoreListMaxCorrection = engSubsPartScoreList[engSubsPartScoreListCorrectionIndex]
                diffScore = engSubsPartScoreListMaxCorrection - engSubsPartScoreListMax
                if abs(diffScore) < 0.06:
                    selected = True
                    engSubsPartScoreListMaxIndex = engSubsPartScoreListCorrectionIndex
                    engSubsPartScoreListMax = engSubsPartScoreList[engSubsPartScoreListMaxIndex]
                    # print("fix score", engSubsPartScoreListMax, "index", engSubsPartScoreListMaxIndex, 'diffScore', diffScore, 'diffIndex', diffIndex)

    return engSubs[engSubsPartIndexList[engSubsPartScoreListMaxIndex]], engSubsPartScoreListMax, selected

def splitChAndEng(text):
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

def joinChAndEng(ch, eng):
    if len(ch) <= 0:
        return eng
    if len(eng) <= 0:
        return ch
    return f"{ch}\n{eng}"

def filterOriginalEngSubText(text):
    #删掉结尾(语气词)
    text = re.sub(" ?\\([^\\(]+?\\)$", '', text)
    #删掉(声音等):
    text = re.sub("^\\-?\\(.+?\\):? ?[\r\n]*", '', text)
    #删掉(一些神奇的内容)
    text = re.sub(" *\\(.+?\\)", '', text)
    #删掉人名:
    text = re.sub("^[A-Za-z0-9\']+:[\r\n]+", '', text)
    #清理多余字符
    text = text.strip()
    #所有字幕整理成一行
    text = re.sub('[\r\n]', ' ', text)
    #清理多余空格
    text = re.sub(' +', ' ', text)
    #再清理一遍多余字符
    text = text.strip()
    return text

if __name__ == '__main__':
    reformat(sys.argv[1])