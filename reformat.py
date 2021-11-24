import pysrt
import sys
import wordsegment
import re

def reformat(path):
    wordsegment.load()
    subs = pysrt.open(path)
    subs.save(f"{path}.bak", encoding='utf-8')
    verbForms = ["I'm","you're","he's","she's","we're","it's","isn't","aren't","they're","there's","wasn't","weren't","I've","you've","he's","she's","it's","we've","they've","there's","hasn't","haven't","I'd","you'd","he'd","she'd","it'd","we'd","they'd","doesn't","don't","didn't","I'll","you'll","he'll","she'll","we'll","they'll","there'll","I'd","you'd","he'd","she'd","it'd", "we've","they'd","there'd","there'd","can't","couldn't","daren't","hadn't","mightn't","mustn't","needn't","oughtn't","shan't","shouldn't","usedn't","won't","wouldn't","that's","what's"]
    verbFormMap = {}

    typoMap = {
        "l'm": "I'm",
        "Let'sqo": "Let's go"
    }
    for verb in verbForms:
        verbFormMap[verb.replace("'", "").lower()] = verb
    print(verbFormMap)


    def formatSegList(segList):
        newSegs = []
        for seg in segList:
            if seg in verbFormMap:
                newSegs.append([seg, verbFormMap[seg]])
            else:
                newSegs.append([seg])
        return newSegs

    def regexGroupSplit(segs, ):
        pass

    def typoFix(text):
        for k,v in typoMap.items():
            text = text.replace(k, v)

        print(text)
        return text

    for sub in subs:
        # print(sub.text)
        sub.text = typoFix(sub.text)
        seg = wordsegment.segment(sub.text)
        if (len(seg) == 1):
            seg = wordsegment.segment(re.sub(re.compile(f"(\ni)([^\\s])", re.I), "\\1 \\2", sub.text))
        seg = formatSegList(seg)
        
        sub.text = re.sub(' +([\\u4e00-\\u9fa5])', ' \\1', sub.text)
        sub.text = sub.text.replace("  ", "\n")
        # sub.text = re.sub(".+?\n", "", sub.text)
        print(seg)
        lines = []
        remain = sub.text
        segLen = len(seg)
        for i in range(0, segLen):
            s = seg[i]
            global regex
            # print("remain:", remain)
            if len(s) > 1:
                regex = re.compile(f"(.*?)({s[0]}|{s[1]})", re.I)
            else:
                regex = re.compile(f"(.*?)({s[0]})", re.I)
            ss = re.search(regex,  remain)
            # print()
            # ssLen = len(ss)
            # print(f"{s}: {ss}")
            if ss is None:
                if (i == segLen - 1):
                    lines.append(remain.strip())
                # print("lines1:", lines)

                continue

            lines.append(remain[:ss.span()[1]].strip())
            remain = remain[ss.span()[1]:].strip()
            if (i == segLen - 1):
                lines.append(remain)
            # print("lines:", lines)

            # if len(ss[0]) > 0:
            #     lines.append(ss[0])
            # if ssLen > 3:
            #     remain = "".join(ss[3:])
            # if (ssLen <= 0):
            #     break
            
            # lines.append(ss[1].strip())
            # if (i == segLen - 1):
            #     if ssLen > 3:
            #         for ii in range(3, ssLen):
            #             lines.append(ss[ii].strip())
            # lines.append(re.sub('(' + s + ")", "\\1", ss[1], flags=re.I))
        # print(lines)
        ss = " ".join(lines)
        ss = re.sub("([^\\sA-Z])([A-Z])", "\\1 \\2", ss)
        ss = ss.replace("  ", " ")
        ss = ss.replace("。", ".")
        ss = re.sub(" *([\\.\\?\\!\\,])", "\\1", ss)
        ss = re.sub(" *([\\']) *", "\\1", ss)
        # ss = ss.replace(" .", ".")
        # ss = ss.replace(" ?", "?")
        ss = re.sub('\n\\s*', '\n', ss)
        ss = re.sub('^\\s*', '', ss)
        ss = ss.replace(" Dr. ", " Dr.")
        ss = ss.replace("\n\n", "\n")

        print(ss)
        #     # print(sub.text)
        # sub.text = sub.text.replace("  ", " ")
        # sub.text = sub.text.replace("  ", " ")
        # print("-->")
        print(sub.text)
        sub.text = ss.strip()
        # print("<--")
        print()
    subs.save(path, encoding='utf-8')

if __name__ == '__main__':
    reformat(sys.argv[1])