import re
from . import common


def parse_times(times):
    result = []
    for t in times:
        hours, minutes, seconds = list(map(float, t.split(':')))
        result.append(hours * 3600 + minutes * 60 + seconds)

    result.sort()
    if result[0] != 0:
        result.insert(0, 0)
    return result


def parse_xml_start_times(text):
    times = re.findall(r'<ChapterTimeStart>(\d+:\d+:\d+\.\d+)</ChapterTimeStart>', text)
    return parse_times(times)


def get_xml_start_times(path):
    return parse_xml_start_times(common.read_all_text(path))


def parse_ogm_start_times(text):
    times = re.findall(r'CHAPTER\d+=(\d+:\d+:\d+\.\d+)', text, flags=re.IGNORECASE)
    return parse_times(times)


def get_ogm_start_times(path):
    return parse_ogm_start_times(common.read_all_text(path))


def format_ogm_chapters(start_times):
    return "\n".join("CHAPTER{0:02}={1}\nCHAPTER{0:02}NAME=".format(idx+1, common.format_srt_time(start).replace(',', '.'))
                     for idx, start in enumerate(start_times)) + "\n"