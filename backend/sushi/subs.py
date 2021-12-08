import codecs
import os
import re
import collections
import chardet

from .common import SushiError, format_time, format_srt_time


def _parse_ass_time(string):
    hours, minutes, seconds = map(float, string.split(':'))
    return hours * 3600 + minutes * 60 + seconds


class ScriptEventBase(object):
    def __init__(self, source_index, start, end, text):
        self.source_index = source_index
        self.start = start
        self.end = end
        self.text = text

        self._shift = 0
        self._diff = 1
        self._linked_event = None
        self._start_shift = 0
        self._end_shift = 0

    @property
    def shift(self):
        return self._linked_event.shift if self.linked else self._shift

    @property
    def diff(self):
        return self._linked_event.diff if self.linked else self._diff

    @property
    def duration(self):
        return self.end - self.start

    @property
    def shifted_end(self):
        return self.end + self.shift + self._end_shift

    @property
    def shifted_start(self):
        return self.start + self.shift + self._start_shift

    def apply_shift(self):
        self.start = self.shifted_start
        self.end = self.shifted_end

    def set_shift(self, shift, audio_diff):
        assert not self.linked, 'Cannot set shift of a linked event'
        self._shift = shift
        self._diff = audio_diff

    def adjust_additional_shifts(self, start_shift, end_shift):
        assert not self.linked, 'Cannot apply additional shifts to a linked event'
        self._start_shift += start_shift
        self._end_shift += end_shift

    def get_link_chain_end(self):
        return self._linked_event.get_link_chain_end() if self.linked else self

    def link_event(self, other):
        assert other.get_link_chain_end() is not self, 'Circular link detected'
        self._linked_event = other

    def resolve_link(self):
        assert self.linked, 'Cannot resolve unlinked events'
        self._shift = self._linked_event.shift
        self._diff = self._linked_event.diff
        self._linked_event = None

    @property
    def linked(self):
        return self._linked_event is not None

    def adjust_shift(self, value):
        assert not self.linked, 'Cannot adjust time of linked events'
        self._shift += value


class ScriptBase(object):
    def __init__(self, events):
        self.events = events

    def sort_by_time(self):
        self.events.sort(key=lambda x: x.start)


class SrtEvent(ScriptEventBase):
    is_comment = False
    style = None

    EVENT_REGEX = re.compile(r"""
                               (\d+?)\s+? # line-number
                               (\d{1,2}:\d{1,2}:\d{1,2},\d+)\s-->\s(\d{1,2}:\d{1,2}:\d{1,2},\d+).  # timestamp
                               (.+?) # actual text
                           (?= # lookahead for the next line or end of the file
                               (?:\d+?\s+? # line-number
                               \d{1,2}:\d{1,2}:\d{1,2},\d+\s-->\s\d{1,2}:\d{1,2}:\d{1,2},\d+) # timestamp
                               |$
                           )""", flags=re.VERBOSE | re.DOTALL)

    @classmethod
    def from_string(cls, text):
        match = cls.EVENT_REGEX.match(text)
        start = cls.parse_time(match.group(2))
        end = cls.parse_time(match.group(3))
        return SrtEvent(int(match.group(1)), start, end, match.group(4).strip())

    def __str__(self):
        return '{0}\n{1} --> {2}\n{3}'.format(self.source_index, self._format_time(self.start),
                                              self._format_time(self.end), self.text)

    @staticmethod
    def parse_time(time_string):
        return _parse_ass_time(time_string.replace(',', '.'))

    @staticmethod
    def _format_time(seconds):
        return format_srt_time(seconds)


class SrtScript(ScriptBase):
    @classmethod
    def from_file(cls, path):
        try:
            with codecs.open(path, encoding='utf-8-sig') as script:
                text = script.read()
                events_list = [SrtEvent(
                    source_index=int(match.group(1)),
                    start=SrtEvent.parse_time(match.group(2)),
                    end=SrtEvent.parse_time(match.group(3)),
                    text=match.group(4).strip()
                ) for match in SrtEvent.EVENT_REGEX.finditer(text)]
                return cls(events_list)
        except IOError:
            raise SushiError("Script {0} not found".format(path))

    def save_to_file(self, path):
        text = '\n\n'.join(map(str, self.events))
        with codecs.open(path, encoding='utf-8', mode='w') as script:
            script.write(text)


class AssEvent(ScriptEventBase):
    def __init__(self, text, position=0):
        kind, _, rest = text.partition(':')
        split = [x.strip() for x in rest.split(',', 9)]

        super(AssEvent, self).__init__(
            source_index=position,
            start=_parse_ass_time(split[1]),
            end=_parse_ass_time(split[2]),
            text=split[9]
        )
        self.kind = kind
        self.is_comment = self.kind.lower() == 'comment'
        self.layer = split[0]
        self.style = split[3]
        self.name = split[4]
        self.margin_left = split[5]
        self.margin_right = split[6]
        self.margin_vertical = split[7]
        self.effect = split[8]

    def __str__(self):
        return '{0}: {1},{2},{3},{4},{5},{6},{7},{8},{9},{10}'.format(self.kind, self.layer,
                                                                      self._format_time(self.start),
                                                                      self._format_time(self.end),
                                                                      self.style, self.name,
                                                                      self.margin_left, self.margin_right,
                                                                      self.margin_vertical, self.effect,
                                                                      self.text)

    @staticmethod
    def _format_time(seconds):
        return format_time(seconds)


class AssScript(ScriptBase):
    def __init__(self, script_info, styles, events, other):
        super(AssScript, self).__init__(events)
        self.script_info = script_info
        self.styles = styles
        self.other = other

    @classmethod
    def from_file(cls, path):
        script_info, styles, events = [], [], []
        other_sections = collections.OrderedDict()

        def parse_script_info_line(line):
            if line.startswith('Format:'):
                return
            script_info.append(line)

        def parse_styles_line(line):
            if line.startswith('Format:'):
                return
            styles.append(line)

        def parse_event_line(line):
            if line.startswith('Format:'):
                return
            events.append(AssEvent(line, position=len(events) + 1))

        def create_generic_parse(section_name):
            if section_name in other_sections:
                raise SushiError("Duplicate section detected, invalid script?")
            other_sections[section_name] = []
            return other_sections[section_name].append

        parse_function = None

        try:
            rawdata = open(path, 'rb').read()
            result = chardet.detect(rawdata) 
            print(result)
            with codecs.open(path, encoding=result['encoding']) as script:
                for line_idx, line in enumerate(script):
                    line = line.strip()
                    if not line:
                        continue
                    low = line.lower()
                    if low == '[script info]':
                        parse_function = parse_script_info_line
                    elif low == '[v4+ styles]':
                        parse_function = parse_styles_line
                    elif low == '[events]':
                        parse_function = parse_event_line
                    elif re.match(r'\[.+?\]', low):
                        parse_function = create_generic_parse(line)
                    elif not parse_function:
                        raise SushiError("That's some invalid ASS script")
                    else:
                        try:
                            parse_function(line)
                        except Exception as e:
                            raise SushiError("That's some invalid ASS script: {0} [line {1}]".format(e.message, line_idx))
        except IOError:
            raise SushiError("Script {0} not found".format(path))
        return cls(script_info, styles, events, other_sections)

    def save_to_file(self, path):
        # if os.path.exists(path):
        #     raise RuntimeError('File %s already exists' % path)
        lines = []
        if self.script_info:
            lines.append('[Script Info]')
            lines.extend(self.script_info)
            lines.append('')

        if self.styles:
            lines.append('[V4+ Styles]')
            lines.append('Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding')
            lines.extend(self.styles)
            lines.append('')

        if self.events:
            events = sorted(self.events, key=lambda x: x.source_index)
            lines.append('[Events]')
            lines.append('Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text')
            lines.extend(map(str, events))

        if self.other:
            for section_name, section_lines in self.other.items():
                lines.append('')
                lines.append(section_name)
                lines.extend(section_lines)

        with codecs.open(path, encoding='utf-8-sig', mode='w') as script:
            script.write(str(os.linesep).join(lines))
