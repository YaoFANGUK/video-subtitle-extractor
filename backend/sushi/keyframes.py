from .common import SushiError, read_all_text


def parse_scxvid_keyframes(text):
    return [i - 3 for i, line in enumerate(text.splitlines()) if line and line[0] == 'i']


def parse_keyframes(path):
    text = read_all_text(path)
    if '# XviD 2pass stat file' in text:
        frames = parse_scxvid_keyframes(text)
    else:
        raise SushiError('Unsupported keyframes type')
    if 0 not in frames:
        frames.insert(0, 0)
    return frames