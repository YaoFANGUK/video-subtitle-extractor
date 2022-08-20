import argparse
import sys
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VideoSubtitleExtractor')
    parser.add_argument('--path', type=str, help='Path of video file')
    parser.add_argument('--xmin', type=float, help='Subtitle Area(x min)')
    parser.add_argument('--ymin', type=float, help='Subtitle Area(y min)')
    parser.add_argument('--xmax', type=float, help='Subtitle Area(x max)')
    parser.add_argument('--ymax', type=float, help='Subtitle Area(y max)')
    parser.add_argument('--lang', type=str, default='ch', help='Language of subtitle')
    args = parser.parse_args()
    subtitle_area = (args.ymin, args.ymax, args.xmin, args.xmax)
    print(args.path, subtitle_area)
    sys.argv = [sys.argv[0]]
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings.ini'), mode='w', encoding='utf-8') as f:
        f.write('[DEFAULT]\n')
        f.write('Interface = 简体中文\n')
        f.write('Language = ' + args.lang + '\n')
        f.write('Mode = fast')

    from main import SubtitleExtractor
    # 新建字幕提取对象
    se = SubtitleExtractor(args.path, subtitle_area)
    # 开始提取字幕
    se.run()