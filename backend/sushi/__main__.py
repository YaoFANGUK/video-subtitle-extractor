import argparse
import logging
import os
import sys
import time

# Use absolute imports to support pyinstaller
# https://github.com/pyinstaller/pyinstaller/issues/2560
from sushi import run, VERSION
from sushi.common import SushiError

if sys.platform == 'win32':
    try:
        import colorama
        colorama.init()
        console_colors_supported = True
    except ImportError:
        console_colors_supported = False
else:
    console_colors_supported = True


class ColoredLogFormatter(logging.Formatter):
    bold_code = "\033[1m"
    reset_code = "\033[0m"
    grey_code = "\033[30m\033[1m"

    error_format = "{bold}ERROR: %(message)s{reset}".format(bold=bold_code, reset=reset_code)
    warn_format = "{bold}WARNING: %(message)s{reset}".format(bold=bold_code, reset=reset_code)
    debug_format = "{grey}%(message)s{reset}".format(grey=grey_code, reset=reset_code)
    default_format = "%(message)s"

    def format(self, record):
        if record.levelno == logging.DEBUG:
            self._fmt = self.debug_format
        elif record.levelno == logging.WARN:
            self._fmt = self.warn_format
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            self._fmt = self.error_format
        else:
            self._fmt = self.default_format

        return super(ColoredLogFormatter, self).format(record)


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Sushi - Automatic Subtitle Shifter')

    parser.add_argument('--window', default=10, type=int, metavar='<size>', dest='window',
                        help='Search window size. [%(default)s]')
    parser.add_argument('--max-window', default=30, type=int, metavar='<size>', dest='max_window',
                        help="Maximum search size Sushi is allowed to use when trying to recover from errors. [%(default)s]")
    parser.add_argument('--rewind-thresh', default=5, type=int, metavar='<events>', dest='rewind_thresh',
                        help="Number of consecutive errors Sushi has to encounter to consider results broken "
                             "and retry with larger window. Set to 0 to disable. [%(default)s]")
    parser.add_argument('--no-grouping', action='store_false', dest='grouping',
                        help="Don't events into groups before shifting. Also disables error recovery.")
    parser.add_argument('--max-kf-distance', default=2, type=float, metavar='<frames>', dest='max_kf_distance',
                        help='Maximum keyframe snapping distance. [%(default)s]')
    parser.add_argument('--kf-mode', default='all', choices=['shift', 'snap', 'all'], dest='kf_mode',
                        help='Keyframes-based shift correction/snapping mode. [%(default)s]')
    parser.add_argument('--smooth-radius', default=3, type=int, metavar='<events>', dest='smooth_radius',
                        help='Radius of smoothing median filter. [%(default)s]')

    # 10 frames at 23.976
    parser.add_argument('--max-ts-duration', default=1001.0 / 24000.0 * 10, type=float, metavar='<seconds>',
                        dest='max_ts_duration',
                        help='Maximum duration of a line to be considered typesetting. [%(default).3f]')
    # 10 frames at 23.976
    parser.add_argument('--max-ts-distance', default=1001.0 / 24000.0 * 10, type=float, metavar='<seconds>',
                        dest='max_ts_distance',
                        help='Maximum distance between two adjacent typesetting lines to be merged. [%(default).3f]')

    # deprecated/test options, do not use
    parser.add_argument('--test-shift-plot', default=None, dest='plot_path', help=argparse.SUPPRESS)
    parser.add_argument('--sample-type', default='uint8', choices=['float32', 'uint8'], dest='sample_type',
                        help=argparse.SUPPRESS)

    parser.add_argument('--sample-rate', default=12000, type=int, metavar='<rate>', dest='sample_rate',
                        help='Downsampled audio sample rate. [%(default)s]')

    parser.add_argument('--src-audio', default=None, type=int, metavar='<id>', dest='src_audio_idx',
                        help='Audio stream index of the source video')
    parser.add_argument('--src-script', default=None, type=int, metavar='<id>', dest='src_script_idx',
                        help='Script stream index of the source video')
    parser.add_argument('--dst-audio', default=None, type=int, metavar='<id>', dest='dst_audio_idx',
                        help='Audio stream index of the destination video')
    # files
    parser.add_argument('--no-cleanup', action='store_false', dest='cleanup',
                        help="Don't delete demuxed streams")
    parser.add_argument('--temp-dir', default=None, dest='temp_dir', metavar='<string>',
                        help='Specify temporary folder to use when demuxing stream.')
    parser.add_argument('--chapters', default=None, dest='chapters_file', metavar='<filename>',
                        help="XML or OGM chapters to use instead of any found in the source. 'none' to disable.")
    parser.add_argument('--script', default=None, dest='script_file', metavar='<filename>',
                        help='Subtitle file path to use instead of any found in the source')

    parser.add_argument('--dst-keyframes', default=None, dest='dst_keyframes', metavar='<filename>',
                        help='Destination keyframes file')
    parser.add_argument('--src-keyframes', default=None, dest='src_keyframes', metavar='<filename>',
                        help='Source keyframes file')
    parser.add_argument('--dst-fps', default=None, type=float, dest='dst_fps', metavar='<fps>',
                        help='Fps of the destination video. Must be provided if keyframes are used.')
    parser.add_argument('--src-fps', default=None, type=float, dest='src_fps', metavar='<fps>',
                        help='Fps of the source video. Must be provided if keyframes are used.')
    parser.add_argument('--dst-timecodes', default=None, dest='dst_timecodes', metavar='<filename>',
                        help='Timecodes file to use instead of making one from the destination (when possible)')
    parser.add_argument('--src-timecodes', default=None, dest='src_timecodes', metavar='<filename>',
                        help='Timecodes file to use instead of making one from the source (when possible)')

    parser.add_argument('--src', required=True, dest="source", metavar='<filename>',
                        help='Source audio/video')
    parser.add_argument('--dst', required=True, dest="destination", metavar='<filename>',
                        help='Destination audio/video')
    parser.add_argument('-o', '--output', default=None, dest='output_script', metavar='<filename>',
                        help='Output script')

    parser.add_argument('-v', '--verbose', default=False, dest='verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--version', action='version', version=VERSION)

    return parser


def parse_args_and_run(cmd_keys):
    def format_arg(arg):
        return arg if ' ' not in arg else '"{0}"'.format(arg)

    args = create_arg_parser().parse_args(cmd_keys)
    handler = logging.StreamHandler()
    if console_colors_supported and hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
        # enable colors
        handler.setFormatter(ColoredLogFormatter())
    else:
        handler.setFormatter(logging.Formatter(fmt=ColoredLogFormatter.default_format))
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    logging.info("Sushi's running with arguments: {0}".format(' '.join(map(format_arg, cmd_keys))))
    start_time = time.time()
    run(args)
    logging.info('Done in {0}s'.format(time.time() - start_time))


def main():
    try:
        parse_args_and_run(sys.argv[1:])
    except SushiError as e:
        logging.critical(e.message)
        sys.exit(2)


if __name__ == '__main__':
    main()