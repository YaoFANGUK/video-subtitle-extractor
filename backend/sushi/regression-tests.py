from contextlib import contextmanager
import json
import logging
import os
import gc
import sys
import resource
import re
import subprocess
import argparse

from .common import format_time
from .demux import Timecodes
from .subs import AssScript
from .wav import WavStream


root_logger = logging.getLogger('')


def strip_tags(text):
    return re.sub(r'{.*?}', " ", text)


@contextmanager
def set_file_logger(path):
    handler = logging.FileHandler(path, mode='a')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(handler)
    try:
        yield
    finally:
        root_logger.removeHandler(handler)


def compare_scripts(ideal_path, test_path, timecodes, test_name, expected_errors):
    ideal_script = AssScript.from_file(ideal_path)
    test_script = AssScript.from_file(test_path)
    if len(test_script.events) != len(ideal_script.events):
        logging.critical("Script length didn't match: {0} in ideal vs {1} in test. Test {2}".format(
            len(ideal_script.events), len(test_script.events), test_name)
        )
        return False
    ideal_script.sort_by_time()
    test_script.sort_by_time()
    failed = 0
    ft = format_time
    for idx, (ideal, test) in enumerate(zip(ideal_script.events, test_script.events)):
        ideal_start_frame = timecodes.get_frame_number(ideal.start)
        ideal_end_frame = timecodes.get_frame_number(ideal.end)

        test_start_frame = timecodes.get_frame_number(test.start)
        test_end_frame = timecodes.get_frame_number(test.end)

        if ideal_start_frame != test_start_frame and ideal_end_frame != test_end_frame:
            logging.debug('{0}: start and end time failed at "{1}". {2}-{3} vs {4}-{5}'.format(
                idx, strip_tags(ideal.text), ft(ideal.start), ft(ideal.end), ft(test.start), ft(test.end))
            )
            failed += 1
        elif ideal_end_frame != test_end_frame:
            logging.debug(
                '{0}: end time failed at "{1}". {2} vs {3}'.format(
                    idx, strip_tags(ideal.text), ft(ideal.end), ft(test.end))
            )
            failed += 1
        elif ideal_start_frame != test_start_frame:
            logging.debug(
                '{0}: start time failed at "{1}". {2} vs {3}'.format(
                    idx, strip_tags(ideal.text), ft(ideal.start), ft(test.start))
            )
            failed += 1

    logging.info('Total lines: {0}, good: {1}, failed: {2}'.format(len(ideal_script.events), len(ideal_script.events) - failed, failed))

    if failed > expected_errors:
        logging.critical('Got more failed lines than expected ({0} actual vs {1} expected)'.format(failed, expected_errors))
        return False
    elif failed < expected_errors:
        logging.critical('Got less failed lines than expected ({0} actual vs {1} expected)'.format(failed, expected_errors))
        return False
    else:
        logging.critical('Met expectations')
        return True


def run_test(base_path, plots_path, test_name, params):
    def safe_add_key(args, key, name):
        if name in params:
            args.extend((key, str(params[name])))

    def safe_add_path(args, folder, key, name):
        if name in params:
            args.extend((key, os.path.join(folder, params[name])))

    logging.info('Testing "{0}"'.format(test_name))

    folder = os.path.join(base_path, params['folder'])

    cmd = ["sushi"]

    safe_add_path(cmd, folder, '--src', 'src')
    safe_add_path(cmd, folder, '--dst', 'dst')
    safe_add_path(cmd, folder, '--src-keyframes', 'src-keyframes')
    safe_add_path(cmd, folder, '--dst-keyframes', 'dst-keyframes')
    safe_add_path(cmd, folder, '--src-timecodes', 'src-timecodes')
    safe_add_path(cmd, folder, '--dst-timecodes', 'dst-timecodes')
    safe_add_path(cmd, folder, '--script', 'script')
    safe_add_path(cmd, folder, '--chapters', 'chapters')
    safe_add_path(cmd, folder, '--src-script', 'src-script')
    safe_add_path(cmd, folder, '--dst-script', 'dst-script')
    safe_add_key(cmd, '--max-kf-distance', 'max-kf-distance')
    safe_add_key(cmd, '--max-ts-distance', 'max-ts-distance')
    safe_add_key(cmd, '--max-ts-duration', 'max-ts-duration')

    output_path = os.path.join(folder, params['dst']) + '.sushi.test.ass'
    cmd.extend(('-o', output_path))
    if plots_path:
        cmd.extend(('--test-shift-plot', os.path.join(plots_path, '{0}.png'.format(test_name))))

    log_path = os.path.join(folder, 'sushi_test.log')

    with open(log_path, "w") as log_file:
        try:
            subprocess.call(cmd, stderr=log_file, stdout=log_file)
        except Exception as e:
            logging.critical('Sushi failed on test "{0}": {1}'.format(test_name, e.message))
            return False

    with set_file_logger(log_path):
        ideal_path = os.path.join(folder, params['ideal'])
        try:
            timecodes = Timecodes.from_file(os.path.join(folder, params['dst-timecodes']))
        except KeyError:
            timecodes = Timecodes.cfr(params['fps'])

        return compare_scripts(ideal_path, output_path, timecodes, test_name, params['expected_errors'])


def run_wav_test(test_name, file_path, params):
    gc.collect(2)

    before = resource.getrusage(resource.RUSAGE_SELF)
    _ = WavStream(file_path, params.get('sample_rate', 12000), params.get('sample_type', 'uint8'))
    after = resource.getrusage(resource.RUSAGE_SELF)

    total_time = (after.ru_stime - before.ru_stime) + (after.ru_utime - before.ru_utime)
    ram_difference = (after.ru_maxrss - before.ru_maxrss) / 1024.0 / 1024.0

    if 'max_time' in params and total_time > params['max_time']:
        logging.critical('Loading "{0}" took too much time: {1} vs {2} seconds'
                         .format(test_name, total_time, params['max_time']))
        return False
    if 'max_memory' in params and ram_difference > params['max_memory']:
        logging.critical('Loading "{0}" consumed too much RAM: {1} vs {2}'
                         .format(test_name, ram_difference, params['max_memory']))
        return False
    return True


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Sushi regression testing util')

    parser.add_argument('--only', dest="run_only", nargs="*", metavar='<test names>',
                        help='Test names to run')
    parser.add_argument('-c', '--conf', default="tests.json", dest='conf_path', metavar='<filename>',
                        help='Config file path')

    return parser


def run():
    root_logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(console_handler)

    args = create_arg_parser().parse_args()

    try:
        with open(args.conf_path) as file:
            config = json.load(file)
    except IOError as e:
        logging.critical(e)
        sys.exit(2)

    def should_run(name):
        return not args.run_only or name in args.run_only

    failed = ran = 0
    for test_name, params in config.get('tests', {}).items():
        if not should_run(test_name):
            continue
        if not params.get('disabled', False):
            ran += 1
            if not run_test(config['basepath'], config['plots'], test_name, params):
                failed += 1
            logging.info('')
        else:
            logging.warn('Test "{0}" disabled'.format(test_name))

    if should_run("wavs"):
        for test_name, params in config.get('wavs', {}).items():
            ran += 1
            if not run_wav_test(test_name, os.path.join(config['basepath'], params['file']), params):
                failed += 1
            logging.info('')

    logging.info('Ran {0} tests, {1} failed'.format(ran, failed))


if __name__ == '__main__':
    run()