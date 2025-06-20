import bisect
from itertools import takewhile, chain
import logging
import operator
import os

import numpy as np

from . import chapters
from .common import SushiError, get_extension, format_time, ensure_static_collection
from .demux import Timecodes, Demuxer
from . import keyframes
from .subs import AssScript, SrtScript
from .wav import WavStream


try:
    import matplotlib.pyplot as plt
    plot_enabled = True
except ImportError:
    plot_enabled = False


ALLOWED_ERROR = 0.01
MAX_GROUP_STD = 0.025
VERSION = '0.6.0'


def abs_diff(a, b):
    return abs(a - b)


def interpolate_nones(data, points):
    data = ensure_static_collection(data)
    values_lookup = {p: v for p, v in zip(points, data) if v is not None}
    if not values_lookup:
        return []

    zero_points = {p for p, v in zip(points, data) if v is None}
    if not zero_points:
        return data

    data_list = sorted(values_lookup.items())
    zero_points = sorted(x for x in zero_points if x not in values_lookup)

    out = np.interp(x=zero_points,
                    xp=list(map(operator.itemgetter(0), data_list)),
                    fp=list(map(operator.itemgetter(1), data_list)))

    values_lookup.update(zip(zero_points, out))

    return [
        values_lookup[point] if value is None else value
        for point, value in zip(points, data)
    ]


# todo: implement this as a running median
def running_median(values, window_size):
    if window_size % 2 != 1:
        raise SushiError('Median window size should be odd')
    half_window = window_size // 2
    medians = []
    items_count = len(values)
    for idx in range(items_count):
        radius = min(half_window, idx, items_count - idx - 1)
        med = np.median(values[idx - radius:idx + radius + 1])
        medians.append(med)
    return medians


def smooth_events(events, radius):
    if not radius:
        return
    window_size = radius * 2 + 1
    shifts = [e.shift for e in events]
    smoothed = running_median(shifts, window_size)
    for event, new_shift in zip(events, smoothed):
        event.set_shift(new_shift, event.diff)


def detect_groups(events_iter):
    events_iter = iter(events_iter)
    groups_list = [[next(events_iter)]]
    for event in events_iter:
        if abs_diff(event.shift, groups_list[-1][-1].shift) > ALLOWED_ERROR:
            groups_list.append([])
        groups_list[-1].append(event)
    return groups_list


def groups_from_chapters(events, times):
    logging.info('Chapter start points: {0}'.format([format_time(t) for t in times]))
    groups = [[]]
    chapter_times = iter(times[1:] + [36000000000])  # very large event at the end
    current_chapter = next(chapter_times)

    for event in events:
        if event.end > current_chapter:
            groups.append([])
            while event.end > current_chapter:
                current_chapter = next(chapter_times)

        groups[-1].append(event)

    groups = [g for g in groups if g]  # non-empty groups
    # check if we have any groups where every event is linked
    # for example a chapter with only comments inside
    broken_groups = [group for group in groups if not any(e for e in group if not e.linked)]
    if broken_groups:
        for group in broken_groups:
            for event in group:
                parent = event.get_link_chain_end()
                parent_group = next(group for group in groups if parent in group)
                parent_group.append(event)
            del group[:]
        groups = [g for g in groups if g]
        # re-sort the groups again since we might break the order when inserting linked events
        # sorting everything again is far from optimal but python sorting is very fast for sorted arrays anyway
        for group in groups:
            group.sort(key=lambda event: event.start)

    return groups


def split_broken_groups(groups):
    correct_groups = []
    broken_found = False
    for g in groups:
        std = np.std([e.shift for e in g])
        if std > MAX_GROUP_STD:
            logging.warn('Shift is not consistent between {0} and {1}, most likely chapters are wrong (std: {2}). '
                         'Switching to automatic grouping.'.format(format_time(g[0].start), format_time(g[-1].end),
                                                                   std))
            correct_groups.extend(detect_groups(g))
            broken_found = True
        else:
            correct_groups.append(g)

    if broken_found:
        groups_iter = iter(correct_groups)
        correct_groups = [list(next(groups_iter))]
        for group in groups_iter:
            if abs_diff(correct_groups[-1][-1].shift, group[0].shift) >= ALLOWED_ERROR \
                    or np.std([e.shift for e in group + correct_groups[-1]]) >= MAX_GROUP_STD:
                correct_groups.append([])

            correct_groups[-1].extend(group)
    return correct_groups


def fix_near_borders(events):
    """
    We assume that all lines with diff greater than 5 * (median diff across all events) are broken
    """
    def fix_border(event_list, median_diff):
        last_ten_diff = np.median([x.diff for x in event_list[:10]], overwrite_input=True)
        diff_limit = min(last_ten_diff, median_diff)
        broken = []
        for event in event_list:
            if not 0.2 < (event.diff / diff_limit) < 5:
                broken.append(event)
            else:
                for x in broken:
                    x.link_event(event)
                return len(broken)
        return 0

    median_diff = np.median([x.diff for x in events], overwrite_input=True)

    fixed_count = fix_border(events, median_diff)
    if fixed_count:
        logging.info('Fixing {0} border events right after {1}'.format(fixed_count, format_time(events[0].start)))

    fixed_count = fix_border(list(reversed(events)), median_diff)
    if fixed_count:
        logging.info('Fixing {0} border events right before {1}'.format(fixed_count, format_time(events[-1].end)))


def get_distance_to_closest_kf(timestamp, keyframes):
    idx = bisect.bisect_left(keyframes, timestamp)
    if idx == 0:
        kf = keyframes[0]
    elif idx == len(keyframes):
        kf = keyframes[-1]
    else:
        before = keyframes[idx - 1]
        after = keyframes[idx]
        kf = after if after - timestamp < timestamp - before else before
    return kf - timestamp


def find_keyframe_shift(group, src_keytimes, dst_keytimes, src_timecodes, dst_timecodes, max_kf_distance):
    def get_distance(src_distance, dst_distance, limit):
        if abs(dst_distance) > limit:
            return None
        shift = dst_distance - src_distance
        return shift if abs(shift) < limit else None

    src_start = get_distance_to_closest_kf(group[0].start, src_keytimes)
    src_end = get_distance_to_closest_kf(group[-1].end + src_timecodes.get_frame_size(group[-1].end), src_keytimes)

    dst_start = get_distance_to_closest_kf(group[0].shifted_start, dst_keytimes)
    dst_end = get_distance_to_closest_kf(group[-1].shifted_end + dst_timecodes.get_frame_size(group[-1].end), dst_keytimes)

    snapping_limit_start = src_timecodes.get_frame_size(group[0].start) * max_kf_distance
    snapping_limit_end = src_timecodes.get_frame_size(group[0].end) * max_kf_distance

    return (get_distance(src_start, dst_start, snapping_limit_start),
            get_distance(src_end, dst_end, snapping_limit_end))


def find_keyframes_distances(event, src_keytimes, dst_keytimes, timecodes, max_kf_distance):
    def find_keyframe_distance(src_time, dst_time):
        src = get_distance_to_closest_kf(src_time, src_keytimes)
        dst = get_distance_to_closest_kf(dst_time, dst_keytimes)
        snapping_limit = timecodes.get_frame_size(src_time) * max_kf_distance

        if abs(src) < snapping_limit and abs(dst) < snapping_limit and abs(src - dst) < snapping_limit:
            return dst - src
        return 0

    ds = find_keyframe_distance(event.start, event.shifted_start)
    de = find_keyframe_distance(event.end, event.shifted_end)
    return ds, de


def snap_groups_to_keyframes(events, chapter_times, max_ts_duration, max_ts_distance, src_keytimes, dst_keytimes,
                             src_timecodes, dst_timecodes, max_kf_distance, kf_mode):
    if not max_kf_distance:
        return

    groups = merge_short_lines_into_groups(events, chapter_times, max_ts_duration, max_ts_distance)

    if kf_mode == 'all' or kf_mode == 'shift':
        #  step 1: snap events without changing their duration. Useful for some slight audio imprecision correction
        shifts = []
        times = []
        for group in groups:
            shifts.extend(find_keyframe_shift(group, src_keytimes, dst_keytimes, src_timecodes, dst_timecodes, max_kf_distance))
            times.extend((group[0].shifted_start, group[-1].shifted_end))

        shifts = interpolate_nones(shifts, times)
        if shifts:
            mean_shift = np.mean(shifts)
            shifts = zip(*[iter(shifts)] * 2)

            logging.info('Group {0}-{1} corrected by {2}'.format(format_time(events[0].start), format_time(events[-1].end), mean_shift))
            for group, (start_shift, end_shift) in zip(groups, shifts):
                if abs(start_shift - end_shift) > 0.001 and len(group) > 1:
                    actual_shift = min(start_shift, end_shift, key=lambda x: abs(x - mean_shift))
                    logging.warning("Typesetting group at {0} had different shift at start/end points ({1} and {2}). Shifting by {3}."
                                    .format(format_time(group[0].start), start_shift, end_shift, actual_shift))
                    for e in group:
                        e.adjust_shift(actual_shift)
                else:
                    for e in group:
                        e.adjust_additional_shifts(start_shift, end_shift)

    if kf_mode == 'all' or kf_mode == 'snap':
        # step 2: snap start/end times separately
        for group in groups:
            if len(group) > 1:
                pass  # we don't snap typesetting
            start_shift, end_shift = find_keyframes_distances(group[0], src_keytimes, dst_keytimes, src_timecodes, max_kf_distance)
            if abs(start_shift) > 0.01 or abs(end_shift) > 0.01:
                logging.info('Snapping {0} to keyframes, start time by {1}, end: {2}'.format(format_time(group[0].start), start_shift, end_shift))
                group[0].adjust_additional_shifts(start_shift, end_shift)


def average_shifts(events):
    events = [e for e in events if not e.linked]
    shifts = [x.shift for x in events]
    weights = [1 - x.diff for x in events]
    avg = np.average(shifts, weights=weights)
    for e in events:
        e.set_shift(avg, e.diff)
    return avg


def merge_short_lines_into_groups(events, chapter_times, max_ts_duration, max_ts_distance):
    search_groups = []
    chapter_times = iter(chapter_times[1:] + [100000000])
    next_chapter = next(chapter_times)
    events = ensure_static_collection(events)

    processed = set()
    for idx, event in enumerate(events):
        if idx in processed:
            continue

        while event.end > next_chapter:
            next_chapter = next(chapter_times)

        if event.duration > max_ts_duration:
            search_groups.append([event])
            processed.add(idx)
        else:
            group = [event]
            group_end = event.end
            i = idx + 1
            while i < len(events) and abs(group_end - events[i].start) < max_ts_distance:
                if events[i].end < next_chapter and events[i].duration <= max_ts_duration:
                    processed.add(i)
                    group.append(events[i])
                    group_end = max(group_end, events[i].end)
                i += 1

            search_groups.append(group)

    return search_groups


def prepare_search_groups(events, source_duration, chapter_times, max_ts_duration, max_ts_distance):
    last_unlinked = None
    for idx, event in enumerate(events):
        if event.is_comment:
            try:
                event.link_event(events[idx + 1])
            except IndexError:
                event.link_event(last_unlinked)
            continue
        if (event.start + event.duration / 2.0) > source_duration:
            logging.info('Event time outside of audio range, ignoring: %s', event)
            event.link_event(last_unlinked)
            continue
        elif event.end == event.start:
            logging.info('{0}: skipped because zero duration'.format(format_time(event.start)))
            try:
                event.link_event(events[idx + 1])
            except IndexError:
                event.link_event(last_unlinked)
            continue

        # link lines with start and end times identical to some other event
        # assuming scripts are sorted by start time so we don't search the entire collection
        def same_start(x):
            return event.start == x.start
        processed = next((x for x in takewhile(same_start, reversed(events[:idx])) if not x.linked and x.end == event.end), None)
        if processed:
            event.link_event(processed)
        else:
            last_unlinked = event

    events = (e for e in events if not e.linked)

    search_groups = merge_short_lines_into_groups(events, chapter_times, max_ts_duration, max_ts_distance)

    # link groups contained inside other groups to the larger group
    passed_groups = []
    for idx, group in enumerate(search_groups):
        try:
            other = next(x for x in reversed(search_groups[:idx])
                         if x[0].start <= group[0].start
                         and x[-1].end >= group[-1].end)
            for event in group:
                event.link_event(other[0])
        except StopIteration:
            passed_groups.append(group)
    return passed_groups


def calculate_shifts(src_stream, dst_stream, groups_list, normal_window, max_window, rewind_thresh):
    def log_shift(state):
        logging.info('{0}-{1}: shift: {2:0.10f}, diff: {3:0.10f}'
                     .format(format_time(state["start_time"]), format_time(state["end_time"]), state["shift"], state["diff"]))

    def log_uncommitted(state, shift, left_side_shift, right_side_shift, search_offset):
        logging.debug('{0}-{1}: shift: {2:0.5f} [{3:0.5f}, {4:0.5f}], search offset: {5:0.6f}'
                      .format(format_time(state["start_time"]), format_time(state["end_time"]),
                              shift, left_side_shift, right_side_shift, search_offset))

    small_window = 1.5
    idx = 0
    committed_states = []
    uncommitted_states = []
    window = normal_window
    while idx < len(groups_list):
        search_group = groups_list[idx]
        tv_audio = src_stream.get_substream(search_group[0].start, search_group[-1].end)
        original_time = search_group[0].start
        group_state = {"start_time": search_group[0].start, "end_time": search_group[-1].end, "shift": None, "diff": None}
        last_committed_shift = committed_states[-1]["shift"] if committed_states else 0
        diff = new_time = None

        if not uncommitted_states:
            if original_time + last_committed_shift > dst_stream.duration_seconds:
                # event outside of audio range, all events past it are also guaranteed to fail
                for g in groups_list[idx:]:
                    committed_states.append({"start_time": g[0].start, "end_time": g[-1].end, "shift": None, "diff": None})
                    logging.info("{0}-{1}: outside of audio range".format(format_time(g[0].start), format_time(g[-1].end)))
                break

            if small_window < window:
                diff, new_time = dst_stream.find_substream(tv_audio, original_time + last_committed_shift, small_window)

            if new_time is not None and abs_diff(new_time - original_time, last_committed_shift) <= ALLOWED_ERROR:
                # fastest case - small window worked, commit the group immediately
                group_state.update({"shift": new_time - original_time, "diff": diff})
                committed_states.append(group_state)
                log_shift(group_state)
                if window != normal_window:
                    logging.info("Going back to window {0} from {1}".format(normal_window, window))
                    window = normal_window
                idx += 1
                continue

        left_audio_half, right_audio_half = np.split(tv_audio, [len(tv_audio[0]) // 2], axis=1)
        right_half_offset = len(left_audio_half[0]) / float(src_stream.sample_rate)
        terminate = False
        # searching from last committed shift
        if original_time + last_committed_shift < dst_stream.duration_seconds:
            diff, new_time = dst_stream.find_substream(tv_audio, original_time + last_committed_shift, window)
            left_side_time = dst_stream.find_substream(left_audio_half, original_time + last_committed_shift, window)[1]
            right_side_time = dst_stream.find_substream(right_audio_half, original_time + last_committed_shift + right_half_offset, window)[1] - right_half_offset
            terminate = abs_diff(left_side_time, right_side_time) <= ALLOWED_ERROR and abs_diff(new_time, left_side_time) <= ALLOWED_ERROR
            log_uncommitted(group_state, new_time - original_time, left_side_time - original_time,
                            right_side_time - original_time, last_committed_shift)

        if not terminate and uncommitted_states and uncommitted_states[-1]["shift"] is not None \
                and original_time + uncommitted_states[-1]["shift"] < dst_stream.duration_seconds:
            start_offset = uncommitted_states[-1]["shift"]
            diff, new_time = dst_stream.find_substream(tv_audio, original_time + start_offset, window)
            left_side_time = dst_stream.find_substream(left_audio_half, original_time + start_offset, window)[1]
            right_side_time = dst_stream.find_substream(right_audio_half, original_time + start_offset + right_half_offset, window)[1] - right_half_offset
            terminate = abs_diff(left_side_time, right_side_time) <= ALLOWED_ERROR and abs_diff(new_time, left_side_time) <= ALLOWED_ERROR
            log_uncommitted(group_state, new_time - original_time, left_side_time - original_time,
                            right_side_time - original_time, start_offset)

        shift = new_time - original_time
        if not terminate:
            # we aren't back on track yet - add this group to uncommitted
            group_state.update({"shift": shift, "diff": diff})
            uncommitted_states.append(group_state)
            idx += 1
            if rewind_thresh == len(uncommitted_states) and window < max_window:
                logging.warn("Detected possibly broken segment starting at {0}, increasing the window from {1} to {2}"
                             .format(format_time(uncommitted_states[0]["start_time"]), window, max_window))
                window = max_window
                idx = len(committed_states)
                del uncommitted_states[:]
            continue

        # we're back on track - apply current shift to all broken events
        if uncommitted_states:
            logging.warning("Events from {0} to {1} will most likely be broken!".format(
                format_time(uncommitted_states[0]["start_time"]),
                format_time(uncommitted_states[-1]["end_time"])))

        uncommitted_states.append(group_state)
        for state in uncommitted_states:
            state.update({"shift": shift, "diff": diff})
            log_shift(state)
        committed_states.extend(uncommitted_states)
        del uncommitted_states[:]
        idx += 1

    for state in uncommitted_states:
        log_shift(state)

    for idx, (search_group, group_state) in enumerate(zip(groups_list, chain(committed_states, uncommitted_states))):
        if group_state["shift"] is None:
            for group in reversed(groups_list[:idx]):
                link_to = next((x for x in reversed(group) if not x.linked), None)
                if link_to:
                    for e in search_group:
                        e.link_event(link_to)
                    break
        else:
            for e in search_group:
                e.set_shift(group_state["shift"], group_state["diff"])


def check_file_exists(path, file_title):
    if path and not os.path.exists(path):
        raise SushiError("{0} file doesn't exist".format(file_title))


def format_full_path(temp_dir, base_path, postfix):
    if temp_dir:
        return os.path.join(temp_dir, os.path.basename(base_path) + postfix)
    else:
        return base_path + postfix


def create_directory_if_not_exists(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def run(args):
    ignore_chapters = args.chapters_file is not None and args.chapters_file.lower() == 'none'
    write_plot = plot_enabled and args.plot_path
    if write_plot:
        plt.clf()
        plt.ylabel('Shift, seconds')
        plt.xlabel('Event index')

    # first part should do all possible validation and should NOT take significant amount of time
    check_file_exists(args.source, 'Source')
    check_file_exists(args.destination, 'Destination')
    check_file_exists(args.src_timecodes, 'Source timecodes')
    check_file_exists(args.dst_timecodes, 'Source timecodes')
    check_file_exists(args.script_file, 'Script')

    if not ignore_chapters:
        check_file_exists(args.chapters_file, 'Chapters')
    if args.src_keyframes not in ('auto', 'make'):
        check_file_exists(args.src_keyframes, 'Source keyframes')
    if args.dst_keyframes not in ('auto', 'make'):
        check_file_exists(args.dst_keyframes, 'Destination keyframes')

    if (args.src_timecodes and args.src_fps) or (args.dst_timecodes and args.dst_fps):
        raise SushiError('Both fps and timecodes file cannot be specified at the same time')

    src_demuxer = Demuxer(args.source)
    dst_demuxer = Demuxer(args.destination)

    if src_demuxer.is_wav and not args.script_file:
        raise SushiError("Script file isn't specified")

    if (args.src_keyframes and not args.dst_keyframes) or (args.dst_keyframes and not args.src_keyframes):
        raise SushiError('Either none or both of src and dst keyframes should be provided')

    create_directory_if_not_exists(args.temp_dir)

    # selecting source audio
    if src_demuxer.is_wav:
        src_audio_path = args.source
    else:
        src_audio_path = format_full_path(args.temp_dir, args.source, '.sushi.wav')
        src_demuxer.set_audio(stream_idx=args.src_audio_idx, output_path=src_audio_path, sample_rate=args.sample_rate)

    # selecting destination audio
    if dst_demuxer.is_wav:
        dst_audio_path = args.destination
    else:
        dst_audio_path = format_full_path(args.temp_dir, args.destination, '.sushi.wav')
        dst_demuxer.set_audio(stream_idx=args.dst_audio_idx, output_path=dst_audio_path, sample_rate=args.sample_rate)

    # selecting source subtitles
    if args.script_file:
        src_script_path = args.script_file
    else:
        stype = src_demuxer.get_subs_type(args.src_script_idx)
        src_script_path = format_full_path(args.temp_dir, args.source, '.sushi' + stype)
        src_demuxer.set_script(stream_idx=args.src_script_idx, output_path=src_script_path)

    script_extension = get_extension(src_script_path)
    if script_extension not in ('.ass', '.srt'):
        raise SushiError('Unknown script type')

    # selection destination subtitles
    if args.output_script:
        dst_script_path = args.output_script
        dst_script_extension = get_extension(args.output_script)
        if dst_script_extension != script_extension:
            raise SushiError("Source and destination script file types don't match ({0} vs {1})"
                             .format(script_extension, dst_script_extension))
    else:
        dst_script_path = format_full_path(args.temp_dir, args.destination, '.sushi' + script_extension)

    # selecting chapters
    if args.grouping and not ignore_chapters:
        if args.chapters_file:
            if get_extension(args.chapters_file) == '.xml':
                chapter_times = chapters.get_xml_start_times(args.chapters_file)
            else:
                chapter_times = chapters.get_ogm_start_times(args.chapters_file)
        elif not src_demuxer.is_wav:
            chapter_times = src_demuxer.chapters
            output_path = format_full_path(args.temp_dir, src_demuxer.path, ".sushi.chapters.txt")
            src_demuxer.set_chapters(output_path)
        else:
            chapter_times = []
    else:
        chapter_times = []

    # selecting keyframes and timecodes
    if args.src_keyframes:
        def select_keyframes(file_arg, demuxer):
            auto_file = format_full_path(args.temp_dir, demuxer.path, '.sushi.keyframes.txt')
            if file_arg in ('auto', 'make'):
                if file_arg == 'make' or not os.path.exists(auto_file):
                    if not demuxer.has_video:
                        raise SushiError("Cannot make keyframes for {0} because it doesn't have any video!"
                                         .format(demuxer.path))
                    demuxer.set_keyframes(output_path=auto_file)
                return auto_file
            else:
                return file_arg

        def select_timecodes(external_file, fps_arg, demuxer):
            if external_file:
                return external_file
            elif fps_arg:
                return None
            elif demuxer.has_video:
                path = format_full_path(args.temp_dir, demuxer.path, '.sushi.timecodes.txt')
                demuxer.set_timecodes(output_path=path)
                return path
            else:
                raise SushiError('Fps, timecodes or video files must be provided if keyframes are used')

        src_keyframes_file = select_keyframes(args.src_keyframes, src_demuxer)
        dst_keyframes_file = select_keyframes(args.dst_keyframes, dst_demuxer)
        src_timecodes_file = select_timecodes(args.src_timecodes, args.src_fps, src_demuxer)
        dst_timecodes_file = select_timecodes(args.dst_timecodes, args.dst_fps, dst_demuxer)

    # after this point nothing should fail so it's safe to start slow operations
    # like running the actual demuxing
    src_demuxer.demux()
    dst_demuxer.demux()

    try:
        if args.src_keyframes:
            src_timecodes = Timecodes.cfr(args.src_fps) if args.src_fps else Timecodes.from_file(src_timecodes_file)
            src_keytimes = [src_timecodes.get_frame_time(f) for f in keyframes.parse_keyframes(src_keyframes_file)]

            dst_timecodes = Timecodes.cfr(args.dst_fps) if args.dst_fps else Timecodes.from_file(dst_timecodes_file)
            dst_keytimes = [dst_timecodes.get_frame_time(f) for f in keyframes.parse_keyframes(dst_keyframes_file)]

        script = AssScript.from_file(src_script_path) if script_extension == '.ass' else SrtScript.from_file(src_script_path)
        script.sort_by_time()

        src_stream = WavStream(src_audio_path, sample_rate=args.sample_rate, sample_type=args.sample_type)
        dst_stream = WavStream(dst_audio_path, sample_rate=args.sample_rate, sample_type=args.sample_type)

        search_groups = prepare_search_groups(script.events,
                                              source_duration=src_stream.duration_seconds,
                                              chapter_times=chapter_times,
                                              max_ts_duration=args.max_ts_duration,
                                              max_ts_distance=args.max_ts_distance)

        calculate_shifts(src_stream, dst_stream, search_groups,
                         normal_window=args.window,
                         max_window=args.max_window,
                         rewind_thresh=args.rewind_thresh if args.grouping else 0)

        events = script.events

        if write_plot:
            plt.plot([x.shift for x in events], label='From audio')

        if args.grouping:
            if not ignore_chapters and chapter_times:
                groups = groups_from_chapters(events, chapter_times)
                for g in groups:
                    fix_near_borders(g)
                    smooth_events([x for x in g if not x.linked], args.smooth_radius)
                groups = split_broken_groups(groups)
            else:
                fix_near_borders(events)
                smooth_events([x for x in events if not x.linked], args.smooth_radius)
                groups = detect_groups(events)

            if write_plot:
                plt.plot([x.shift for x in events], label='Borders fixed')

            for g in groups:
                start_shift = g[0].shift
                end_shift = g[-1].shift
                avg_shift = average_shifts(g)
                logging.info('Group (start: {0}, end: {1}, lines: {2}), '
                             'shifts (start: {3}, end: {4}, average: {5})'
                             .format(format_time(g[0].start), format_time(g[-1].end), len(g), start_shift, end_shift,
                                     avg_shift))

            if args.src_keyframes:
                for e in (x for x in events if x.linked):
                    e.resolve_link()
                for g in groups:
                    snap_groups_to_keyframes(g, chapter_times, args.max_ts_duration, args.max_ts_distance, src_keytimes,
                                             dst_keytimes, src_timecodes, dst_timecodes, args.max_kf_distance, args.kf_mode)
        else:
            fix_near_borders(events)
            if write_plot:
                plt.plot([x.shift for x in events], label='Borders fixed')

            if args.src_keyframes:
                for e in (x for x in events if x.linked):
                    e.resolve_link()
                snap_groups_to_keyframes(events, chapter_times, args.max_ts_duration, args.max_ts_distance, src_keytimes,
                                         dst_keytimes, src_timecodes, dst_timecodes, args.max_kf_distance, args.kf_mode)

        for event in events:
            event.apply_shift()

        script.save_to_file(dst_script_path)

        if write_plot:
            plt.plot([x.shift + (x._start_shift + x._end_shift) / 2.0 for x in events], label='After correction')
            plt.legend(fontsize=5, frameon=False, fancybox=False)
            plt.savefig(args.plot_path, dpi=300)

    finally:
        if args.cleanup:
            src_demuxer.cleanup()
            dst_demuxer.cleanup()
