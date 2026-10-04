"""Import paddle, replacing its cryptic ccache lookup output with one clear note.

Importing ``paddle`` runs ``where ccache`` (``which`` on POSIX) with stderr
inherited, which prints "INFO: Could not find files for the given pattern(s)."
on Windows, followed by a "No ccache found" UserWarning. ccache is only used when
compiling C++ extensions, which this project never does.

Import this module before any other ``paddle``/``paddleocr`` import. Only the
ccache lookup output is replaced; other output is untouched.
"""
import os
import shutil
import subprocess
import sys
import warnings

CCACHE_NOTE = (
    "[Note] ccache is not installed. This is optional: it only speeds up "
    "compiling Paddle C++ extensions, which this app does not do. "
    "Install it system-wide (e.g. `winget install ccache`) only if you build such extensions."
)

_NOTE_ENV = 'VSE_CCACHE_NOTE_SHOWN'


def _import_paddle():
    if 'paddle' in sys.modules:
        return sys.modules['paddle']

    original = subprocess.check_output

    def check_output(args, *a, **kw):
        if isinstance(args, (list, tuple)) and list(args[1:]) == ['ccache']:
            kw.setdefault('stderr', subprocess.DEVNULL)
        return original(args, *a, **kw)

    subprocess.check_output = check_output
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='No ccache found')
            import paddle
    finally:
        subprocess.check_output = original
    # The environment is inherited by child processes, so the note prints once per run.
    if shutil.which('ccache') is None and not os.environ.get(_NOTE_ENV):
        os.environ[_NOTE_ENV] = '1'
        print(CCACHE_NOTE, file=sys.__stderr__)
    return paddle


paddle = _import_paddle()
