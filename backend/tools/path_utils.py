import os
import sys


def resource_path(relative_path):
    """Get absolute path to a bundled resource.

    When running in a PyInstaller bundle, resources are extracted to sys._MEIPASS.
    In development, resolve relative to the project root (parent of backend/).
    """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    # dev mode: project root is two levels up from this file
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), relative_path)


def app_path(relative_path):
    """Get path relative to the executable directory for user-writable data.

    In a PyInstaller bundle, this is next to the .app/.exe.
    In development, same as resource_path (project root).
    """
    if getattr(sys, 'frozen', False):
        return os.path.join(os.path.dirname(sys.executable), relative_path)
    return resource_path(relative_path)
