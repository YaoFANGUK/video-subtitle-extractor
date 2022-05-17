if __name__ == '__main__':
    # 导入QPT
    from qpt.executor import CreateExecutableModule as CEM
    import os
    WORK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(WORK_DIR)
    LAUNCH_PATH = os.path.join(WORK_DIR, 'gui.py')
    SAVE_PATH = os.path.join(os.path.dirname(WORK_DIR), 'vse_out')
    ICON_PATH = os.path.join(WORK_DIR, "design", "vse.ico")
    module = CEM(work_dir=WORK_DIR, launcher_py_path=LAUNCH_PATH, save_path=SAVE_PATH, icon=ICON_PATH, hidden_terminal=False)
    # 开始打包
    module.make()
