# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['gui.py'],
             pathex=['/Users/yao/anaconda3/envs/subEnv/lib/python3.7/site-packages', '/Users/yao/Github/video-subtitle-extractor'],
             binaries=[('/Users/yao/Github/video-subtitle-extractor/dylib/libgeos_c.dylib', '.')],
             datas=[('/Users/yao/Github/video-subtitle-extractor/backend', 'backend'),
                    ('/Users/yao/Github/video-subtitle-extractor/vse.ico', '.')
                   ],
             hiddenimports=['imgaug', 'skimage.filters.rank.core_cy_3d',
                            'pyclipper', 'lmdb'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='vse',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False , icon='vse.ico')
app = BUNDLE(exe,
             name='Subtitle Extractor.app',
             icon='vse.ico',
             bundle_identifier=None)
