# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['D:/ANPR/ANPR/run.py'],
             pathex=['D:\\ANPR\\ANPR'],
             binaries=[],
             datas=[('D:/ANPR/ANPR/bullet-camera.ico', '.'), ('D:/ANPR/ANPR/config.ini', '.'), ('D:/ANPR/ANPR/database.txt', '.'), ('D:/ANPR/ANPR/weights', 'weights/')],
             hiddenimports=['configparser'],
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
          [],
          exclude_binaries=True,
          name='run',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True , icon='D:\\ANPR\\ANPR\\bullet-camera.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='run')
