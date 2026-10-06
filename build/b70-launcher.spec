# -*- mode: python ; coding: utf-8 -*-

# data files land in _internal; launcher.py resolves HERE to _MEIPASS when the
# bundle's exe directory has no web/index.html next to it.

a = Analysis(
    ['../launcher.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('../webwindow.py', '.'),
        ('../appwindow.py', '.'),
        ('../patches/patch_mtp_nightly.py', 'patches'),
        ('../patches/patch_mtp_boundary.py', 'patches'),
        ('../patches/patch_vllm_worker_affinity.py', 'patches'),
        ('../web/index.html', 'web'),
        ('../web/assets/b70-launcher.svg', 'web/assets'),
        ('../web/assets/b70-launcher-512.png', 'web/assets'),
        ('../web/assets/b70-launcher-256.png', 'web/assets'),
        ('../web/assets/b70-launcher-128.png', 'web/assets'),
        ('../web/assets/b70-launcher-48.png', 'web/assets'),
        ('../web/assets/b70-launcher-32.png', 'web/assets'),
        ('../web/assets/b70-launcher-16.png', 'web/assets'),
        ('../recipes.json', '.'),
        ('../settings.json', '.'),
        ('../README.md', '.'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='b70-launcher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='b70-launcher',
)
