# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Скрытые импорты — для динамически загружаемых модулей
hidden_imports = (
    collect_submodules('nltk') +
    collect_submodules('transformers') +
    collect_submodules('pdfplumber') +
    collect_submodules('torch') +
    collect_submodules('textstat') +
    collect_submodules('textblob') +
    collect_submodules('sklearn') +
    collect_submodules('pandas')
)

# Сбор необходимых данных
datas = (
    collect_data_files('nltk') +
    collect_data_files('transformers') +
    collect_data_files('pdfplumber') +
    collect_data_files('textblob')
)

# Добавляем собственные папки и шаблоны
for folder in ['app/model', 'app/static', 'app/templates']:
    if os.path.exists(folder):
        datas.append((folder, folder))

# Включаем также всю папку app
datas.append(('app', 'app'))

block_cipher = None

a = Analysis(
    ['run.py'],                  # главный файл
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='run',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='run'
)


