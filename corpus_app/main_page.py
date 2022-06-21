import streamlit as st
from pathlib import Path
import shutil

if not Path("model").is_dir():
    shutil.unpack_archive("model.zip")

if not Path("corpora").is_dir():
    shutil.unpack_archive("corpora.zip")

st.title("Работа с корпусом")

st.markdown('''
### Функционал:
- Конкорданс
- Коллигации
- Коллокации
- Н-граммы
''')
