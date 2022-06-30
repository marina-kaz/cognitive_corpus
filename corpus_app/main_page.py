import logging
from pathlib import Path
import shutil

import streamlit as st

logging.info('Onto model unpacking')
if not Path("model").is_dir():
    shutil.unpack_archive("model.zip")
logging.info('Onto corpora unpacking')
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
