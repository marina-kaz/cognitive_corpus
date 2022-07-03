import logging
from pathlib import Path
import shutil

import spacy
import streamlit as st
from spacy.tokens import DocBin


logging.info('Onto model unpacking')
if not Path("model").is_dir():
    shutil.unpack_archive(Path.cwd() / "corpus_app" / "model.zip")
logging.info('Onto corpora unpacking')
if not Path("corpus").is_dir():
    shutil.unpack_archive(Path.cwd() / "corpus_app" / "corpus.zip")


st.title("Работа с корпусом")

st.markdown('''
### Функционал:
- Конкорданс
- Коллокации
- Н-граммы
''')

st.markdown('''##### Пожалуйста, дождитесь окончания работы функции load_resources.''')
st.write('Это займет около минуты')

@st.cache(persist=True, allow_output_mutation=True)
def load_resources(model_path, corpus_path):
    nlp = spacy.load(model_path)
    doc_bin = DocBin().from_disk(corpus_path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    return nlp, docs


logging.info("Onto resource loading")
rel_path = Path.cwd()
model_path = rel_path / "model"
corpus_path = rel_path / "corpus" / "large_corpus.spacy"
nlp, docs = load_resources(model_path, corpus_path)
logging.info("Loaded resources")

st.write('Спасибо за ожидание!')
st.write(f'Загруженный корпус содержит {len(docs)} документов')
