from collections import Counter
import logging
from pathlib import Path

import pandas as pd
import spacy
from spacy.tokens import DocBin
from spacy.matcher import PhraseMatcher
from st_aggrid import AgGrid
import streamlit as st

logging.info('Starting n-grams')

st.title("Н-граммы")

rel_path = Path.cwd()
logging.info("Onto model loading")
nlp = spacy.load(rel_path / "model")
logging.info("Onto corpus loading")
doc_bin = DocBin().from_disk(rel_path / "corpora/corpus.spacy")
docs = list(doc_bin.get_docs(nlp.vocab))
logging.info("Loaded resources")

query = st.text_input(
    label="Введите слово/фразу для поиска по n-граммам",
    value="образ"
)

n = st.select_slider(
    label="Выберите порядок n-граммы",
    options=[2, 3, 4, 5, 6, 7, 8, 9, 10],
    value=2
)

consider_punctuation = st.checkbox('Учитывать пунктуацию')
consider_digits = st.checkbox('Учитывать цифры')
consider_stop_words = st.checkbox('Учитывать стоп-слова')

logging.info('Received input, onto matching')
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
phrase_matcher.add(query, [nlp(query)])


def get_ngrams(tokens, n):
    ngrams = []
    for idx in range(len(tokens) - n + 1):
        ngram = tokens[idx:idx+n]
        ngrams.append(ngram)
    return ngrams


def is_relevant(token):
    if token.is_space:
        return False
    if not consider_punctuation and token.is_punct:
        return False
    if not consider_digits and token.is_digit:
        return False
    if not consider_stop_words and token.is_stop:
        return False
    return True


matched_docs = 0
matched_ngrams = []

for idx, doc in enumerate(docs):
    matches = phrase_matcher(doc, as_spans=True)

    if matches:
        matched_docs += 1
        doc_to_pass = [i.lower_
                       for i in doc[matches[0].start:matches[-1].end + 1]
                       if is_relevant(i)]
        ngrams = get_ngrams(doc_to_pass, n)
        for match in set([match.text for match in matches]):
            for ngram in ngrams:
                if match in ngram:
                    matched_ngrams.append(tuple(ngram))

st.write(f'Всего найдены вхождения в {matched_docs} документов')
logging.info(f'Appropriate matches are found, onto projecting')
output = Counter(matched_ngrams).most_common()[:100]
sequences = [' '.join(i[0]) for i in output]
frequencies = [i[1] for i in output]
resulting_df = pd.DataFrame({'Вхождения': frequencies, 'Фрагменты': sequences})
AgGrid(resulting_df)


st.download_button(
     label="Скачать как таблицу",
     data=resulting_df.to_csv(index=False, encoding='windows-1251'),
     file_name=f'ngrams_for_{" ".join(query.split())}.csv'
 )
