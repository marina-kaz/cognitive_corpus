from itertools import chain
import logging
from pathlib import Path

import pandas as pd
import spacy
from spacy.tokens import Doc, DocBin
from spacy.matcher import Matcher, PhraseMatcher
import streamlit as st
from st_aggrid import AgGrid

logging.info('Starting concordance')

st.title("Конкорданс")


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

conc_phrase = st.text_input(
    label="Введите слово/фразу для поиска по конкордансу",
    placeholder="Слово/фраза"
)

phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
phrase_matcher.add(conc_phrase, [nlp(conc_phrase)])

matches = []
for idx, doc in enumerate(docs):
    match = phrase_matcher(doc, as_spans=True)
    if match:
        matches.append(match)
matches = list(chain.from_iterable(matches))
logging.info('Found appropriate matches')


def get_concordance_item(match):
    """
    Gets concordance item
    """
    rel_span = match.text
    left_context, right_context = '', ''
    for sentence in match.doc.sents:
        sentence_matches = phrase_matcher(sentence, as_spans=True)
        if not sentence_matches:
            continue
        for sentence_match in sentence_matches:
            left_context = match.doc[sentence.start:sentence_match.start]
            right_context = match.doc[sentence_match.end:sentence.end]
    doc_meta = match.doc.user_data
    meta = (
        f"Автор: {doc_meta['author']}\n\n"
        f"Заголовок: {doc_meta['title']}\n\n"
        f"Источник: {doc_meta['source']}\n\n"
        f"URL: {doc_meta['url']}"
    )
    return left_context, rel_span, right_context, meta

conc_items = [get_concordance_item(match) for match in matches]
n_occurences = len(conc_items)
conc_items = set(conc_items)
logging.info(f'Found {len(conc_items)} concordances, onto projeting')
conc_df = pd.DataFrame(conc_items, columns=["Левый контекст", "Слово/фраза",
                                            "Правый контекст", "Источник"])
AgGrid(conc_df)

corpus_volume = 0
for doc in docs:
    corpus_volume += len(doc)
st.write(f'Найдено {n_occurences} вхождений, IPM: {(n_occurences / corpus_volume) * 1000000}')

st.download_button(
     label="Скачать как таблицу",
     data=conc_df.to_csv(index=False, sep=','),
     file_name=f'contexts_for_{" ".join(" ".join(conc_phrase.split()).split())}.csv'
 )
