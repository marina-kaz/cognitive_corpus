import os

import streamlit as st
import spacy
import json
import re
from itertools import chain
from tqdm.notebook import tqdm
import pandas as pd
from pathlib import Path
from spacy.tokens import Doc, DocBin
from spacy import Language
from spacy.matcher import Matcher, PhraseMatcher

st.title("Конкорданс")
print("fsdgs", os.getcwd())

# @st.cache
# def get_resources(path="/Users/asbabiy/Documents/study/fcm/corpus_app"):
#     """
#     Gets all recources
#     """
#     rel_path = Path(path)
#     model = spacy.load(rel_path / "model")
#
#     doc_bin = DocBin().from_disk(rel_path / "corpora/corpus.spacy")
#     texts = list(doc_bin.get_docs(model.vocab))
#
#     return model, texts

rel_path = Path("/Users/asbabiy/Documents/study/fcm/corpus_app")

nlp = spacy.load(rel_path / "model")
doc_bin = DocBin().from_disk(rel_path / "corpora/corpus.spacy")

docs = list(doc_bin.get_docs(nlp.vocab))

# nlp, docs = get_resources()

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


def get_concordance_item(match):
    """
    Gets concordance item
    """

    rel_span = match.text
    left_context = "..." + match.doc[:match.start][-10:].text
    right_context = match.doc[match.end:][:10].text + "..."

    left_context = re.sub(r"-\n", "", left_context)
    right_context = re.sub(r"-\n", "", right_context)

    left_context = re.sub(r"\s+", " ", left_context)
    right_context = re.sub(r"\s+", " ", right_context)

    doc_meta = match.doc.user_data
    meta = (
        f"Автор: {doc_meta['author']}\n\n"
        f"Заголовок: {doc_meta['title']}\n\n"
        f"Источник: {doc_meta['source']}\n\n"
        f"URL: {doc_meta['url']}"
    )

    return left_context, rel_span, right_context, meta


conc_items = [get_concordance_item(match) for match in matches]
conc_df = pd.DataFrame(conc_items, columns=["Левый контекст", "Слово/фраза", "Правый контекст", "Источник"])
st.write(conc_df)