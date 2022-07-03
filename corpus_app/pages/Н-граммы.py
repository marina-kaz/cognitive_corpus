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

query = st.text_input(
    label="Введите слово/фразу для поиска по n-граммам",
    value="образ"
)

n = st.select_slider(
    label="Выберите порядок n-граммы",
    options=[2, 3, 4, 5, 6, 7, 8, 9, 10],
    value=2
)

position = st.selectbox("На каком месте должен быть искомый токен?",
                        tuple(["на любом"] + [i + 1 for i in range(n)]))

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


def find_position(sequence, subsequence):
  streak = 0
  for idx in range(len(sequence)):
    for step in range(len(subsequence)):
      if idx + step > len(sequence) - 1:
        return False
      if sequence[idx + step] == subsequence[step]:
        streak += 1
      else:
        streak = 0
      if streak == len(subsequence):
        return idx + 1
  return False


matched_docs = 0
matched_ngrams = []

for idx, doc in enumerate(docs):
    matches = phrase_matcher(doc, as_spans=True)

    if matches:
        matched_docs += 1
        doc_to_pass = [i.lemma_
                       for i in doc[matches[0].start:matches[-1].end + 1]
                       if is_relevant(i)]
        ngrams = get_ngrams(doc_to_pass, n - 1 + len(query.split()))
        match_lemma = matches[0].lemma_
        if position == 'на любом':
            matched_ngrams.extend([tuple(ngram)
                                   for ngram in ngrams
                                   if find_position(ngram,
                                                    match_lemma.split())])
        else:
            matched_ngrams.extend([tuple(ngram)
                                   for ngram in ngrams
                                   if position == find_position(ngram,
                                                                match_lemma.split())])

st.write(f'Всего найдены вхождения в {matched_docs} документов')
logging.info(f'Appropriate matches are found, onto projecting')
output = Counter(matched_ngrams).most_common()[:100]
sequences = [' '.join(i[0]) for i in output]
frequencies = [i[1] for i in output]
resulting_df = pd.DataFrame({'Вхождения': frequencies, 'Фрагменты': sequences})
AgGrid(resulting_df)


st.download_button(
     label="Скачать как таблицу",
     data=resulting_df.to_csv(index=False, sep=','),
     file_name=f'ngrams_for_{" ".join(query.split())}.csv'
 )
