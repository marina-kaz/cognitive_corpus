from collections import Counter
import logging
from pathlib import Path
from math import log, sqrt

import pandas as pd
import spacy
from spacy.tokens import DocBin
from spacy.matcher import PhraseMatcher
from st_aggrid import AgGrid
import streamlit as st

logging.info('Starting collocations')

st.title("Коллокации")

rel_path = Path.cwd()
logging.info("Onto model loading")
nlp = spacy.load(rel_path / "model")
logging.info("Onto corpus loading")
doc_bin = DocBin().from_disk(rel_path / "corpora/corpus.spacy")
docs = list(doc_bin.get_docs(nlp.vocab))
logging.info("Loaded resources")

query = st.text_input(
    label="Введите слово для поиска коллокаций",
    value="Концептуальный",
    placeholder="Слово"
)

consider_punctuation = st.checkbox('Учитывать пунктуацию')
consider_digits = st.checkbox('Учитывать цифры')
consider_stop_words = st.checkbox('Учитывать стоп-слова')

to_sort_by = st.selectbox(
                         'По какой мере ассоциации отсортировать выдачу?',
                         ('MI', 't-score', 'Dice')
                          )
logging.info('Received input, onto matching')


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


def get_relevant_doc(doc):
    return [token for token in doc if is_relevant(token)]


def get_ngrams(tokens, n):
    ngrams = []
    for idx in range(len(tokens) - n + 1):
        ngram = tokens[idx:idx+n]
        ngrams.append(ngram)
    return ngrams


def get_relevant_left(match):
    for idx in range(1, 100000):
        if is_relevant(match.doc[match.start - idx]):
            return match.doc[match.start - idx]


def get_relevant_right(match):
    for idx in range(0, 100000):
        if is_relevant(match.doc[match.end + idx]):
            return match.doc[match.end + idx]


phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
phrase_matcher.add(query, [nlp(query)])

query_occurences = 0
counterparts = []
bigrams = []
for doc in docs:
    matches = phrase_matcher(doc, as_spans=True)
    if matches:
        query_occurences += len(matches)
        for match in matches:

            left_neighbor = get_relevant_left(match)
            counterparts.append(left_neighbor.lemma_)
            bigrams.append((left_neighbor.lemma_, match.lemma_))

            right_neighbor = get_relevant_right(match)
            counterparts.append(right_neighbor.lemma_)
            bigrams.append((match.lemma_, right_neighbor.lemma_))
            query_lemma = match.lemma_

logging.info(f'Appropriate matches are found, onto scores calculating')

all_tokens = []
_ = [all_tokens.extend([token.lemma_ for token in get_relevant_doc(doc)]) for doc in docs]
counterparts_occurences = Counter(all_tokens)
collocation_occurences = Counter(bigrams)


def get_score(collocation, score='mi'):
    f_n_c = collocation_occurences[collocation]
    f_n = counterparts_occurences[collocation[0]]
    f_c = counterparts_occurences[collocation[-1]]
    if collocation[0] == query_lemma:
        f_n = query_occurences
    if collocation[-1] == query_lemma:
        f_c = query_occurences
    N = len(all_tokens)
    if score not in ['mi', 't', 'dice']:
        raise ValueError('wrong type of score, available values: mi, t, dice')
    if score == 'mi':
        return log((f_n_c * N) / (f_n * f_c), 2)
    if score == 't':
        return (f_n_c - ((f_n * f_c)  / N)) / sqrt(f_n_c)
    if score == 'dice':
        return log((2 * f_n_c) / (f_c + f_n), 2)


mi_scores = [get_score(collocation, score='mi')
             for collocation in collocation_occurences]
t_scores = [get_score(collocation, score='t')
            for collocation in collocation_occurences]
dice_scores = [get_score(collocation, score='dice')
               for collocation in collocation_occurences]

logging.info(f'Scores calculated, onto projecting')

df = pd.DataFrame({'коллокация': [' + '.join(collocation) for collocation in collocation_occurences.keys()],
                   'MI': mi_scores,
                   't-score': t_scores,
                   'Dice': dice_scores})

st.write('Результаты отсортированы согласно', to_sort_by)
AgGrid(df.sort_values(by=[to_sort_by], ascending=False))

sorted_df = df.sort_values(by=[to_sort_by], ascending=False)
st.download_button(
     label="Скачать как таблицу",
     data=sorted_df.to_csv(index=False),
     file_name=f'collocations_for_{" ".join(query.split())}.csv'
 )

st.write('Меры ассоциации были рассчитаны по следующим формулам')
st.latex(r'''
MI score = log_2(\frac{f(n, c) \times N}{f(n) \times f(c)})
''')
st.latex(r'''
t-score=\frac{f(n, c) - \frac{f(n)\times f(c)}{N}}{\sqrt{f(n,c)}}
''')
st.latex(r'''
Dice=log_2(\frac{2 \times f(n, c)}{f(n) + f(c)})
''')
