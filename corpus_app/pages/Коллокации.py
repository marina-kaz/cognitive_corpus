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
    label="Введите слово для поиска коллокаций",
    value="Концептуальный",
    placeholder="Слово"
)

consider_punctuation = st.checkbox('Учитывать пунктуацию')
consider_digits = st.checkbox('Учитывать цифры')
consider_stop_words = st.checkbox('Учитывать стоп-слова')

threshold = st.number_input(
    label="Токены с какой минимальной встречаемостью учитывать при подсчете мер ассоциации?",
    value=30
)

condition = st.selectbox(
                         'Какое положение искомого токена учитывать?',
                         ('любое положение',
                          'токен должен быть только слева',
                          'токен должен быть только справа'),
                          )

condition = {'токен должен быть только слева': 'left_only',
             'токен должен быть только справа': 'right_only',
             'любое положение': 'both'}[condition]

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


# @st.cache
def get_relevant_doc(doc):
    return [token for token in doc if is_relevant(token)]


def get_relevant_left(match):
    for idx in range(1, match.start):
        if is_relevant(match.doc[match.start - idx]):
            return match.doc[match.start - idx]


def get_relevant_right(match):
    for idx in range(0, len(match.doc) - match.start - len(query.split())):
        if is_relevant(match.doc[match.end + idx]):
            return match.doc[match.end + idx]


phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
phrase_matcher.add(query, [nlp(query)])

query_occurences = 0
bigrams = []
contexts = {}
for doc in docs:
    matches = phrase_matcher(doc, as_spans=True)
    if matches:
        query_occurences += len(matches)
        for match in matches:

            left_neighbor = get_relevant_left(match)
            if left_neighbor:
                bigrams.append((left_neighbor.lemma_, match.lemma_))
                contexts[bigrams[-1]] = match.doc[match.start - 15: match.end + 15]

            right_neighbor = get_relevant_right(match)
            if right_neighbor:
                bigrams.append((match.lemma_, right_neighbor.lemma_))
                contexts[bigrams[-1]] = match.doc[match.start - 15: match.end + 15]
            query_lemma = match.lemma_

logging.info(f'Appropriate matches are found, onto scores calculating')

# all_tokens = []
# _ = [all_tokens.extend([token.lemma_ for token in get_relevant_doc(doc)]) for doc in docs]
# counterparts_occurences = Counter(all_tokens)


@st.cache(persist=True, allow_output_mutation=True, suppress_st_warning=True)
def get_counterpart_occurences():
    st.write('Для оптимизации работы поиска необходимо один раз подождать подгрузки всех нужных статистик.')
    all_tokens = []
    for doc in docs:
        all_tokens.extend([token.lemma_ for token in get_relevant_doc(doc)])
    st.write('Спасибо за Ваше терпение!')
    return all_tokens, Counter(all_tokens)


all_tokens, counterparts_occurences = get_counterpart_occurences()


def is_collocation_corrupted(collocation):
  if collocation[0] == query_lemma:
    return True if counterparts_occurences[collocation[-1]] < threshold else False
  if collocation[-1] == query_lemma:
    return True if counterparts_occurences[collocation[0]] < threshold else False


bigrams_filtered = [collocation for collocation in bigrams
                    if not is_collocation_corrupted(collocation)]
collocation_occurences = Counter(bigrams_filtered)

def is_collocation_acceptable(collocation, condition):
  if not condition in ['left_only', 'right_only', 'both']:
    raise ValueError
  if condition == 'left_only':
    return True if collocation[0] == query_lemma else False
  if condition == 'right_only':
    return True if collocation[-1] == query_lemma else False
  if condition == 'both':
    return True

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
             for collocation in collocation_occurences.keys()
             if is_collocation_acceptable(collocation, condition)]
t_scores = [get_score(collocation, score='t')
            for collocation in collocation_occurences.keys()
            if is_collocation_acceptable(collocation, condition)]
dice_scores = [get_score(collocation, score='dice')
               for collocation in collocation_occurences.keys()
               if is_collocation_acceptable(collocation, condition)]
context = ['...' + str(contexts[collocation]).replace('\n', '') + '...'
           for collocation in collocation_occurences.keys()
           if is_collocation_acceptable(collocation, condition)]

logging.info(f'Scores calculated, onto projecting')

def split_into_words(collocation):
    words = []
    for word in collocation:
        words.extend(word.split())
    return words

df = pd.DataFrame({'коллокация': [' + '.join(split_into_words(collocation))
                                  for collocation in collocation_occurences.keys()
                                  if is_collocation_acceptable(collocation, condition)],
                   'MI': mi_scores,
                   't-score': t_scores,
                   'Dice': dice_scores,
                   'контекст': context})

st.write('Результаты отсортированы согласно', to_sort_by)
AgGrid(df.sort_values(by=[to_sort_by], ascending=False))

sorted_df = df.sort_values(by=[to_sort_by], ascending=False)
st.download_button(
     label="Скачать как таблицу",
     data=sorted_df.to_csv(index=False, sep=','),
     file_name=f'collocations_for_{"_".join(query.split())}.csv'
 )

word = st.text_input(
    label="Введите лемму, чтобы узнать ее частотность",
    value="метафора",
)
st.write(counterparts_occurences[word])

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
