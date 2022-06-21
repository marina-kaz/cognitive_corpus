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

st.title("Коллигации")

coll_phrase = st.text_input(
    label="Введите слово/фразу для поиска коллигаций",
    value="Концепты",
    placeholder="Слово/фраза"
)

context_type = st.selectbox(
    label="Выберите тип контекста",
    options=["Левый", "Правый"]
)

max_step_size = st.select_slider(
    label="Выберите шаг для поиска контекста",
    options=[1, 2, 3]
)

col1, col2 = st.columns(2)

with col1:
    pos_1 = st.text_input(label="Введите часть речи в UD формате", placeholder="NOUN", key="pos_1")
    pos_2 = st.text_input(label="Введите часть речи в UD формате", placeholder="NOUN", key="pos_2")
    pos_3 = st.text_input(label="Введите часть речи в UD формате", placeholder="NOUN", key="pos_3")

with col2:
    feats_1 = st.text_input(
        label="Введите грамматические признаки в UD формате",
        placeholder="Case=Nom|Degree=Pos|Number=Plur",
        key="feats_1"
    )

    feats_2 = st.text_input(
        label="Введите грамматические признаки в UD формате",
        placeholder="Nom|Pos|Plur",
        key="feats_2"
    )

    feats_3 = st.text_input(
        label="Введите грамматические признаки в UD формате",
        placeholder="Nom,Pos,Plur",
        key="feats_3"
    )

coll_df = pd.DataFrame(columns=["Левый контекст", "Слово/фраза", "Правый контекст", "Источник"])
st.dataframe(coll_df)






