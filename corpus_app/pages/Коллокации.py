import streamlit as st
import pandas as pd

st.title("Коллокации")

coll_phrase = st.text_input(
    label="Введите слово для поиска н-грамм",
    value="Концептуальный",
    placeholder="Слово"
)

max_step_size = st.select_slider(
    label="Выберите тип н-грамм",
    options=[1, 2, 3, 4]
)

coll_df = pd.DataFrame(columns=["Левый контекст", "Н-грамма", "Правый контекст", "Источник"])
st.dataframe(coll_df)