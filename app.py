import streamlit as st
from page1 import page_load
from page2 import page_load_2
from page3 import mostrar_inferencia

# Selector de página
st.sidebar.title("Navegación")
page = st.sidebar.selectbox("Selecciona la página", ["Exploración", "Entrenamiento", "Inferencia"])

if page == "Exploración":
    page_load()
elif page == "Entrenamiento":
    page_load_2()
elif page == "Inferencia":
    mostrar_inferencia()

