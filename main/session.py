import streamlit as st

def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'emotion' not in st.session_state:
        st.session_state.emotion = None
    if 'initial_question_asked' not in st.session_state:
        st.session_state.initial_question_asked = False
