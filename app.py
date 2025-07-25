import streamlit as st
from views import start_page, train_page

st.set_page_config(page_title="ML Hub", layout='wide')

# Set default stage
if "stage" not in st.session_state:
    st.session_state.stage = "select"

# Page routing
if st.session_state.stage == "select":
    start_page.show()
elif st.session_state.stage == "train":
    train_page.show()
