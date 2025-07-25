import streamlit as st
import pandas as pd
from utils.start_utils import get_classifiers, get_classifier_map, get_datasets, load_sample_dataset

def show():
    st.title("🤖 ML Hub - Get Your Model")

    model_options = get_classifiers()
    selected_model = st.selectbox("Select your model", model_options)
    st.session_state.selected_model = selected_model

    classifier_map = get_classifier_map()
    st.session_state.selected_model_class = classifier_map[selected_model]

    dataset_option = st.radio("Choose a dataset", ["Use sample dataset", "Upload dataset"])

    if dataset_option == "Use sample dataset":
        dataset_names = get_datasets()
        display_names = [name.replace("load_", "") + " dataset" for name in dataset_names]
        dataset_map = {name.replace("load_", "") + " dataset": name for name in dataset_names}

        dataset_display = st.selectbox("Select a dataset", display_names)
        loader_name = dataset_map[dataset_display]
        st.session_state.dataset_loader_name = loader_name

        df = load_sample_dataset(loader_name)
        st.session_state.df = df

    else:
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df

    if "df" in st.session_state:
        st.subheader("📊 Preview of your data")
        st.dataframe(st.session_state.df.sample(5))

    if st.button("Continue"):
        if "df" in st.session_state:
            st.session_state.stage = "train"
        else:
            st.error("Please upload or select a dataset")
