import streamlit as st
import inspect
from utils.train_utils import train_model, plot_confusion_matrix, plot_scatter
from utils.generate_zip import generate_zip

def show():
    st.title(f"⚙️ Configure {st.session_state.selected_model}")

    df = st.session_state.df
    model_class = st.session_state.selected_model_class

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.sample(5))

    st.sidebar.title("🔧 Hyperparameter Tuning")
    hyperparams = {}

    model_signature = inspect.signature(model_class.__init__)
    for name, param in model_signature.parameters.items():
        if name == 'self':
            continue

        default = param.default

        if isinstance(default, bool):
            hyperparams[name] = st.sidebar.checkbox(name, value=default)
        elif isinstance(default, int):
            hyperparams[name] = st.sidebar.number_input(name, value=default, step=1)
        elif isinstance(default, float):
            hyperparams[name] = st.sidebar.number_input(name, value=default)
        elif isinstance(default, str):
            hyperparams[name] = st.sidebar.text_input(name, value=default)
        # NoneType: use text input and auto-convert
        elif default is None:
            user_input = st.sidebar.text_input(name, value="", help="Leave empty for None")
            user_input = user_input.strip()

            if user_input == "":
                hyperparams[name] = None
            else:
                # Special handling for common known types
                try:
                    # Try boolean
                    if user_input.lower() in ["true", "false"]:
                        hyperparams[name] = user_input.lower() == "true"
                    # Try float
                    elif "." in user_input:
                        hyperparams[name] = float(user_input)
                    # Try int
                    elif user_input.isdigit() or (user_input.startswith('-') and user_input[1:].isdigit()):
                        hyperparams[name] = int(user_input)
                    else:
                        # Fallback to string (e.g. 'balanced')
                        hyperparams[name] = user_input
                except Exception:
                    hyperparams[name] = user_input  # final fallback

        # Handle special cases where empty string should be None
        elif default is inspect.Parameter.empty:
            user_input = st.sidebar.text_input(name, value="", help="Leave empty for default")
            user_input = user_input.strip()
            
            if user_input == "":
                continue  # Skip this parameter, let it use default
            else:
                # Same conversion logic as above
                try:
                    if user_input.lower() in ["true", "false"]:
                        hyperparams[name] = user_input.lower() == "true"
                    elif "." in user_input:
                        hyperparams[name] = float(user_input)
                    elif user_input.isdigit() or (user_input.startswith('-') and user_input[1:].isdigit()):
                        hyperparams[name] = int(user_input)
                    else:
                        hyperparams[name] = user_input
                except Exception:
                    hyperparams[name] = user_input
        else:
            continue

    # Clean up problematic parameters - convert empty strings to None for object parameters
    object_params = ['estimator', 'base_estimator', 'base_classifier', 'criterion', 'splitter']
    for param_name in object_params:
        if param_name in hyperparams and hyperparams[param_name] == '':
            hyperparams[param_name] = None

    st.sidebar.markdown("---")
    st.sidebar.markdown("📚 **Need help?** [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)")

    target_col = st.selectbox("🎯 Select Target Column", df.columns, index=len(df.columns)-1)

    if st.button("🔁 Train Model"):
        result = train_model(model_class, hyperparams, df, target_col)

        if result["success"]:
            st.success(f"✅ Model trained successfully!")

            # Scatter plot
            st.subheader("📌 Feature Scatter Plot")
            scatter_fig = plot_scatter(result["X_test"], result["y_test"])
            st.pyplot(scatter_fig)

            # Accuracy
            st.markdown(f"**🎯 Accuracy:** `{result['accuracy']:.4f}`")

            # Confusion matrix
            st.subheader("📊 Confusion Matrix")
            cm_fig = plot_confusion_matrix(result["model"], result["X_test"], result["y_test"], figsize=(4, 4))
            st.pyplot(cm_fig)

            # Save download
            zip_path = generate_zip(
                result["model"],
                result["X_test"].copy(),         # X
                result["y_test"].copy(),         # y
                st.session_state.selected_model, # model name
                hyperparams                      # params
            )

            with open(zip_path, "rb") as f:
                st.download_button("📦 Download Training Bundle (.zip)", f, file_name="trained_model.zip")


        else:
            st.error(f"❌ Training failed: {result['error']}")

    if st.button("🔙 Go Back"):
        st.session_state.stage = "select"