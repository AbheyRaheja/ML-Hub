# 🤖 ML Hub – Interactive ML Model Trainer with Streamlit

**ML Hub** is a powerful no-code machine learning tool built with **Streamlit**.  
It allows you to train, visualize, and export ML models using scikit-learn, with just a few clicks.

Whether you're a beginner exploring models or a developer testing pipelines, ML Hub helps you do it interactively and fast.

---

## 🚀 Features

✅ Select any classification algorithm from scikit-learn  
✅ Use built-in datasets or upload your own CSV  
✅ Tune model hyperparameters via the sidebar  
✅ Train and evaluate models instantly  
✅ Visual output:
- 📌 Feature scatter plot
- 📊 Confusion matrix
- 🎯 Accuracy score  
✅ Export full training bundle as a `.zip` containing:
- `model.pkl` – trained model
- `data.csv` – dataset used
- `train_pipeline.py` – complete training code

---

## 🧠 Tech Stack

-  Python
-  Streamlit
-  scikit-learn
-  pandas, matplotlib

---

## 🗂️ Project Structure

ML Hub/
    ├── app.py 
    ├── views/
    │ ├── start_page.py 
    │ └── train_page.py
    ├── utils/
    │ ├── start_utils.py
    │ ├── train_utils.py 
    │ └── generate_zip.py 
    ├── temp_export/
    ├── requirements.txt 

---

## Sample Output (ZIP Export)

trained_model.zip
├── model.pkl           
├── data.csv           
└── train_pipeline.py

---

## Future Enhancements

 Add regression and clustering models

 Enable export as .ipynb notebook

 Add FastAPI backend for serving predictions

 Save training history and models per session

 Deploy to Streamlit Cloud or Hugging Face Spaces