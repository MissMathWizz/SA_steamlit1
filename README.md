# Streamlit QA Interface (Gemini-powered)

## 📦 Requirements
Before running this app, make sure the following are complete:

1. **Install the required packages**:
```bash
pip install --upgrade --user google-cloud-aiplatform pymupdf rich streamlit 
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas streamlit Ipython google-cloud-aiplatform pymupdf rich streamlit 

```

2. **Authenticate with Google Cloud (Colab/Notebook)**:
```python
from google.colab import auth
auth.authenticate_user()
```
Or from CLI:
```bash
gcloud auth application-default login
```

3. **Set your project ID**:
```bash
gcloud config set project salesforce-assistant-457220
```

4. **Make sure `text_metadata_df`, `image_metadata_df`, and the Gemini model are loaded**  
The app assumes they're globally available and passed into the function. You may need to adapt the loading.

5. ##use ngrok to share the demo
```bash
brew install ngrok 
streamlit run streamlit_app.py
ngrok http 8501
```
 http://127.0.0.1:4040  


## 🚀 Run the App
```bash
streamlit run streamlit_app.py
```
Then open `http://localhost:8501` in your browser.

## 📁 Structure
- `streamlit_app.py`: Main interface
- `backend_functions.py`: All your logic (copied from the notebook)



## ⚠️ Vertex AI SDK (Deprecated Notice)
This app uses the legacy import path:
```python
from vertexai.preview.generative_models import GenerativeModel
```
However, as of April 2025, this SDK path has been deprecated or removed from some environments (especially local Python 3.13 environments).
<img width="759" alt="image" src="https://github.com/user-attachments/assets/a417395a-12cd-45ce-bcce-46d7e7758cd2" />
