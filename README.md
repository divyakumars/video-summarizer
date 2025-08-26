# 🎬 Video Summarizer  

A simple and interactive web app that automatically **summarizes videos** into concise text using **AI & NLP techniques**. Built with **Python, Streamlit, and Machine Learning**, this tool helps save time by extracting the core content from videos.  

🔗 **Live Demo**: [Video Summarizer App](https://videosummarizerpy21.streamlit.app/)  

---

## 🚀 Features  
- 📥 Upload video files for summarization  
- 🧠 Extracts audio and converts it into text using **Speech-to-Text**  
- ✂️ Generates concise summaries using **Natural Language Processing (NLP)**  
- ⚡ Fast, simple, and user-friendly Streamlit interface  
- ☁️ Deployed online for easy access  

---

## 🛠️ Tech Stack  
- **Frontend / UI**: Streamlit  
- **Backend**: Python  
- **AI/ML Tools**:  
  - NLP (Text Summarization models)  
  - Speech Recognition (for converting audio to text)  
- **Other Libraries**:  
  - `moviepy` – for video/audio processing  
  - `nltk / transformers` – for text summarization  
  - `streamlit` – for UI and deployment  

---

## 📂 Project Structure  
```bash
video-summarizer/
│── video_summarozer.py              # Main Streamlit app
│── requirements.txt                 # Dependencies
│── utils/                           # Helper functions
│── README.md                        # Project documentation
│── package.txt                      # Dependencies
```
## ⚙️ Installation & Setup

Clone the repository:

```bash
git clone https://github.com/divyakumars/video-summarizer.git
cd video-summarizer
```
## Create and activate a virtual environment:

```
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Mac/Linux
```

## Install dependencies:
```
pip install -r requirements.txt

```
## Run the Streamlit app locally:
```
streamlit run app.py
```
## 📸 Screenshots
<img width="1133" height="892" alt="Screenshot 2025-08-26 104434" src="https://github.com/user-attachments/assets/39f178ea-05e1-4230-9bb8-9d50e4d550c0" />


## Contributing

Contributions are welcome!
Feel free to fork this repo, create an issue, or submit a pull request.
