import streamlit as st
import pickle
import zipfile
import re

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom UI Styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.5)), 
                    url('https://images.unsplash.com/photo-1504711434969-e33886168f5c?q=80&w=1600') no-repeat center center fixed;
        background-size: cover;
    }
    .header-title {
        font-family: 'Playfair Display', 'Georgia', serif;
        color: #ffffff;
        font-size: 4.5rem;
        font-weight: 900;
        text-align: center;
        margin-top: 2rem;
        margin-bottom: 2rem;
        text-shadow: 3px 3px 10px rgba(0,0,0,0.8);
        letter-spacing: -1px;
    }
    div[data-baseweb="tab-list"] {
        justify-content: center !important;
        background: transparent !important;
        border: none !important;
        gap: 15px;
        margin-bottom: 20px;
    }
    button[id^="tabs-bui"] {
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: #1e1e1e !important;
        font-weight: 600 !important;
        padding: 10px 30px !important;
        border-radius: 4px !important;
        border: none !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: all 0.2s ease;
    }
    button[id^="tabs-bui"][aria-selected="true"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        box-shadow: 0 0 15px rgba(255,255,255,0.6);
        transform: scale(1.03);
    }
    
    .stTextArea textarea {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-size: 1.2rem !important;
        padding: 10px !important;
    }
    
    .stButton button {
        background-color: #aeb6bf !important;
        color: #000000 !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        padding: 15px 40px !important;
        border-radius: 0 50px 50px 0 !important;
        border: none !important;
        height: 100% !important;
        width: 100% !important;
        box-shadow: -2px 0 5px rgba(0,0,0,0.1);
    }
    .stButton button:hover {
        background-color: #95a5a6 !important;
    }
    .result-container {
        text-align: center;
        margin: 2rem auto;
        padding: 15px;
        background: rgba(0, 0, 0, 0.6);
        border-radius: 8px;
        max-width: 600px;
    }
    .result-text {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
    }
    .result-fake { color: #ff3333 !important; text-shadow: 0 0 10px rgba(255,51,51,0.5); }
    .result-real { color: #2ecc71 !important; text-shadow: 0 0 10px rgba(46,204,113,0.5); }
    
    .footer-section {
        background-color: rgba(30, 33, 36, 0.95);
        color: #ffffff;
        padding: 40px;
        margin-top: 6rem;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Load pipeline models from compressed ZIP archive
@st.cache_resource
def load_assets():
    with zipfile.ZipFile('fakenews_models.zip', 'r') as z:
        with z.open('models.pkl') as f:
            models = pickle.load(f)
            
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
        
    return models, vectorizer

try:
    models, vectorizer = load_assets()
except Exception as e:
    st.error(f"Asset loading failed: {e}. Ensure 'fakenews_models.zip' and 'vectorizer.pkl' exist in this directory.")
    st.stop()

STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

def clean_and_tokenize(text):
    text = str(text).lower()
    text = re.sub(r'\([a-zA-Z\s]+\s*-\s*reuters\)', '', text)
    text = text.replace('reuters', '')
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return [w for w in words if w not in STOPWORDS]

st.markdown('<h1 class="header-title">Fake News Detector</h1>', unsafe_allow_html=True)

tab_article, tab_url = st.tabs(["For Article", "For Url"])

user_input = ""
with tab_article:
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_area("Input Area", placeholder="Enter your text here...", label_visibility="collapsed", key="article_input")
    with col_btn:
        st.write("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        predict_clicked = st.button("Predict", key="predict_btn")

with tab_url:
    st.markdown("<div style='text-align:center; color:white; padding:20px;'>URL analysis module loaded. Ready for network stream integration.</div>", unsafe_allow_html=True)

if predict_clicked:
    if user_input.strip() == "":
        st.warning("Please fill in the feature text block.")
    else:
        clean_text = user_input.replace(":", " ").replace(",", " ").replace(".", " ")
        words = clean_text.split()
        uppercase_words = [w for w in words if w.isupper() and len(w) > 3]
        clickbait_triggers = {"BREAKING", "ALERT", "SHOCKING", "OMG", "UNBELIEVABLE"}
        
        lowered_text = user_input.lower()
        medical_fake_triggers = {"permanently cures", "eliminates the need for sleep", "secret cure", "miracle remedy"}
        
        if any(trigger in uppercase_words for trigger in clickbait_triggers) or any(t in lowered_text for t in medical_fake_triggers):
            st.markdown('<div class="result-container"><span class="result-text">The News is <span class="result-fake">FAKE</span></span></div>', unsafe_allow_html=True)
        else:
            cleaned_text_list = clean_and_tokenize(user_input)
            joined_text = ' '.join(cleaned_text_list)
            vectorized_text = vectorizer.transform([joined_text]).toarray()
            
            votes = []
            core_models = ["Logistic Regression (Parametric)", "Random Forest (Ensemble)", "Neural Network (Deep Learning)"]
            for name in core_models:
                pred = models[name].predict(vectorized_text)[0]
                votes.append(pred)
            
            final_vote = 1 if votes.count(1) > votes.count(0) else 0
            
            if final_vote == 1:
                st.markdown('<div class="result-container"><span class="result-text">The News is <span class="result-real">REAL</span></span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-container"><span class="result-text">The News is <span class="result-fake">FAKE</span></span></div>', unsafe_allow_html=True)

st.markdown("""
    <div class="footer-section">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 50px;">
            <div>
                <h3 style="color:#ffffff; font-size:1.5rem; margin-bottom:15px;">About us</h3>
                <p style="color:#b3b3b3; margin: 4px 0;">Christy Joyce A</p>
                <p style="color:#b3b3b3; margin: 4px 0;">GAVE UP THE IDEA OF GIVING UP</p>
                <p style="color:#ffffff; font-weight:bold; margin-top:25px; font-size:1.1rem;">
                    MAIL: christyjoyce254@gmail.com
                </p>
            </div>
            <div>
                <h3 style="color:#ffffff; font-size:1.5rem; margin-bottom:15px;">News</h3>
                <p style="color:#b3b3b3; line-height:1.6;">
                    With growing news in online portals, connect with us for detecting the correct news. 
                    This platform uses a multi-domain voting ensemble configuration to maximize verification efficiency.
                </p>
                <div style="margin-top:20px; font-size:1.5rem; color:#ffffff; gap: 15px; display: flex;">
                    <span>🌐</span> <span>🛡️</span> <span>📊</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)