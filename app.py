import streamlit as st
import spacy
import re
from wordfreq import zipf_frequency
import io

@st.cache_resource
def load_nlp():
    # Streamlit will look for the model installed via requirements.txt
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

st.set_page_config(page_title="Vocab Extractor", page_icon="📖")
st.title("🎯 High-Yield Vocab Extractor")

uploaded_file = st.file_uploader("Upload 'class6_text.txt'", type="txt")

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    
    if st.button("Extract Words"):
        clean_text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        doc = nlp(clean_text)
        
        unique_words = set()
        for token in doc:
            if token.is_stop or token.is_punct or len(token.text) < 3:
                continue
            if token.pos_ not in ['ADJ', 'VERB', 'ADV']:
                continue
            
            lemma = token.lemma_.lower()
            if zipf_frequency(lemma, 'en') < 5.0:
                unique_words.add(lemma)

        results = sorted(list(unique_words))
        st.success(f"Extracted {len(results)} words!")
        st.text_area("Results", value="\n".join(results), height=300)
        
        st.download_button("Download TXT", "\n".join(results), "vocab.txt")
