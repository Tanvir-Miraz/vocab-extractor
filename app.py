import streamlit as st
import spacy
import re
from wordfreq import zipf_frequency
import io
from PIL import Image

# 1. Page Configuration (Sets the browser tab icon)
try:
    img = Image.open("logo.png")
    st.set_page_config(page_title="Vocab Extractor", page_icon=img)
except:
    st.set_page_config(page_title="Vocab Extractor", page_icon="📖")

# Load NLP model
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

# 2. Sidebar/Header Branding
st.sidebar.image("logo.png", width=150) if "img" in locals() else None
st.title("🎯 High-Yield Vocab Extractor")
st.write("Upload your text file to extract academic Verbs, Adjectives, and Adverbs.")

# 3. File Uploader
uploaded_file = st.file_uploader("Choose a .txt file", type="txt")

if uploaded_file is not None:
    # Read text
    text = uploaded_file.read().decode("utf-8")
    
    # 4. Process Button
    if st.button("Extract Vocabulary"):
        # Clean text
        clean_text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        doc = nlp(clean_text)
        
        unique_words = set()
        
        with st.spinner("Analyzing text..."):
            for token in doc:
                # Filter: Stops, Punctuation, Short words
                if token.is_stop or token.is_punct or len(token.text) < 3:
                    continue

                # POS Filter (Focus on descriptive/action words)
                if token.pos_ not in ['ADJ', 'VERB', 'ADV']:
                    continue

                # Frequency Filter (Zipf < 5.0 for academic words)
                lemma = token.lemma_.lower()
                freq = zipf_frequency(lemma, 'en')

                if freq < 5.0:
                    unique_words.add(lemma)

        sorted_words = sorted(list(unique_words))
        
        # 5. Display & Download Results
        st.success(f"Found {len(sorted_words)} high-yield words!")
        
        # Download Button
        output_data = f"Found {len(sorted_words)} words:\n" + "\n".join(sorted_words)
        st.download_button(
            label="📥 Download Results (.txt)",
            data=output_data,
            file_name="high_yield_vocab.txt",
            mime="text/plain"
        )
        
        st.text_area("Word List Preview", "\n".join(sorted_words), height=300)
