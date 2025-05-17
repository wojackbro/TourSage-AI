import os
import streamlit as st
from dotenv import load_dotenv
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
from langchain.schema import Document
import dateutil.parser as parser

from document_processor import DocumentProcessor, DocumentSummary
from rag_engine import RAGEngine
from search_engine import SearchEngine
from utils import validate_api_keys

# Load environment variables
load_dotenv()

# Validate API keys
is_valid, error_message = validate_api_keys()
if not is_valid:
    st.error(error_message)
    st.stop()

# Initialize components
document_processor = DocumentProcessor()
rag_engine = RAGEngine()
search_engine = SearchEngine(api_key=os.getenv("SERPAPI_API_KEY"))

# Set page config
st.set_page_config(
    page_title="TourSage 🎵",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS with animations and modern styling
st.markdown("""
    <style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100%); }
        to { transform: translateX(0); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    body, .main {
        padding: 2rem;
        background: linear-gradient(135deg, #fff700 0%, #4CAF50 100%);
        animation: fadeIn 0.5s ease-out;
        min-height: 100vh;
    }
    
    .stTextArea textarea {
        height: 200px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
    }
    
    .response-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s ease-out;
    }
    
    .stProgress > div > div {
        background-color: #4CAF50;
        transition: width 0.3s ease;
    }
    
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .stAlert {
        border-radius: 10px;
        animation: fadeIn 0.5s ease-out;
    }
    
    .stMarkdown {
        color: #333333;
    }
    
    .stSubheader {
        color: #2c3e50;
        font-weight: 600;
    }
    
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border-radius: 10px;
        padding: 1rem;
        animation: fadeIn 0.5s ease-out;
    }
    
    .stWarning {
        background-color: #fff3cd;
        color: #856404;
        border-radius: 10px;
        padding: 1rem;
        animation: fadeIn 0.5s ease-out;
    }
    
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        border-radius: 10px;
        padding: 1rem;
        animation: fadeIn 0.5s ease-out;
    }
    
    .profile-card {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s ease-out;
    }
    
    .profile-card h2 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
    }
    
    .profile-card p {
        margin: 0.5rem 0;
        opacity: 0.9;
    }
    
    .profile-card a {
        color: white;
        text-decoration: none;
        border-bottom: 2px solid rgba(255,255,255,0.5);
        transition: all 0.3s ease;
    }
    
    .profile-card a:hover {
        border-bottom-color: white;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease-out;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Header with profile card
st.markdown("""
    <div class="profile-card">
        <h2>TourSage 🎵</h2>
        <p>Your intelligent concert tour information assistant</p>
        <p>Developed by <a href="https://www.abidhossain.me" target="_blank">Abid Hossain</a></p>
    </div>
""", unsafe_allow_html=True)

# Feature cards (with black text)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📝</div>
            <h3 style="color: #000;">Add Documents</h3>
            <p style="color: #000;">Upload and process concert tour documents with AI-powered analysis</p>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">❓</div>
            <h3 style="color: #000;">Ask Questions</h3>
            <p style="color: #000;">Get instant answers about concert tours from your documents</p>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <h3 style="color: #000;">Search Concerts</h3>
            <p style="color: #000;">Find real-time concert information for your favorite artists</p>
        </div>
    """, unsafe_allow_html=True)

# Sidebar with better styling
st.sidebar.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #ffffff;
        animation: slideIn 0.5s ease-out;
    }
    .sidebar .sidebar-content .stRadio > div {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f8f9fa;
        transition: all 0.3s ease;
    }
    .sidebar .sidebar-content .stRadio > div:hover {
        background-color: #e9ecef;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a function:", ["Add Document", "Ask Questions", "Search Concerts"])

# --- Move process_document function here ---
def process_document(file, hf_client, vector_store, progress_bar):
    """Process a document and add it to the vector store."""
    try:
        # Read document content
        content = file  # file is a string from st.text_area
        
        # Check relevance
        progress_bar.progress(20, "Checking document relevance...")
        relevance = hf_client.classify_text(
            content,
            ["concert tour", "music event", "concert", "tour", "unrelated"]
        )
        
        # Generate summary
        progress_bar.progress(40, "Generating summary...")
        with st.spinner("Summarizing..."):
            summary = hf_client.summarize_text(content)
        
        # Extract key information
        progress_bar.progress(60, "Extracting key information...")
        info = hf_client.generate_text(
            "Extract ONLY the following information from this concert tour document as a JSON object. "
            "Do NOT include any explanations, steps, or extra text. "
            "Format: {"
            "\"artist\": \"\", "
            "\"tour_name\": \"\", "
            "\"dates\": [], "
            "\"venues\": [], "
            "\"ticket_prices\": [], "
            "\"special_notes\": \"\""
            "} "
            f"Document: {content}"
        )
        
        try:
            info_dict = json.loads(info)
        except json.JSONDecodeError:
            info_dict = {"raw_text": info}
        
        # Create document object
        progress_bar.progress(80, "Creating document object...")
        doc = Document(
            page_content=content,
            metadata={
                "filename": "manual_entry.txt",
                "summary": summary,
                "info": info_dict,
                "relevance_scores": relevance
            }
        )
        
        # Add to vector store
        progress_bar.progress(90, "Adding to vector store...")
        vector_store.add_documents([doc])
        
        # Save to disk
        progress_bar.progress(95, "Saving to disk...")
        vector_store.save_local("data/documents/vector_store")
        progress_bar.progress(100, "Complete!")
        
        # Display success message and summary
        st.success("✅ Thank you for sharing! Your document has been successfully added to the database.")
        
        # Display the processed information
        st.markdown("""
            <div style='background-color: #ffffff; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 1rem 0;'>
                <h3 style='color: #2c3e50; margin-bottom: 1.5rem;'>Document Summary</h3>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f'<div class="response-box">{summary}</div>', unsafe_allow_html=True)
        
        st.markdown("""
            <div style='background-color: #ffffff; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 1rem 0;'>
                <h3 style='color: #2c3e50; margin-bottom: 1.5rem;'>Extracted Information</h3>
            </div>
        """, unsafe_allow_html=True)
        
        if "error" in info_dict:
            st.error(info_dict["error"])
        elif "raw_text" in info_dict:
            st.markdown(f'<div class="response-box">{info_dict["raw_text"]}</div>', unsafe_allow_html=True)
        else:
            for key, value in info_dict.items():
                if isinstance(value, list):
                    st.markdown(f"""
                        <div style='margin-bottom: 1rem;'>
                            <p style='color: #2c3e50; font-weight: 500;'>{key.title()}:</p>
                            <ul style='margin-left: 1.5rem;'>
                                {''.join(f'<li>{item}</li>' for item in value)}
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style='margin-bottom: 1rem;'>
                            <p style='color: #2c3e50; font-weight: 500;'>{key.title()}: {value}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
    except Exception as e:
        st.markdown(f"""
            <div class='stError'>
            Error processing document: {str(e)}<br><br>
            <strong>Suggestions:</strong>
            <ul>
                <li>Check your internet connection</li>
                <li>Verify that your Hugging Face API key is valid</li>
                <li>Try using a paid Hugging Face Inference Endpoint for better reliability</li>
                <li>Consider running models locally for more control</li>
            </ul>
            </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(0)

def is_junk_tour_name(name):
    import re
    if not name or name.strip() in {'"""', "''", ")", "(", "*", "-", "_", "", "The provided code defines"}:
        return True
    if re.match(r'^[^a-zA-Z0-9]+$', name):  # only symbols
        return True
    if len(name.strip()) < 3:
        return True
    junk_patterns = [
        r'def ', r'import ', r'pytest', r'FILEPATH', r'class ', r'assert', r'provided code', r'function returns',
        r'\breturn\b', r'\bself\.assert\w+\b', r'\btest_', r'\bmain\b', r'\bExample', r'\bcode to solve',
        r'\bfunction', r'\bscript', r'\btest case', r'\bprint\('
    ]
    for pat in junk_patterns:
        if re.search(pat, name, re.IGNORECASE):
            return True
    return False

if page == "Add Document":
    st.markdown("""
        <div style='background-color: #ffffff; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <h2 style='color: #2c3e50; margin-bottom: 1rem;'>Add New Document</h2>
            <p style='color: #666666; margin-bottom: 1.5rem;'>Paste your concert tour document below:</p>
        </div>
    """, unsafe_allow_html=True)
    
    document = st.text_area("Document", height=300)
    
    if st.button("Process Document", key="process_btn"):
        if document:
            progress_bar = st.progress(0)
            process_document(document, document_processor.hf_client, document_processor.vector_store, progress_bar)
        else:
            st.warning("Please enter a document to process.")

elif page == "Ask Questions":
    st.markdown("""
        <div style='background-color: #ffffff; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <h2 style='color: #2c3e50; margin-bottom: 1rem;'>Ask Questions</h2>
            <p style='color: #666666; margin-bottom: 1.5rem;'>Ask questions about the concert tours in our database:</p>
        </div>
    """, unsafe_allow_html=True)
    
    question = st.text_input("Your question")
    
    if st.button("Get Answer", key="answer_btn"):
        if question:
            with st.spinner("Searching for answer..."):
                context_docs = document_processor.get_relevant_chunks(question)
                
                if context_docs:
                    answer = rag_engine.answer_question(question, context_docs)
                    st.markdown("### Answer")
                    if isinstance(answer, dict) and "error" in answer:
                        st.error(answer["error"])
                        if "suggestions" in answer:
                            st.markdown(f'<div class="response-box">{answer["suggestions"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="response-box">{answer}</div>', unsafe_allow_html=True)
                else:
                    st.warning("No relevant information found in the database.")
        else:
            st.warning("Please enter a question.")

else:  # Search Concerts
    st.markdown("""
        <div style='background-color: #ffffff; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <h2 style='color: #2c3e50; margin-bottom: 1rem;'>Search Concert Information</h2>
            <p style='color: #666666; margin-bottom: 1.5rem;'>Search for real-time concert information about an artist:</p>
        </div>
    """, unsafe_allow_html=True)
    
    artist = st.text_input("Artist/Band Name")
    
    if st.button("Search", key="search_btn"):
        if artist:
            with st.spinner("Searching for concert information..."):
                results = search_engine.search_concerts(artist, document_processor=document_processor)
                
                def remove_explanation_blocks(text):
                    import re
                    text = re.sub(r'#+\s*Explanation[\s\S]*?(?=\n\n|$)', '', text)
                    text = re.sub(r'\*\s+The `.*?` function.*?\n', '', text)
                    # Remove code-like lines and artifacts
                    code_patterns = [
                        r'^if __name__ == .__main__.:.*$', r'^python.*$', r'^import re.*$', r'^\s*$', r'^\s*""".*"""$', r'^\s*#.*$', r'^\s*def .*$', r'^\s*class .*$', r'^\s*print\(.*\).*$'
                    ]
                    for pat in code_patterns:
                        text = re.sub(pat, '', text, flags=re.MULTILINE)
                    return text.strip()
                
                def sanitize_output(text):
                    text = search_engine.sanitize_output(text)
                    text = remove_explanation_blocks(text)
                    return text.strip()
                
                if results["status"] == "success":
                    st.success(results["message"])
                    # Deduplicate and aggregate results
                    seen_keys = set()
                    deduped = []
                    for r in results["results"]:
                        key = (r.get("tour_name", ""), tuple(r.get("dates", [])), tuple(r.get("venues", [])))
                        if key not in seen_keys:
                            deduped.append(r)
                            seen_keys.add(key)
                    # Show deduped results (cleaned, no summary)
                    unique_headings = set()
                    for r in deduped:
                        tour_name_clean = sanitize_output(r.get('tour_name', 'Upcoming Tour'))
                        if is_junk_tour_name(tour_name_clean):
                            tour_name_clean = 'Upcoming Tour'
                        # Deduplicate by heading
                        if tour_name_clean in unique_headings:
                            continue
                        unique_headings.add(tour_name_clean)
                        with st.expander(f"🎵 {artist} - {tour_name_clean}", expanded=False):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("### Tour Dates")
                                for date in r.get('dates', []):
                                    date_clean = sanitize_output(date)
                                    if date_clean:
                                        st.markdown(f"- {date_clean}")
                                st.markdown("### Venues")
                                for venue in r.get('venues', []):
                                    venue_clean = sanitize_output(venue)
                                    if venue_clean:
                                        st.markdown(f"- {venue_clean}")
                            with col2:
                                if r.get('ticket_info'):
                                    ticket_info_clean = sanitize_output(r.get('ticket_info'))
                                    if ticket_info_clean:
                                        st.markdown("### Ticket Information")
                                        st.markdown(ticket_info_clean)
                                st.markdown("### Source Information")
                                st.markdown(f"Last Updated: {sanitize_output(r.get('last_updated', 'N/A'))}")
                                st.markdown(f"Confidence Score: {r.get('confidence_score', 0):.1%}")
                                source_url = r.get("source_url")
                                if source_url and source_url != "Local Document":
                                    st.markdown(f"[View Source]({source_url})")
                                elif source_url == "Local Document":
                                    st.markdown("Source: Local Document")
                else:
                    st.error(results["message"])
        else:
            st.warning("Please enter an artist/band name.")

# Footer with better styling and animation
st.markdown("""
    <div style='text-align: center; padding: 2rem 0; margin-top: 2rem; border-top: 1px solid #e0e0e0; animation: fadeIn 0.5s ease-out;'>
        <p style='color: #666666;'>TourSage - Your intelligent concert tour information assistant</p>
        <p style='color: #666666; font-size: 0.9rem;'>Developed by <a href="https://www.abidhossain.me" target="_blank" style='color: #4CAF50; text-decoration: none;'>Abid Hossain</a></p>
    </div>
""", unsafe_allow_html=True) 