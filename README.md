# TourSage 🎵

TourSage is an intelligent Streamlit app for managing, querying, and searching concert tour documents using Retrieval-Augmented Generation (RAG) and modern NLP models. It features robust error handling, a beautiful UI, and support for both free and paid Hugging Face API tiers.

---

## Features
- **Document Ingestion & Deduplication**: Upload and analyze concert tour documents with AI-powered classification, summarization, and information extraction. Duplicate documents are automatically detected and skipped for a clean, efficient database.
- **Ask Questions**: Query your ingested documents for instant, concise, and non-repetitive answers using RAG.
- **Search Concerts**: Get real-time concert information for any artist using SerpAPI and AI-powered extraction. Results are deduplicated and combined from both your local database and the internet.
- **Advanced Text Cleaning**: All extracted and displayed information is aggressively cleaned to remove code, test logic, and technical clutter before display.
- **Robust Fallbacks for LLM/API Failures**: If the LLM is rate-limited or fails, the system gracefully falls back to regex or default extraction, always returning a valid result.
- **Modern, Accessible UI**: Feature card texts and all main UI elements are styled for maximum readability and accessibility.
- **User-Friendly Error Handling**: All errors are caught and displayed with actionable suggestions.

---

## Quick Start: Local Setup

### 1. Requirements
- **Python 3.11** (strongly recommended; do NOT use 3.12+ due to LangChain and Pydantic compatibility)
- `git`, `pip`, and a modern browser

### 2. Clone the Repository
```bash
git clone <your-repo-url>
cd TourSage
```

### 3. Create and Activate a Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

#### Key Dependencies
- `streamlit==1.32.0`
- `langchain==0.1.12`
- `faiss-cpu==1.7.4`
- `sentence-transformers==2.5.1`
- `python-dotenv==1.0.1`
- `pydantic==2.6.4`
- `plotly==6.0.1`
- `pandas==2.2.1`
- `numpy==1.24.3`
- `huggingface-hub==0.19.4`
- `transformers==4.35.2`
- `torch==2.7.0`
- `google-search-results==2.4.2`
- `tiktoken==0.5.1`
- `pytest`, `black`, `isort`, `flake8` (for development)

### 5. Set Up API Keys
Create a `.env` file in the project root with:
```
HF_API_KEY=your_huggingface_api_key
SERPAPI_API_KEY=your_serpapi_key
```
- Get a free Hugging Face API key: https://huggingface.co/settings/tokens
- Get a free SerpAPI key: https://serpapi.com/

---

## Running the App
```bash
streamlit run src/app.py
```
Visit [http://localhost:8501](http://localhost:8501) in your browser.

---

## How to Use (Walkthrough)
1. **Add Document:**  
   Upload or paste a concert tour document. The system will deduplicate, summarize, and extract structured info (artist, tour name, dates, venues, etc.).
2. **Ask Questions:**  
   Query your ingested documents for instant, concise answers. The system avoids repetition and uses only the most relevant context.
3. **Search Concerts:**  
   Find real-time concert info for any artist. Results are deduplicated, cleaned, and combined from both your local database and the internet.

---

## Hugging Face API: Free Tier Limitations & Model Notes
- **Timeouts and Model Unavailability**: The free Hugging Face Inference API is slow and often times out, especially for text generation and summarization. This is NOT a bug in TourSage—it's a limitation of the free API tier.
- **Model Availability**: Only a few models (e.g., `facebook/bart-large-cnn` for summarization, `gpt2` for text generation, `facebook/bart-large-mnli` for classification) are available for free. Most instruction-tuned models (FLAN-T5, Mistral, etc.) are NOT available on the free tier.
- **Error Handling**: If the API times out or returns unstructured output, the app will show the raw model output and user-friendly suggestions. For best results, use a paid Hugging Face Inference Endpoint or run models locally.
- **How to Improve**: For production or better reliability, upgrade to a paid Hugging Face endpoint or run your own model server.

---

## Compatibility & Troubleshooting
- **Python Version**: Use Python 3.11. Other versions (3.12+) may cause issues with LangChain, Pydantic, and FAISS.
- **Pydantic & LangChain**: The app is built for Pydantic v2 and LangChain 0.1.x. If you see serialization/deserialization errors (e.g., with FAISS vector store), delete the `data/documents/vector_store` directory and let the app rebuild it.
- **Dangerous Deserialization**: When loading the FAISS vector store, the app uses `allow_dangerous_deserialization=True` because the file is created locally and is safe. Never use this flag with untrusted files.
- **Dependency Issues**: If you see `ModuleNotFoundError` or version conflicts, ensure your virtual environment is active and run `pip install -r requirements.txt` again.
- **Timeouts & API Errors**: These are almost always due to the free API tier, not your code. Try again later, or upgrade your API plan.

## Document Relevance & Classification Logic
- The app uses a multi-label classifier to determine if a document is relevant to concert tours.
- **Threshold**: The relevance threshold is set to 0.3 (was 0.7), making the app less strict and more inclusive of relevant documents.
- **Labels**: The classifier uses multiple labels: `concert tour`, `music event`, `concert`, `tour`, `unrelated`.
- **Logic**: A document is accepted if the `concert tour` score is above 0.3 **or** if the top label is `concert tour`.
- If your relevant document is still flagged, check the classification scores (displayed in the UI) and consider adjusting the threshold or labels further for your use case.

---

## Error Handling & UI Improvements
- **Raw Output Display**: If the model output cannot be parsed, the app shows the raw output and suggestions for next steps.
- **User-Friendly Feedback**: All features (document processing, question answering, concert search) provide clear error messages and actionable advice.
- **Progress Indicators**: The UI shows step-by-step progress for document processing.
- **Modern Design**: The app uses custom CSS and animations for a professional look.
- **Feature Card Texts**: All homepage feature cards use black text for maximum readability.

---

## Common Issues & Solutions
- **Timeouts or 504 Errors**: Wait and try again, or use a paid Hugging Face endpoint.
- **Vector Store Errors**: Delete the `data/documents/vector_store` directory and restart the app.
- **Dependency Errors**: Reinstall dependencies in a clean Python 3.11 virtual environment.
- **API Key Errors**: Make sure your `.env` file is set up and keys are valid.

---

## Credits
Developed by [Abid Hossain](https://www.abidhossain.me)

---

## Best Practices & Tips
- Always use a virtual environment.
- Use Python 3.11 for best compatibility.
- For production, use a paid API or run models locally for speed and reliability.
- If you update dependencies, check for breaking changes in LangChain, Pydantic, and FAISS.

---

## Changelog & Session Summary
- Migrated from OpenAI to Hugging Face API for all NLP tasks.
- Added robust error handling for all API/model calls.
- Improved UI with modern design, progress bars, and clear feedback.
- Updated requirements for Python 3.11, Pydantic v2, and LangChain 0.1.x.
- Added support for safe FAISS deserialization and troubleshooting steps.
- Documented all known issues and solutions in this README.
- **Document deduplication**: Prevents duplicate documents from being ingested.
- **Cleaner extraction & display**: All extracted information is strictly formatted as clean JSON, with no code or extraneous text.
- **Advanced text cleaning**: Search results and extracted info are aggressively cleaned to remove any code, test logic, or technical clutter before display.
- **Robust fallback for LLM/API failures**: If the LLM is rate-limited or fails, the system gracefully falls back to regex or default extraction, always returning a valid result.
- **Improved QA**: Answers are concise, non-repetitive, and deduplicated, even if the same document is ingested multiple times.
- **Enhanced search logic**: Combines local vector search and internet search, deduplicates results, and provides debug info for troubleshooting.
- **Modern, accessible UI**: Feature card texts are now black for better readability. The UI is more visually accessible and user-friendly.

---

For any issues or feature requests, please open an issue or contact Abid Hossain via [abidhossain.me](https://www.abidhossain.me). 