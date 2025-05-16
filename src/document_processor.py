import os
from typing import Tuple, Optional, Dict, Any, List
from datetime import datetime
import json
from pathlib import Path
import logging
from dateutil import parser
import re
from dateutil.relativedelta import relativedelta
import hashlib

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field, validator
from langchain_core.embeddings import Embeddings

from hf_utils import HuggingFaceClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentSummary(BaseModel):
    """Schema for document summary."""
    is_relevant: bool = Field(description="Whether the document is relevant to concert tours")
    summary: str = Field(description="Summary of the document content")
    tour_dates: list[str] = Field(description="List of tour dates mentioned in the document")
    venues: list[str] = Field(description="List of venues mentioned in the document")
    artists: list[str] = Field(description="List of artists/bands mentioned in the document")
    confidence_score: float = Field(description="Confidence score for the relevance", default=0.0)

    @validator('tour_dates')
    def validate_dates(cls, dates):
        """Validate and standardize dates to YYYY-MM-DD format."""
        valid_dates = []
        for date in dates:
            try:
                # Try to parse the date using dateutil
                parsed_date = parser.parse(date, fuzzy=True)
                
                # Handle relative dates (e.g., "next month", "in 2 weeks")
                if parsed_date.year < 2025:
                    # If year is not specified or is in the past, assume it's 2025-2026
                    if parsed_date.month >= datetime.now().month:
                        parsed_date = parsed_date.replace(year=2025)
                    else:
                        parsed_date = parsed_date.replace(year=2026)
                
                # Ensure date is within 2025-2026
                if 2025 <= parsed_date.year <= 2026:
                    standardized_date = parsed_date.strftime("%Y-%m-%d")
                    valid_dates.append(standardized_date)
                else:
                    logger.warning(f"Date {date} is outside 2025-2026 range")
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse date '{date}': {str(e)}")
                continue
        return sorted(list(set(valid_dates)))  # Remove duplicates and sort

    @validator('venues')
    def validate_venues(cls, venues):
        """Validate that venues are not empty strings."""
        return [v.strip() for v in venues if v.strip()]

class HuggingFaceEmbeddings(Embeddings):
    def __init__(self, embedding_fn):
        self.embedding_fn = embedding_fn

    def embed_documents(self, texts):
        return self.embedding_fn(texts)

    def embed_query(self, text):
        return self.embedding_fn([text])[0]

    def __call__(self, text):
        if isinstance(text, str):
            return self.embed_query(text)
        elif isinstance(text, list):
            return self.embed_documents(text)
        else:
            raise TypeError("Input must be a string or a list of strings")

class DocumentProcessor:
    def __init__(self, data_dir: str = "data/documents"):
        """Initialize the document processor.
        
        Args:
            data_dir: Directory to store processed documents
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Hugging Face client
        self.hf_client = HuggingFaceClient(api_key=os.getenv("HF_API_KEY"))
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Initialize vector store if it exists
        self.vector_store = self._load_or_create_vector_store()

        # Keywords for relevance checking
        self.relevance_keywords = [
            "concert", "tour", "venue", "performance", "show", "gig",
            "ticket", "artist", "band", "music", "stage", "audience"
        ]

        # Date patterns for extraction
        self.date_patterns = [
            # Standard formats
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?,? 202[56]\b',
            r'\b\d{1,2}(?:st|nd|rd|th)? (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* 202[56]\b',
            r'\b202[56]-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b',
            # Full month names
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}(?:st|nd|rd|th)?,? 202[56]\b',
            # Weekday formats
            r'\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),? \d{1,2}(?:st|nd|rd|th)? (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* 202[56]\b',
            # Numeric formats
            r'\b(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/202[56]\b',
            r'\b(?:0?[1-9]|[12]\d|3[01])/(?:0?[1-9]|1[0-2])/202[56]\b',
            # Relative dates
            r'\b(?:next|this|coming) (?:week|month|year)\b',
            r'\bin (?:a|one|two|three|four|five|six) (?:week|month)s?\b',
            # Season references
            r'\b(?:Spring|Summer|Fall|Autumn|Winter) 202[56]\b',
            # Quarter references
            r'\bQ[1-4] 202[56]\b',
            # Month references
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December) 202[56]\b'
        ]

        self.document_hashes = set()
        # Load existing hashes if vector store already has documents
        if hasattr(self.vector_store, 'documents'):
            for doc in self.vector_store.documents:
                content_hash = hashlib.sha256(doc.page_content.encode('utf-8')).hexdigest()
                self.document_hashes.add(content_hash)

    def _load_or_create_vector_store(self) -> FAISS:
        """Load existing vector store or create a new one."""
        vector_store_path = self.data_dir / "vector_store"
        if vector_store_path.exists():
            return FAISS.load_local(
                str(vector_store_path),
                HuggingFaceEmbeddings(self.hf_client.get_embeddings),
                allow_dangerous_deserialization=True
            )
        return FAISS.from_texts([""], HuggingFaceEmbeddings(self.hf_client.get_embeddings))

    def _check_relevance(self, document: str) -> Tuple[bool, float, str]:
        """Check if document is relevant to concert tours.
        
        Args:
            document: The document text to check
            
        Returns:
            Tuple containing:
            - bool: Whether the document is relevant
            - float: Confidence score
            - str: Reason for the decision
        """
        # First check using keyword matching
        doc_lower = document.lower()
        keyword_matches = sum(1 for keyword in self.relevance_keywords if keyword in doc_lower)
        keyword_score = min(keyword_matches / len(self.relevance_keywords), 1.0)

        # Then use classification
        relevance = self.hf_client.classify_text(
            document,
            labels=["concert tour", "music event", "concert", "tour", "unrelated"]
        )
        
        # Calculate confidence score
        tour_score = relevance.get("concert tour", 0.0)
        music_score = relevance.get("music event", 0.0)
        concert_score = relevance.get("concert", 0.0)
        
        # Combine scores with weights
        confidence_score = (
            keyword_score * 0.3 +
            tour_score * 0.4 +
            music_score * 0.2 +
            concert_score * 0.1
        )

        # Determine relevance and reason
        is_relevant = confidence_score > 0.4
        reason = "Document contains sufficient concert tour related information." if is_relevant else \
                "Document does not contain enough concert tour related information."

        return is_relevant, confidence_score, reason

    def _extract_dates_from_text(self, text: str) -> List[str]:
        """Extract dates from text using regex patterns."""
        dates = []
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = match.group()
                try:
                    # Handle relative dates
                    if 'next' in date_str.lower() or 'coming' in date_str.lower():
                        if 'week' in date_str.lower():
                            date = datetime.now() + relativedelta(weeks=1)
                        elif 'month' in date_str.lower():
                            date = datetime.now() + relativedelta(months=1)
                        elif 'year' in date_str.lower():
                            date = datetime.now() + relativedelta(years=1)
                    elif 'in' in date_str.lower():
                        # Extract number of weeks/months
                        num = 1
                        if 'two' in date_str.lower():
                            num = 2
                        elif 'three' in date_str.lower():
                            num = 3
                        elif 'four' in date_str.lower():
                            num = 4
                        elif 'five' in date_str.lower():
                            num = 5
                        elif 'six' in date_str.lower():
                            num = 6
                        
                        if 'week' in date_str.lower():
                            date = datetime.now() + relativedelta(weeks=num)
                        elif 'month' in date_str.lower():
                            date = datetime.now() + relativedelta(months=num)
                    else:
                        date = parser.parse(date_str, fuzzy=True)
                    
                    # Ensure date is within 2025-2026
                    if date.year < 2025:
                        if date.month >= datetime.now().month:
                            date = date.replace(year=2025)
                        else:
                            date = date.replace(year=2026)
                    
                    if 2025 <= date.year <= 2026:
                        dates.append(date.strftime("%Y-%m-%d"))
                except Exception as e:
                    logger.debug(f"Failed to parse date {date_str}: {str(e)}")
                    continue
        return sorted(list(set(dates)))

    def process_document(self, document: str) -> Tuple[bool, str, Optional[DocumentSummary]]:
        """Process a new document.
        
        Args:
            document: The document text to process
            
        Returns:
            Tuple containing:
            - bool: Whether the document was successfully processed
            - str: Message about the processing result
            - Optional[DocumentSummary]: Document summary if relevant
        """
        try:
            content_hash = hashlib.sha256(document.encode('utf-8')).hexdigest()
            if content_hash in self.document_hashes:
                return False, 'This document has already been ingested.', None
            self.document_hashes.add(content_hash)

            # Check document relevance
            is_relevant, confidence_score, reason = self._check_relevance(document)
            
            if not is_relevant:
                return False, f"Sorry, I cannot ingest documents with other themes. {reason}", None

            # Generate summary
            summary_text = self.hf_client.summarize_text(document)
            
            # Extract dates using regex first
            extracted_dates = self._extract_dates_from_text(document)
            
            # Extract information using text generation
            prompt = f"""Extract information from this concert tour document:
            {document}
            
            Extract:
            1. Tour dates (in any format, but must include year 2025 or 2026)
            2. Venues (full venue names)
            3. Artists/bands
            
            Format as JSON with these fields:
            {{
                "tour_dates": [],
                "venues": [],
                "artists": []
            }}
            """
            
            info_text = self.hf_client.generate_text(prompt)
            
            # Parse the generated JSON
            try:
                info = json.loads(info_text)
                # Combine dates from both sources
                all_dates = set(extracted_dates + info.get("tour_dates", []))
                info["tour_dates"] = sorted(list(all_dates))
            except json.JSONDecodeError:
                logger.warning("Failed to parse generated JSON, using fallback extraction")
                info = {
                    "tour_dates": extracted_dates,
                    "venues": [],
                    "artists": []
                }
            
            # Create summary object
            summary = DocumentSummary(
                is_relevant=True,
                summary=summary_text,
                tour_dates=info.get("tour_dates", []),
                venues=info.get("venues", []),
                artists=info.get("artists", []),
                confidence_score=confidence_score
            )
            
            # Save document and update vector store
            doc_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._save_document(doc_id, document, summary)
            
            # Update vector store
            chunks = self.text_splitter.split_text(document)
            self.vector_store.add_texts(chunks)
            self.vector_store.save_local(str(self.data_dir / "vector_store"))
            
            # Generate user-friendly response
            response = (
                f"Thank you for sharing! Your document has been successfully added to the database.\n\n"
                f"Here is a brief summary of the data from the document:\n{summary_text}\n\n"
                f"Tour Dates: {', '.join(summary.tour_dates) if summary.tour_dates else 'None specified'}\n"
                f"Venues: {', '.join(summary.venues) if summary.venues else 'None specified'}\n"
                f"Artists: {', '.join(summary.artists) if summary.artists else 'None specified'}"
            )
            
            return True, response, summary
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return False, f"Error processing document: {str(e)}", None

    def _save_document(self, doc_id: str, document: str, summary: DocumentSummary):
        """Save document and its summary to disk."""
        doc_data = {
            "id": doc_id,
            "content": document,
            "summary": summary.dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.data_dir / f"{doc_id}.json", "w") as f:
            json.dump(doc_data, f, indent=2)

    def get_relevant_chunks(self, query: str, k: int = 3) -> list[str]:
        """Retrieve relevant document chunks for a query.
        
        Args:
            query: The search query
            k: Number of chunks to retrieve
            
        Returns:
            List of relevant text chunks
        """
        return self.vector_store.similarity_search(query, k=k) 