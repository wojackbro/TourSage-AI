from typing import List, Dict, Any, Optional, Tuple
from serpapi import GoogleSearch
from pydantic import BaseModel, Field, validator
from hf_utils import HuggingFaceClient
from document_processor import DocumentProcessor
import os
import json
import requests
from datetime import datetime
import re
from dateutil import parser
from langchain.schema import Document
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConcertInfo(BaseModel):
    """Schema for concert information."""
    artist: str = Field(description="Name of the artist/band")
    tour_name: str = Field(description="Name of the tour")
    dates: List[str] = Field(description="List of concert dates")
    venues: List[str] = Field(description="List of venues")
    ticket_info: str = Field(description="Information about ticket availability and prices")
    source_url: str = Field(description="URL of the source information")
    last_updated: str = Field(description="When this information was last updated")
    confidence_score: float = Field(description="Confidence score for the information", default=0.0)

    @validator('dates')
    def validate_dates(cls, dates):
        """Validate that all dates are within 2025-2026."""
        for date_str in dates:
            try:
                date = parser.parse(date_str)
                if not (2025 <= date.year <= 2026):
                    raise ValueError(f"Date {date_str} is outside 2025-2026 range")
            except Exception as e:
                logger.warning(f"Invalid date format: {date_str}")
                raise ValueError(f"Invalid date format: {date_str}")
        return dates

    @validator('venues')
    def validate_venues(cls, venues):
        """Validate that venues are not empty strings."""
        return [v.strip() for v in venues if v.strip()]

class SearchEngine:
    def __init__(self, api_key: str):
        """Initialize the search engine.
        
        Args:
            api_key: SerpAPI key
        """
        self.api_key = api_key
        self.hf_client = HuggingFaceClient(api_key=os.getenv("HF_API_KEY"))

        # Enhanced date patterns
        self.date_patterns = [
            # Standard formats
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? 202[56]\b',
            r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* 202[56]\b',
            r'\b202[56]-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b',
            # Additional formats
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},? 202[56]\b',
            r'\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),? \d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* 202[56]\b'
        ]
        
        # Common venue keywords
        self.venue_keywords = [
            'arena', 'stadium', 'theatre', 'theater', 'hall', 'center', 'centre',
            'garden', 'field', 'park', 'amphitheater', 'amphitheatre', 'club',
            'venue', 'auditorium', 'pavilion', 'dome'
        ]

    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text that fall within 2025-2026."""
        dates = []
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    date_str = match.group()
                    date = parser.parse(date_str)
                    if 2025 <= date.year <= 2026:
                        dates.append(date_str)
                except Exception as e:
                    logger.debug(f"Failed to parse date {match.group()}: {str(e)}")
                    continue
        return sorted(list(set(dates)))

    def _extract_venues(self, text: str) -> List[str]:
        """Extract venue names from text using both regex and LLM."""
        venues = set()
        
        # First try regex-based extraction
        for keyword in self.venue_keywords:
            pattern = rf'\b[A-Z][a-zA-Z\s\'-]+(?:{keyword})\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            venues.update(match.group().strip() for match in matches)
        
        # Then try TogetherAI for additional extraction
        try:
            prompt = f"""Extract venue names from this concert information. Return only a JSON array of venue names.
            Focus on actual venue names, not generic locations. Format: ["Venue 1", "Venue 2"]
            Text: {text}
            """
            response = self.hf_client.generate_text(prompt)
            
            # Check if we hit rate limits
            if "429" not in response and "Too Many Requests" not in response:
                try:
                    llm_venues = json.loads(response)
                    if isinstance(llm_venues, list):
                        venues.update(llm_venues)
                except Exception as e:
                    logger.warning(f"Failed to parse LLM venue response: {str(e)}")
        except Exception as e:
            logger.warning(f"LLM venue extraction failed: {str(e)}")
        
        return sorted(list(venues))

    def _extract_ticket_info(self, text: str) -> str:
        """Extract ticket information from text."""
        try:
            # First try TogetherAI
            prompt = f"""Extract detailed ticket information from this concert information. Include:
            1. Ticket sale dates
            2. Price ranges
            3. Availability status
            4. Where to buy tickets
            Format as a clear, concise paragraph.
            Text: {text}
            """
            result = self.hf_client.generate_text(prompt)
            
            # Check if we hit rate limits
            if "429" in result or "Too Many Requests" in result:
                # Fallback to regex-based extraction
                ticket_info = []
                
                # Extract price ranges
                price_patterns = [
                    r'\$(\d+(?:\.\d{2})?)\s*(?:to|-)\s*\$(\d+(?:\.\d{2})?)',  # $99 to $499
                    r'priced\s+(?:from|between)\s+\$(\d+)\s*(?:to|-)\s*\$(\d+)',  # priced from $99 to $499
                    r'tickets\s+(?:from|starting at)\s+\$(\d+)',  # tickets from $99
                ]
                
                for pattern in price_patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        if len(match.groups()) == 2:
                            ticket_info.append(f"Ticket prices range from ${match.group(1)} to ${match.group(2)}")
                        else:
                            ticket_info.append(f"Tickets starting at ${match.group(1)}")
                
                # Extract sale dates
                sale_patterns = [
                    r'(?:pre-sale|presale|early access)\s+(?:begins|starts|starts on)\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?)',  # pre-sale begins January 15
                    r'(?:general|public)\s+(?:sale|tickets)\s+(?:begins|starts|starts on)\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?)',  # general sale begins January 20
                    r'(?:fan club|VIP)\s+(?:pre-sale|presale)\s+(?:begins|starts|starts on)\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?)',  # fan club pre-sale begins January 10
                ]
                
                for pattern in sale_patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        ticket_info.append(f"Sale begins on {match.group(1)}")
                
                if ticket_info:
                    return " | ".join(ticket_info)
                return "Ticket information not available"
            
            return result
        except Exception as e:
            logger.warning(f"Error extracting ticket info: {str(e)}")
            return "Ticket information not available"

    def _calculate_confidence_score(self, text: str, dates: List[str], venues: List[str]) -> float:
        """Calculate confidence score for the extracted information."""
        score = 0.0
        
        # Date confidence
        if dates:
            score += 0.4
        
        # Venue confidence
        if venues:
            score += 0.3
        
        # Text quality confidence
        if len(text.split()) > 50:  # Sufficient content
            score += 0.2
        
        # Source quality confidence
        if any(keyword in text.lower() for keyword in ['official', 'announced', 'confirmed']):
            score += 0.1
            
        return min(score, 1.0)

    def _is_relevant_tour(self, text: str) -> bool:
        """Check if the text contains information about 2025-2026 tours."""
        dates = self._extract_dates(text)
        return len(dates) > 0

    def _process_search_result(self, result: Dict[str, Any], artist: str) -> Optional[ConcertInfo]:
        """Process a single search result."""
        try:
            snippet = result.get("snippet", "")
            if not self._is_relevant_tour(snippet):
                return None

            dates = self._extract_dates(snippet)
            venues = self._extract_venues(snippet)
            
            concert_info = ConcertInfo(
                artist=artist,
                tour_name=self._extract_tour_name(snippet, artist),
                dates=dates,
                venues=venues,
                ticket_info=self._extract_ticket_info(snippet),
                source_url=result.get("link", ""),
                last_updated=datetime.now().isoformat(),
                confidence_score=self._calculate_confidence_score(snippet, dates, venues)
            )
            
            return concert_info
        except Exception as e:
            logger.error(f"Error processing search result: {str(e)}")
            return None

    def search_concerts(self, artist: str, document_processor=None) -> dict:
        # Broadened queries
        local_query = f"{artist} tour OR concert 2025 2026"
        internet_query = f"{artist} concert OR tour 2025 2026 official website"
        logger.info(f"[DEBUG] Local search query: {local_query}")
        logger.info(f"[DEBUG] Internet search query: {internet_query}")
        
        local_dict_results = []
        local_error = None
        internet_error = None  # Ensure this is always initialized
        
        # --- Local Vector Search ---
        if document_processor:
            try:
                local_results = document_processor.get_relevant_chunks(local_query, k=5)
                logger.info(f"[DEBUG] Local search found {len(local_results)} results.")
                for doc in local_results:
                    if hasattr(doc, 'page_content'):
                        info = self._extract_concert_info(doc.page_content)
                        if info:
                            info_dict = info.dict() if hasattr(info, 'dict') else dict(info)
                            info_dict["source_url"] = "Local Document"
                            info_dict["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            info_dict["confidence_score"] = info_dict.get("confidence_score", 0.9)
                            local_dict_results.append(info_dict)
                    elif isinstance(doc, dict):
                        local_dict_results.append(doc)
            except Exception as e:
                local_error = str(e)
                logger.error(f"[DEBUG] Local search error: {local_error}")
        else:
            logger.warning("[DEBUG] No document_processor provided for local search.")
        
        # --- Internet Search ---
        internet_results = []
        try:
            from serpapi import GoogleSearch
            params = {
                "q": internet_query,
                "api_key": self.api_key,
                "num": 10,
                "engine": "google",
                "tbm": "nws"
            }
            search = GoogleSearch(params)
            results = search.get_dict()
            news_results = results.get("news_results", [])
            logger.info(f"[DEBUG] Internet search found {len(news_results)} news results.")
            for item in news_results:
                concert_info = self._extract_concert_info(item.get("title", "") + "\n" + item.get("snippet", ""))
                if concert_info:
                    info_dict = concert_info.dict() if hasattr(concert_info, 'dict') else dict(concert_info)
                    info_dict["source_url"] = item.get("link", "")
                    info_dict["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    info_dict["confidence_score"] = info_dict.get("confidence_score", 0.5)
                    internet_results.append(info_dict)
        except Exception as e:
            internet_error = str(e)
            logger.error(f"[DEBUG] Internet search error: {internet_error}")
        
        # --- Combine and Deduplicate Results ---
        combined_results = []
        seen = set()
        for result in local_dict_results + internet_results:
            key = (tuple(result.get("dates", [])), tuple(result.get("venues", [])))
            if key not in seen:
                combined_results.append(result)
                seen.add(key)
        
        # --- Debug Output for Results ---
        logger.info(f"[DEBUG] Combined results count: {len(combined_results)}")
        logger.info(f"[DEBUG] Local error: {local_error}")
        logger.info(f"[DEBUG] Internet error: {internet_error}")
        
        # --- Fallback and Response ---
        if combined_results:
            return {
                "status": "success",
                "message": f"Found {len(combined_results)} unique tour information sources for {artist}.",
                "results": combined_results,
                "debug": {
                    "local_query": local_query,
                    "internet_query": internet_query,
                    "local_results_count": len(local_dict_results),
                    "internet_results_count": len(internet_results),
                    "local_error": local_error,
                    "internet_error": internet_error
                }
            }
        else:
            return {
                "status": "error",
                "message": f"No relevant tour information found for {artist} in 2025-2026 (checked both local and internet sources).",
                "results": [],
                "debug": {
                    "local_query": local_query,
                    "internet_query": internet_query,
                    "local_results_count": len(local_dict_results),
                    "internet_results_count": len(internet_results),
                    "local_error": local_error,
                    "internet_error": internet_error
                }
            }

    def remove_code_lines(self, text: str) -> str:
        import re
        code_patterns = [
            r'^def ', r'^class ', r'^import ', r'^if __name__ == .__main__.:', r'unittest', r'print\(', r'assert ', r'\breturn\b', r'\bself\.assert\w+\b', r'\btest_', r'\bmain\b', r'\bExample', r'\bcode to solve', r'\bfunction', r'\bscript', r'\btest case', r'\btest_', r'\bprint\('
        ]
        for pat in code_patterns:
            text = re.sub(pat + '.*', '', text, flags=re.MULTILINE)
        # Remove code blocks (triple backticks)
        text = re.sub(r'```[\s\S]*?```', '', text)
        # Remove lines that are indented (likely code)
        text = re.sub(r'^    .+', '', text, flags=re.MULTILINE)
        # Remove lines with only brackets or braces
        text = re.sub(r'^[\[\]\{\}]+$', '', text, flags=re.MULTILINE)
        # Remove lines with only numbers or variable assignments
        text = re.sub(r'^\s*\w+\s*=.*$', '', text, flags=re.MULTILINE)
        # Remove lines with only whitespace
        text = re.sub(r'^\s+$', '', text, flags=re.MULTILINE)
        return text.strip()

    def remove_code_blocks(self, text: str) -> str:
        import re
        # Remove triple-backtick code blocks
        text = re.sub(r'```[\s\S]+?```', '', text)
        # Remove indented code blocks (4+ spaces)
        text = re.sub(r'(?m)^ {4,}.*$', '', text)
        # Remove function/class definitions and test code
        text = re.sub(r'(?m)^.*def [\w_]+\(.*\):.*$', '', text)
        text = re.sub(r'(?m)^.*class [\w_]+\(.*\):.*$', '', text)
        text = re.sub(r'(?m)^.*@pytest.*$', '', text)
        text = re.sub(r'(?m)^.*assert .*$', '', text)
        text = re.sub(r'(?m)^.*for .* in .*:.*$', '', text)
        text = re.sub(r'(?m)^.*return .*$', '', text)
        text = re.sub(r'(?m)^.*print\(.*\).*$','', text)
        text = re.sub(r'(?m)^.*unittest.*$', '', text)
        text = re.sub(r'(?m)^.*test_extract.*$', '', text)
        text = re.sub(r'(?m)^.*test_', '', text)
        # Remove lines with only brackets/braces or variable assignments
        text = re.sub(r'^[\[\]\{\}]+$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\w+\s*=.*$', '', text, flags=re.MULTILINE)
        # Remove lines with only whitespace
        text = re.sub(r'^\s+$', '', text, flags=re.MULTILINE)
        return text.strip()

    def is_code_leaking(self, output: str) -> bool:
        suspicious_tokens = ['def ', 'return ', 'print(', 'import ', 'class ', 'for ', '@pytest', 'unittest', 'assert', 'test_']
        return any(tok in output for tok in suspicious_tokens)

    def sanitize_output(self, output_text: str) -> str:
        if self.is_code_leaking(output_text):
            output_text = self.remove_code_blocks(output_text)
        return output_text.strip()

    def _extract_concert_info(self, text: str) -> Optional[ConcertInfo]:
        """Extract concert information from text."""
        def clean_text(text):
            import re
            # Remove code blocks (triple backticks)
            text = re.sub(r'```[\s\S]*?```', '', text)
            # Remove lines starting with def, class, import, or test code
            text = re.sub(r'^(def |class |import |if __name__ == .__main__.:|unittest\.|#|\s*assert |\s*print\()', '', text, flags=re.MULTILINE)
            # Remove lines that are indented (likely code)
            text = re.sub(r'^    .+', '', text, flags=re.MULTILINE)
            # Remove lines that look like test cases or code comments
            text = re.sub(r'^\s*#.*$', '', text, flags=re.MULTILINE)
            # Remove lines with common code patterns
            text = re.sub(r'\bself\.assert\w+\b.*', '', text)
            text = re.sub(r'\breturn\b.*', '', text)
            # Remove any remaining triple backticks
            text = text.replace('```', '')
            # Remove lines with only brackets or braces
            text = re.sub(r'^[\[\]\{\}]+$', '', text, flags=re.MULTILINE)
            # Remove lines with only numbers or variable assignments
            text = re.sub(r'^\s*\w+\s*=.*$', '', text, flags=re.MULTILINE)
            # Remove lines with only whitespace
            text = re.sub(r'^\s+$', '', text, flags=re.MULTILINE)
            return text.strip()
        try:
            # Clean the text - remove code, test functions, and technical lines
            cleaned_text = clean_text(text)
            cleaned_text = self.remove_code_lines(cleaned_text)
            cleaned_text = self.sanitize_output(cleaned_text)
            # Extract tour name
            tour_name = self._extract_tour_name(cleaned_text)
            # Extract dates
            dates = self._extract_dates(cleaned_text)
            # Extract venues
            venues = self._extract_venues(cleaned_text)
            # Extract ticket info
            ticket_info = self._extract_ticket_info(cleaned_text)
            # Final output sanitization for ticket_info and tour_name
            ticket_info = self.sanitize_output(ticket_info)
            tour_name = self.sanitize_output(tour_name)
            return ConcertInfo(
                artist="",  # Will be set by caller
                tour_name=tour_name,
                dates=dates,
                venues=venues,
                ticket_info=ticket_info,
                source_url="",  # Will be set by caller or fallback
                last_updated="",  # Will be set by caller or fallback
                confidence_score=0.0  # Will be set by caller
            )
        except Exception as e:
            print(f"Error extracting concert info: {str(e)}")
            # Fallback: always return a dict with all required fields
            return {
                "artist": "",
                "tour_name": "",
                "dates": [],
                "venues": [],
                "ticket_info": "",
                "source_url": "",
                "last_updated": "",
                "confidence_score": 0.0
            }

    def _extract_tour_name(self, text: str) -> str:
        """Extract tour name from text."""
        try:
            # First try TogetherAI
            prompt = f"""Extract the tour name from this text. If no specific tour name is mentioned, return "Upcoming Tour".
            Look for patterns like "[Artist] [Tour Name] Tour" or "The [Tour Name] Tour".
            Text: {text}
            """
            result = self.hf_client.generate_text(prompt).strip()
            
            # Check if we hit rate limits
            if "429" in result or "Too Many Requests" in result:
                # Fallback to regex patterns
                patterns = [
                    r'"([^"]+ Tour)"',  # "Tour Name Tour"
                    r'The ([^"]+ Tour)',  # The Tour Name Tour
                    r'([^"]+ Tour) will',  # Tour Name Tour will
                    r'announces ([^"]+ Tour)',  # announces Tour Name Tour
                    r'presents ([^"]+ Tour)',  # presents Tour Name Tour
                    r'([^"]+ Tour) featuring',  # Tour Name Tour featuring
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, text)
                    if match:
                        return match.group(1)
                
                return "Upcoming Tour"
            
            return result
        except Exception as e:
            logger.warning(f"Error extracting tour name: {str(e)}")
            return "Upcoming Tour" 