import pytest
from src.document_processor import DocumentProcessor, DocumentSummary

def test_document_processor_initialization():
    """Test document processor initialization."""
    processor = DocumentProcessor()
    assert processor.data_dir.exists()
    assert processor.vector_store is not None

def test_process_relevant_document():
    """Test processing a relevant concert tour document."""
    processor = DocumentProcessor()
    
    # Sample concert tour document
    document = """
    Lady Gaga's 2025 World Tour
    Dates: June 15, 2025 - August 30, 2025
    Venues:
    - Madison Square Garden, New York
    - Staples Center, Los Angeles
    - O2 Arena, London
    Special guest: Tony Bennett
    """
    
    success, message, summary = processor.process_document(document)
    
    assert success
    assert isinstance(summary, DocumentSummary)
    assert summary.is_relevant
    assert "Lady Gaga" in summary.artists
    assert len(summary.venues) > 0
    assert len(summary.tour_dates) > 0

def test_process_irrelevant_document():
    """Test processing an irrelevant document."""
    processor = DocumentProcessor()
    
    # Sample irrelevant document
    document = """
    Recipe for Chocolate Cake
    Ingredients:
    - 2 cups flour
    - 1 cup sugar
    - 3 eggs
    """
    
    success, message, summary = processor.process_document(document)
    
    assert not success
    assert "cannot ingest documents with other themes" in message.lower()
    assert summary is None

def test_get_relevant_chunks():
    """Test retrieving relevant document chunks."""
    processor = DocumentProcessor()
    
    # First add a document
    document = """
    Taylor Swift's 2025 Stadium Tour
    Dates: May 1, 2025 - July 15, 2025
    Venues:
    - SoFi Stadium, Los Angeles
    - MetLife Stadium, New York
    """
    
    processor.process_document(document)
    
    # Test query
    chunks = processor.get_relevant_chunks("Where is Taylor Swift performing in 2025?")
    
    assert len(chunks) > 0
    assert any("Taylor Swift" in chunk.page_content for chunk in chunks) 