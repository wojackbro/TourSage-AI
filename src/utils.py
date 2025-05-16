import os
from typing import Optional
from datetime import datetime
import json
from pathlib import Path

def ensure_dir(directory: str) -> Path:
    """Ensure a directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_json_file(file_path: str) -> Optional[dict]:
    """Load and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data or None if file doesn't exist
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def save_json_file(file_path: str, data: dict) -> None:
    """Save data to a JSON file.
    
    Args:
        file_path: Path to save the JSON file
        data: Data to save
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def format_timestamp(timestamp: datetime) -> str:
    """Format a timestamp for display.
    
    Args:
        timestamp: Datetime object
        
    Returns:
        Formatted timestamp string
    """
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def validate_api_keys() -> tuple[bool, str]:
    """Validate that required API keys are set.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_keys = {
        "HF_API_KEY": os.getenv("HF_API_KEY"),
        "SERPAPI_API_KEY": os.getenv("SERPAPI_API_KEY")
    }
    
    missing_keys = [key for key, value in required_keys.items() if not value]
    
    if missing_keys:
        return False, f"Missing required API keys: {', '.join(missing_keys)}"
    return True, "" 