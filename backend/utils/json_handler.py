import json
import os
from typing import Any, List, Dict
from threading import Lock

# Thread-safe locks for each file
file_locks = {}

def get_lock(filename: str) -> Lock:
    """Get or create a lock for a specific file"""
    if filename not in file_locks:
        file_locks[filename] = Lock()
    return file_locks[filename]

def read_json(filename: str) -> List[Dict[str, Any]] | Dict[str, Any]:
    """
    Read JSON file safely. Creates file with empty list if it doesn't exist.
    """
    lock = get_lock(filename)
    with lock:
        if not os.path.exists(filename):
            # Initialize with empty list
            with open(filename, 'w') as f:
                json.dump([], f)
            return []
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                return data if data else []
        except json.JSONDecodeError:
            # If file is corrupted, reset it
            with open(filename, 'w') as f:
                json.dump([], f)
            return []

def write_json(filename: str, data: List[Dict[str, Any]] | Dict[str, Any]) -> bool:
    """
    Write JSON file safely with proper formatting.
    """
    lock = get_lock(filename)
    with lock:
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error writing to {filename}: {e}")
            return False

def append_json(filename: str, item: Dict[str, Any]) -> bool:
    """
    Append an item to a JSON array file.
    """
    data = read_json(filename)
    if not isinstance(data, list):
        data = []
    data.append(item)
    return write_json(filename, data)

def update_json_item(filename: str, condition: callable, updates: Dict[str, Any]) -> bool:
    """
    Update items in JSON file that match a condition.
    condition: function that takes an item and returns True if it should be updated
    updates: dictionary of fields to update
    """
    data = read_json(filename)
    if not isinstance(data, list):
        return False
    
    modified = False
    for item in data:
        if condition(item):
            item.update(updates)
            modified = True
    
    if modified:
        return write_json(filename, data)
    return False

def delete_json_item(filename: str, condition: callable) -> bool:
    """
    Delete items from JSON file that match a condition.
    """
    data = read_json(filename)
    if not isinstance(data, list):
        return False
    
    original_length = len(data)
    data = [item for item in data if not condition(item)]
    
    if len(data) < original_length:
        return write_json(filename, data)
    return False

def find_json_item(filename: str, condition: callable) -> Dict[str, Any] | None:
    """
    Find first item in JSON file that matches a condition.
    """
    data = read_json(filename)
    if not isinstance(data, list):
        return None
    
    for item in data:
        if condition(item):
            return item
    return None

def find_all_json_items(filename: str, condition: callable = None) -> List[Dict[str, Any]]:
    """
    Find all items in JSON file that match a condition.
    If no condition provided, returns all items.
    """
    data = read_json(filename)
    if not isinstance(data, list):
        return []
    
    if condition is None:
        return data
    
    return [item for item in data if condition(item)]