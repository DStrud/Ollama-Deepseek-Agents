import os
import json

MEMORY_FILE = "memory.json"

def load_memory():
    """Load agent memories from a JSON file if it exists."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as file:
            return json.load(file)
    return {}

def save_memory(memory_data):
    """Persist agent memories to a JSON file."""
    with open(MEMORY_FILE, "w") as file:
        json.dump(memory_data, file, indent=4)
