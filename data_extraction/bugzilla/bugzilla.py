import requests
import datetime
from dotenv import load_dotenv
import os

# Bugzilla API key
load_dotenv(dotenv_path="../secrets/.env")

API_KEY = os.getenv("BUGZILLA_API_KEY")

# Bugzilla API endpoint
BUGZILLA_API_URL = "https://bugzilla.mozilla.org/rest/bug"

# Define performance-related keywords
PERFORMANCE_KEYWORDS = ["perf-alert", "perf"]

# Function to fetch performance-related bugs within a date range
def fetch_performance_bugs():
    params = {
        "api_key": API_KEY,
        "product": "Firefox",
        "keywords": ["perf"],
        "limit": 5,  # Adjust as needed
    }
    
    response = requests.get(BUGZILLA_API_URL, params=params)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.json()}")
        return []
    
    return response.json().get("bugs", [])

# Main function
def main():
    bugs = fetch_performance_bugs()
    
    if not bugs:
        print("No performance bugs found in the given time period.")
        return
    
    print(f"Found {len(bugs)} performance-related bugs:\n")
    
    for bug in bugs:
        print(f"Bug ID: {bug['id']}")
        print(f"Summary: {bug['summary']}")
        print(f"keywords: {bug['keywords']}\n")

    
if __name__ == "__main__":
    main()
