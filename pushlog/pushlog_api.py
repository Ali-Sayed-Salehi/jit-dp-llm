import requests
import json
from pprint import pprint

# Define repository and date range
repo = "autoland"
start_date = "2024-06-01"
end_date = "2024-06-5"

# API URL
url = f"https://hg.mozilla.org/integration/{repo}/json-pushes?startdate={start_date}&enddate={end_date}"

# Fetch push data
response = requests.get(url)
push_data = response.json()
pprint(push_data)

# if response.status_code == 200:
#     push_data = response.json()
    
#     # Parse push details
#     pushes = []
#     for push_id, push_info in push_data["pushes"].items():
#         pushes.append({
#             "push_id": push_id,
#             "pusher": push_info["user"],
#             "date": push_info["date"],
#             "changesets": push_info["changesets"]
#         })
    
#     # Print formatted JSON output
#     print(json.dumps(pushes, indent=4))
# else:
#     print(f"Failed to fetch push data: {response.status_code}")
