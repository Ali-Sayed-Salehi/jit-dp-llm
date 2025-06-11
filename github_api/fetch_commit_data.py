import requests
import os
from dotenv import load_dotenv

REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load GitHub token from .env file
load_dotenv(dotenv_path= f"{REPO_PATH}/secrets/.env")
GITHUB_TOKEN = os.getenv("GITHUB_API_TOKEN")

#commit info
OWNER = "apache"
REPO = "cassandra"
COMMIT_SHA = "1d7bacc45fa1cd6cac36d7f9ece30ba1ed430f2a"

# Set up headers
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def get_commit_message(owner, repo, sha):
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        full_message = response.json()["commit"]["message"]
        main_message = full_message.split("\n\n")[0].strip()
        return main_message
    else:
        raise Exception(f"❌ Failed to get commit message: {response.status_code} {response.text}")

def get_commit_diff(owner, repo, sha):
    url = f"https://github.com/{owner}/{repo}/commit/{sha}.diff"
    headers = HEADERS.copy()
    headers["Accept"] = "application/vnd.github.v3.diff"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"❌ Failed to get diff: {response.status_code} {response.text}")

if __name__ == "__main__":
    try:
        message = get_commit_message(OWNER, REPO, COMMIT_SHA)
        diff = get_commit_diff(OWNER, REPO, COMMIT_SHA)

        print("✅ Commit message:")
        print(message)
        print("\n✅ Code diff:")
        print(diff)
    except Exception as e:
        print(e)
