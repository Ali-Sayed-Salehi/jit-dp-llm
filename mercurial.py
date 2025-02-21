import subprocess
import json

# Define the path to the Autoland repository (Update this if necessary)
REPO_PATH = "../autoland"

def fetch_pushes(start_date, end_date):
    """
    Fetches push events from Mozilla Autoland between start_date and end_date using Mercurial CLI.

    :param start_date: Start date in 'YYYY-MM-DD HH:MM:SS' format
    :param end_date: End date in 'YYYY-MM-DD HH:MM:SS' format
    :return: List of pushes within the given time range
    """

    # Construct the Mercurial pushlog command
    hg_command = [
        "hg", "pushlog",
        "-R", REPO_PATH,  # Specify repository path
        "--template", "{pushid}|{pusher}|{date|isodate}|{changesets}\n",
        "--date", f"{start_date} to {end_date}"
    ]

    try:
        # Run the Mercurial command
        result = subprocess.run(hg_command, capture_output=True, text=True, check=True)
        pushes = []

        # Parse output
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split("|")
                push_id = parts[0]
                pusher = parts[1]
                push_date = parts[2]
                changesets = parts[3].split()  # List of changesets in this push

                pushes.append({
                    "push_id": push_id,
                    "pusher": pusher,
                    "date": push_date,
                    "changesets": changesets
                })

        return pushes

    except subprocess.CalledProcessError as e:
        print(f"Error running Mercurial command: {e}")
        return []
    

def fetch_commits(start_date, end_date):
    """
    Fetches commits from Mozilla Autoland between start_date and end_date using Mercurial CLI.

    :param start_date: Start date in YYYY-MM-DD format
    :param end_date: End date in YYYY-MM-DD format
    :return: List of commits in the given time range
    """

    # Construct the Mercurial log command
    hg_command = [
        "hg", "log",
        "-R", REPO_PATH,  # Specify repository path
        "--template", "{node|short}|{author}|{date|isodate}\n",
        "--date", f"{start_date} to {end_date}"
    ]

    try:
        # Run Mercurial command
        result = subprocess.run(hg_command, capture_output=True, text=True, check=True)
        commits = []

        # Parse output
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split("|")
                commits.append({
                    "rev": parts[0],
                    "author": parts[1],
                    "date": parts[2]
                })

        return commits

    except subprocess.CalledProcessError as e:
        print(f"Error running Mercurial command: {e}")
        return []
    


if __name__ == "__main__":
    # Example usage for testing
    start_date = "2025-01-20 19:00:13"
    end_date = "2025-01-21 01:57:09"

    commits = fetch_commits(start_date, end_date)
    print(len(commits))
    #print(json.dumps(commits, indent=4))
