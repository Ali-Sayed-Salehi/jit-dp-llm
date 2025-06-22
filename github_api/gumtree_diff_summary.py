import os
import subprocess
import sys
import tempfile
import json
import re

# ---------------------- Config ----------------------
REPOS_ROOT = "/speed-scratch/a_s87063/repos/perf-pilot/github_api/repos"  # All repos will be cloned here
GUMTREE_CLI = os.path.join(os.path.dirname(__file__), "tools", "gumtree", "bin", "gumtree")

# ---------------------- Git Logic ----------------------

def get_repo_path(owner, name):
    return os.path.join(REPOS_ROOT, f"{owner}__{name}")

def clone_if_needed(owner, name):
    repo_path = get_repo_path(owner, name)
    if not os.path.exists(repo_path):
        os.makedirs(REPOS_ROOT, exist_ok=True)
        url = f"https://github.com/{owner}/{name}.git"
        print(f"📥 Cloning {url}...")
        subprocess.run(["git", "clone", url, repo_path], check=True)
    return repo_path

def get_changed_files(repo_path, commit):
    output = subprocess.check_output(
        ["git", "-C", repo_path, "diff", "--name-status", f"{commit}^", commit],
        text=True
    ).splitlines()
    changes = []
    for line in output:
        status, *path_parts = line.strip().split('\t')
        path = path_parts[-1]
        changes.append((status, path))
    return changes

def get_file_content(repo_path, revision, filepath):
    try:
        return subprocess.check_output(
            ["git", "-C", repo_path, "show", f"{revision}:{filepath}"],
            text=True
        )
    except subprocess.CalledProcessError:
        return ""

# ---------------------- GumTree Logic ----------------------

def run_gumtree_diff(file_before, file_after):
    try:
        result = subprocess.check_output(
            [GUMTREE_CLI, "textdiff", file_before, file_after], text=True
        )
        return result.strip().splitlines()
    except subprocess.CalledProcessError:
        return ["(gumtree error)"]

# ---------------------- Main Formatter ----------------------

def summarize_diff(owner, repo_name, commit_hash):
    repo_path = clone_if_needed(owner, repo_name)
    changed_files = get_changed_files(repo_path, commit_hash)

    output_lines = []
    output_lines.append(f"🔍 Diff Summary for {owner}/{repo_name}@{commit_hash}\n")

    for status, path in changed_files:
        output_lines.append(f"--- {status} {path} ---")

        if status == 'A':
            after = get_file_content(repo_path, commit_hash, path)
            with tempfile.NamedTemporaryFile(delete=False, mode='w') as tmp_after:
                tmp_after.write(after)
                after_path = tmp_after.name
            output = run_gumtree_diff("/dev/null", after_path)
        elif status == 'D':
            before = get_file_content(repo_path, f"{commit_hash}^", path)
            with tempfile.NamedTemporaryFile(delete=False, mode='w') as tmp_before:
                tmp_before.write(before)
                before_path = tmp_before.name
            output = run_gumtree_diff(before_path, "/dev/null")
        else:
            before = get_file_content(repo_path, f"{commit_hash}^", path)
            after = get_file_content(repo_path, commit_hash, path)
            with tempfile.NamedTemporaryFile(delete=False, mode='w') as tmp_before:
                tmp_before.write(before)
                before_path = tmp_before.name
            with tempfile.NamedTemporaryFile(delete=False, mode='w') as tmp_after:
                tmp_after.write(after)
                after_path = tmp_after.name
            output = run_gumtree_diff(before_path, after_path)

        output_lines.extend(output)
        output_lines.append("")

    output_filename = f"{owner}__{repo_name}__{commit_hash}__summary.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"✅ Summary written to {output_filename}")

# ---------------------- Entry ----------------------

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <owner> <repo_name> <commit_hash>")
        sys.exit(1)

    owner, repo_name, commit_hash = sys.argv[1], sys.argv[2], sys.argv[3]
    summarize_diff(owner, repo_name, commit_hash)
