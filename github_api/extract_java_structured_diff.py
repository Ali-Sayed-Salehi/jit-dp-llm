import os
import subprocess
import sys
import javalang
from difflib import unified_diff
from collections import defaultdict

# ---------------------- Config ----------------------
REPOS_ROOT = "/speed-scratch/a_s87063/repos/perf-pilot/github_api/repos"  # All repos will be cloned here

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

# ---------------------- Git Logic ----------------------

def get_commit_message(repo_path, commit):
    return subprocess.check_output(
        ["git", "-C", repo_path, "log", "-n", "1", "--format=%B", commit],
        text=True
    ).strip()

def get_changed_java_files(repo_path, commit):
    output = subprocess.check_output(
        ["git", "-C", repo_path, "diff", "--name-only", f"{commit}^", commit],
        text=True
    ).splitlines()
    return [f for f in output if f.endswith(".java")]

def get_file(repo_path, revision, filepath):
    try:
        return subprocess.check_output(
            ["git", "-C", repo_path, "show", f"{revision}:{filepath}"],
            text=True
        ).splitlines()
    except subprocess.CalledProcessError:
        return []

# ---------------------- AST Logic ----------------------

def parse_ast_info(code_lines, lineno):
    code = "\n".join(code_lines)
    try:
        tree = javalang.parse.parse(code)
    except Exception as e:
        print(f"⚠️ AST parse failed at line {lineno}: {e}")
        return "UnknownMethod", "UnknownPath"

    method_name = "UnknownMethod"
    shortest_path = None

    for path, node in tree.filter(javalang.tree.Node):
        # Get method name if inside one
        if isinstance(node, javalang.tree.MethodDeclaration):
            if node.position and node.position.line and node.position.line <= lineno:
                method_name = node.name

        # Check if this node corresponds to the diff line
        if hasattr(node, 'position') and node.position and node.position.line == lineno:
            path_names = [type(p).__name__ for p in path] + [type(node).__name__]
            if not shortest_path or len(path_names) < len(shortest_path):
                shortest_path = path_names

    return method_name, ">".join(shortest_path) if shortest_path else "UnknownPath"


# ---------------------- Diff Formatter ----------------------

def format_diff(owner, repo_name, commit_hash):
    repo_path = clone_if_needed(owner, repo_name)
    output_lines = []

    output_lines.append(f"<COMMIT_MESSAGE>{get_commit_message(repo_path, commit_hash)}</COMMIT_MESSAGE>\n")

    changed_files = get_changed_java_files(repo_path, commit_hash)
    for file_path in changed_files:
        before = get_file(repo_path, f"{commit_hash}^", file_path)
        after = get_file(repo_path, commit_hash, file_path)

        output_lines.append(f"<FILE name=\"{file_path}\">")

        diff = list(unified_diff(before, after, n=0))

        old_line = None
        new_line = None

        # Structure: {function: {'removed': {path: [lines]}, 'added': {path: [lines]}}}
        function_changes = defaultdict(lambda: {'removed': defaultdict(list), 'added': defaultdict(list)})

        for line in diff:
            if line.startswith('@@'):
                parts = line.split()
                old_line = int(parts[1].split(',')[0][1:])
                new_line = int(parts[2].split(',')[0][1:])
            elif line.startswith('-'):
                if old_line is None:
                    continue
                content = line[1:].rstrip()
                method, path = parse_ast_info(before, old_line)
                function_changes[method]['removed'][path].append(content)
                old_line += 1
            elif line.startswith('+'):
                if new_line is None:
                    continue
                content = line[1:].rstrip()
                method, path = parse_ast_info(after, new_line)
                function_changes[method]['added'][path].append(content)
                new_line += 1

        for method, changes in function_changes.items():
            output_lines.append(f"  <FUNCTION name=\"{method}\">")

            if changes['removed']:
                output_lines.append("    <REMOVED>")
                for path, lines in changes['removed'].items():
                    output_lines.append(f"      <LINE path=\"{path}\">")
                    for line in lines:
                        output_lines.append(f"        {line}")
                    output_lines.append("      </LINE>")
                output_lines.append("    </REMOVED>")

            if changes['added']:
                output_lines.append("    <ADDED>")
                for path, lines in changes['added'].items():
                    output_lines.append(f"      <LINE path=\"{path}\">")
                    for line in lines:
                        output_lines.append(f"        {line}")
                    output_lines.append("      </LINE>")
                output_lines.append("    </ADDED>")

            output_lines.append("  </FUNCTION>")

        output_lines.append(f"</FILE>\n")

    filename = f"{owner}__{repo_name}__{commit_hash}.xml"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"✅ Output written to {filename}")



# ---------------------- Entry ----------------------

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <owner> <repo_name> <commit_hash>")
        sys.exit(1)

    owner, repo_name, commit_hash = sys.argv[1], sys.argv[2], sys.argv[3]
    format_diff(owner, repo_name, commit_hash)