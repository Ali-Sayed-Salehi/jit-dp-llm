import os
import subprocess
import sys
import javalang
from difflib import unified_diff
from collections import defaultdict
import re

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
    raw_message = subprocess.check_output(
        ["git", "-C", repo_path, "log", "-n", "1", "--format=%B", commit],
        text=True
    ).strip()

    raw_message = re.sub(r"^\s*\[[A-Z]+-\d+\]\s*", "", raw_message)
    raw_message = re.sub(r"\[[A-Z]+-\d+\]", "[some-ticket]", raw_message)
    raw_message = re.sub(r"#\d+", "[some-ticket]", raw_message)

    return raw_message.strip()

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

def get_file(repo_path, revision, filepath):
    try:
        return subprocess.check_output(
            ["git", "-C", repo_path, "show", f"{revision}:{filepath}"],
            text=True
        ).splitlines()
    except subprocess.CalledProcessError:
        return []

def get_unified_diff(repo_path, commit, file_path):
    try:
        return subprocess.check_output(
            ["git", "-C", repo_path, "diff", "--unified=0", f"{commit}^", commit, "--", file_path],
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
    path_names_with_values = None

    for path_tuple, node in tree.filter(javalang.tree.Node):
        path = list(path_tuple) + [node]
        if hasattr(node, 'position') and node.position and node.position.line == lineno:
            filtered_path = []
            for p in path:
                node_type = type(p).__name__
                if node_type in {"CompilationUnit", "list"}:
                    continue
                name = getattr(p, "name", None)
                filtered_path.append(f"{node_type}:{name}" if name else f"{node_type}")

            if not shortest_path or len(filtered_path) < len(shortest_path):
                shortest_path = filtered_path
                path_names_with_values = filtered_path

        if isinstance(node, (javalang.tree.Import, javalang.tree.PackageDeclaration)):
            if node.position and node.position.line == lineno:
                return type(node).__name__, type(node).__name__

        if isinstance(node, javalang.tree.MethodDeclaration):
            if node.position and node.position.line and node.position.line <= lineno:
                method_name = node.name

    return method_name, " > ".join(path_names_with_values) if path_names_with_values else "UnknownPath"

def is_comment_line(line):
    stripped = line.strip()
    return (
        not stripped
        or stripped.startswith("//")
        or stripped.startswith("/*")
        or stripped.endswith("*/")
        or (stripped.startswith("*") and not stripped.startswith("*="))
        or stripped.startswith("@")
    )

# ---------------------- Diff Formatter ----------------------

def find_ast_path(code_lines, line_numbers):
    block_comment = False
    for ln in line_numbers:
        if 0 < ln <= len(code_lines):
            line = code_lines[ln - 1].strip()
            if "/*" in line:
                block_comment = True
            if not block_comment and not is_comment_line(line):
                return parse_ast_info(code_lines, ln)[1]
            if "*/" in line:
                block_comment = False
    return ""

def format_diff(owner, repo_name, commit_hash):
    use_ast = '--ast' in sys.argv

    repo_path = clone_if_needed(owner, repo_name)
    output_lines = []

    output_lines.append(f"<COMMIT_MESSAGE>{get_commit_message(repo_path, commit_hash)}</COMMIT_MESSAGE>\n")

    changed_files = get_changed_files(repo_path, commit_hash)
    for status, file_path in changed_files:
        before = get_file(repo_path, f"{commit_hash}^", file_path) if status != 'A' else []
        after = get_file(repo_path, commit_hash, file_path) if status != 'D' else []

        output_lines.append(f"<FILE name=\"{file_path}\">")

        is_java = file_path.endswith(".java")

        if status == 'A':
            lines = [line for line in after if line.strip()]
            path = parse_ast_info(after, 1)[1] if use_ast and is_java and lines else ""
            path_attr = f' path=\"{path}\"' if path else ""
            output_lines.append(f"  <ADDED{path_attr}>")
            for line in lines:
                output_lines.append(f"    {line}")
            output_lines.append("  </ADDED>")
        elif status == 'D':
            lines = [line for line in before if line.strip()]
            path = parse_ast_info(before, 1)[1] if use_ast and is_java and lines else ""
            path_attr = f' path=\"{path}\"' if path else ""
            output_lines.append(f"  <REMOVED{path_attr}>")
            for line in lines:
                output_lines.append(f"    {line}")
            output_lines.append("  </REMOVED>")
        else:
            diff = get_unified_diff(repo_path, commit_hash, file_path)

            old_line = None
            new_line = None
            changes = {'removed': [], 'added': []}

            for line in diff:
                if line.startswith('@@'):
                    parts = line.split()
                    old_line = int(parts[1].split(',')[0][1:])
                    new_line = int(parts[2].split(',')[0][1:])
                elif line.startswith('-'):
                    if old_line is None:
                        continue
                    content = line[1:].rstrip()
                    changes['removed'].append((old_line, content))
                    old_line += 1
                elif line.startswith('+'):
                    if new_line is None:
                        continue
                    content = line[1:].rstrip()
                    changes['added'].append((new_line, content))
                    new_line += 1

            for change_type in ['removed', 'added']:
                entries = [(ln, content) for ln, content in sorted(changes[change_type]) if content.strip()]
                if entries:
                    path_of_block = find_ast_path(before if change_type == 'removed' else after, [ln for ln, _ in entries]) if use_ast and is_java else ""
                    path_attr = f' path=\"{path_of_block}\"' if path_of_block else ""
                    output_lines.append(f"  <{change_type.upper()}{path_attr}>")
                    for _, content in entries:
                        output_lines.append(f"    {content}")
                    output_lines.append(f"  </{change_type.upper()}>")

        output_lines.append(f"</FILE>\n")

    filename = f"{owner}__{repo_name}__{commit_hash}.xml"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"✅ Output written to {filename}")

# ---------------------- Entry ----------------------

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script.py <owner> <repo_name> <commit_hash> [--ast]")
        sys.exit(1)

    owner, repo_name, commit_hash = sys.argv[1], sys.argv[2], sys.argv[3]
    format_diff(owner, repo_name, commit_hash)
