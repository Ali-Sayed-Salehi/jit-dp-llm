import os
import subprocess
import sys
from difflib import unified_diff
from collections import defaultdict
import re
from tree_sitter import Language, Parser

# ---------------------- Config ----------------------
REPOS_ROOT = "/speed-scratch/a_s87063/repos/perf-pilot/github_api/repos"
JAVA_LANGUAGE_LIB = "/speed-scratch/a_s87063/repos/perf-pilot/github_api/tree-sitter-langs.so"
JAVA_LANGUAGE_REPO = "tree-sitter-java"

# ---------------------- Tree-sitter Init Script Call ----------------------
def initialize_tree_sitter():
    if not os.path.exists(JAVA_LANGUAGE_LIB):
        print("⚙️ Initializing Tree-sitter shared library...")
        subprocess.run([
            "python", "tree_sitter_init.py",
            JAVA_LANGUAGE_LIB,
            JAVA_LANGUAGE_REPO
        ], check=True)

initialize_tree_sitter()

JAVA_LANGUAGE = Language(JAVA_LANGUAGE_LIB, "java")
PARSER = Parser()
if PARSER.language != JAVA_LANGUAGE:
    PARSER.set_language(JAVA_LANGUAGE)

def get_repo_path(owner, name):
    return os.path.join(REPOS_ROOT, f"{owner}__{name}")

def clone_if_needed(owner, name):
    repo_path = get_repo_path(owner, name)
    if not os.path.exists(repo_path):
        os.makedirs(REPOS_ROOT, exist_ok=True)
        url = f"https://github.com/{owner}/{name}.git"
        print(f"📅 Cloning {url}...")
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

# ---------------------- Tree-sitter Utilities ----------------------

def remove_comments_and_imports(code_lines):
    code = "\n".join(code_lines)
    tree = PARSER.parse(bytes(code, "utf8"))
    root = tree.root_node
    lines_to_exclude = set()

    def visit(node):
        if node.type in {"line_comment", "block_comment"}:
            for i in range(node.start_point[0], node.end_point[0] + 1):
                lines_to_exclude.add(i)
        elif node.type in {"import_declaration", "package_declaration"}:
            for i in range(node.start_point[0], node.end_point[0] + 1):
                lines_to_exclude.add(i)
        for child in node.children:
            visit(child)

    visit(root)
    return [line for idx, line in enumerate(code_lines) if idx not in lines_to_exclude]

def get_enclosing_method_and_path(code_lines, lineno, max_depth=6):
    code = "\n".join(code_lines)
    tree = PARSER.parse(bytes(code, "utf8"))
    root = tree.root_node

    method_name = "UnknownMethod"
    ast_path = []

    def find(node, path):
        nonlocal method_name, ast_path
        if node.start_point[0] <= lineno <= node.end_point[0]:
            path.append(node.type)
            if node.type == "method_declaration":
                method_name = code[node.start_byte:node.end_byte].split("(")[0].strip().split()[-1]
            for child in node.children:
                find(child, path[:])
            if len(path) <= max_depth:
                ast_path = path

    find(root, [])
    return method_name, ">".join(ast_path) if ast_path else "UnknownPath"

# ---------------------- Diff Formatter ----------------------

def format_diff(owner, repo_name, commit_hash):
    repo_path = clone_if_needed(owner, repo_name)
    output_lines = []
    output_lines.append(f"<COMMIT_MESSAGE>{get_commit_message(repo_path, commit_hash)}</COMMIT_MESSAGE>\n")

    changed_files = get_changed_java_files(repo_path, commit_hash)
    for file_path in changed_files:
        before_raw = get_file(repo_path, f"{commit_hash}^", file_path)
        after_raw = get_file(repo_path, commit_hash, file_path)

        if not before_raw and after_raw:
            output_lines.append(f"<FILE name=\"{file_path}\">\n  <ADDED_FILE/>\n</FILE>\n")
            continue
        elif before_raw and not after_raw:
            output_lines.append(f"<FILE name=\"{file_path}\">\n  <REMOVED_FILE/>\n</FILE>\n")
            continue

        before = remove_comments_and_imports(before_raw)
        after = remove_comments_and_imports(after_raw)

        output_lines.append(f"<FILE name=\"{file_path}\">")

        diff = list(unified_diff(before, after, n=0))
        old_line = None
        new_line = None

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
                method, path = get_enclosing_method_and_path(before, old_line - 1)
                function_changes[method]['removed'][path].append(content)
                old_line += 1
            elif line.startswith('+'):
                if new_line is None:
                    continue
                content = line[1:].rstrip()
                method, path = get_enclosing_method_and_path(after, new_line - 1)
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

        output_lines.append("</FILE>\n")

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
