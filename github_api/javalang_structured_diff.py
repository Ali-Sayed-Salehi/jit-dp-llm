
import os
import subprocess
import javalang
import re
import sys
from charset_normalizer import from_bytes
from itertools import groupby
from operator import itemgetter

REPOS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "github_repos"))

def safe_git_output(args):
    raw = subprocess.check_output(args, stderr=subprocess.DEVNULL)
    detected = from_bytes(raw).best()
    return str(detected) if detected else raw.decode("utf-8", errors="replace")

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

def get_commit_message(repo_path, commit):
    raw_message = safe_git_output(["git", "-C", repo_path, "log", "-n", "1", "--format=%B", commit])
    lines = raw_message.splitlines()
    lines = [line for line in lines if not line.strip().startswith("git-svn-id:")]
    lines = [line.strip() for line in lines if line.strip()]
    cleaned = []
    for idx, line in enumerate(lines):
        line = re.sub(r"^(\[?[A-Z]+-\d+\]?;?\s*)", "", line)
        line = re.sub(r"\[[A-Z]+-\d+\]", "some ticket", line)
        line = re.sub(r"#\d+", "some ticket", line)
        line = re.sub(r"\b[A-Z]+-\d+\b", "some ticket", line)
        line = re.sub(r"<[^>]+>", "", line)
        cleaned.append(line.strip())
    return " ".join(cleaned).strip()

def get_changed_files(repo_path, commit):
    status_lines = safe_git_output(["git", "-C", repo_path, "diff", "--name-status", f"{commit}^", commit]).splitlines()
    numstat_lines = safe_git_output(["git", "-C", repo_path, "diff", "--numstat", f"{commit}^", commit]).splitlines()
    summary_lines = safe_git_output(["git", "-C", repo_path, "diff", "--summary", f"{commit}^", commit]).splitlines()
    binary_files = {line.split("\t")[2] for line in numstat_lines if line.startswith("-\t-\t")}
    symlink_files = set()
    permission_files = set()
    for line in summary_lines:
        if "mode" in line and ("new mode" in line or "old mode" in line):
            match = re.search(r"b/(.+)", line)
            if match:
                permission_files.add(match.group(1))
        if "create mode 120000" in line or "delete mode 120000" in line:
            match = re.search(r"b/(.+)", line)
            if match:
                symlink_files.add(match.group(1))
    changes = []
    for line in status_lines:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            status, path = parts[0], parts[-1]
            changes.append((status, path, path in binary_files, path in symlink_files, path in permission_files))
    return changes

def get_file(repo_path, revision, filepath):
    try:
        return safe_git_output(["git", "-C", repo_path, "show", f"{revision}:{filepath}"]).splitlines()
    except subprocess.CalledProcessError:
        return []

def remove_apache_license_block(lines):
    block_start = None
    block_end = None
    for i in range(min(len(lines), 30)):
        line = lines[i].strip()
        if block_start is None:
            if re.search(r"/\*+.*Licensed to the Apache Software Foundation", line) or (
                line.startswith("/*") and i + 1 < len(lines) and "Licensed to the Apache Software Foundation" in lines[i + 1]
            ):
                block_start = i
        elif "*/" in line:
            block_end = i
            break
    if block_start is not None and block_end is not None:
        return lines[:block_start] + lines[block_end + 1:]
    return lines

def get_unified_diff(repo_path, commit, file_path):
    try:
        return safe_git_output(["git", "-C", repo_path, "diff", "--unified=0", f"{commit}^", commit, "--", file_path]).splitlines()
    except subprocess.CalledProcessError:
        return []

def group_line_entries(entries, max_gap=1):
    if not entries:
        return []
    grouped = []
    current = [entries[0]]
    for prev, curr in zip(entries, entries[1:]):
        if curr[0] - prev[0] <= max_gap:
            current.append(curr)
        else:
            grouped.append(current)
            current = [curr]
    grouped.append(current)
    return grouped

def parse_ast_info(code_lines, lineno):
    code = "\n".join(code_lines)
    try:
        tree = javalang.parse.parse(code)
    except Exception:
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

def find_ast_path(code_lines, line_numbers):
    for ln in line_numbers:
        if 0 < ln <= len(code_lines):
            i = ln - 1
            while i < len(code_lines):
                line = code_lines[i].strip()
                if not is_comment_line(line):
                    return parse_ast_info(code_lines, i + 1)[1]
                i += 1
    return ""

def extract_structured_diff(owner, repo_name, commit_hash, use_ast=False):
    repo_path = clone_if_needed(owner, repo_name)
    output_lines = []
    output_lines.append(f"<COMMIT_MESSAGE>{get_commit_message(repo_path, commit_hash)}</COMMIT_MESSAGE>\n")
    changed_files = get_changed_files(repo_path, commit_hash)
    for status, file_path, is_binary, is_symlink, is_permission_change in changed_files:
        output_lines.append(f"<FILE name=\"{file_path}\">")
        if is_symlink:
            output_lines.append("  Symbolic link change detected but content not shown.")
            output_lines.append("</FILE>\n")
            continue
        if is_permission_change:
            output_lines.append("  File permission change detected. No content change.")
            output_lines.append("</FILE>\n")
            continue
        if is_binary:
            output_lines.append("  Binary file change detected but content not shown.")
            output_lines.append("</FILE>\n")
            continue

        full_before = get_file(repo_path, f"{commit_hash}^", file_path) if status != 'A' else []
        full_after = get_file(repo_path, commit_hash, file_path) if status != 'D' else []
        before = remove_apache_license_block(full_before) if status != 'A' else []
        after = remove_apache_license_block(full_after) if status != 'D' else []
        is_java = file_path.endswith(".java")

        if status == 'A':
            lines = [line for line in after if line.strip()]
            first_code_line = next((i + 1 for i, l in enumerate(full_after) if not is_comment_line(l.strip())), 1)
            path = parse_ast_info(full_after, first_code_line)[1] if use_ast and is_java and lines else ""
            path_attr = f' path=\"{path}\"' if path else ""
            output_lines.append(f"  <ADDED{path_attr}>")
            for line in lines:
                output_lines.append(f"    {line}")
            output_lines.append("  </ADDED>")
        elif status == 'D':
            lines = [line for line in before if line.strip()]
            first_code_line = next((i + 1 for i, l in enumerate(full_before) if not is_comment_line(l.strip())), 1)
            path = parse_ast_info(full_before, first_code_line)[1] if use_ast and is_java and lines else ""
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
                elif line.startswith('-') and old_line is not None:
                    changes['removed'].append((old_line, line[1:].rstrip()))
                    old_line += 1
                elif line.startswith('+') and new_line is not None:
                    changes['added'].append((new_line, line[1:].rstrip()))
                    new_line += 1
            for change_type in ['removed', 'added']:
                entries = [(ln, content) for ln, content in sorted(changes[change_type]) if content.strip()]
                groups = group_line_entries(entries)
                for group in groups:
                    lines = [line for _, line in group]
                    line_nums = [ln for ln, _ in group]
                    code = full_before if change_type == 'removed' else full_after
                    path_of_block = find_ast_path(code, line_nums) if use_ast and is_java else ""
                    path_attr = f' path=\"{path_of_block}\"' if path_of_block else ""
                    output_lines.append(f"  <{change_type.upper()}{path_attr}>")
                    for line in lines:
                        output_lines.append(f"    {line}")
                    output_lines.append(f"  </{change_type.upper()}>")
        output_lines.append("</FILE>\n")
    return "\n".join(output_lines)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python extract_structured_diff.py <owner> <repo_name> <commit_hash> [--ast]")
        sys.exit(1)
    owner, repo_name, commit_hash = sys.argv[1:4]
    use_ast = "--ast" in sys.argv
    result_str = extract_structured_diff(owner, repo_name, commit_hash, use_ast=use_ast)
    filename = f"{owner}__{repo_name}__{commit_hash}.xml"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(result_str)
    print(f"✅ Output written to {filename}")
