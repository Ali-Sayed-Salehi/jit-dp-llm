import pandas as pd
from pprint import pprint
import sys
import os
import argparse

REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_PATH)

from github_api import get_commit_message, get_commit_diff

parser = argparse.ArgumentParser(description="Choose which dataset to load")
parser.add_argument(
    "--mode",
    choices=["apachejit_llm", "apachejit_logreg", "mozilla_perf"],
    required=True,
    help="""choose which data operation to do.
    apachejit_llm: ApacheJIT defect prediction dataset for LLMs,
    apachejit_logreg: ApacheJIT defect prediction dataset for logistic regression,
    mozilla_perf: Mozilla Performance regression dataset
    """
)
parser.add_argument("--debug", action="store_true", help="Process only a small portion of the data")

args = parser.parse_args()
DEBUG = args.debug

print(f"preparing data for {args.mode} {'(debug mode)' if DEBUG else ''}")

if args.mode == "apachejit_llm":
    input_data_path = os.path.join(REPO_PATH, "datasets", "jit_dp", "apachejit_with_diff.jsonl")
    apachejit_with_diff_df = pd.read_json(input_data_path, lines=True)
    apachejit_list = apachejit_with_diff_df.to_dict(orient='records')

    if DEBUG:
        apachejit_list = apachejit_list[:10]

    sorted_apachejit_list = sorted(apachejit_list, key=lambda x: x['author_date'])

    new_jit_list = []
    OWNER = "apache"

    for commit in sorted_apachejit_list:
        project = commit.get('project')
        project_parts = project.split("/")
        repo = project_parts[1]
        commit_hash = commit.get('commit_id')
        message = commit.get('commit_message')
        diff = commit.get('diff')

        if not message or not diff:
            continue

        response = "1" if commit.get('buggy') else "0"

        prompt = f"""[project]:
        {repo}
        [commit message]:
        {message}
        [code diff]:
        {diff}"""

        new_jit_list.append({'prompt': prompt, 'response': response})

    new_jit_df = pd.DataFrame(new_jit_list)
    output_data_path = os.path.join(REPO_PATH, "datasets", "jit_dp", "apachejit_llm.jsonl")
    new_jit_df.to_json(output_data_path, orient="records", lines=True)
    print(f"✅ Saved dataset to {output_data_path}")

elif args.mode == "apachejit_logreg":
    input_data_path = os.path.join(REPO_PATH, "datasets", "jit_dp", "apachejit_total.csv")
    apachejit_df = pd.read_csv(input_data_path)
    apachejit_list = apachejit_df.to_dict(orient='records')

    if DEBUG:
        apachejit_list = apachejit_list[:10]

    new_jit_list = []

    for commit in apachejit_list:
        new_commit = {
            'date_created': commit.get('author_date'),
            'num_lines_added': commit.get('la'),
            'num_lines_deleted': commit.get('ld'),
            'num_files_touched': commit.get('nf'),
            'num_directories_touched': commit.get('nd'),
            'num_subsystems_touched': commit.get('ns'),
            'change_entropy': commit.get('ent'),
            'num_developers_touched_files': commit.get('ndev'),
            'time_from_last_change': commit.get('age'),
            'num_changes_in_files': commit.get('nuc'),
            'author_experience': commit.get('aexp'),
            'author_recent_experience': commit.get('arexp'),
            'author_subsystem_experience': commit.get('asexp'),
            'label': "1" if commit.get('buggy') else "0"
        }

        new_jit_list.append(new_commit)

    sorted_new_jit_df = pd.DataFrame(sorted(new_jit_list, key=lambda x: x['date_created']))
    sorted_new_jit_df.replace(["", "null", "N/A", "--"], pd.NA, inplace=True)
    sorted_new_jit_clean_df = sorted_new_jit_df.dropna()

    output_data_path = os.path.join(REPO_PATH, "datasets", "jit_dp", "apachejit_logreg.csv")
    sorted_new_jit_clean_df.to_csv(output_data_path, index=False)
    print(f"✅ Saved dataset to {output_data_path}")

elif args.mode == "mozilla_perf":
    input_data_path = os.path.join(REPO_PATH, "datasets", "bugs_with_diff.jsonl")
    bugs_with_diff_df = pd.read_json(input_data_path, lines=True)
    bugs_list = bugs_with_diff_df.to_dict(orient='records')

    if DEBUG:
        bugs_list = bugs_list[:10]

    dataset = []

    for bug in bugs_list:
        prompt = f"""[product]:
        {bug.get('product')}
        [component]:
        {bug.get('component')}
        [code diff]:
        {bug.get('raw_diff')}"""

        response = "1" if bug.get('bug_is_perf_regressor') else "0"

        dataset.append({'prompt': prompt, 'response': response})

    dataset_df = pd.DataFrame(dataset)
    output_data_path = os.path.join(REPO_PATH, "datasets", "dataset.jsonl")
    dataset_df.to_json(output_data_path, orient="records", lines=True)
    print(f"✅ Saved dataset to {output_data_path}")
