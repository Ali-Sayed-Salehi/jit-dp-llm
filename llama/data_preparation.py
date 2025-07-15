import pandas as pd
from pprint import pprint
import sys
import os
import argparse
import textwrap

REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_PATH)

from github_api import get_commit_message, get_commit_diff

parser = argparse.ArgumentParser(description="Choose which dataset to load")
parser.add_argument(
    "--mode",
    choices=["apachejit_llm", "apachejit_llm_struc", "apachejit_logreg", "mozilla_perf"],
    required=True,
    help="""choose which data operation to do.
    apachejit_llm: ApacheJIT defect prediction dataset for LLMs,
    apachejit_logreg: ApacheJIT defect prediction dataset for logistic regression,
    mozilla_perf: Mozilla Performance regression dataset,
    apachejit_llm_struc: ApacheJIT defect prediction dataset for LLMs with structured commits
    """
)
parser.add_argument("--debug", action="store_true", help="Process only a small portion of the data")
parser.add_argument("--include_metadata", action="store_true", help="Inlcude other commit features in buckets instead ofnumerical values.")

args = parser.parse_args()
DEBUG = args.debug

# Normalize and bucketize
def normalize_and_bucketize(value, min_val, max_val):
    if max_val == min_val:
        norm = 0.0  # Avoid division by zero
    else:
        norm = (value - min_val) / (max_val - min_val)
    
    # Bucketize into 5 bins
    if norm < 0.2:
        return "VERY_LOW"
    elif norm < 0.4:
        return "LOW"
    elif norm < 0.6:
        return "MEDIUM"
    elif norm < 0.8:
        return "HIGH"
    else:
        return "VERY_HIGH"

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

elif args.mode == "apachejit_llm_struc":
    input_data_path = os.path.join(REPO_PATH, "datasets", "jit_dp", "apachejit_small_with_struc_diff.jsonl")
    apachejit_with_diff_df = pd.read_json(input_data_path, lines=True)
    
    if args.include_metadata:
        metadata_cols = ["la","ld","nf","nd","ns","ent","ndev","age","nuc","aexp","arexp","asexp"]
        # Exclude outliers using 5th and 95th percentiles
        min_max_stats = {
            col: tuple(apachejit_with_diff_df[col].clip(lower=apachejit_with_diff_df[col].quantile(0.05),
                                                        upper=apachejit_with_diff_df[col].quantile(0.95)).agg(['min', 'max']))
            for col in metadata_cols
        }

        for col in metadata_cols:
            min_val, max_val = min_max_stats[col]
            apachejit_with_diff_df[f"{col}_bucketized"] = apachejit_with_diff_df[col].apply(lambda x: normalize_and_bucketize(x, min_val, max_val))


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
        diff = commit.get('diff')
        if not diff:
            continue
        response = "1" if commit.get('buggy') else "0"

        if args.include_metadata:
            num_lines_added = commit.get('la_bucketized')
            num_lines_deleted = commit.get('ld_bucketized')
            num_files_touched = commit.get('nf_bucketized')
            num_directories_touched = commit.get('nd_bucketized')
            num_subsystems_touched = commit.get('ns_bucketized')
            change_entropy = commit.get('ent_bucketized')
            num_developers_touched_files = commit.get('ndev_bucketized')
            time_from_last_change = commit.get('age_bucketized')
            num_changes_in_files = commit.get('nuc_bucketized')
            author_experience = commit.get('aexp_bucketized')
            author_recent_experience = commit.get('arexp_bucketized')
            author_subsystem_experience = commit.get('asexp_bucketized')

            lines = [
                "<METADATA>",
                f"num_lines_added: {num_lines_added}",
                f"num_lines_deleted: {num_lines_deleted}",
                f"num_files_touched: {num_files_touched}",
                f"num_directories_touched: {num_directories_touched}",
                f"num_subsystems_touched: {num_subsystems_touched}",
                f"change_entropy: {change_entropy}",
                f"num_developers_touched_files: {num_developers_touched_files}",
                f"time_from_last_change: {time_from_last_change}",
                f"num_changes_in_files: {num_changes_in_files}",
                f"author_experience: {author_experience}",
                f"author_recent_experience: {author_recent_experience}",
                f"author_subsystem_experience: {author_subsystem_experience}",
                "</METADATA>",
                "",
                diff,
            ]
            prompt = "\n".join(lines)
        else:
            prompt = diff

        new_jit_list.append({'prompt': prompt, 'response': response})

    new_jit_df = pd.DataFrame(new_jit_list)
    output_data_path = os.path.join(REPO_PATH, "datasets", "jit_dp", "apachejit_llm_small_struc_meta.jsonl")
    new_jit_df.to_json(output_data_path, orient="records", lines=True)
    print(f"✅ Saved dataset to {output_data_path}")