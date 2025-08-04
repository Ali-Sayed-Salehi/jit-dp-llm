import pickle
import pandas as pd
import sys

def load_pkl_as_df(path):
    """Load a pickle file as a pandas DataFrame."""
    with open(path, "rb") as f:
        df = pickle.load(f)
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"{path} does not contain a pandas DataFrame.")
        return df

def combine_and_sort(pkl_files):
    # Load all dataframes
    dfs = [load_pkl_as_df(p) for p in pkl_files]

    # Concatenate them
    combined_df = pd.concat(dfs, ignore_index=True)

    # Sort by author_date_unix_timestamp
    if "author_date_unix_timestamp" not in combined_df.columns:
        raise KeyError("Column 'author_date_unix_timestamp' not found in dataframes")

    combined_df = combined_df.sort_values(by="author_date_unix_timestamp")

    # Drop unwanted columns (ignore if missing)
    drop_cols = [
        "parent_hashes", "author_name", "author_email",
        "author_date", "fileschanged", "classification",
        "commit_message"
    ]
    combined_df = combined_df.drop(columns=[c for c in drop_cols if c in combined_df.columns], errors="ignore")

    # Rename columns
    rename_map = {
        "commit_hash": "commit_id",
        "author_date_unix_timestamp": "author_date",
        "is_buggy_commit": "buggy",
        "entropy": "ent",
        "exp":"aexp",
        "rexp":"arexp",
        "sexp":"asexp",
    }
    combined_df = combined_df.rename(columns=rename_map)


    # Add GitHub project mapping
    project_to_repo = {
        "commons-digester":    "apache/commons-digester",
        "commons-beanutils":   "apache/commons-beanutils",
        "commons-collections": "apache/commons-collections",
        "commons-bcel":        "apache/commons-bcel",
        "commons-validator":   "apache/commons-validator",
        "commons-dbcp":        "apache/commons-dbcp",
        "commons-io":          "apache/commons-io",
        "commons-net":         "apache/commons-net",
        "commons-jcs":         "apache/commons-jcs",
        "commons-vfs":         "apache/commons-vfs",
        "commons-lang":        "apache/commons-lang",
        "commons-codec":       "apache/commons-codec",
        "commons-math":        "apache/commons-math",
        "commons-compress":    "apache/commons-compress",
        "commons-configuration":"apache/commons-configuration",
        "commons-scxml":       "apache/commons-scxml",
        "ant-ivy":             "apache/ant-ivy",
        "gora":                "apache/gora",
        "giraph":              "apache/giraph",
        "opennlp":             "apache/opennlp",
        "parquet-mr":          "apache/parquet-mr",
    }

    # ✅ Replace project values with owner/repo, fallback to original if missing
    if "project" in combined_df.columns:
        original_projects = combined_df["project"].unique().tolist()

        combined_df["project"] = combined_df["project"].apply(
            lambda p: project_to_repo.get(p, p)  # fallback to original name if not mapped
        )

        # Check for missing mappings
        missing_projects = [p for p in original_projects if p not in project_to_repo]
        if missing_projects:
            print("⚠️ Warning: Some projects were not found in mapping and kept as-is:")
            for proj in missing_projects:
                print(f"   - {proj}")
    else:
        print("⚠️ No 'project' column found in the dataset.")



    # # Print all unique project names
    # if "project" in combined_df.columns:
    #     projects = combined_df["project"].unique()
    #     print(f"📊 Projects in dataset ({len(projects)}): {projects}")
    # else:
    #     print("⚠️ No 'project' column found in the dataset.")

    # Save total dataset
    total_csv = "jit_defects4j_total.csv"
    combined_df.to_csv(total_csv, index=False)
    print(f"✅ Full dataset saved to {total_csv} (shape={combined_df.shape})")

    # Save last 8000 rows
    small_csv = "jit_defects4j_small.csv"
    last_8k = combined_df.tail(8000)
    last_8k.to_csv(small_csv, index=False)
    print(f"✅ Last 8000 rows saved to {small_csv} (shape={last_8k.shape})")

    print(f"Columns: {combined_df.columns.tolist()}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python combine_pkls.py file1.pkl file2.pkl file3.pkl")
        sys.exit(1)

    pkl_files = sys.argv[1:4]
    combine_and_sort(pkl_files)
