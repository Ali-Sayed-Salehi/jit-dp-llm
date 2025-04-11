#How to Run

```bash
git clone git@github.com:Ali-Sayed-Salehi/perf-pilot.git

python -m venv my-venv

source my-venv/Scripts/activate

pip install -r requirements.txt

# create a Google Cloud service account that can accesss Google Drive and store its key as a json file
dvc remote modify --local gdrive_remote gdrive_use_service_account true
dvc remote modify --local gdrive_remote gdrive_service_account_json_file_path "/path/to/your-key.json"

dvc pull

```