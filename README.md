# How to Setup

```bash
git clone git@github.com:Ali-Sayed-Salehi/perf-pilot.git

python -m venv venv

source my-venv/Scripts/activate

pip install -r requirements.txt

# create a Google Cloud service account and enable accesss to Google Drive. Store its key as a json file
# then contact repo owners and provide your service account email to gain access to data files. Then run:
dvc remote modify --local gdrive_remote gdrive_use_service_account true
dvc remote modify --local gdrive_remote gdrive_service_account_json_file_path "/path/to/your-key.json"

dvc pull

```