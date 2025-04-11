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

# Modules

### bugzilla
Scripts here deal with operations on bugs from the Mozilla bugzilla API

### llama
Fine-tuning llama models for defect prediction

### conduit
Mozilla Phabricator Conduit API which deals with code diffs and commits

### datasets
Includes all the resulting datasets created from various scripts. Can be fetched using dvc from Google Drive

### mercurial
hg cli scripts for dealing with diffs and commits

### pushlog
A limited Mozilla API that only deals with push information for the mercurial repos

### simulation
Simulating various changes to software engineering flows on historical data

### treeherder
Mozilla treeherder API for info related to their CI pipeline, i.e. Taskcluster. This API has info on tests and regressions


### scripts
various helper scripts