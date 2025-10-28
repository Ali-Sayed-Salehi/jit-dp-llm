import requests
from pprint import pprint
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import ast
from collections import Counter
from dotenv import load_dotenv
import os

"""
Fetches recent Bugzilla bugs and links them with performance regressions from Treeherder.

Flow:
1. Uses the Bugzilla API to fetch bugs created within TIMESPAN_IN_DAYS.
2. Saves all raw bugs to datasets/mozilla_perf/all_bugs.csv.
3. Loads the Treeherder alerts_with_bug_and_test_info.csv for existing performance regressions.
4. Matches each regression bug with its regressor bugs using Bugzilla’s 'regressed_by' field.
5. Collects product and component info to focus on relevant areas.
6. Produces a final CSV (perf_bugs.csv) labeling each bug as:
   - Performance regressor (caused a regression)
   - Performance regression (was regressed)
   - Includes regressed tests, alert IDs, and metadata.
"""

REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
secrets_path = os.path.join(REPO_PATH, "secrets", ".env")
load_dotenv(dotenv_path=secrets_path)

all_bugs_path = os.path.join(REPO_PATH, "datasets", "mozilla_perf", "all_bugs.csv")

API_KEY = os.getenv("BUGZILLA_API_KEY")
BUGZILLA_API_URL = "https://bugzilla.mozilla.org/rest"

BUGZILLA_API = f"{BUGZILLA_API_URL}/bug"
FIELDS = ["id", "summary", "regressed_by", "product", "component", "creation_time"]
LIMIT = 150  # Page size

START_DATE_ISO = "2024-10-10T00:00:00Z"   # <-- set the date you want
threshold_time = START_DATE_ISO

# TIMESPAN_IN_YEARS = 10
# TIMESPAN_IN_DAYS = TIMESPAN_IN_YEARS * 365

offset = 0
all_bugs_list = []

now = datetime.now()
# threshold_time = now - relativedelta(days=TIMESPAN_IN_DAYS)

# get all the bugs in the last year
while True:
    params = {
        "api_key": API_KEY,
        "include_fields": ",".join(FIELDS),
        "v1": threshold_time,
        "o1": "greaterthan",
        "f1": "creation_ts",
        "limit": LIMIT,
        "offset": offset
    }

    response = requests.get(BUGZILLA_API, params=params)

    bugs = response.json().get("bugs", [])
    if not bugs:
        break  # No more bugs

    all_bugs_list.extend(bugs)
    offset += LIMIT

all_bugs_df = pd.DataFrame(all_bugs_list)

all_bugs_df.to_csv(all_bugs_path, index=False)

print(f"✅ Saved all the bugs from Bugzilla to {all_bugs_path}")

all_bugs_df = pd.read_csv(all_bugs_path)
all_bugs_df['creation_time'] = pd.to_datetime(all_bugs_df['creation_time'])
all_bugs_df['regressed_by'] = all_bugs_df['regressed_by'].apply(ast.literal_eval)
all_bugs_df_sorted = all_bugs_df.sort_values(by='creation_time')

# sorted list of bugs by creation time
all_bugs_list = all_bugs_df.to_dict(orient='records')

# get all perf regression alerts from treeherder module
regressions_path = os.path.join(REPO_PATH, "datasets", "mozilla_perf", "alerts_with_bug_and_test_info.csv")
regressions_df = pd.read_csv(regressions_path)
regressions_df['reg_perf_tests_list'] = regressions_df['reg_perf_tests_list'].apply(ast.literal_eval)
regressions_list = regressions_df.to_dict(orient='records')

# convert regressions_list from list to dict for fast lookup
regressions_dict = {}
for bug in regressions_list:
    regressions_dict[bug['regression_bug_id']] = bug

# convert all_bugs from list to dict for fast lookup
all_bugs_dict = {}
for bug in all_bugs_list:
    all_bugs_dict[bug['id']] = bug

# get regressor bugs from regression bugs and add their relevant info
all_regressor_bugs_list = []
regression_bugs_not_in_bugzilla_bugs = []
regressions_without_regressor = []

for bug in regressions_list:

    regression_bug_id = bug['regression_bug_id']
    regression_bug = all_bugs_dict.get(regression_bug_id)

    if not regression_bug:
        regression_bugs_not_in_bugzilla_bugs.append(regression_bug_id)
        # pprint(bug['perf_reg_alert_summary_id'])
        continue

    regressor_bug_ids_list = regression_bug.get('regressed_by')
    
    # Regression is the regressor (the perf regression message is posted to the regressor bug ticket)
    if not regressor_bug_ids_list:
        regressor_bug_ids_list.append(regression_bug_id)

    for bug_id in regressor_bug_ids_list:
        regressor_bug = {"bug_id": bug_id, 
                            "regressed_perf_tests": bug['reg_perf_tests_list'], 
                            "regressed_product": regression_bug['product'], 
                            "regressed_component": regression_bug['component'],
                            "regressor_revision": bug['regressor_push_head_revision']}

        all_regressor_bugs_list.append(regressor_bug)

# pprint("regression_bugs_not_in_bugzilla_bugs:\n")
# pprint(regression_bugs_not_in_bugzilla_bugs)
# print()

# gather relevant components and products
relevant_components = set()
relevant_products = set()

# for counting purposes
relevant_components_list = []
relevant_products_list = []

for bug in all_regressor_bugs_list:
    product = bug.get("regressed_product")
    component = bug.get("regressed_component")

    relevant_products.add(product)
    relevant_components.add(component)

    relevant_components_list.append(component)
    relevant_products_list.append(product)


products_count = Counter(relevant_products_list)
components_count = Counter(relevant_components_list)

pprint(products_count)
pprint(components_count)

# convert all_regressor_bugs_list from list to dict for fast lookup
all_regressor_bugs_dict = {}
for bug in all_regressor_bugs_list:
    all_regressor_bugs_dict[bug['bug_id']] = bug

# create the final csv file
all_bugs_with_added_regressor_info_list = []
for bug in all_bugs_list:
    if (bug['component'] not in relevant_components) or (bug['product'] not in relevant_products):
        continue

    bug_id = bug['id']
    bug_summary = bug.get('summary')
    bug_component = bug.get('component')
    bug_product = bug.get('product')
    bug_creation_time = bug.get('creation_time')

    bug_is_perf_regressor = bug_id in all_regressor_bugs_dict
    bug_is_perf_regression = bug_id in regressions_dict

    regressed_perf_tests_list = []
    perf_reg_alert_summary_id = 0

    regressor_revision = None

    if bug_is_perf_regressor:
        regressor_bug = all_regressor_bugs_dict.get(bug_id)
        regressed_perf_tests_list = regressor_bug.get('regressed_perf_tests')
        regressor_revision = regressor_bug.get('regressor_revision')

    if bug_is_perf_regression:
        regression_bug = regressions_dict.get(bug_id)
        perf_reg_alert_summary_id = regression_bug.get('perf_reg_alert_summary_id')

    bug_with_added_info = {'bug_id': bug_id, 
                           'bug_summary': bug_summary, 
                           'bug_is_perf_regressor': bug_is_perf_regressor,
                           'bug_is_perf_regression': bug_is_perf_regression,
                           'regressed_perf_tests': regressed_perf_tests_list,
                           'regressor_revision': regressor_revision,
                           'perf_reg_alert_summary_id': perf_reg_alert_summary_id, 
                           'product': bug_product,
                           'component': bug_component,
                           'creation_time': bug_creation_time}
    all_bugs_with_added_regressor_info_list.append(bug_with_added_info)

# for bug in all_bugs_with_added_regressor_info_list:
#     if bug['bug_is_perf_regression'] and bug['bug_is_perf_regressor']:
#         pprint(bug)

all_bugs_with_added_regressor_info_df = pd.DataFrame(all_bugs_with_added_regressor_info_list)
perf_bugs_path = os.path.join(REPO_PATH, "datasets", "mozilla_perf", "perf_bugs.csv")
all_bugs_with_added_regressor_info_df.to_csv(perf_bugs_path, index=False)

print(f"✅ Perf bugs with added info saved to {perf_bugs_path}")