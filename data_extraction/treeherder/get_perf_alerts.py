import requests
from pprint import pprint
import datetime
from thclient import TreeherderClient
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import ast
import os

"""
Fetches recent performance alert summaries from Treeherder and extracts regression info.

Flow:
1. Connects to the Treeherder API and downloads recent alert summaries until reaching the
   given timespan (TIMESPAN_IN_DAYS).
2. Saves all raw alerts to datasets/mozilla_perf/alert_summaries.csv.
3. Filters only alerts linked to bugs with regression-related statuses (fixed, wontfix, backedout).
4. For each alert, collects regressed tests and platforms from its details.
5. Creates a compact CSV (alerts_with_bug_and_test_info.csv) with:
   - Bug ID
   - Regressed perf tests
   - Alert summary ID
This CSV is used later to link regressions with Bugzilla bugs.
"""



REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

client = TreeherderClient()

TIMESPAN_IN_DAYS = 60
COLUMNS = ["regression bug id"]

ALERT_SUMMARY_STATUS_DICT = {
  "untriaged": 0,
  "downstream": 1,
  "reassigned": 2,
  "invalid": 3,
  "improvement": 4,
  "investigating": 5,
  "wontfix": 6,
  "fixed": 7,
  "backedout": 8
}

INCLUDED_ALERT_SUMMARY_STATUSES = {
    ALERT_SUMMARY_STATUS_DICT['wontfix'],
    ALERT_SUMMARY_STATUS_DICT['fixed'],
    ALERT_SUMMARY_STATUS_DICT['backedout']
}

alert_summary_params = {
    "page": 1
}

now = datetime.now()
threshold_time = now - relativedelta(days=TIMESPAN_IN_DAYS)

alert_push_time = now
uri = "performance/alertsummary"

alert_summaries_list = []

# get alert summaries
while (alert_push_time >= threshold_time):

    alert_summaries_response_dict = client._get_json(uri, **alert_summary_params)
    alert_summaries_list.extend(alert_summaries_response_dict["results"])

    next_url = alert_summaries_response_dict['next']
    next_page = next_url.split('page=')[1]
    alert_summary_params['page'] = next_page

    alert_push_time_epoch = alert_summaries_response_dict['results'][-1]['push_timestamp']
    alert_push_time = datetime.fromtimestamp(alert_push_time_epoch)

    time.sleep(0.5)

# print("alert summaries:\n")
# pprint(alert_summaries_list)
# print("\n")

alert_summaries_df = pd.DataFrame(alert_summaries_list)
alert_summaries_path =  os.path.join(REPO_PATH, "datasets", "mozilla_perf", "alert_summaries.csv")
alert_summaries_df.to_csv(alert_summaries_path, index=False)

print(f"✅ Alert summaries saved to {alert_summaries_path}")

alert_summaries_df = pd.read_csv(alert_summaries_path)

alert_summaries_df['alerts'] = alert_summaries_df['alerts'].apply(ast.literal_eval)
alert_summaries_df['related_alerts'] = alert_summaries_df['related_alerts'].apply(ast.literal_eval)
alert_summaries_df['bug_number'] = alert_summaries_df['bug_number'].astype('Int64')

alert_summaries_list = alert_summaries_df.to_dict(orient='records')

filtered_alert_summaries_list = []

# filter alert summaries to only include regressions    
for alert_summary in alert_summaries_list:
    if alert_summary['status'] not in INCLUDED_ALERT_SUMMARY_STATUSES:
            continue

    filtered_alert_summaries_list.append(alert_summary)

# print("filtered_alert_summaries_list:\n")
# pprint(filtered_alert_summaries_list)
# print("\n")

# add relevant perf tests to alert summaries
alert_summaries_with_added_info_list = []

for alert_summary in filtered_alert_summaries_list:
            
    single_alerts_list = []
    regression_tests_set = set()

    single_alerts_list.extend(alert_summary['alerts'])
    single_alerts_list.extend(alert_summary['related_alerts'])

    for alert in single_alerts_list:

        if not alert.get('is_regression'):
             continue
        
        alert_machine_platform = alert['series_signature'].get('machine_platform')
        alert_single_test = alert['series_signature'].get('test')

        if alert_machine_platform or alert_single_test:
             regression_tests_set.add((alert_single_test, alert_machine_platform))

    alert_summary['tests_list'] = [{"test": test, "platform": platform} for test, platform in regression_tests_set]

    alert_summaries_with_added_info_list.append(alert_summary)

# print("alert_summaries_with_added_info_list:\n")
# pprint(alert_summaries_with_added_info_list)
# print("\n")

# extract needed columns 
regression_bug_ids_list = []
alert_summary_ids_list = []
regression_tests_list = []

for alert_summary in alert_summaries_with_added_info_list:
     regression_bug_id = alert_summary.get('bug_number')
     if regression_bug_id:
          regression_bug_ids_list.append(regression_bug_id)
          alert_summary_ids_list.append(alert_summary.get("id"))
          regression_tests_list.append(alert_summary.get("tests_list"))
          
# print("regression_bug_ids_list:\n")
# pprint(regression_bug_ids_list)
# print("\n")

regressions_df = pd.DataFrame({'regression_bug_id': regression_bug_ids_list, 
                               'reg_perf_tests_list': regression_tests_list,
                               'perf_reg_alert_summary_id': alert_summary_ids_list})

alerts_with_bug_and_test_info_path =  os.path.join(REPO_PATH, "datasets", "mozilla_perf", "alerts_with_bug_and_test_info.csv")
regressions_df.to_csv(alerts_with_bug_and_test_info_path, index=False)

print(f"✅ Bug ids, perf tests, perf alert summary ids saved to {alerts_with_bug_and_test_info_path}")