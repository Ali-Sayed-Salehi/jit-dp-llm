import requests
from pprint import pprint
import datetime
from thclient import TreeherderClient
import pandas as pd
from datetime import datetime
import numpy as np
import mercurial.mercurial as mercurial

def get_perf_test_times():
    client = TreeherderClient()

    performance_summary_params = {
    "repository": "autoland",
    "signature":307933,
    "interval":2592000,
    "all_data":True,
    "replicates":False
    }

    data_list = client._get_json("performance/summary", **performance_summary_params)
    data_dict = data_list[0]
    jobs_list = data_dict['data']
    for index, job in enumerate(jobs_list):
      job_params = {
      "id":job["job_id"]
      }
      single_job_list = client.get_jobs("autoland", **job_params)
      job_dict = single_job_list[0]
      submit_time = datetime.fromtimestamp(job_dict['submit_timestamp'])
      start_time = datetime.fromtimestamp(job_dict['start_timestamp'])
      end_time = datetime.fromtimestamp(job_dict['end_timestamp'])
      duration = end_time - start_time
      wait_time = end_time - submit_time

      if index == 0:
        prev_submit_time = submit_time

      submit_time_diff = submit_time - prev_submit_time
      commits = mercurial.fetch_commits(prev_submit_time, submit_time)

      prev_submit_time = submit_time
      print("id: ", job_dict['id'],
        " -> ",
        "   submit time: ", submit_time,
        "   start time: ", start_time,
        "   end time: ", end_time,
        "   duration: ", duration,
        "   wait time: ", wait_time,
        "   submit time difference: ", submit_time_diff,
        "   num of commits in the span: ", len(commits))
      

def fetch_submit_timestamps():
  client = TreeherderClient()

  performance_summary_params = {
  "repository": "autoland",
  "signature":307933,
  "interval":2592000,
  "all_data":True,
  "replicates":False
  }

  data_list = client._get_json("performance/summary", **performance_summary_params)
  data_dict = data_list[0]
  jobs_list = data_dict['data']
  submit_times_list = []

  for index, job in enumerate(jobs_list):
    job_params = {
    "id":job["job_id"]
    }
    single_job_list = client.get_jobs("autoland", **job_params)
    job_dict = single_job_list[0]
    submit_time = datetime.fromtimestamp(job_dict['submit_timestamp'])
    submit_times_list.append(job_dict['submit_timestamp'])

  return submit_times_list



if __name__ == "__main__":
    get_perf_test_times()