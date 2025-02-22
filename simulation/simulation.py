import matplotlib.pyplot as plt
import heapq
import random
from datetime import datetime, timedelta
import numpy as np
import treeherder

def simulate_test_queue(submit_timestamps, num_machines, test_duration=22):
    """
    Simulates test execution queue and calculates average waiting time.

    :param submit_timestamps: List of submit times in Unix epoch format.
    :param num_machines: Number of machines available for parallel execution.
    :param test_duration: Duration of each test in minutes.
    :return: Average waiting time in minutes.
    """

    # Convert Unix timestamps to datetime objects
    submit_times = [datetime.utcfromtimestamp(ts) for ts in sorted(submit_timestamps)]

    # Min heap to track machine availability (earliest end time first)
    machine_heap = []
    total_wait_time = 0
    task_count = len(submit_times)

    for submit_time in submit_times:
        # Free up machines that have completed their tests
        while machine_heap and machine_heap[0] <= submit_time:
            heapq.heappop(machine_heap)

        # If there are available machines, test starts immediately
        if len(machine_heap) < num_machines:
            wait_time = 0
            heapq.heappush(machine_heap, submit_time + timedelta(minutes=test_duration))
        else:
            # Otherwise, it has to wait for the next available machine
            next_available_time = heapq.heappop(machine_heap)
            wait_time = (next_available_time - submit_time).total_seconds() / 60  # Convert to minutes
            heapq.heappush(machine_heap, next_available_time + timedelta(minutes=test_duration))

        total_wait_time += wait_time

    # Compute average waiting time
    avg_wait_time = total_wait_time / task_count if task_count > 0 else 0
    return avg_wait_time


# def machine_count_test_queue(tests: list[Test], num_machines, test_duration=22):
    




# Generate random past timestamps (50 submissions in the past 3 hours)
# current_time = datetime.utcnow().timestamp()
# submit_timestamps = sorted(random.randint(int(current_time - 10800), int(current_time)) for _ in range(50))
submit_timestamps = treeherder.fetch_submit_timestamps()

# Varying number of machines
machine_counts = np.arange(1, 21)  # Test from 1 to 20 machines
avg_wait_times = [simulate_test_queue(submit_timestamps, num_machines) for num_machines in machine_counts]

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(machine_counts, avg_wait_times, marker='o', linestyle='-')
plt.xlabel("Number of Available Machines")
plt.ylabel("Average Queue Wait Time (minutes)")
plt.title("Effect of Machine Availability on Queue Wait Time")
plt.grid(True)
plt.show()
