import numpy as np


def maximum_throughput_scheduler(users, resources_available):
    resources_total = resources_available
    resource_allocation = np.zeros(len(users))

    channel_qualities = []
    for user in users:
        channel_qualities += [user.channel_quality]

    while (sum(channel_qualities) > 0) and (resources_available > 0):
        best_channel_id = int(np.argmax(channel_qualities))
        resources_requested = users[best_channel_id].units_requested
        resources_allocated = min(resources_requested, resources_available)
        resource_allocation[best_channel_id] = resources_allocated

        resources_available -= resources_allocated
        channel_qualities[best_channel_id] = 0

    # resource_allocation = resource_allocation / resources_total
    return resource_allocation


def max_min_fair_scheduler(users, resources_available):
    resources_total = resources_available
    resource_allocation = np.zeros(len(users))

    # Case 1: Resources available >= Resources requested----------------------------------------------------------------
    sum_requests = 0
    for user in users:
        sum_requests += user.units_requested
    if sum_requests <= resources_available:
        for user in users:
            resource_allocation[user.user_id] = user.units_requested
        return resource_allocation

    # Case 2: Grant at most equal resources = resources_available / num_users-------------------------------------------
    weighted_requests = np.zeros(len(users))  # Higher is better: Low request or good channel lead to high weight
    for user in users:
        if user.units_requested > 0:
            weighted_requests[user.user_id] = user.channel_quality / user.units_requested
    num_users_requesting = np.count_nonzero(weighted_requests)
    equal_distribution = np.floor(resources_available / num_users_requesting)

    while sum(weighted_requests) > 0:  # Distribute from best request to worst
        best_request_id = np.argmax(weighted_requests)
        if users[best_request_id].units_requested < equal_distribution:
            units_allocated = users[best_request_id].units_requested

            resources_available -= units_allocated
            num_users_requesting -= 1
            if num_users_requesting > 0:
                equal_distribution = np.floor(resources_available / num_users_requesting)
        else:
            units_allocated = equal_distribution

            resources_available -= units_allocated
            num_users_requesting -= 1
            # Dont update equal distribution here

        resource_allocation[best_request_id] = units_allocated
        weighted_requests[best_request_id] = 0

    # Case 3: Flooring leads to unallocated resources.
    # 1. -> allocate floor(resources_available / num_active_users) to all users with requests remaining
    # 2. -> allocate 1 resource to mod_resources_available(num_active_users) users with the most units requested.
    if sum(resource_allocation) < resources_total:
        remaining_requests = np.zeros(len(users))
        for user in users:
            remaining_requests[user.user_id] = user.units_requested - resource_allocation[user.user_id]
        num_users_requesting = np.count_nonzero(remaining_requests)
        # 1.----------------------------------------------
        if num_users_requesting <= resources_available:
            equal_distribution = np.floor(resources_available / num_users_requesting)
            for user in users:
                if remaining_requests[user.user_id] > 0:
                    units_allocated = min(remaining_requests[user.user_id], equal_distribution)

                    remaining_requests[user.user_id] -= units_allocated
                    resource_allocation[user.user_id] += units_allocated
                    resources_available -= units_allocated
        # 2.----------------------------------------------
        for _ in range(int(resources_available)):
            largest_remaining_request_id = np.argmax(remaining_requests)
            remaining_requests[largest_remaining_request_id] = 0  # Dont allocate more than one unit to a user in 2.
            resource_allocation[largest_remaining_request_id] += 1
            resources_available -= 1
            if sum(remaining_requests) == 0:  # All remaining users have been served once but resources remain -> again
                for user in users:
                    remaining_requests[user.user_id] = user.units_requested - resource_allocation[user.user_id]

    # resource_allocation = resource_allocation / resources_total
    return resource_allocation


def delay_sensitive_scheduler(users, resources_available):
    resources_total = resources_available
    resource_allocation = np.zeros(len(users))

    resources_requested = np.zeros(len(users))
    for user in users:
        resources_requested[user.user_id] = user.units_requested

    # Step 0: Resources available >= Resources requested----------------------------------------------------------------
    sum_requests = sum(resources_requested)
    if sum_requests <= resources_available:
        for user in users:
            resource_allocation[user.user_id] = user.units_requested
        return resource_allocation

    # Step 1: Channel Quality weighting---------------------------------------------------------------------------------
    # Allocate weight equal to datarate_user / datarate_avg * channel_quality_user / channel_quality_avg
    channel_quality_weightings = np.zeros(len(users))
    datarates = np.zeros(len(users))
    channel_qualities = np.zeros(len(users))
    for user in users:
        if resources_requested[user.user_id] > 0:
            datarates[user.user_id] = max(user.datarate, .000000001)
            channel_qualities[user.user_id] = user.channel_quality

    mean_active_datarate = np.mean(datarates[resources_requested > 0])
    mean_active_channel_quality = np.mean(channel_qualities[resources_requested > 0])
    for user in users:
        if resources_requested[user.user_id] > 0:
            channel_quality_weightings[user.user_id] = datarates[user.user_id] / mean_active_datarate * \
                                                       channel_qualities[user.user_id] / mean_active_channel_quality

    channel_quality_weightings = channel_quality_weightings / sum(channel_quality_weightings)  # normalize

    # Step 2: Timeout urgency weighting---------------------------------------------------------------------------------
    # Allocate weight equal to
    # (timeouts_user / lowest_time_to_timeout_user) / sum(timeouts_user / lowest_time_to_timeout_user)
    timeout_urgency_weightings = np.zeros(len(users))
    timeouts = np.zeros(len(users))
    lowest_times_to_timeout = 10_000 * np.ones(len(users))  # lowest per user
    for user in users:
        if resources_requested[user.user_id] > 0:
            # timeouts[user.user_id] = max(user.jobs_lost_to_timeout, 0.001)
            timeouts[user.user_id] = 1
            for job in user.jobs:
                if user.latency_max - job.delay < lowest_times_to_timeout[user.user_id]:
                    lowest_times_to_timeout[user.user_id] = user.latency_max - job.delay

    timeout_ratio = timeouts / lowest_times_to_timeout
    sum_timeout_ratio = sum(timeout_ratio)
    for user in users:
        if resources_requested[user.user_id] > 0:
            timeout_urgency_weightings[user.user_id] = timeout_ratio[user.user_id] / sum_timeout_ratio

    timeout_urgency_weightings = timeout_urgency_weightings / sum(timeout_urgency_weightings)  # normalize

    # Step 3: Scale weights to available resources----------------------------------------------------------------------
    combined_weightings = 0.1 * channel_quality_weightings + 1 * timeout_urgency_weightings  # weights could be added here

    while sum(combined_weightings) > 0:
        combined_weightings = combined_weightings / sum(combined_weightings)  # normalize

        highest_weight_id = np.argmax(combined_weightings)
        units_allocated = min(np.ceil(combined_weightings[highest_weight_id] * resources_available),
                              resources_requested[highest_weight_id])
        if (lowest_times_to_timeout[highest_weight_id] == 1) and \
                (units_allocated < resources_requested[highest_weight_id]): # not enough time left to send packet, ignore
            units_allocated = 0

        resources_available -= units_allocated
        resources_requested[highest_weight_id] -= units_allocated
        resource_allocation[highest_weight_id] = units_allocated

        combined_weightings[highest_weight_id] = 0

    # resource_allocation = resource_allocation / resources_total
    return resource_allocation
