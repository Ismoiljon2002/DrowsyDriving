# import random

# initial_investment = 10000
# lower_bound = 5000
# upper_bound = 20000
# failure_probability = 0.05
# num_trials = 10000

# total_return = 0

# for _ in range(num_trials):
#     if random.random() <= failure_probability:
#         project_return = -initial_investment
#     else:
#         project_return = random.uniform(lower_bound, upper_bound)
    
#     total_return += project_return

# expected_return = total_return / num_trials

# print("Expected Return: $" + str(expected_return))




# import random
# import math

# num_points = 10000
# lower_bound = 0
# upper_bound = math.pi/2
# total_area = (upper_bound - lower_bound) * 1

# points_under_curve = 0

# for _ in range(num_points):
#     x = random.uniform(lower_bound, upper_bound)
#     y = random.uniform(0, 1)

#     if y <= math.cos(x):
#         points_under_curve += 1

# estimated_area = (points_under_curve / num_points) * total_area

# print("Estimated Area: " + str(estimated_area))




# import random

# initial_population = 5000
# mean_growth_rate = 0.1
# std_dev_growth_rate = 0.02
# carrying_capacity_lower = 8000
# carrying_capacity_upper = 12000
# num_trials = 10000

# total_population = 0

# for _ in range(num_trials):
#     growth_rate = random.normalvariate(mean_growth_rate, std_dev_growth_rate)
#     carrying_capacity = random.uniform(carrying_capacity_lower, carrying_capacity_upper)
    
#     population_next = initial_population + growth_rate * initial_population * (1 - initial_population / carrying_capacity)
#     total_population += population_next

# average_population = total_population / num_trials

# print("Estimated fish population after one year: " + str(average_population))




import random

average_time_between_cars = 2  # minutes
std_dev_time_between_cars = 0.5  # minutes
num_trials = 10000
num_cars_threshold = 30
hour_in_minutes = 60

count_exceeding_cars = 0

for _ in range(num_trials):
    total_time = 0

    for _ in range(num_cars_threshold):
        time_between_cars = random.normalvariate(average_time_between_cars, std_dev_time_between_cars)
        total_time += time_between_cars

    total_time_in_hours = total_time / hour_in_minutes

    if total_time_in_hours <= 1:
        count_exceeding_cars += 1

probability_exceeding_cars = count_exceeding_cars / num_trials

print("Estimated probability of processing more than 30 cars in an hour:", probability_exceeding_cars)
