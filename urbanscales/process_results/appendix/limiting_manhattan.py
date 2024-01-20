import numpy as np
import matplotlib.pyplot as plt


def calculate_distances(origin, destination):
    manhattan_distance = np.abs(destination[0] - origin[0]) + np.abs(destination[1] - origin[1])
    euclidean_distance = np.sqrt((destination[0] - origin[0]) ** 2 + (destination[1] - origin[1]) ** 2)
    return manhattan_distance, euclidean_distance


def generate_random_points(n):
    return np.random.rand(n, 2) * 10  # Scale up by 10 for a larger range


def plot_ratio_vs_n(ratio_values):
    plt.plot(range(1, len(ratio_values) + 1), ratio_values)
    plt.xlabel('Number of OD pairs sampled (N)')
    plt.ylabel('Ratio of Manhattan distance to Euclidean distance')
    plt.title('Ratio of Manhattan distance to Euclidean distance vs N')
    plt.show(block=False)


# Set the number of OD pairs to sample
N = 10000

ratio_values = []

for n in range(1, N + 1, 10):
    origin = generate_random_points(n)
    destination = generate_random_points(n)

    total_ratio = 0

    for i in range(n):
        manhattan_distance, euclidean_distance = calculate_distances(origin[i], destination[i])
        ratio = manhattan_distance / euclidean_distance
        total_ratio += ratio

    average_ratio = total_ratio / n
    ratio_values.append(average_ratio)

plot_ratio_vs_n(ratio_values)
