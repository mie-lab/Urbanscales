a = []
for base in [
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
]:  # , 6, 7, 8, 9, 10]:
    for i in range(7):  # :range(60, 120, 10):
        scale = base * (2 ** i)
        if scale > 300:
            continue
        a.append(scale)

import numpy as np

b = np.random.rand(30, 10) * 0
for val in a:
    b[val // 10][val % 10] = 1


import matplotlib.pyplot as plt

plt.imshow(b)
plt.show()
