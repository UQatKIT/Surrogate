import time

import numpy as np
import umbridge as ub
import matplotlib.pyplot as plt


surrogate = ub.HTTPModel(url="http://localhost:4243", name="surrogate")

input_values = np.linspace(-5, 5, 100).reshape(-1, 1)
results = []

for i in range(input_values.shape[0]):
   time.sleep(0.1)
   values = [input_values[i,:].tolist()]
   results.append(surrogate(values))

result_values = [result[0] for result in results]
result_vars = [result[1] for result in results]

fig, ax = plt.subplots()
