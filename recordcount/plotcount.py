count = []
with open('counts.txt') as f:
    for line in f:
        count.append(int(line.split('\n')[0]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.figure(figsize=(12,8))
plt.bar(range(len(count)),count)
plt.xlabel('date', fontsize=14)
plt.ylabel('no.record', fontsize=14)
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.show()