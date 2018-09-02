import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True, font_scale=1.2)

user = pd.read_csv('../data/user_features/user_features_analysis.csv')


plt.figure(figsize=(12,8))
sns.countplot(y="timezone", data=user)
plt.show()

plt.figure(figsize=(12,8))
sns.countplot(y="language", data=user)
plt.show()

plt.figure(figsize=(12,8))
sns.countplot(y="platform", data=user)
plt.show()

plt.figure(figsize=(12,8))
sns.countplot(y="screen", data=user)
plt.show()

