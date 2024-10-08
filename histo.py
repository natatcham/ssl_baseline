import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_path = 'test_results/MOS_all_scores.csv'
df = pd.read_csv(csv_path)

df['sentence'] = df['sentence'].apply(lambda x: str(x).zfill(4))

def round_to_nearest_quarter(score):
    return score
df_fh1 = df[df['task'] == 'FH1']

system_mean_scores = df_fh1.groupby('system')['score'].mean()
system_mean_scores_rounded = system_mean_scores.apply(round_to_nearest_quarter)
score_counts = system_mean_scores_rounded.value_counts().sort_index()
plt.figure(figsize=(10, 6))


bins = np.arange(0, 1.25, 0.25) - 0.125
plt.hist(system_mean_scores_rounded, bins=bins, edgecolor='black', color='b', align='mid')

plt.xticks(np.arange(0, 1.25, 0.25), [0.0, 0.25, 0.5, 0.75, 1.0])

plt.xlabel('Rounded Score MOS')
plt.ylabel('Number of Systems')
plt.title('Histogram of MOS Mean Scores per System')

plt.show()
