import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_path = 'test_results/MOS_all_scores.csv'
df = pd.read_csv(csv_path)

df['sentence'] = df['sentence'].apply(lambda x: str(x).zfill(4))

def extract_id_from_wav(filename):
    wav_id = filename.split('_')[-1].split('.')[0]
    return wav_id  

def round_to_nearest_quarter(score):
    return round(score * 4) / 4

systems = df['system'].unique()

print("Available systems:")
sys=[]
for system in systems:
    sys.append(system)
    print(system)
print(len(sys))
selected_system = input("Enter the name of the system to visualize the histogram for: ")

wav_directory = f'{selected_system}/2023-FH1_submission_directory/FH1_MOS/wav'

df_fh1 = df[(df['task'] == 'FH1') & (df['system'] == selected_system)]

sentence_mean_scores = df_fh1.groupby('sentence')['score'].mean()

sentence_mean_scores_rounded = sentence_mean_scores.apply(round_to_nearest_quarter)

matching_scores_rounded = []

for root, dirs, files in os.walk(wav_directory):
    for file in files:
        if file.endswith('.wav'):
            wav_id = extract_id_from_wav(file)
            if wav_id in sentence_mean_scores_rounded.index:
                score_mean_rounded = sentence_mean_scores_rounded[wav_id]
                matching_scores_rounded.append(score_mean_rounded)

if matching_scores_rounded: 
    plt.figure(figsize=(10, 6))

    bins = np.arange(0, 1.25, 0.25) - 0.125

    plt.hist(matching_scores_rounded, bins=bins, edgecolor='black', color='b', align='mid')

    plt.xticks(np.arange(0, 1.25, 0.25), [0.0, 0.25, 0.5, 0.75, 1.0])

    plt.xlabel('Rounded Score MOS')
    plt.ylabel("Nombre d'échantillons")
    plt.title(f'Histogramme des moyennes des Scores MOS pour {selected_system}')

    plt.show()
else:
    print(f"No matching scores found for system '{selected_system}'.")
