import os
import pandas as pd
import librosa

def extract_id_from_wav(filename):
    wav_id = filename.split('_')[-1].split('.')[0]
    return wav_id  

def round_to_nearest_quarter(score):
    return round(score * 4) / 4

def get_wav_duration(filepath):
    try:
        audio_data, sr = librosa.load(filepath, sr=None)
        return librosa.get_duration(y=audio_data, sr=sr)
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return None

csv_path = 'test_results/MOS_all_scores.csv'
df = pd.read_csv(csv_path)

df['sentence'] = df['sentence'].apply(lambda x: str(x).zfill(4))

systems = df['system'].unique()

wav_data = []

for system in systems:
    print(f"Processing system: {system}")
    
    wav_directory = f'{system}/2023-FS1_submission_directory/FS1_MOS/wav'
    
    df_fh1 = df[(df['task'] == 'FS1') & (df['system'] == system)]
    
    sentence_mean_scores = df_fh1.groupby('sentence')['score'].mean()
    
    sentence_mean_scores_rounded = sentence_mean_scores.apply(round_to_nearest_quarter)
    
    for root, dirs, files in os.walk(wav_directory):
        for file in files:
            if file.endswith('.wav'):
                wav_id = extract_id_from_wav(file)
                if wav_id in sentence_mean_scores_rounded.index:
                    wav_filepath = os.path.join(root, file)
                    
                    duration = get_wav_duration(wav_filepath)
                    
                    score_mean_rounded = sentence_mean_scores_rounded[wav_id]
                    
                    if duration is not None:
                        wav_data.append({
                            'system': system,
                            'wav_filename': file,
                            'wav_id': wav_id,
                            'duration': duration,
                            'mean_MOS_score_rounded': score_mean_rounded
                        })

wav_df = pd.DataFrame(wav_data)

output_csv = 'wavFS1_duration_mos_scores.csv'
wav_df.to_csv(output_csv, index=False)

print(f"Data successfully saved to {output_csv}")
