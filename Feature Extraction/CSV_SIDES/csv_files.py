import pandas as pd
import os

big_frame = pd.DataFrame()

path = "C:/Users/Mynha/Desktop/Garlic-Varieties-Classification/Feature Extraction/CSV/"

for file in os.listdir(path):
    if file.endswith('.csv'):
        df = pd.read_csv(file)
        big_frame = big_frame.append(df, ignore_index=True)
        big_frame.to_csv('color_features.csv')
