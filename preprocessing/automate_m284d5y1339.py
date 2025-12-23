import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    
    if 'Dataset' in df.columns:
        df['Dataset'] = df['Dataset'].replace({1: 0, 2: 1})
    
    scaler = StandardScaler()
    fitur = df.drop('Dataset', axis=1)
    target = df['Dataset']
    
    scaled_data = pd.DataFrame(scaler.fit_transform(fitur), columns=fitur.columns)
    scaled_data['Target'] = target.values

    scaled_data.to_csv(output_path, index=False)
    print("Data berhasil dibersihkan")

if __name__ == "__main__":
    clean_data("../dataset_raw/liver_raw.csv", "liver_preprocessed.csv")