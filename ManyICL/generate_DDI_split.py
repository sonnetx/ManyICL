import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def generate_ddi_splits(data_path, test_size=0.5, random_state=42):
    # Load the dataset
    df = pd.read_csv(data_path)
    
    # Create slices for FST I/II and FST V/VI
    fst_12 = df[df['skin_tone'] == 12]
    fst_56 = df[df['skin_tone'] == 56]
    
    def split_by_patient(df_slice, test_size):
        train_patients, test_patients = train_test_split(df_slice, test_size=test_size, random_state=random_state)

        return train_patients, test_patients
    
    # Split each slice
    train_12, test_12 = split_by_patient(fst_12, test_size)
    train_56, test_56 = split_by_patient(fst_56, test_size)
    
    # Combine the splits
    demo_df = pd.concat([train_12, train_56])
    test_df = pd.concat([test_12, test_56])
    
    # Check balance
    def check_balance(df):
        fst_balance = df['skin_tone'].value_counts(normalize=True)
        malignant_balance = df['malignant'].value_counts(normalize=True)
        return fst_balance, malignant_balance
    
    train_fst_balance, train_malignant_balance = check_balance(demo_df)
    test_fst_balance, test_malignant_balance = check_balance(test_df)
    
    print("Train set balance:")
    print("FST balance:", train_fst_balance)
    print("Malignant balance:", train_malignant_balance)
    print("\nTest set balance:")
    print("FST balance:", test_fst_balance)
    print("Malignant balance:", test_malignant_balance)
    
    # Create new DataFrames with desired structure
    def create_output_df(df):
        output_df = df[['DDI_file', 'malignant']]
        output_df['benign'] = (~output_df['malignant']).astype(int)
        output_df['malignant'] = output_df['malignant'].astype(int)
        return output_df
    
    demo_output = create_output_df(demo_df)
    test_output = create_output_df(test_df)
    
    return demo_output, test_output

# Usage
data_path = '/home/groups/roxanad/ddi/ddi_metadata.csv'
demo_df, test_df = generate_ddi_splits(data_path)

# Save the splits
demo_df.to_csv('/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/DDI/ddi_demo.csv')
test_df.to_csv('/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/DDI/ddi_test.csv')
