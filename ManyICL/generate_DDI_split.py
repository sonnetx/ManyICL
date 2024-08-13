import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def generate_ddi_splits(data_path, test_size=0.5, random_state=42):
    # Load the dataset
    df = pd.read_csv(data_path)
    
    # Create slices for FST I/II and FST V/VI
    fst_12 = df[df['skin_tone'] == '12']
    fst_56 = df[df['skin_tone'] == '56']
    
    def split_by_patient(df_slice, test_size):
        unique_patients = df_slice['DDI_file'].unique()
        train_patients, test_patients = train_test_split(unique_patients, test_size=test_size, random_state=random_state)
        
        demo_df = df_slice[df_slice['DDI_file'].isin(train_patients)]
        test_df = df_slice[df_slice['DDI_file'].isin(test_patients)]
        
        return demo_df, test_df
    
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
        output_df = pd.DataFrame(index=df['file_name'])
        output_df['benign'] = (df['malignant'] == 0).astype(int)
        output_df['malignant'] = df['malignant']
        return output_df
    
    demo_output = create_output_df(demo_df)
    test_output = create_output_df(test_df)
    
    return demo_output, test_output

# Usage
data_path = '/home/groups/roxanad/ddi/ddi_metadata.csv'
demo_df, test_df = generate_ddi_splits(data_path)

# Save the splits
demo_df.to_csv('ddi_train.csv', index=False)
test_df.to_csv('ddi_test.csv', index=False)