import os
import pandas as pd

# List of directories containing the CSV files
directories = ["transf", "cddd", "fingpr"]

# File names for can, opposite, and nonstereo
can_file = "ADH2_can.csv"
opposite_file = "ADH2_opposite.csv"
nonstereo_file = "ADH2_nostereo.csv"

# Loop through each directory
for directory in directories:
    # Load the can, opposite, and nonstereo files
    can_path = os.path.join(directory, can_file)
    opposite_path = os.path.join(directory, opposite_file)
    nonstereo_path = os.path.join(directory, nonstereo_file)
    
    df_can = pd.read_csv(can_path)
    df_opposite = pd.read_csv(opposite_path)
    df_nonstereo = pd.read_csv(nonstereo_path)

    # Ensure the dataframes are aligned by dropping the non-numeric columns if needed
    X_can = df_can#.drop(labels=['SMILES', 'TR/TE', 'F/L_class', '@/@@_class', 'R/S_class'], axis=1)
    X_opposite = df_opposite#.drop(labels=['SMILES', 'TR/TE', 'F/L_class', '@/@@_class', 'R/S_class'], axis=1)
    X_nonstereo = df_nonstereo#.drop(labels=['SMILES', 'TR/TE', 'F/L_class', '@/@@_class', 'R/S_class'], axis=1)

    # Calculate differences: can - opposite and can - nonstereo
    DLSV_opposite = X_can - X_opposite
    DLSV_nonstereo = X_can - X_nonstereo

    # Save the difference as new CSV files
    DLSV_opposite_path = os.path.join(directory, "DLSV_opposite.csv")
    DLSV_nonstereo_path = os.path.join(directory, "DLSV_nonstereo.csv")

    DLSV_opposite.to_csv(DLSV_opposite_path, index=False, header=True)
    DLSV_nonstereo.to_csv(DLSV_nonstereo_path, index=False, header=True)

    print(f"Processed differences for {directory}: DLSV_opposite.csv and DLSV_nonstereo.csv saved.")
