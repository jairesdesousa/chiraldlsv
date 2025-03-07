import os
import pandas as pd
from sklearn.manifold import TSNE

# List of directories containing the CSV files
directories = ["transf", "cddd", "fingpr"]

# List of files to process in each directory
file_names = ["DLSV_opposite.csv", "ADH2_can.csv", "DLSV_nonstereo.csv"]

# Loop through each directory and file
for directory in directories:
    for file_name in file_names:
        # Construct the full path to the CSV file
        file_path = os.path.join(directory, file_name)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
              
        # Apply t-SNE transformation
        tsne = TSNE(n_components=2, learning_rate=200, init='pca', perplexity=30)
        X_train_tsne = tsne.fit_transform(df)
        
        # Create a DataFrame with the transformed data
        dftransf = pd.DataFrame(X_train_tsne, columns=['x', 'y'])
        
        # Create output file path with _transf.csv suffix
        output_file_name = file_name.replace(".csv", "_coordi.csv")
        output_file_path = os.path.join(directory, output_file_name)
        
        # Save the transformed data to a new CSV file
        dftransf.to_csv(output_file_path, index=False, header=True)
        
        # Print confirmation message
        print(f"Processed {file_path} and saved as {output_file_path}")
