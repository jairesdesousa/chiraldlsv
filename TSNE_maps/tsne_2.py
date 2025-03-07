import os
import plotly.express as px
import pandas as pd

# List of directories containing the _transf.csv files
directories = ["transf", "cddd", "fingpr"]

# List of files to process in each directory
file_names = ["DLSV_opposite_coordi.csv", "ADH2_can_coordi.csv", "DLSV_nonstereo_coordi.csv"]

# Load the class file that contains the descriptor classes
class_file = 'class_all.csv'
df_class = pd.read_csv(class_file)

# Descriptor classes to visualize
descriptor_classes = {
    'TR/TE': df_class['TR/TE'],
    'F/L_class': df_class['F/L_class'],
    '@/@@_class': df_class['@/@@_class'],
    'R/S_class': df_class['R/S_class']
}

# Loop through each directory and file
for directory in directories:
    for file_name in file_names:
        # Construct the full path to the transformed CSV file
        file_path = os.path.join(directory, file_name)
        
        # Load the transformed t-SNE coordinates
        df_transf = pd.read_csv(file_path)
        Xfig = df_transf.to_numpy()

        # Loop through each descriptor class and generate the corresponding t-SNE plot
        for descriptor, y in descriptor_classes.items():
            fig = px.scatter(
                x=Xfig[:, 0], y=Xfig[:, 1], color=y,
                hover_name=df_class['SMILES'],
                hover_data={
                    'TR/TE': df_class['TR/TE'],
                    'F/L_class': df_class['F/L_class'],
                    '@/@@_class': df_class['@/@@_class'],
                    'R/S_class': df_class['R/S_class']
                }
            )
            
            # Update layout of the figure
            fig.layout.update(
                title=f"t-SNE Visualization for {file_name} by {descriptor}",
                xaxis_title="First t-SNE Component",
                yaxis_title="Second t-SNE Component"
            )
            
            # Replace slashes in the descriptor with underscores for valid file names
            safe_descriptor = descriptor.replace('/', '_')
            
            # Create output file path with .html suffix
            output_html_name = f"{file_name.replace('_coordi.csv', '')}_{safe_descriptor}.html"
            output_html_path = os.path.join(directory, output_html_name)
            
            # Save the HTML file
            fig.write_html(output_html_path)
            print(f"t-SNE visualization for {file_name} and {descriptor} saved as {output_html_path}")
