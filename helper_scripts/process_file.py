import pandas as pd
import numpy as np

# Assuming df is your DataFrame
dfd = pd.read_csv("G:\University\Diplom\Program\SynthSupportAI\Helper_scripts\wiki_movie_plots_deduped.csv")
dfd = dfd.replace(["unknown", "Unknown", "nan"], np.nan) 
dfd=dfd.drop(columns=['Wiki Page'])
dfd = dfd.dropna()
dfd.head()

# Create an empty list to store the formatted text for each row
formatted_text = []

# Iterate through the rows of the DataFrame
for index, row in dfd.iterrows():
    # Format the text for each row
    row_text = f"Release year is {row['Release Year']}, title is {row['Title']}, ethnicity is {row['Origin/Ethnicity']}, director is {row['Director']}, cast is {row['Cast']}, genre is {row['Genre']}, plot is {row['Plot']}."
    
    # Append the formatted text to the list
    formatted_text.append(row_text)

# Join the formatted text into a single string
output_text = ' '.join(formatted_text)


# Write the result to a file
with open('output.txt', 'w') as file:
    file.write(output_text)