import json
import pandas as pd
import tools

# Load the new log file
file_path_new = 'log_zenvia_rodrigo.json'
with open(file_path_new, 'r') as file_new:
    data_new = json.load(file_new)

# Convert the new data to a pandas DataFrame for analysis
df_new = pd.json_normalize(data_new)

# Calculate the status percentages
status_counts_new = df_new['status'].value_counts(normalize=True) * 100
status_counts_new = status_counts_new.reset_index()
status_counts_new.columns = ['status', 'percentage']

# Aggregate useful information
summary_new = df_new.groupby('status').agg(
    total_messages=('id', 'count'),
    reason_descriptions=('reason', lambda x: ', '.join(pd.Series(x).drop_duplicates()))
).reset_index()

# Merge status percentages with the summary
summary_new = pd.merge(summary_new, status_counts_new, on='status')

print(summary_new)

# Display the summary to the user
#tools.display_dataframe_to_user(name="New Log Analysis Summary", dataframe=summary_new)

# Save the new summary as PDF
summary_new_path = 'log_analysis_summary_new.pdf'
summary_new.to_csv(summary_new_path, index=False)

summary_new_path
