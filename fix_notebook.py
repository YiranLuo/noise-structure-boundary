#!/usr/bin/env python3
import json

# Load the notebook
with open('/home/yluo147/projects/LowProFool/Playground.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the get_df function cell and modify it
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def get_df(dataset):' in source and 'df = df.astype(float)' in source:
            # Create new source with the fix
            modified_source = []
            added_fix = False
            
            for line in cell['source']:
                modified_source.append(line)
                # Insert our fix right after selecting the columns
                if '    df = df[feature_names + [target]]' in line and not added_fix:
                    modified_source.append('\n')
                    modified_source.append('    # Convert categorical features to numeric values before casting to float\n')
                    modified_source.append('    categorical_cols = [\'checking_status\', \'savings_status\', \'employment\', \'own_telephone\', \'foreign_worker\']\n')
                    modified_source.append('    for col in categorical_cols:\n')
                    modified_source.append('        df[col] = pd.Categorical(df[col]).codes\n')
                    added_fix = True
            
            # Update the cell source
            cell['source'] = modified_source
            print("Fixed get_df function")
            break

# Save the modified notebook
with open('/home/yluo147/projects/LowProFool/Playground.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Successfully updated the notebook!") 