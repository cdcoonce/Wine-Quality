    # Imports
from ucimlrepo import fetch_ucirepo 
import pandas as pd

## Dataset import

    # fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
    
    # data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
    
    # metadata 
print(wine_quality.metadata) 
    
    # variable information 
print(wine_quality.variables) 


## Save Locally

    # Combine features and target
wine_quality_df = pd.concat([X, y], axis=1)

    # Save to CSV
file_path = '/Users/cdcoonce/Documents/GitHub/Practice_Data_Sets/Wine Quality/Dataset/combined.csv'
wine_quality_df.to_csv(file_path, index=False)
print(f"Dataset saved to {file_path}")

