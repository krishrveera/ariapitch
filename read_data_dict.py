import json, pandas as pd

with open("features/static_features.json") as f:
    feature_dict = json.load(f)
print(f"Number of features described: {len(feature_dict)}")

with open("phenotype/diagnosis/benign_lesions.json") as f:
    bl_dict = json.load(f)
# Inspect field names, descriptions, data types
for k, v in bl_dict.items():
    print(f"{k}: {v.get('description', '')}")