import json
import pandas as pd

path = "./MLDS_hw2_data/training_label.json"

with open(path) as data_file:    
    y_data = json.load(data_file)

for data in y_data:
    data['caption']=data['caption'][0]

data = {'video_id': [],
        'caption': []} 

for x in y_data:
    data['video_id'].append(x['id'])
    data['caption'].append(x['caption'])

df = pd.DataFrame.from_dict(data)

df.to_csv("train_data.csv", index=False)