import pandas as pd
import json
import os

list_of_boxes = []
destination = './Anchors.json'
source = './data/'

for i, file in enumerate(os.listdir(source)):
   
    # print(os.listdir(source + file))
    # print(list(filter(lambda x: x.endswith('csv'), os.listdir(source + file))))
    for i, dir_list in enumerate(os.listdir(source + file)):
        files = list(filter(lambda  x: x.endswith('csv'), os.listdir(source + file + '/' + dir_list)))
        # obj['clip'] = dir_list
        for df in files:
            if not df.startswith("clip_info"):
                obj = {
                    "dir":file,
                    "players":[],
                    "clip": dir_list,
                    "file":f"{df[:2]}.png"
                }
                d = pd.read_csv(source + file + '/' + dir_list + '/' + df)
                for x, y, w, h in zip(d['x'], d['y'], d['w'], d['h']):
                    obj['players'].append([x, y, w, h])
                list_of_boxes.append(obj)
                # print(f"{df} in {dir_list} finished.")
        # print(f"{dir_list} finished. {i} / {len(dir_list)}")

    print(f"{file} finished. {len(os.listdir(source))}")
with open("data.json", "w+") as f:
    f.write(json.dumps(list_of_boxes))

        
