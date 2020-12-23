import json
import pandas as pd
import requests
import torch

with open('/Users/animitkulkarni/Python/InstagramCaptioner/data_management/annotations/captions_train2014.json') as f:
    data = json.load(f)

annotations = data['annotations'][0:30000]

class MSCOCODataset(torch.data.Dataset):

    def __init__(self, images, annotations):
        pass

    def __getitem__(self, idx):
        return

    def __len__(self):
        return 

    def collate_fn(self):
        return
