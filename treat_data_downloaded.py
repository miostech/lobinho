import os
import json


def treat_data_downloaded(symbol: str):
    # read all files contains the symbol in the name and the extension is json
    data = []
    for file in os.listdir():
        if symbol in file and file.endswith('.json'):
            with open(file) as f:
                data.append(json.load(f))
    return data
