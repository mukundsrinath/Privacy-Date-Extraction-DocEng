'''

This file takes in a data path and an output path and extracts date instances in an HTML file using regex. 
Each date instance contains a date format (ex: dd/mm/yyyy) preceded by and followed by either 250 characters or line break or a sentence break whichever comes first. 

'''

import os
import re
import json
from tqdm import tqdm

data_path = ''
output_path = ''

files = os.listdir(data_path)

p = re.compile('20[0-2][0-9]|19\d{2}|\'\d{2}|\d{1,2}\/\d{1,2}\/\d{2,4}|\d{1,2}\.\d{1,2}\.\d{2,4}|\d{1,2}\-\d{1,2}\-\d{2,4}')

for _file in tqdm(files):
    matches = []
    with open(data_path+_file) as f:
        data = f.readlines()
    for line in data:
        line = json.loads(line)
        _hash = line['hash']
        text = line['text']
        for m in p.finditer(text):
            start_pos = m.start()
            end_pos = m.end()
            if start_pos - 250 < 0:
                real_start_pos = 0
            else:
                real_start_pos = start_pos - 250
            if end_pos + 250 >= len(text):
                real_end_pos = len(text) - 1
            else:
                real_end_pos = end_pos + 250
            extract_left = text[real_start_pos:start_pos]
            extract_left = extract_left.split('\n')[-1]
            extract_left = extract_left.split('. ')[-1]
            extract_right = text[end_pos:real_end_pos]
            extract_right = extract_right.split('\n')[0]
            extract_right = extract_right.split('. ')[0]
            matches.append({'hash':_hash, 'instance':extract_left+text[start_pos:end_pos]+extract_right})
    with open(output_path, 'a+') as f:
        for row in matches:
            f.write(json.dumps(row)+'\n')
