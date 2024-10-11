import os 
import json
import re
import pandas as pd 
from torch.utils.data import Dataset


hate_def = {'Not-Hatespeech':0,'Hatespeech':1}

def get_tagset(classification_set):
    if os.path.isfile(classification_set):
        # read the tagging scheme from a file
        sep = '\t' if classification_set.endswith('.tsv') else ','
        df = pd.read_csv(classification_set, sep=sep)
        tags = {tag: idx for idx, tag in enumerate(df.columns)}
        # tags = {row['tag']: row['idx'] for idx, row in df.iterrows()}
        return tags

    if 'hate_def' in classification_set:
        return hate_def
    
def clean_text(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    url_replacement = ''
    text_without_urls = url_pattern.sub(url_replacement, text)
    return text_without_urls
    
class HateDataset(Dataset):
    def __init__(self, path, max_instances=None,
                tokenizer=None, max_length=0, 
                padding='max_length', truncation=True,
                labels = None) -> None:
        self.path = path 
        self.max_instances = max_instances
        self.examples = None
        self.tokenizer=tokenizer
        self.max_length=max_length
        self.padding = padding
        self.truncation = truncation
        self.length = 0
        self.labels = labels
        self.num_labels = len(labels) if isinstance(self.labels, dict) else self.labels
        self.__post_init__()
    
    def __post_init__(self):
        with open(self.path, 'r') as fin:
            dataset = json.load(fin)
            data = dataset['data']
            del dataset

        
        data = data[:self.max_instances] if self.max_instances else data
        print(f'Finished reading {len(data)} - limit {self.max_instances}')
        text, labels = [], []
        for instance in data: 
             curr_text = clean_text(instance['text'])
             text.append(curr_text)
             try:
                labels.append(instance['label'])
             except:
                 continue

        result = self.tokenizer(
            text, 
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation
            )
        result['labels'] = [int(label) for label in labels]
        self.examples = result
        del result

    def __len__(self):
        return len(self.examples.input_ids)

    def __getitem__(self, idx):
        input_ids = self.examples.input_ids[idx]
        attention_mask = self.examples.attention_mask[idx]
        try:
            labels = self.examples.labels[idx]
        except:
            labels=None
        return {
            'input_ids':input_ids,
            'attention_mask':attention_mask,
            'label':labels,
            }