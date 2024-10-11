import os
import time
import math
import glob
import json

from log import logger

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import optim

from accelerate import Accelerator
import evaluate
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    default_data_collator,
    get_scheduler,
)
import numpy
from sklearn.metrics import classification_report, roc_auc_score

from utils.data_utils import HateDataset, hate_def
from utils.utils import parse_args
from models.model import HateModel
from models.models import HateMLPResModel, HateMLPSimpleModel, HateSwitchMLPModel

def predict_file(file, args):
    
    args.output = "greek-cross-hate-speech/output"
    args.model_name = "greekbert"
    args.model_type = "base"
    args.run_name = "greekbert_switchmlp_ogtd" #model name 
    args.lm = "nlpaueb/bert-base-greek-uncased-v1"
    args.max_length = 250
    args.test = file

    logger.info(f'Starting experiment: {args.run_name} - {file}')
    out_dir_path = args.output + '/' + args.run_name
    if not os.path.exists(out_dir_path):
        os.mkdir(out_dir_path)

    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=out_dir_path) if args.with_tracking else Accelerator()
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.lm, use_fast=True,
                            cache_dir='greek-cross-hate-speech/hfcache')
    
    test_data = HateDataset(path=args.test,  max_instances=None, tokenizer=tokenizer, max_length=args.max_length, padding=args.padding, labels=hate_def)

    config = AutoConfig.from_pretrained(args.lm, num_labels=2, cache_dir='greek-cross-hate-speech/hfcache')

    model = None
    if args.model_type == 'base':
        model = HateModel(lm=args.lm, config=config)
    if args.model_type == 'mlpres':
        model = HateMLPResModel(lm=args.lm, config=config)
    if args.model_type == 'mlp':
        model = HateMLPSimpleModel(lm=args.lm, config=config)
    if args.model_type == "switchmlp":
        model = HateSwitchMLPModel(lm=args.lm, config=config, experts=args.experts)
    assert model!=None, "Cannot proceed without proper model"   
    
    # dataloader - collate fn
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=default_data_collator, shuffle=True)
    
    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    accelerator.load_state(out_dir_path)
    # Prepare everything with our `accelerator`.
    
    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # metric = evaluate.load("accuracy")
    metric = evaluate.load("f1")

    logger.info(f'Running evaluation on {args.test}')
    model.eval()
    all_predictions = []
    samples_seen = 0
    for step, batch in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            # outputs = model(**batch)
            loss, logits = model(**batch)
        predictions = logits.argmax(dim=-1) #if not is_regression else outputs.logits.squeeze()
        predictions, references = accelerator.gather((predictions, batch["input_ids"]))
        # If we are in a multiprocess environment, the last batch has duplicates
        for reference, prediction in zip(references,predictions):
            text = tokenizer.decode(reference, skip_special_tokens =True)
            label = prediction.detach().cpu().numpy().item()
            all_predictions.append({"text":text, "label": label})
                                         
    output_dir_path = 'greek-cross-hate-speech/data/twitch/per_user/'
    output_dir_path = file.split('.json')[0]+"_predictions.json"
    with open(os.path.join(out_dir_path, 'output_dir_path'), 'w') as fout_preds: 
        json.dump(all_predictions, fout_preds)

    hate, non_hate, counter = 0, 0, 0 
    for prediction in all_predictions:
        curr_label = prediction['label'] 
        if curr_label == 0 or curr_label=='0':
            non_hate+=1
        if curr_label == 1 or curr_label=='1':
            hate+=1
        counter+=1

    return {"total": counter, "hate": hate, "non-hate": non_hate}

def main():
    argument_parser = parse_args()
    timestamp = time.time()
    args = parse_args()

    root_path = 'greek-cross-hate-speech/data/twitch/per_user'
    users = glob.glob(root_path+'/*.json')
    results = {}
    for user in tqdm(users): 
        scores = predict_file(user, args)
        results[user] = scores
    
    all_results = {"results": results}
    with open('greek-cross-hate-speech/data/twitch/user_hate_stas.json', 'w') as fout_user_stats:
        json.dump(all_results, fout_user_stats)
    with open('greek-cross-hate-speech/data/twitch/user_hate_stas.csv', 'w') as fout_user_stats:
        for user, scores in results.items():
            fout_user_stats.write(f'{user},{scores["hate"]},{scores["non-hate"]}\n')


if __name__ == '__main__':
    main()