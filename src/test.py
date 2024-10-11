import os
import time
import math

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

def main():
    argument_parser = parse_args()
    timestamp = time.time()
    args = parse_args()
    
    logger.info(f'Starting experiment: {args.run_name}')
    out_dir_path = args.output + '/' + args.run_name
    if not os.path.exists(out_dir_path):
        os.mkdir(out_dir_path)

    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=out_dir_path) if args.with_tracking else Accelerator()
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.lm, use_fast=True,
                            cache_dir='greek-cross-hate-speech/hfcache')
    
    test_data = HateDataset(path=args.test,  max_instances=args.max_instances, tokenizer=tokenizer, max_length=args.max_length, padding=args.padding, labels=hate_def)

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

    metric = evaluate.load("f1")

    logger.info(f'Running evaluation on {args.test}')
    model.eval()
    preds, refs = [], [] 
    samples_seen = 0
    for step, batch in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            # outputs = model(**batch)
            loss, logits = model(**batch)
        predictions = logits.argmax(dim=-1) #if not is_regression else outputs.logits.squeeze()
        predictions, references = accelerator.gather((predictions, batch["labels"]))
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(test_dataloader) - 1:
                predictions = predictions[: len(test_dataloader.dataset) - samples_seen]
                references = references[: len(test_dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
        preds.append(predictions.cpu().numpy())
        refs.append(references.cpu().numpy())

    eval_metric = metric.compute()
    logger.info(f"Eval Metric: {eval_metric}")
    refs = numpy.concatenate(refs)
    preds = numpy.concatenate(preds)
    report = classification_report(refs, preds, digits=4)
    roc_auc = roc_auc_score(refs, preds)
    print(report)
    print(f"AUC: {roc_auc}")
    with open(os.path.join(out_dir_path,'eval_report.txt'), 'w') as fout_scores: 
            fout_scores.write(report)
            fout_scores.write(f"AUC: {roc_auc}")
    
    return 

if __name__ == '__main__':
    main()