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
from models.bibmodels import OGTDmodel, KusailCNNBert, DACHSModel
from models.models import HateMLPResModel, HateMLPSimpleModel, HateSwitchMLPModel

def main():
    argument_parser = parse_args()
    timestamp = time.time()
    args = parse_args()
    
    logger.info(f'Starting experiment: {args.run_name}')
    out_dir_path = os.path.join(args.output,args.run_name)
    if not os.path.exists(out_dir_path):
        os.mkdir(out_dir_path)

    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=out_dir_path) if args.with_tracking else Accelerator()
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.lm, use_fast=True,
                            cache_dir='greek-cross-hate-speech/hfcache')
    
    train_data = HateDataset(path=args.train, max_instances=args.max_instances, tokenizer=tokenizer, max_length=args.max_length, padding=args.padding, labels=hate_def)
    test_data = HateDataset(path=args.test,  max_instances=args.max_instances, tokenizer=tokenizer, max_length=args.max_length, padding=args.padding, labels=hate_def)
    if args.dev:
        dev_data = HateDataset(path=args.dev, max_instances=args.max_instances, tokenizer=tokenizer, max_length=args.max_length, padding=args.padding, labels=hate_def)

    config = AutoConfig.from_pretrained(args.lm, num_labels=train_data.num_labels, cache_dir='greek-cross-hate-speech/hfcache')
    config.max_len = args.max_length

    model = None
    if args.model_type == 'base':
        model = HateModel(lm=args.lm, config=config)
    if args.model_type == 'mlpres':
        model = HateMLPResModel(lm=args.lm, config=config)
    if args.model_type == 'mlp':
        model = HateMLPSimpleModel(lm=args.lm, config=config)
    if args.model_type == "switchmlp":
        model = HateSwitchMLPModel(lm=args.lm, config=config, experts=args.experts)
    if args.model_type == "ogtd": 
        model = OGTDmodel(lm=args.lm, config=config)
    if args.model_type == "kusail":
        model = KusailCNNBert(768, "nlpaueb/bert-base-greek-uncased-v1")
    if args.model_type == "dachs":
        model = DACHSModel(lm=args.lm, config=config)
    assert model!=None, "Cannot proceed without proper model"    
    
    # dataloader - collate fn
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=default_data_collator, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=default_data_collator, shuffle=True)
    if args.dev: 
        dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size, collate_fn=default_data_collator, shuffle=True)

    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.grad_accum_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    if args.dev:
        model, optimizer, train_dataloader, test_dataloader, dev_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader,  test_dataloader, dev_dataloader, lr_scheduler
        )
    else: 
            model, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader,  test_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.grad_accum_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("glue_no_trainer", experiment_config)

    # # Get the metric function
    metric = evaluate.load("f1")


    # Train!
    total_batch_size = args.batch_size * accelerator.num_processes * args.grad_accum_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {train_data.length}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.grad_accum_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    
    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            # outputs = model(**batch)
            # loss = outputs.loss
            loss, logits = model(**batch)
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.grad_accum_steps
            accelerator.backward(loss)
            # accelerator.backward(-loss.item())
            if step % args.grad_accum_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if out_dir_path is not None:
                        output_dir = os.path.join(out_dir_path, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        if args.dev:
            model.eval()
            preds, refs = [], [] 
            samples_seen = 0
            for step, batch in enumerate(dev_dataloader):
                with torch.no_grad():
                    # outputs = model(**batch)
                    loss, logits = model(**batch)
                predictions = logits.argmax(dim=-1) #if not is_regression else outputs.logits.squeeze()
                predictions, references = accelerator.gather((predictions, batch["labels"]))
                # If we are in a multiprocess environment, the last batch has duplicates
                if accelerator.num_processes > 1:
                    if step == len(dev_dataloader) - 1:
                        predictions = predictions[: len(dev_dataloader.dataset) - samples_seen]
                        references = references[: len(dev_dataloader.dataset) - samples_seen]
                    else:
                        samples_seen += references.shape[0]
                metric.add_batch(
                    predictions=predictions,
                    references=references,
                )
                preds.append(predictions.cpu().numpy())
                refs.append(references.cpu().numpy())

            eval_metric = metric.compute()
            logger.info(f"epoch {epoch}: {eval_metric}")
            refs = numpy.concatenate(refs)
            preds = numpy.concatenate(preds)
            report = classification_report(refs, preds)
            print(report)
            with open(os.path.join(out_dir_path,'dev_reports.txt'), 'a') as fout_scores: 
                fout_scores.write(report)
    
    print("---Test set---")
    model.eval()
    preds, refs = [], [] 
    samples_seen = 0
    for step, batch in enumerate(test_dataloader):
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
    logger.info(f"epoch {epoch}: {eval_metric}")
    refs = numpy.concatenate(refs)
    preds = numpy.concatenate(preds)
    report = classification_report(refs, preds, digits=4)
    roc_auc = roc_auc_score(refs, preds)
    print(report)
    print(f"AUC: {roc_auc}")
    with open(os.path.join(out_dir_path,'reports.txt'), 'w') as fout_scores: 
            fout_scores.write(report)
            fout_scores.write(f"AUC: {roc_auc}")

    if out_dir_path: 
        accelerator.save_state(output_dir=out_dir_path)
    return 

if __name__ == '__main__':
    main()