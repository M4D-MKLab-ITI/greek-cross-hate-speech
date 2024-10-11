import os
import argparse

# import pandas as pd 

def parse_args():
    p = argparse.ArgumentParser(description='Model configuration.', add_help=False)
    p.add_argument('--train', type=str, help='Path to the train data.', default=None)
    p.add_argument('--test', type=str, help='Path to the test data.', default=None)
    p.add_argument('--dev', type=str, help='Path to the dev data.', default=None)

    p.add_argument('--output', type=str, help='Output directory.', default='.')
    
    p.add_argument('--label_set', type=str, help='classification tagging set', default='hate_def')
    
    p.add_argument('--max_instances', type=int, help='Maximum number of instances', default=-1)
    p.add_argument('--max_length', type=int, help='Maximum number of tokens per instance.', default=250)
    p.add_argument('--padding', type=str, help='Padding scheme for tokenizer. Options: [True or longest, \'max_length\', False]', default='max_length')

    p.add_argument('--lm', type=str, help='Pretrained encoder model to use', default='xlm-roberta-large')
    p.add_argument('--model_path', type=str, help='Model path.', default=None)
    p.add_argument('--model_type', type=str, help='Model type, controls model architecture', default='base')
    p.add_argument('--model_name', type=str, help='Model name.', default=None)
    p.add_argument('--stage', type=str, help='Training stage', default='fit')
    p.add_argument('--prefix', type=str, help='Prefix for storing evaluation files.', default='test')

    p.add_argument('--batch_size', type=int, help='Batch size.', default=128)
    p.add_argument('--grad_accum_steps', type=int, help='Number of Gradient Accumulation steps', default=1)
    p.add_argument('--gpus', type=int, help='Number of GPUs.', default=1)
    p.add_argument('--cuda', type=str, help='Cuda Device', default='cuda:0')
    p.add_argument('--epochs', type=int, help='Number of epochs for training.', default=5)
    p.add_argument('--max_train_steps', type=int, help='Number of training steps, overwrites epochs', default=None)
    p.add_argument('--lr', type=float, help='Learning rate', default=1e-5)
    p.add_argument('--lr_scheduler', type=str, help='Learning rate scheduler used by optimizer', default='linear', choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    p.add_argument('--lr_warmup', type=int, help='Number of warmup LR warmup steps', default=0)
    p.add_argument('--weight_decay', type=float, help="Weight decay to use.", default=0.0)
    p.add_argument('--dropout', type=float, help='Dropout rate', default=0.1)
    p.add_argument('--es_patience', type=int, help='Early stopping patience in number of epochs', default=3)
    p.add_argument('--checkpointing_steps', type=int, help='Save model every defined number of steps', default=None)
    
    p.add_argument('--output_path', type=str, help="Path to save the file", default=None)
    p.add_argument('--with_tracking', action="store_true", help='Use tracking of HF Accelerator')
    p.add_argument('--run_name', type=str, help="Experiment name for logging purposes.", default=None)

    p.add_argument('--experts', type=int, help="Number of experts in sparse architectures (e.g., SwitchMLP)", default=2)
    return p.parse_args()

