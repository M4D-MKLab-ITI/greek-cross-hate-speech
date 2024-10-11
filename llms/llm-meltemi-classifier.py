import csv
import re
import os
import json
import textwrap
from sklearn.model_selection import train_test_split
import numpy as np
import transformers
import evaluate
from sklearn.metrics import classification_report, roc_auc_score
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from huggingface_hub import login
import argparse

login(token='') #add hf token if required
device = "cuda" # the device to load the model onto
model_name = "meltemi"
model_id = "ilsp/Meltemi-7B-Instruct-v1"

setup = 5 #options: 0, 5, 10
dataset = 'ogtd'
sample_dataset = 'ogtd'
out_path = 'llm-classification-outputs/'

def store_json(dataset, path, fn):
    data = {"data":dataset}
    with open(os.path.join(path,fn), 'w') as fout:
        json.dump(data, fout)
    return 

def read_dataset(data_path):
    with open(data_path, 'r') as fin:
        dataset = json.load(fin)
        data = dataset['data']
    return data

def call_llm(prompt, query, model, tokenizer):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query},
        ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_prompt = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(input_prompt['input_ids'], max_new_tokens=256, do_sample=True)#, temperature=0.5, top_k=15, top_p=0.95)
    result = tokenizer.batch_decode(outputs)
    return result

def parse_args():
   p = argparse.ArgumentParser(description='Experiment settings', add_help=False)
   p.add_argument('--dataset', type=str, help='dataset_name', default=None)
   p.add_argument('--setup', type=int, help='setup int', default=None)
   p.add_argument('--sample_dataset', type=str, help='sample_dataset_name', default=None)
   p.add_argument('--out_path', type=str, help="Path to save the logs", default=None)
   return p.parse_args()

def main():
    args = parse_args()
    dataset = args.dataset 
    setup = args.setup 
    sample_dataset = args.sample_dataset
    out_path = args.out_path
    error_counter = 0 
    errors = []

    if dataset == 'ogtd':
      data = read_dataset('data/ogtd/test.json') #ogtd
    if dataset == 'twitch':
      data = read_dataset('data/twitch/annotations/test.json') #twitch
    if dataset == "dachs":
      data = read_dataset('data/dachs/Test_collected.json') #dachs
    
    dachs_samples = [
       "USERNAME USERNAME Σου χω ξαναγράψει, είμαι πασοκος με τα μπούνια αλλά το κάνεις σωστά το έργο. Όσο για τον βαξεβάνη μην κωλοτρώγεσαι συνέχεια όλοι έχουν τους δικούς τους μπλαμπλαδόρους. Και η νδ και το πασοκ κλπ",
       "RT USERNAME: Να εδώ η Παμελα Αντερσον στηρίζει τις βίαιες διαδηλώσεις ενάντια στις ελίτ URL",
       "USERNAME USERNAME Τι λες ρε λουγκρα",
       "USERNAME Σιχαμενε πορδοσαλτε.. τραβα αυτοκτονα μωρη προαγανδα",
       "USERNAME Να σε δω κρεμασμένο ανάποδα και να καίγεσαι και ας πεθάνω",
       "RT USERNAME: «Στη σέντρα» βγάζει ο Μητσοτάκης το δήθεν αριστερό πρόσημο της κυβέρνησης Τσίπρα URL",
       "Σου εύχομαι κάτι κακό",
       "USERNAME USERNAME Κάτι άλλο έχεις να πεις μωρή λουλού που αξίζει να απαντήσω..;; ή σε μπούκωσα και λες ότι να'ναι;;; Όξω μωρή συριζαία πουστάρδα!!!!",
       "Πραγματικά έλεος ρε γαμημενε ψεύτη",
       "RT USERNAME: ΒΙΝΤΕΟ Ξεφτίλες νεοραγιαδες πηγαίνουν στα Γυφτοσκοπια των γυφτομακεδονων και αφήνουν τα ευρώ τους !ΝΤΡΟΠΗ ΡΕ ΝΤΡΟΠΗ",
       ]
    dachs_labels = ["0","0","1","1","1","0","1","1","0","1"]

    ogtd_samples = [
       "@USER Οι μουσουλμάνες που τις βιάζουν έτσι κ αλλιώς, οπότε τις βολεύει να κυνηγούν εμάς κ να αφήσουν αυτές ήσυχες οι άντρες τους.",
       "Η Κάτια προσπαθεί να πείσει οτι δεν είναι ελέφαντας #GynaikaXwrisOnoma",
       "@USER Ισκαρ, φτυς τ αρχιδια σου ρε φασιστοσκυλο...",
       "@USER Φεύγω μετανάστης σε χώρα με βασιλιά,περιμένω να πεθάνετε όλοι από καρκίνο, γυρνάω ψηφίζω Αλεξαρο. Πάλι νίκη της αριστεράς.",
       "Καλά γιατί λες ότι, είσαι νέος αφού γεννήθηκες το 39; Ναι αλλά, στις 29 Φεβρουαρίου κι' έχω γενέθλια κάθε τέσσερα χρόνια οπότε;",
       "Οι γυναίκες χρειάζονται περισσότερο ύπνο γιατί έχουν πιο σύνθετο εγκέφαλο",
       "Ότι η πορνογραφία ανηλίκων είναι... Πλημμελημα (!) με ξεπερνά. Γαμω τη δικαιοσύνη μας!",
       "Πάρτους σπιτάκι σου αφού τόσο τους αγαπάς αλλά όπως είπε ένας αλληλέγγυος σαν εσένα όταν τον ρώτησαν αυτό ''αλληλέγγυος είμαι μαλάκας δεν είμαι ''.Έτσι δεν είναι ε;",
       "@USER Εσύ είσαι η μόνη σωστή εδώ μέσα! Στους άλλους αναφέρομαι που σου κάνουν επίθεση. Στους φασίστες Χρυσαυγιτες",
       "@USER Και αντρες και γυναικες αν θελουν να στη κανουν....."
    ]
    ogtd_labels = ["1","0","1","1","0","0","1","1","0","0"]
    
    # 0-shot just instructions 
    instruction = "Classify the following texts. Respond with 1 if the text contains offensive language or hate speech contents or 0 if not. Do not respond with anything but 1 or 0." 
    if setup==0:
      system = instruction
    if setup==5 or setup==10:
        system = f"{instruction}\n\n"
        if sample_dataset=='dachs':
            active_data = dachs_samples[:setup]
            active_labels = dachs_labels[:setup]
        if sample_dataset=='ogtd':
            active_data = ogtd_samples[:setup]
            active_labels = ogtd_labels[:setup]
        for i in range(len(active_data)):   
            system = f"{system}Text: {active_data[i]}\n"
            system = f"{system}Label: {active_labels[i]}\n\n"

    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.to(device)

    preds, refs = [], []
    for i, dp in enumerate(tqdm(data)): 
        query = data[i]['text']
        label = int(data[i]['label'])
        refs.append(label)

        query = f"Text :{query}\nLabel: "
        resp = call_llm(system, query, model=model, tokenizer=tokenizer)
        res = resp[0].split('\n<|assistant|>\n')[-1].replace('</s>', '')
        
        print(res)
        if res == '1' or res == '0':
            preds.append(int(res))
        else:
            error_counter+=1
            errors.append([label, res])
            if label=='1':
              preds.append(0)
            else:
              preds.append(1)


    report = classification_report(refs, preds, digits=4)
    roc_auc = roc_auc_score(refs, preds)
    print(f"Model: {model_id} \t Setup: {setup}")
    print(report)
    print(f"AUC: {roc_auc}")
    if setup==0:
        out_log = out_path + f"{dataset}_{model_name}_0.txt"
    if setup!=0:
        out_log = out_path + f"{dataset}_{model_name}_{str(setup)}-{sample_dataset}.txt"
    with open(out_log, 'w') as fout: 
       fout.write(f"Model: {model_id} \t Setup: {setup}\n")
       fout.write(report)
       fout.write(f"\nAUC: {roc_auc}")
       fout.write(f"\nErrors:{error_counter}\n")
       for error in errors:
          fout.write(f"{error[0]} {error[1]}\n~\n")

if __name__=="__main__":
   main()