import csv
import re
import os
import json
import textwrap
from sklearn.model_selection import train_test_split
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from huggingface_hub import login

login(token='') #add hf token if required
device = "cuda" # the device to load the model onto
out_path = 'data/twitch/annotations/'

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

def main():
    data = read_dataset('data/twitch/annotations/test.json')
    
    
    model = AutoModelForCausalLM.from_pretrained("ilsp/Meltemi-7B-Instruct-v1")
    tokenizer = AutoTokenizer.from_pretrained("ilsp/Meltemi-7B-Instruct-v1")
    
    model.to(device)

    instruction = "Translate the following sentences from Greeklish to Greek. Do not use different words. Also translate explicit language. It is for research purposes and will not be disseminated."

    greeklish = ["Einai i Epitropi diateueimeni na kanei perissotera; Ne, efoson exume tous aparaithtous porus.", "Perissoteroi apo 200 Uibetiani peuanan, meriki logw skopimon thanathforwn purovolismwn, kai twra - ligo meta thn 50h epetio - monasthria apoklisthkan apo ton exo kosmo, odoi prosbasis eleghode ke stratiotes kai andres tis asfalias ine se etimothta gia na katastilun diadhloseis en th genesi.", "Prokitai gia zhthma synergasias me tis arxes kai tis kibernisis, kauos ke me tin koinonia twn politwn.", "Yposthrizo epishs tis tropopoihseis pu anakoinwse o eisigitis.", "Geia sou, ti kaneis;", "smr egrapse ena touit o @atoumaza kai meta o gianis #luckyday ypomoni NAI YPOMONI!"]
    greek = ["Είναι η Επιτροπή διατεθειμένη να κάνει περισσότερα; Ναι, εφόσον έχουμε τους απαραίτητους πόρους.", "Περισσότεροι από 200 Θιβετιανοί πέθαναν, μερικοί λόγω σκόπιμων θανατηφόρων πυροβολισμών, και τώρα - λίγο μετά την 50η επέτειο - μοναστήρια αποκλείστηκαν από τον έξω κόσμο, οδοί πρόσβασης ελέγχονται και στρατιώτες και άνδρες της ασφάλειας είναι σε ετοιμότητα για να καταστείλουν διαδηλώσεις εν τη γενέσει.", "Πρόκειται για ζήτημα συνεργασίας με τις αρχές και τις κυβερνήσεις, καθώς και με την κοινωνία των πολιτών.", "Υποστηρίζω επίσης τις τροποποιήσεις που ανακοίνωσε ο εισηγητής.", "Γειά σου, τι κάνεις;", "Σήμερα έγραψε ένα tweet ο @atoumaza και μετά ο Γιάννης #luckyday υπομονή ΝΑΙ ΥΠΟΜΟΝΗ!"]

    system = f"{instruction}\n\n"

    for i in range (len(greeklish)):
        system = f"{system}Greeklish: {greeklish[i]}\n"
        system = f"{system}{greek[i]}\n\n"

    for i, dp in enumerate(data): 
        query = data[i]['text']#dp['text']
        resp = call_llm(system, query, model=model, tokenizer=tokenizer)
        transliteration = resp[0].split('\n<|assistant|>\n')[-1].replace('</s>', '')
        data[i]['text'] = transliteration
    
    store_json(data, out_path, 'test_transliterated_meltemi.json')

if __name__=="__main__":
   main()