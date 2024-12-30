import json
import pickle
from collections import Counter

import numpy as np

# Reading Results from a JSON File
with open('gpt4.0-base-out.json', 'r') as file:
    big_list = json.load(file)

# Reading Preprocessed Text Data from a JSON File
with open('text.pkl', 'rb') as f:
    text_data = pickle.load(f)

partial_answers = [value[-1] for value in big_list.values()]
answers = [value[-2] for value in big_list.values()]
preds = [value[0] for value in big_list.values()]
golds = [value[1] for value in big_list.values()]

print("Accuracy:", "%.2f" % (np.mean(answers) * 100))
print("MRR:", "%.2f" % (np.mean(partial_answers) * 100))



# Analyzing Performance on Polysemous Words
index = 0
p_sense_nums_w = []
p_sense_nums_r = []
for t, p, g in zip(text_data, preds, golds):
    if p != g:
        #print(t, text_data[t]['context'], p, g, len(text_data[t]['sense_definitions']))
        p_sense_nums_w.append(len(text_data[t]['wordnet_definitions']))
    else:
        p_sense_nums_r.append(len(text_data[t]['wordnet_definitions']))
    index+=1

print(sorted(Counter(p_sense_nums_w).items()))
print(sorted(Counter(p_sense_nums_r).items()))

right_when_zero = sorted(Counter(p_sense_nums_r).items())[0][1]
wrong_when_zero = sorted(Counter(p_sense_nums_w).items())[0][1]

right_when_one = sorted(Counter(p_sense_nums_r).items())[1][1]
wrong_when_one = sorted(Counter(p_sense_nums_w).items())[1][1]

right_when_over_one = 0
wrong_when_over_one = 0

for s, c in sorted(Counter(p_sense_nums_w).items()):
    if s > 1: wrong_when_over_one += c
for s, c in sorted(Counter(p_sense_nums_r).items()):
    if s > 1: right_when_over_one += c

print('Hits@1 |D^t|==0: %.2f'%(right_when_zero/(right_when_zero + wrong_when_zero)*100))
print('Hits@1 |D^t|==1: %.2f'%(right_when_one/(right_when_one + wrong_when_one)*100))
print('Hits@1 |D^t|>1: %.2f'%(right_when_over_one/(right_when_over_one + wrong_when_over_one)*100))