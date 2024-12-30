import pickle
import pennylane as qml
from pennylane import numpy as np
import torch
from tqdm import tqdm
import json




#The data preprocessing and encoding sections can be found in 'Q-VWSD.ipynb'



url = '/mnt/'
with open(url+'text.pkl', 'rb') as f:
    text_data = pickle.load(f)

dev = qml.device('default.qubit', wires=9)
def circuit(x,H):
    # Apply Hamiltonian evolution to the initial state, assuming it is used for expectation value measurement.
    qml.AmplitudeEmbedding(features=x, wires=range(9), normalize=True)
    return qml.expval(qml.Hermitian(H, wires=range(9)))

qnode = qml.QNode(circuit, dev, diff_method=None)
def QIC(x,v):
    H = torch.outer(v, v.conj())
    res = qnode(x,H)
    return res



def evaluate_posterior(state,image_features, text_data ,out):

        state =  torch.load(state).cpu()
        image_features =  torch.load(image_features).cpu()

        with open(out, 'r') as file:
            mydict = json.load(file)
        for data_index in tqdm(range(len(text_data.keys()))):
            if str(data_index) in mydict.keys():
                continue
            data = text_data[data_index]
            gold = data['gold']; gold_index = data['candidates'].index(gold)
            qbayesian_probs = []
            for i in image_features[data_index]:
                a = QIM(state[data_index].to(torch.float32), i.to(torch.float32))
                qbayesian_probs.append(a)
            pred = np.argmax(qbayesian_probs)

            merged_array = np.hstack(qbayesian_probs)
            sorted_indexes = reversed(np.argsort(merged_array))
            i = 1
            for index in sorted_indexes:
                if index == gold_index:
                    partial_answers = 1 / i
                    break
                i += 1

            if pred == gold_index:
                answers = 1
            else:
                answers = 0

            result = [data['candidates'][pred], gold, answers, partial_answers]
            mydict[data_index] = result
            with open(out, 'w') as file:
                json.dump(mydict, file)

        print("Accuracy:", "%.2f" % (np.mean(answers) * 100))
        print("MRR:", "%.2f" % (np.mean(partial_answers) * 100))








print('gpt4.0-base--->')
evaluate_posterior(url+'state_gpt4.0_base.pt',url+'project_base.pt',text_data,url+'gpt4.0-base-out.json')


print('wn+cndg-base--->')
evaluate_Qbayesian_posterior(url+'state_wn+cndg_base.pt',url+'project_base.pt',text_data,url+'wn+cndg-base-out.json')

