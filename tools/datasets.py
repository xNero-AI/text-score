import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from sklearn.model_selection import train_test_split
import arkad
import os
import urllib.request
from pathlib import Path

from .tokenizer import MyTokenizer

class TextDataset(Dataset):
    def __init__(self, texts, scores, tokenizer):
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        scores = self.scores[idx]
        tokens = self.tokenizer(text)
        x = torch.tensor(tokens)
        y = scores.clone().detach().reshape(1)
        return x, y
    
def get_dataset(local_file_path : str='./data/redacoes.xlsx',
                sentence_length : int = 128):
    
    df = pd.read_excel(local_file_path) #.sample(sample_size)
    tema_e_redacao = df['tema'] + "\n\n" + df['redacoes']
    nota_final = df[['nota1', 'nota2', 'nota3', 'nota4', 'nota5']].sum(axis=1)
    # print(tema_e_redacao.shape, nota_final.shape, df.shape)
    X_train, X_test, y_train, y_test = train_test_split(tema_e_redacao, 
                                                        nota_final,
                                                        test_size=0.5, 
                                                        random_state=42)
    
    tokenizer = MyTokenizer(sentence_length=sentence_length)
    tokenizer.fit(X_train)

    dataset_train = TextDataset(list(X_train), torch.tensor(list(y_train)), tokenizer)
    dataset_test = TextDataset(list(X_test), torch.tensor(list(y_test)), tokenizer)
    # print(list(X_test), len(list(X_test)))
    # print(list(y_test), len(list(y_test)))

    return dataset_train, dataset_test, tokenizer

def get_dataset_fake(local_file_path : str='./data/redacoes_to_train.csv',
                     sentence_length : int = 128):
    
    df = pd.read_csv(local_file_path) #.sample(sample_size)
    tema_e_redacao = df['tema_e_redacao'] 
    nota_final = df['nota']
    X_train, X_test, y_train, y_test = train_test_split(tema_e_redacao, 
                                                        nota_final,
                                                        test_size=0.2, 
                                                        random_state=42)
    
    tokenizer = MyTokenizer(sentence_length=sentence_length)
    tokenizer.fit(X_train)

    dataset_train = TextDataset(list(X_train), torch.tensor(list(y_train)), tokenizer)
    dataset_test = TextDataset(list(X_test), torch.tensor(list(y_test)), tokenizer)

    return dataset_train, dataset_test, tokenizer