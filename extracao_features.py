# Extrator de features do dataset
# 
# Fernando Silva Grande
# Murilo Luis Calvo Neves
# Vitor Padovani

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from spacytextblob.spacytextblob import SpacyTextBlob
import numpy as np
import spacy
import statistics
import time

pln = spacy.load("en_core_web_sm")
pln.add_pipe('spacytextblob')

arquivo_dataset = "dataset/WELFake_Dataset.csv"

arquivo_saida = "dataset_features.csv"

def carregar_dataset() -> pd.DataFrame:
    try:
        return pd.read_csv(arquivo_dataset)
    except FileExistsError:
        print("Arquivo de entrada nao encontrado")
        exit()

if __name__ == "__main__":
    
    dataset: pd.DataFrame = carregar_dataset().iloc[:500]

    dados_dataframe = {
        'classe': [],
        'polaridade': [],
        'subjetividade': [],
        'tamanho_medio_palavra': [],
        'tamanho_medio_sentenca': [],
        'num_exclamacoes': [],
        'num_maiusculas': [],
        'proporcao_stopwords': [],
        'diversidade_lexica': [],
        'num_numeros': [],
        'entidades_nomeadas': [],
        'proporcao_pronomes': [],
        'num_links': []
    }

    contador_id = 0

    for i, linha in dataset.iterrows():
        
        id_texto = contador_id
        contador_id += 1

        texto = str(linha.get('title', '')) + ' ' + str(linha.get('text', ''))
        classe = linha.get('label', '')

        if texto.replace(" ", "") == '' or classe == '':
            continue

        # print(texto)

        tokens = pln(texto)

        tokens = pln(texto)

        polaridade = tokens._.blob.polarity
        subjetividade = tokens._.blob.subjectivity

        tokens_alpha = [t for t in tokens if t.is_alpha]

        tamanho_medio_palavra = np.mean([len(t.text) for t in tokens_alpha]) if len(tokens_alpha) > 0 else 0
        tamanho_medio_sentenca = np.mean([len(s) for s in tokens.sents]) if len(list(tokens.sents)) > 0 else 0
        num_exclamacoes = texto.count("!")
        num_maiusculas = sum(1 for t in tokens_alpha if t.text.isupper())
        proporcao_stopwords = sum(1 for t in tokens if t.is_stop)/len(tokens) if len(tokens) > 0 else 0
        diversidade_lexica = len(set([t.lemma_ for t in tokens_alpha]))/len(tokens_alpha) if len(tokens_alpha) > 0 else 0
        num_numeros = sum(1 for t in tokens if t.like_num)
        entidades_nomeadas = len(tokens.ents)
        proporcao_pronomes = sum(1 for t in tokens if t.pos_ == "PRON")/len(tokens) if len(tokens) > 0 else 0
        num_links = texto.lower().count("http") + texto.lower().count("www")

        dados_dataframe['classe'].append(classe)
        dados_dataframe['polaridade'].append(polaridade)
        dados_dataframe['subjetividade'].append(subjetividade)
        dados_dataframe['tamanho_medio_palavra'].append(tamanho_medio_palavra)
        dados_dataframe['tamanho_medio_sentenca'].append(tamanho_medio_sentenca)
        dados_dataframe['num_exclamacoes'].append(num_exclamacoes)
        dados_dataframe['num_maiusculas'].append(num_maiusculas)
        dados_dataframe['proporcao_stopwords'].append(proporcao_stopwords)
        dados_dataframe['diversidade_lexica'].append(diversidade_lexica)
        dados_dataframe['num_numeros'].append(num_numeros)
        dados_dataframe['entidades_nomeadas'].append(entidades_nomeadas)
        dados_dataframe['proporcao_pronomes'].append(proporcao_pronomes)
        dados_dataframe['num_links'].append(num_links)

        print(f"Iteracao {contador_id} feita")
    

    df = pd.DataFrame(dados_dataframe)
    df.to_csv(arquivo_saida)
    