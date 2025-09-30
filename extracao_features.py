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

arquivo_saida = "dataset_features_sem_duplicatas_novo.csv"

def carregar_dataset() -> pd.DataFrame:
    try:
        return pd.read_csv(arquivo_dataset)
    except FileExistsError:
        print("Arquivo de entrada nao encontrado")
        exit()


def preprocessar_texto(texto: str, pln) -> str:
    doc = pln(texto)

    tokens_filtrados = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]

    return " ".join(tokens_filtrados)


if __name__ == "__main__":
    
    dataset: pd.DataFrame = carregar_dataset()
    dataset.drop_duplicates(subset=["title", "text", "label"], inplace=True)

    VERBOS_DUVIDA_INCERTEZA = {"may", "might", "could", "seem", "suggest", "believe", "appear", "think"}
    HEDGES = {"perhaps", "apparently", "maybe", "possibly", "probably", "sort of", "kind of", "more or less"}

    dados_dataframe = {
        'classe': [],
        'polaridade': [],
        'subjetividade': [],
        'tamanho_medio_palavra': [],
        'tamanho_medio_sentenca': [],
        'num_exclamacoes': [],
        'proporcao_stopwords': [],
        'diversidade_lexica': [],
        'num_numeros': [],
        'entidades_nomeadas': [],
        'proporcao_pronomes': [],
        'num_links': [],
        'proporcao_adj_adv': [], 
        'proporcao_duvida_incerteza': [],
        'proporcao_citacoes': [],
        'pontuacao_nao_terminal_por_sentenca': [],
        'tamanho_entidade_nomeada_mais_longa': [],
        'proporcao_hedges': [],
        'proporcao_conj_coordenativas': []
    }

    contador_id = 0

    for i, linha in dataset.iterrows():
        
        id_texto = contador_id
        contador_id += 1

        texto = str(linha.get('title', '')) + ' ' + str(linha.get('text', ''))
        classe = linha.get('label', '')

        if texto.replace(" ", "") == '' or classe == '':
            continue

        tokens_original = pln(texto)
        
        texto_processado = preprocessar_texto(texto, pln)
        tokens_processados = pln(texto_processado)

        polaridade = tokens_processados._.blob.polarity
        subjetividade = tokens_processados._.blob.subjectivity

        tokens_alpha_original = [t for t in tokens_original if t.is_alpha]

        tamanho_medio_palavra = np.mean([len(t.text) for t in tokens_alpha_original]) if len(tokens_alpha_original) > 0 else 0
        tamanho_medio_sentenca = np.mean([len(s) for s in tokens_original.sents]) if len(list(tokens_original.sents)) > 0 else 0
        num_exclamacoes = texto.count("!")
        
        proporcao_stopwords = sum(1 for t in tokens_original if t.is_stop)/len(tokens_original) if len(tokens_original) > 0 else 0
        
        tokens_alpha_processado = [t for t in tokens_processados if t.is_alpha]
        diversidade_lexica = len(set([t.lemma_ for t in tokens_alpha_processado]))/len(tokens_alpha_processado) if len(tokens_alpha_processado) > 0 else 0

        num_numeros = sum(1 for t in tokens_original if t.like_num)
        entidades_nomeadas = len(tokens_original.ents)
        proporcao_pronomes = sum(1 for t in tokens_original if t.pos_ == "PRON")/len(tokens_original) if len(tokens_original) > 0 else 0
        num_links = texto.lower().count("http") + texto.lower().count("www")

        num_total_tokens = len(tokens_original)
        num_adj_adv = sum(1 for t in tokens_original if t.pos_ in ("ADJ", "ADV"))
        proporcao_adj_adv = num_adj_adv / num_total_tokens if num_total_tokens > 0 else 0
        num_duvida_incerteza = sum(1 for t in tokens_original if t.lemma_ in VERBOS_DUVIDA_INCERTEZA)
        proporcao_duvida_incerteza = num_duvida_incerteza / num_total_tokens if num_total_tokens > 0 else 0

        num_citacoes = texto.count('"') + texto.count("'")
        proporcao_citacoes = num_citacoes / num_total_tokens if num_total_tokens > 0 else 0
        num_pontuacao_nao_terminal = sum(1 for t in tokens_original if t.text in (",", ";", ":", "—", "-"))
        num_sentencas = len(list(tokens_original.sents))

        pontuacao_nao_terminal_por_sentenca = num_pontuacao_nao_terminal / num_sentencas if num_sentencas > 0 else 0
        tamanho_entidade_nomeada_mais_longa = max([len(ent) for ent in tokens_original.ents]) if len(tokens_original.ents) > 0 else 0
        num_hedges = sum(1 for t in tokens_original if t.lemma_ in HEDGES)
        proporcao_hedges = num_hedges / num_total_tokens if num_total_tokens > 0 else 0
        num_conj_coordenativas = sum(1 for t in tokens_original if t.pos_ == "CCONJ")
        proporcao_conj_coordenativas = num_conj_coordenativas / num_total_tokens if num_total_tokens > 0 else 0

        dados_dataframe['classe'].append(classe)
        dados_dataframe['polaridade'].append(polaridade)
        dados_dataframe['subjetividade'].append(subjetividade)
        dados_dataframe['tamanho_medio_palavra'].append(tamanho_medio_palavra)
        dados_dataframe['tamanho_medio_sentenca'].append(tamanho_medio_sentenca)
        dados_dataframe['num_exclamacoes'].append(num_exclamacoes)
        dados_dataframe['proporcao_stopwords'].append(proporcao_stopwords)
        dados_dataframe['diversidade_lexica'].append(diversidade_lexica)
        dados_dataframe['num_numeros'].append(num_numeros)
        dados_dataframe['entidades_nomeadas'].append(entidades_nomeadas)
        dados_dataframe['proporcao_pronomes'].append(proporcao_pronomes)
        dados_dataframe['num_links'].append(num_links)
        dados_dataframe['proporcao_adj_adv'].append(proporcao_adj_adv)
        dados_dataframe['proporcao_duvida_incerteza'].append(proporcao_duvida_incerteza)
        dados_dataframe['proporcao_citacoes'].append(proporcao_citacoes)
        dados_dataframe['pontuacao_nao_terminal_por_sentenca'].append(pontuacao_nao_terminal_por_sentenca)
        dados_dataframe['tamanho_entidade_nomeada_mais_longa'].append(tamanho_entidade_nomeada_mais_longa)
        dados_dataframe['proporcao_hedges'].append(proporcao_hedges)
        dados_dataframe['proporcao_conj_coordenativas'].append(proporcao_conj_coordenativas)

        print(f"Iteracao {contador_id} feita")
    

    df = pd.DataFrame(dados_dataframe)
    df.to_csv(arquivo_saida)
