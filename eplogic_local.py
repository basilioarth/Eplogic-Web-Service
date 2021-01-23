#!/usr/bin/env python
# coding: utf-8

import json
import joblib
import pandas as pd
import sklearn

print()
path = input('Insita o path do diretório do projeto: ')
print()

# Importando a entrada em formato .json:

with open('{}/input/pe-dataset.json'.format(path)) as file:
    dataset = json.load(file)

# Transformando a entrada em um dataframe:

db = pd.DataFrame.from_dict(dataset, orient='index')

# Agrupando o dataframe por painéis:

db = db.sort_values(by='panel_info')

db.reset_index(level=0, inplace=True)

# Instâncias do painel:

instances_painel = db.drop(columns=['index', 'panel_info', 'panel_eplet', 'reactive'])

# Transformando os valores categóricos em numéricos:

for column in instances_painel.columns:
    instances_painel[column] = instances_painel[column].apply(lambda x:int(x))

# Carregando o Modelo Persistido

# Modelo treinado com a base de SP
joblib_model = joblib.load("{}/persisted_model/final_model.pkl".format(path))

# Classificando Epítopos

predicted_labels_painel = joblib_model.predict(instances_painel)

def returnReactiveEplets(labels):
    painels = []
    predict_eplet = []
    painel_number = db['panel_info'][0]                            # Recebe o identificador do primeiro painel
    first_react_eplet = 1                                          # Flag para controlarmos o cálculo do cutoff
    cutoff = 0                                                                                                                
    
    for index in range(0, len(labels)):                            # Percorre todas as instâncias da base
        if(db['panel_info'][index] == painel_number):              # Se ainda estivermos no mesmo painel
            
            if(labels[index] == 1):                                # Se o epítopo for classificado como reativo
                predict_eplet.append(db['panel_eplet'][index])     # Adiconamos à lista de epítopos classificados como reativos
                
                if(first_react_eplet):                             # Verificamos se é o primeiro epítopo reativo daquele painel
                    cutoff = db['panel_min_mfi'][index]            # Inicializamos o valor do cutoff
                    first_react_eplet = 0                          # Atualizamos o valor da flag
                    
                elif(db['panel_min_mfi'][index] < cutoff):         # Verificamos se o cutoff do eplet é o menor cutoff
                    cutoff = db['panel_min_mfi'][index]            # Atualizamos o valor do cutoff
        else:                                                       
            results = {                                            # Se mudarmos de painel, resetamos as análises
                    'panel': '',
                    'analysis': [],
                    'cutoff': ''
            }
            results['panel'] = painel_number                       # Preenchemos o identificador do painel
            results['analysis'] = predict_eplet                    # Preenchemos a lista de epítopos classificados como reativos
            results['cutoff'] = str(cutoff)                        # Preenchemos o valor do cutoff
            painels.append(results)                                # Adicionamos à lista de resultados
            
            painel_number = db['panel_info'][index]                # Atualizamos o identificador do painel
            predict_eplet = []                                     # Resetamos a lista de epítopos classificados como reativos
            cutoff = 0                                             # Resetamos o valor do cutoff
            first_react_eplet = 1                                  # Resetamos a flag para controlarmos o cálculo do cutoff
            
    return painels

panels = returnReactiveEplets(predicted_labels_painel)

def saveResults(panels):
    with open('{}/output/panels_output.json'.format(path), 'w') as json_file:
            json.dump(panels, json_file)

saveResults(panels)

print('Instalador funcionando')
input('Arquivo .json criado na pasta output! Pressione ENTER para encerrar.')