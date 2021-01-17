#!/usr/bin/env python
# coding: utf-8

import json
import joblib
import pandas as pd
from flask import Flask

# Importando a entrada em formato .json:

with open('input/pe-dataset.json') as file:
    pe_dataset = json.load(file)

# Transformando a entrada em um dataframe:

db_pe = pd.DataFrame.from_dict(pe_dataset, orient='index')

# Agrupando o dataframe por painéis:

db_pe = db_pe.sort_values(by='panel_info')

db_pe.reset_index(level=0, inplace=True)

# Instâncias do painel:

instances_painel = db_pe.drop(columns=['index', 'panel_info', 'panel_eplet', 'reactive'])

# Transformando os valores categóricos em numéricos:

for column in instances_painel.columns:
    instances_painel[column] = instances_painel[column].apply(lambda x:int(x))

# Carregando o Modelo Persistido

# Modelo treinado com a base de SP
joblib_model = joblib.load("persisted_model/joblib_final_model.pkl")

# Classificando Epítopos

predicted_labels_painel = joblib_model.predict(instances_painel)

def returnReactiveEplets(labels):
    painels = []
    predict_eplet = []
    painel_number = db_pe['panel_info'][0]                          # Recebe o identificador do primeiro painel
    first_react_eplet = 1                                           # Flag para controlarmos o cálculo do cutoff
    cutoff = 0                                                                                                                
    
    for index in range(0, len(labels)):                             # Percorre todas as instâncias da base
        if(db_pe['panel_info'][index] == painel_number):            # Se ainda estivermos no mesmo painel
            
            if(labels[index] == 1):                                 # Se o epítopo for classificado como reativo
                predict_eplet.append(db_pe['panel_eplet'][index])   # Adiconamos à lista de epítopos classificados como reativos
                
                if(first_react_eplet):                              # Verificamos se é o primeiro epítopo reativo daquele painel
                    cutoff = db_pe['panel_min_mfi'][index]          # Inicializamos o valor do cutoff
                    first_react_eplet = 0                           # Atualizamos o valor da flag
                    
                elif(db_pe['panel_min_mfi'][index] < cutoff):       # Verificamos se o cutoff do eplet é o menor cutoff
                    cutoff = db_pe['panel_min_mfi'][index]          # Atualizamos o valor do cutoff
        else:                                                       
            results = {                                             # Se mudarmos de painel, resetamos as análises
                    'painel': '',
                    'analysis': [],
                    'cutoff': ''
            }
            results['painel'] = painel_number                      # Preenchemos o identificador do painel
            results['analysis'] = predict_eplet                    # Preenchemos a lista de epítopos classificados como reativos
            results['cutoff'] = str(cutoff)                        # Preenchemos o valor do cutoff
            painels.append(results)                                # Adicionamos à lista de resultados
            
            painel_number = db_pe['panel_info'][index]             # Atualizamos o identificador do painel
            predict_eplet = []                                     # Resetamos a lista de epítopos classificados como reativos
            cutoff = 0                                             # Resetamos o valor do cutoff
            first_react_eplet = 1                                  # Resetamos a flag para controlarmos o cálculo do cutoff
            
    return painels

panels = returnReactiveEplets(predicted_labels_painel)

'''
def saveResults(panels):
    for index in range(0, len(panels)):
        with open('output/web_format/panel_data_{}.json'.format(index), 'w') as json_file:
            json.dump(panels[index], json_file)


def saveResults(panels):
    with open('output/panels_output.json', 'w') as json_file:
            json.dump(panels, json_file)

saveResults(panels)
'''

panels_json = json.dumps(panels)

app = Flask(__name__)

@app.route('/')
def ola():
    return panels_json

app.run()