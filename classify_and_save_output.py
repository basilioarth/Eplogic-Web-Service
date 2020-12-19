import json
import joblib
import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import sklearn.metrics as mt

db_pe = pd.read_csv('datasets/pe-dataset.csv')

instances_painel = db_pe.drop(columns=['panel_info', 'panel_eplet', 'reactive'])
labels_painel = np.array(db_pe['reactive'])

model = joblib.load("persisted_model/joblib_final_model.pkl")
predicted_labels_painel = ms.cross_val_predict(model, instances_painel, labels_painel, cv=10, n_jobs=-1)

predicted_labels_df = pd.DataFrame(data=predicted_labels_painel.flatten())

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
                'result_sugestion': {"cutoff": ""},
                'result_user': {
                    'analysis': [],
                    'cutoff': ''
                }
            }
            results['result_user']['analysis'] = predict_eplet     # Preenchemos a lista de epítopos classificados como reativos
            results['result_user']['cutoff'] = str(cutoff)         # Preenchemos o valor do cutoff
            painels.append(results)                                # Adicionamos à lista de resultados
            
            painel_number = db_pe['panel_info'][index]             # Atualizamos o identificador do painel
            predict_eplet = []                                     # Resetamos a lista de epítopos classificados como reativos
            cutoff = 0                                             # Resetamos o valor do cutoff
            first_react_eplet = 1                                  # Resetamos a flag para controlarmos o cálculo do cutoff
            
    return painels

def saveResults(panels):
    for index in range(0, len(panels)):
        with open('output/panel_data_{}.json'.format(index), 'w') as json_file:
            json.dump(panels[index], json_file)

panels = returnReactiveEplets(predicted_labels_painel)
saveResults(panels)