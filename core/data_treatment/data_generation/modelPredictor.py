import os
import keras
import constants

import numpy as np
import pandas as pd
from synergie.services.prediction_service import PredictionService

class ModelPredictor:
    def __init__(self, model_type: keras.models.Model, model_success: keras.models.Model) -> None:

        self.model_type = model_type
        self.model_success = model_success

    def load_from_csv(self, dataset_path : str, filename : str = "jumplist.csv"):
        df_jumps = pd.read_csv(os.path.join(dataset_path, filename))
        data_jump = []

        for index,rows in df_jumps.iterrows():
            df_onejump = pd.read_csv(os.path.join(dataset_path, rows["path"]))
            data_jump.append(df_onejump)

        predict_type, predict_success = self.predict(data_jump)
        df_jumps['type'] = predict_type
        df_jumps['success'] = predict_success

        self.df_jumps = df_jumps

        self.df_jumps.to_csv(os.path.join(dataset_path, filename), index=False)

    def predict(self, data : list[pd.DataFrame]):
        return PredictionService(self.model_type, self.model_success).predict_segments(data)

    def checktype(self, checkPath : str):
        df_check = pd.read_csv(os.path.join(checkPath, "jumplist.csv"))
        typeCheck = np.array(df_check["type"])
        typePredict = np.array(self.df_jumps["type"])

        index = typeCheck!=8
        typeCheck = typeCheck[index]
        typePredict = typePredict[index]
        indexbis = typePredict!=2
        typeCheck = typeCheck[indexbis]
        typePredict = typePredict[indexbis]
        
        total = (typePredict==typeCheck).sum()/len(typeCheck)
        precision = []
        for i in range(6):
            count = (typeCheck==i).sum()
            precision.append((typePredict[typeCheck==i]==i).sum()/count)
        precision.append(total)
        return precision
    
    def checksuccess(self, checkPath : str):
        df_check = pd.read_csv(os.path.join(checkPath, "jumplist.csv"))
        typeCheck = np.array(df_check["success"])
        typePredict = np.array(self.df_jumps["success"])

        index = typeCheck!=2
        typeCheck = typeCheck[index]
        typePredict = typePredict[index]
        indexbis = typePredict!=2
        typeCheck = typeCheck[indexbis]
        typePredict = typePredict[indexbis]

        total = (typePredict==typeCheck).sum()/len(typeCheck)
        precision = []
        for i in range(2):
            count = (typeCheck==i).sum()
            precision.append((typePredict[typeCheck==i]==i).sum()/count)
        precision.append(total)
        return precision
