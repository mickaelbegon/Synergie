import os
import keras
import constants

import numpy as np
import pandas as pd
from synergie.config import JUMP_WINDOW_FRAMES, SUCCESS_WINDOW_START, TYPE_WINDOW_FRAMES

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
        predict_jump_type = []
        predict_jump_success = []
        valid_indices = []
        predict_type = np.full(len(data), 8.0)
        predict_success = np.full(len(data), 2.0)

        for index,df in enumerate(data):
            df_predictjump = self._numeric_features(df)
            if len(df_predictjump) == JUMP_WINDOW_FRAMES:
                predict_jump_type.append(df_predictjump[:TYPE_WINDOW_FRAMES])
                predict_jump_success.append(df_predictjump[SUCCESS_WINDOW_START:])
                valid_indices.append(index)

        if not predict_jump_type:
            return (predict_type, predict_success)

        type_temporal = np.asarray(predict_jump_type, dtype=np.float32)
        success_temporal = np.asarray(predict_jump_success, dtype=np.float32)
        prediction_type = self._predict(self.model_type, type_temporal)
        prediction_success = self._predict(self.model_success, success_temporal)

        for prediction_index, data_index in enumerate(valid_indices):
            predict_type[data_index] = np.argmax(prediction_type[prediction_index])
            predict_success[data_index] = np.argmax(prediction_success[prediction_index])
        
        return (predict_type, predict_success)

    def _numeric_features(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Return the model's expected temporal channels as numeric values."""
        numeric = dataframe[constants.fields_to_keep].apply(pd.to_numeric, errors="coerce")
        return np.nan_to_num(numeric.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def _predict(self, trained_model, temporal_features: np.ndarray):
        """Support both modern two-input and older one-input saved models."""
        model_inputs = getattr(trained_model, "inputs", None)
        if model_inputs is not None and len(model_inputs) > 1:
            scalar_features = np.zeros((len(temporal_features), 2), dtype=np.float32)
            return trained_model.predict(
                {"temporal_input": temporal_features, "scalar_input": scalar_features}
            )
        return trained_model.predict(temporal_features)

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
