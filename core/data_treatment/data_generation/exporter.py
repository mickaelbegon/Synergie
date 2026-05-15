import copy
import os
from pathlib import Path
from typing import List
import pandas as pd

import constants
from core.database.DatabaseManager import DatabaseManager, JumpData
from core.utils.jump import Jump
from synergie.config import JUMP_WINDOW_FRAMES


def mstostr(ms: float):

    s = round(ms / 1000)
    return "{:02d}:{:02d}".format(s // 60, s % 60)


def export(
    df: pd.DataFrame,
    sampleTimeFineSynchro: int = 0,
    *,
    type_model_path: str | None = None,
    success_model_path: str | None = None,
) -> pd.DataFrame:
    from core.data_treatment.data_generation.trainingSession import trainingSession
    """
    exports the data to a folder, in order to be used by the ML model
    :param folder_name: the folder where to export the data
    :param sampleTimeFineSynchro: the timefinesample of the synchro tap
    :return:
    """
    # get the list of csv files

    jumpList : List[Jump]= []
    predict_jump = []

    session = trainingSession(df, sampleTimeFineSynchro)

    for jump in session.jumps:
        jump_copy = copy.deepcopy(jump)
        jump_copy.df_type = jump.df_type.copy(deep=True)
        jump_copy.df_success = jump.df_success.copy(deep=True)
        jumpList.append(jump_copy)
        predict_jump.append(jump_copy.df)

    predict_type = [8] * len(predict_jump)
    predict_success = [2] * len(predict_jump)
    prediction_status = "skipped_missing_model_runtime"
    try:
        from synergie.services.prediction_service import PredictionService

        prediction = PredictionService.from_paths(
            type_model_path or constants.modeltype_filepath,
            success_model_path or constants.modelsuccess_filepath,
        )
        predict_type, predict_success = prediction.predict_segments(predict_jump)
        prediction_status = "predicted"
    except ModuleNotFoundError as exc:
        if exc.name not in {"tensorflow", "keras"}:
            raise

    jumpDictCSV = []
    for i,jump in enumerate(jumpList):
        if jump.df is None:
            continue
        if len(jump.df) == JUMP_WINDOW_FRAMES:
            # since videoTimeStamp is for user input, I can change it's value to whatever I want
            jumpDictCSV.append({
                'videoTimeStamp': mstostr(jump.startTimestamp),
                'type': predict_type[i],
                'success': predict_success[i],
                "rotations": "{:.1f}".format(jump.rotation),
                "rotation_direction": jump.rotation_direction,
                "signed_rotations": "{:.3f}".format(jump.signed_rotation),
                "rotation_speed": jump.max_rotation_speed,
                "length": jump.length,
            })

    jumpListdf = pd.DataFrame(jumpDictCSV)
    jumpListdf = jumpListdf.sort_values(by=['videoTimeStamp'])
    jumpListdf.attrs["prediction_status"] = prediction_status

    return jumpListdf


def old_export():
    from core.data_treatment.data_generation.trainingSession import trainingSession
    """
    exports the data to a folder, in order to be used by the ML model
    :param folder_name: the folder where to export the data
    :param sampleTimeFineSynchro: the timefinesample of the synchro tap
    :return:
    """
    jumpDictCSV = []
    folder = Path("data/pending")
    folder.mkdir(exist_ok=True)

    for training in os.listdir("data/new"):
        if os.path.isfile(f"data/new/{training}"):
            jumpList = []
            print(training)
            parts = training.replace(".csv", "").split("_")
            synchro = int(parts[0])
            training_id = parts[1]
            try:
                skater_name = DatabaseManager().get_skater_name_from_training_id(training_id)
            except (TypeError, ValueError, KeyError):
                skater_name = parts[2] + "_" + parts[3][:4] + "_" + parts[0]
            df = pd.read_csv(f"data/new/{training}")

            session = trainingSession(df, synchro)

            for jump in session.jumps:
                jump_copy = copy.deepcopy(jump)
                jump_copy.skater_name = skater_name
                jump_copy.df = jump.df.copy(deep=True)
                jumpList.append(jump_copy)


            for i in jumpList:
                if i.df is None:
                    continue
                jump_id = str(i.skater_name) + "_" + str(int(i.startTimestamp))
                if jump_id != "0":

                    filename = os.path.join(folder, str(jump_id) + ".csv")
                    i.generate_csv(filename)
                    # since videoTimeStamp is for user input, I can change it's value to whatever I want
                    jumpDictCSV.append({'date': parts[2],
                                        'starting': parts[3],
                        'path': "data/annotated/" + parts[2] + "/" + parts[3][:4] + "/" + str(jump_id) + ".csv",
                                        'videoTimeStamp': mstostr(i.startTimestamp),
                                        'type': i.type.value,
                                        'skater': i.skater_name,
                                        "sucess": 2,
                                        "rotations": "{:.1f}".format(i.rotation)})


    jumpListdf = pd.DataFrame(jumpDictCSV)
    jumpListdf = jumpListdf.sort_values(by=['date', 'starting', 'videoTimeStamp']).reset_index(drop=True)
    jumpListdf.drop(columns=['date', 'starting'], inplace=True)
    jumpListdf.to_csv(os.path.join(folder, "jumplist.csv"))
