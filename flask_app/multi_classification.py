# -*- coding: utf-8 -*-

import time
from sklearn.preprocessing import StandardScaler 
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

def multi_classification_model(df):
    # 모델 로드
    model = load_model('mlp_model.h5')
    
    # 상관관계 높은 변수 제거를 통한 차원 축소
    cols = ['X_Minimum', 'Y_Minimum', 'Pixels_Areas', 'Minimum_of_Luminosity', 'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300', 'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas']
    X = df[cols]
    
    # 데이터 표준화 (Standardization)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 예측값 출력
    start_time = time.time()
    prediction = model.predict(X)
    result_pred = np.argmax(prediction, axis=1)
    end_time = time.time()
    prediction_time = np.round(end_time - start_time, 3)
    
    class_mapping = {0: 'Pastry', 1: 'Z_Scratch', 2: 'K_Scatch', 3: 'Stains', 4: 'Dirtiness', 5: 'Bumps', 6: 'Other_Faults'}
    defection_type = [class_mapping[pred] for pred in result_pred]

    pd.DataFrame({'defection_type': defection_type, 'prediction': result_pred}).to_csv('static/prediction_results.csv', index=False)
    
    return {'prediction': defection_type, 'prediction_time': prediction_time}