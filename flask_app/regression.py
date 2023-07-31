import pickle
import numpy as np
import time
import pandas as pd 

def one_hot_encode_sex(data):
    data["Sex_1"] = (data["Sex"] == "M").astype(int)
    data["Sex_2"] = (data["Sex"] == "F").astype(int)
    data["Sex_3"] = (data["Sex"] == "I").astype(int)
    data = data[["Sex_1", "Sex_2", "Sex_3", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight"]]  # Reorder columns
    return data


def regression_model(X_test):
    # 들어오는 데이터 전처리 (Sex 인코딩)
    X_test_encoded = one_hot_encode_sex(X_test)
    X_test_encoded["Ratio"] = round(0.5 * np.sqrt((X_test_encoded["Length"] / 2) ** 2 + (X_test_encoded["Diameter"] / 2) ** 2), 3)
    X_test_encoded["Thickness"] = X_test_encoded["Shell weight"] / (3.14 * X_test_encoded["Length"] * X_test_encoded["Diameter"])
    X_test_encoded["Body weight"] = X_test_encoded["Shucked weight"] - X_test_encoded["Viscera weight"]
    
    # 모델 불러오기
    with open('linear_regression_model_Ridge.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # 학습된 모델을 사용하여 예측
    start_time = time.time()
    prediction = model.predict(X_test_encoded)
    end_time = time.time()
    prediction_time = np.round(end_time - start_time, 3)
    
    pd.DataFrame(prediction, columns=['prediction']).to_csv('static/prediction_results.csv', index=False)

    return {'prediction': prediction, 'prediction_time': prediction_time}
