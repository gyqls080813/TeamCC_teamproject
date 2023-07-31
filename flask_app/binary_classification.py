import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

def binary_classification_model(X_test):
    if(len(X_test.columns) > 8):
        X_test = X_test.iloc[:,:8]
    X_test.dropna(inplace=True)
    sc = StandardScaler()
    X_test = sc.fit_transform(X_test)
    model = load_model('binary_mlp_model.h5')
    start_time = time.time()
    prediction = model.predict(X_test)
    end_time = time.time()

    prediction_time = np.round(end_time - start_time, 3)
    binary_prediction = [1 if p >= 0.5 else 0 for p in prediction]

    pd.DataFrame(binary_prediction, columns=['prediction']).to_csv('static/prediction_results.csv', index=False)

    return {'prediction': binary_prediction, 'prediction_time': prediction_time}
