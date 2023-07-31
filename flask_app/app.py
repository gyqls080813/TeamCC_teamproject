from flask import Flask, request, render_template, url_for, send_from_directory, redirect
from werkzeug.utils import secure_filename
import pandas as pd
import os
from binary_classification import binary_classification_model
from regression import regression_model
from multi_classification import multi_classification_model

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def select_model():
    if request.method == 'POST':
        selected_option = request.form.get('option')
        if selected_option == 'age_prediction':
            return redirect('/age_prediction')
        elif selected_option == 'pulsar_judgement':
            return redirect('/pulsar_judgement')
        elif selected_option == 'defect_verification':
            return redirect('/defect_verification')
    return render_template('index.html')

@app.route('/<model_name>', methods=['GET', 'POST'])
def select_input_method(model_name):
    if request.method == 'POST':
        input_method = request.form.get('input_method')
        if input_method == 'direct_input':
            return redirect(f'/{model_name}/input')
        elif input_method == 'csv_file':
            return redirect(f'/{model_name}/upload')
    return render_template('input_method.html', model_name=model_name)

@app.route('/<model_name>/upload', methods=['GET', 'POST'])
def upload_file(model_name):
    if request.method == 'POST':
        csv_file = request.files['file']
        if not csv_file:
            return "No file"
        
        filename = secure_filename(csv_file.filename)
        csv_file.save(os.path.join('.', filename))
        X_test = pd.read_csv(filename)
        
        os.remove(filename)
        if model_name == 'age_prediction':
            result = regression_model(X_test)
        elif model_name == 'pulsar_judgement':
            result = binary_classification_model(X_test)
        elif model_name == 'defect_verification':
            result = multi_classification_model(X_test)
        return render_template('result_upload.html', prediction_time=result['prediction_time'], 
                               prediction=result['prediction'], 
                               download_link=url_for('download'), model_name=model_name)

    return render_template('upload.html', model_name=model_name)

@app.route('/<model_name>/input', methods=['GET', 'POST'])
def input_data(model_name):
    # 각 모델에 대한 입력 필드 정의
    fields = {
        'age_prediction': [
            ('Sex', 'select', ['M', 'F', 'I']),
            ('Length', 'number'),
            ('Diameter', 'number'),
            ('Height', 'number'),
            ('Whole weight', 'number'),
            ('Shucked weight', 'number'),
            ('Viscera weight', 'number'),
            ('Shell weight', 'number')
        ],
        'pulsar_judgement': [
            ('Mean of the integrated profile', 'number'),
            ('Standard deviation of the integrated profile', 'number'),
            ('Excess kurtosis of the integrated profile', 'number'),
            ('Skewness of the integrated profile', 'number'),
            ('Mean of the DM-SNR curve', 'number'),
            ('Standard deviation of the DM-SNR curve', 'number'),
            ('Excess kurtosis of the DM-SNR curve', 'number'),
            ('Skewness of the DM-SNR curve', 'number')
        ],
        'defect_verification': [
            ('X_Minimum', 'number'),
            ('Y_Minimum', 'number'),
            ('Pixels_Areas', 'number'),
            ('Minimum_of_Luminosity', 'number'),
            ('Maximum_of_Luminosity', 'number'),
            ('Length_of_Conveyer', 'number'),
            ('TypeOfSteel_A300', 'number'),
            ('Steel_Plate_Thickness', 'number'),
            ('Edges_Index', 'number'),
            ('Empty_Index', 'number'),
            ('Square_Index', 'number'),
            ('Outside_X_Index', 'number'),
            ('Edges_X_Index', 'number'),
            ('Edges_Y_Index', 'number'),
            ('Outside_Global_Index', 'number'),
            ('LogOfAreas', 'number')
            
        ]
    }

    if request.method == 'POST':
        data = {field[0]: [float(request.form[field[0]]) if field[1] == 'number' else request.form[field[0]]] 
                for field in fields[model_name]}
        X_test = pd.DataFrame(data)
        
        if model_name == 'age_prediction':
            result = regression_model(X_test)
        elif model_name == 'pulsar_judgement':
            result = binary_classification_model(X_test)
        elif model_name == 'defect_verification':
            result = multi_classification_model(X_test)

        return render_template('result_input.html', prediction_time=result['prediction_time'], 
                               prediction=result['prediction'], model_name=model_name)
    else:
        # 선택한 모델에 대한 필드를 HTML 템플릿으로 전달
        return render_template('input.html', model_name=model_name, fields=fields[model_name])


@app.route('/<model_name>/upload/result', methods=['GET'])
def result_upload(model_name):
    return render_template('result_upload.html', model_name = model_name)

@app.route('/<model_name>/input/result', methods=['GET'])
def result_input(model_name):
    return render_template('result_input.html', model_name = model_name)

@app.route("/download")
def download():
    return send_from_directory(directory='static', path='prediction_results.csv')

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)
