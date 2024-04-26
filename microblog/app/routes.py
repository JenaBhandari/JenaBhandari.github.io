from flask import jsonify, render_template, request
from app import app
from app.api import inferences
@app.route('/')
@app.route('/index')
def index():
    return render_template('home.html')

@app.route('/submit-data', methods=['POST'])
def submit_data():
    data = request.json 
    zip_code = data['zip_code']
    age = data['age']
    gender = data['gender']
    age_range = data['age_range']    
    age_top = data['age_top']
    age_bottom = data['age_bottom']
    print("Received data:", data)
    print("Zip Code:", zip_code)
    print("Age:", age)
    print("Gender:", gender)
    print("Age Range:", age_range)
    print("Age Top:", age_top)
    print("Age Bottom:", age_bottom)

    print("before inference call")
    # Call the function from inferences.py
    result_income, result_edu = inferences.perform_inference(zip_code, age, gender, age_range, age_top, age_bottom)
    result_income_str = str(result_income)
    result_edu_str = str(result_edu)
    response_data = {
        "inference_income": result_income_str,
        "inference_edu": result_edu_str
    }
    return jsonify(response_data)

@app.route('/load-data', methods=['POST'])
def load_data():
    print("before load inference call")
    result = inferences.prep()
    
    return jsonify("")