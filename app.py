from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from keras.models import load_model
import numpy as np
import cv2
import bcrypt
import re
import base64

from pymongo import MongoClient

app = Flask(__name__)
app.secret_key = "your_secret_key"

model = load_model(r'final_balanced_model.h5')
client = MongoClient("") # we removed the mongo link 
db = client["DeepRayWeb"]
users_collection = db["users"]
patients_collection = db["patients"]

def is_valid_password(password):
    return len(password) >= 8

@app.route('/')
def serve_html():
    return render_template('index.html')

@app.route('/diagnosis')
def diagnosis():
    if 'username' not in session:
        flash("You need to log in to access this page.", "error")
        return redirect(url_for('login'))
    return render_template('diagnosis.html')

@app.route('/patients-rec')
def patients_rec():
    if 'username' not in session:
        flash("You need to log in to access this page.", "error")
        return redirect(url_for('login'))
    return render_template('patients-rec.html')




@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users_collection.find_one({"username": username, "password": password})
        if user:  
            session['username'] = username
            flash("Logged in successfully!", "success") 
            return redirect(url_for('serve_html'))
        else:  
            flash("Invalid username or password", "error") 
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['password']
            confirm_password = request.form['confirm_password']
            print(f"Received data: username={username}, password={password}, confirm_password={confirm_password}")
            if password != confirm_password:
                flash("Passwords do not match", "error")
                print("Passwords do not match")
                return redirect(url_for('signup'))
            existing_user = users_collection.find_one({"username": username})
            print(f"Existing user: {existing_user}")
            if existing_user:
                flash("Username already registered", "error")
                print("Username already exists")
                return redirect(url_for('signup'))
            result = users_collection.insert_one({"username": username, "password": password})
            print(f"Inserted user ID: {result.inserted_id}")
            flash("Account created successfully!", "success")
            return redirect(url_for('login'))
        except Exception as e:
            print("Error during signup:", e)
            flash("An error occurred while processing your request.", "error")
            return redirect(url_for('signup'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("Logged out successfully", "success")
    session.pop('_flashes', None)
    return redirect(url_for('login'))

@app.route('/predict/', methods=['POST'])
def predict():
    if 'username' not in session:
        return jsonify({"error": "You must be logged in to make a diagnosis"}), 403
    try:
        xray_file = request.files['xray']
        image_data = xray_file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)
        class_labels = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']
        result = class_labels[predicted_class[0]]
        return jsonify({"prediction": result})
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({"error": "An error occurred during prediction."}), 500

   
@app.route('/save_diagnosis', methods=['POST'])
def save_diagnosis():
    if 'username' not in session:
        return jsonify({"error": "You must be logged in to save diagnosis"}), 403
    if 'xray' not in request.files:
        return jsonify({"error": "X-ray image is missing"}), 400
    if 'patient-id' not in request.form:
        return jsonify({"error": "Patient ID is missing"}), 400
    if 'diagnosis' not in request.form:
        return jsonify({"error": "Diagnosis is missing"}), 400
    xray_file = request.files['xray']
    image_data = xray_file.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    patient_id = request.form['patient-id']
    diagnosis_result = request.form['diagnosis']
    print(f"Patient ID: {patient_id}")
    print(f"Diagnosis: {diagnosis_result}")
    print(f"Image size: {len(image_data)} bytes")
    patients_collection.insert_one({
        "Username": session['username'],
        "Patient_ID": patient_id,
        "Diagnosis": diagnosis_result,
        "XRay_Image": image_base64
    })
    return jsonify({"message": "Diagnosis saved successfully!"})

@app.route('/get_reports', methods=['GET'])
def get_reports():
    reports = list(patients_collection.find({}, {'_id': 0, 'Username': 0})) 
    return jsonify(reports)

@app.route('/delete_report/<string:patient_id>', methods=['DELETE'])
def delete_report(patient_id):
    result = patients_collection.delete_one({"Patient_ID": patient_id})
    if result.deleted_count > 0:
        return jsonify({'success': True, 'message': 'Report deleted successfully!'}), 200
    else:
        return jsonify({'success': False, 'message': 'Report not found'}), 404

@app.route('/edit_report/<string:patient_id>', methods=['POST'])
def edit_report(patient_id):
    data = request.get_json()
    new_diagnosis = data.get('diagnosis')
    patient_report = patients_collection.find_one({"Patient_ID": patient_id})
    if patient_report:
        patients_collection.update_one(
            {"Patient_ID": patient_id},
            {"$set": {"Diagnosis": new_diagnosis}}
        )
        return jsonify({'success': True, 'message': 'Report updated successfully!'}), 200
    else:
        return jsonify({'success': False, 'message': 'Report not found'}), 404


if __name__ == '__main__':
    app.run(debug=True)
