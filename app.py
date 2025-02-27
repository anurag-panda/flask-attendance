from flask import Flask, render_template, request, redirect, url_for, jsonify
import sqlite3
import os
import cv2
import numpy as np
from datetime import datetime
from face_utils import FaceRecognizer
from datetime import date
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

nimgs = 10



datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/images'):
    os.makedirs('static/images')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Id,Time')
    
def totalreg():
    return len(os.listdir('static/images'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, miniSize=(20, 20))
        return face_points
    except:
        return []
    
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []
    studentlist = os.listdir('static/images')
    for student in studentlist:
        studentfaces = os.listdir(f'static/images/{student}')
        for facename in studentfaces:
            face = cv2.imread(f'static/images/{student}/{facename}')
            resized_face = cv2.resize(face, (100, 100))
            faces.append(resized_face)
            labels.append(student)
        faces = np.array(faces)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(faces, labels)
        joblib.dump(knn, 'static/face_recognition_model.pkl')
        
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    ids = df['Id']
    times = df['Time']
    l = len(df)
    return names, ids, times, l

def add_attendance(name):
    studentname = name.split('_')[0]
    studentid = name.split('_')[1]
    current_time = datetime.now().strftime('%H:%M:%S')
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(studentid) not in list(df['Id']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{studentname},{studentid},{current_time}')
            
def getallusers():
    studentlist = os.listdir('static/images')
    names = []
    ids = []
    l = len(studentlist)
    
    for i in studentlist:
        name, id = i.split('_')
        names.append(name)
        ids.append(id)
        
    return studentlist, names, ids, l

#initialize the database

def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # Create students table
    c.execute('''CREATE TABLE IF NOT EXISTS students(
                  name TEXT, 
                  Id INTEGER,
                  email TEXT, 
                  department TEXT, 
                  year INTEGER, 
                  image_path TEXT)
                ''')
                 
    # Create attendance table
    c.execute('''CREATE TABLE IF NOT EXISTS attendance(
                  student_id INTEGER,
                  date TEXT,
                  time TEXT,
                  FOREIGN KEY(student_id) REFERENCES students(id))''')
    conn.commit()
    conn.close()

init_db()

# Student Management Routes
@app.route('/')
def index():
    return render_template('base.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/start', methods=['GET'])
def start():
    
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('attendance.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
            cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
            cv2.putText(frame, f'{identified_person}', (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        frame[305:305 + 480, 360:360 + 640] = frame
        cv2.imshow('attendance.html', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('attendance.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/add_student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'POST':
        name = request.form['name']
        Id = request.form['student_id']
        email = request.form['email']
        department = request.form['department']
        year = request.form['year']
        image = request.files['image']
        
        if image:
            filename = f"{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
            
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            c.execute("INSERT INTO students (name, Id, email, department, year, image_path) VALUES (?, ?, ?, ?, ?, ?)",
                     (name, Id,  email, department, year, image_path))
            conn.commit()
            conn.close()
            alert = "Student added successfully"
            return alert
            
             # Capture student's face
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(1)
        
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (100, 100))
                if not os.path.exists(f'static/images/{Id}'):
                    os.makedirs(f'static/images/{Id}')
                cv2.imwrite(f'static/images/{Id}/{name}.jpg', face)
            
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Retrain the model
        train_model()
        
        return redirect(url_for('students'))
       
    return render_template('add_student.html')

@app.route('/students')
def students():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    c.execute("SELECT Id, name, email, department, year FROM students")
    students = c.fetchall()
    
    conn.close()
    return render_template('student.html', students=students)

@app.route('/delete_student', methods=['POST'])
def delete_student():
    student_id = request.form['student_id']
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # Get image path
    c.execute("SELECT image_path FROM students WHERE id=?", (student_id,))
    image_path = c.fetchone()[0]
    
    # Delete from database
    c.execute("DELETE FROM students WHERE id=?", (student_id,))
    c.execute("DELETE FROM attendance WHERE student_id=?", (student_id,))
    conn.commit()
    conn.close()
    
    # Delete image file
    if os.path.exists(image_path):
        os.remove(image_path)
    
    return redirect(url_for('students'))

# Attendance Route
@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    data = request.get_json()
    student_id = data.get('student_id')
    
    if not student_id:
        return jsonify({'error': 'Missing student_id'}), 400
    
    try:
        # Add attendance record to database
        date = datetime.now().strftime('%Y-%m-%d')
        time = datetime.now().strftime('%H:%M:%S')
        
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("INSERT INTO attendance (student_id, date, time) VALUES (?, ?, ?)",
                 (student_id, date, time))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Attendance marked successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Database error: {str(e)}'
        }),500

@app.route('/detect_face', methods=['POST'])
def detect_face():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    try:
        # Read image from request
        file = request.files['image']
        nparr = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Initialize face recognizer
        recognizer = FaceRecognizer()
        
        # Detect face and get student ID
        student_id = recognizer.recognize_face(frame)
        
        if student_id:
            return jsonify({
                'success': True,
                'student_id': student_id
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No recognized face found'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error processing image: {str(e)}'
        }), 500


@app.route('/attendance_summary')
def attendance_summary():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    c.execute('''SELECT students.id, students.name, attendance.date, attendance.time 
                 FROM students 
                 JOIN attendance ON students.id = attendance.student_id''')
    records = c.fetchall()
    
    conn.close()
    return render_template('attendance_summary.html', records=records)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)