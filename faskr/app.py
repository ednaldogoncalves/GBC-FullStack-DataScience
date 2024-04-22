"""
Final Project: Suspect detection CCTV (closed-circuit television)
Final Project - AASD4016: Full Stack Data Science
Professor: Vejey Gandyer
Final Project - ASSD4017: Data Science Driven
Professor: Moe Fadaee
Student ID: 101373529
Student Name: Gonçalves, Ednaldo
"""


# Import Libraries
from flask import Flask, render_template, request, redirect, url_for
import os
from helper import test_detection_faces_one_image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search1')
def search1():
    return render_template('search1.html')

@app.route('/search2')
def search2():
    return render_template('search2.html')

@app.route('/search3')
def search3():
    return render_template('search3.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        # Handle form submission
        suspect_photo = request.files['suspect_photo']

        processed_image_path = suspect_photo

        # call the function (assuming test_detection_faces_one_image returns the processed image path)
        processed_image_path = test_detection_faces_one_image(suspect_photo)

        if processed_image_path == '':
            result_message = f"Suspect not found in database."
        else:
            result_message = f"Suspect found in database."

        # Pass result message and processed image path to HTML template
        return render_template('result1.html', result_message=result_message, processed_image_path=processed_image_path)
    else:
        # Render the search4.html template
        return render_template('result5.html')

@app.route('/search4')
def search4():
    return render_template('search4.html')

@app.route("/result5/<test>")
def test_function(test):
    return test

if __name__ == '__main__':
    app.run(debug=True)