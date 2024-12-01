from flask import Flask, render_template, request, redirect, url_for
from detection import detect_and_classify
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        predictions, boxed_image_path = detect_and_classify(file_path)
        return render_template('result.html', predictions=predictions, boxed_image=boxed_image_path)

if __name__ == '__main__':
    app.run(debug=True)
