import os

import flask
from flask import Flask, request, jsonify, session, render_template, redirect
from flask_cors import CORS
from werkzeug.utils import secure_filename

import serve

# define the app
app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# API route
@app.route('/predict', methods=['POST','GET'])
def predict_api():
    """API function that recevies forms input and sends predictions to frontend
    """
    if request.method == 'GET':
        # Show the upload form.
        return render_template('simple_client.html')

    pdf_file = request.files['pdf_file']
    modelPath = request.form['dname']


    if pdf_file.filename == '':
        return '''No file selected.<a href="/predict">Click here to go back.</a>'''
    elif modelPath == '':
        return '''No model selected.<a href="/predict">Click here to go back.</a>'''
    try:
        model = request.form['model']
    except:
        return '''No model selected.<a href="/predict">Click here to go back.</a>'''

    model_dir = modelPath + model
    filename = secure_filename(pdf_file.filename)

    abs_fpath = filename
    session['currentFile'] = filename
    session['modelDirectory'] = model_dir

    pdf_file.save(abs_fpath)


    app.logger.info("api_input: " + str(model_dir))
    model_api =serve.get_model_api(model_dir,abs_fpath)
    os.remove(abs_fpath)

    if(len(model_api) == 0):
        return '''Not Supported.<a href="/predict">Click here to go back.</a>'''
    else:
        return model_api[0].to_json(orient='records', lines=True)+'''
            <html><body>
            <br><br><a href="/return-files">Click here to download as csv.</a><br><br>
            <a href="/predict">Click here to upload new document.</a>
            </body></html>
            '''

@app.route('/predictText', methods=['POST','GET'])
def predict_text_api():

    """API function that recevies text input and sends predictions to frontend

    """
    if request.method == 'GET':
        # Show the enter text form.
        return render_template('full_client.html')

    sentence = request.json['input']
    modelPath = request.json['dname']

    #Empty validation
    if modelPath == '':
        return '''No model selected.<a href="/predict">Click here to go back.</a>'''
    try:
        model = request.json['model']
    except:
        return '''No model selected.<a href="/predict">Click here to go back.</a>'''

    model_dir = modelPath + model


    app.logger.info("api_input: " + str(model_dir))
    #Get predictions for text
    model_api =serve.get_model_api1(model_dir,sentence)

    return jsonify(model_api)

@app.route('/listdir',methods=['POST','GET'])
def dir_listing():
    """API function that list directories.

    """
    # Joining the base and the requested path
    abs_path = request.json
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return exit(1)

    # Show directory contents
    files = os.listdir(abs_path)
    print(files)
    return jsonify(files)

@app.route('/return-files/')
def return_files_tut():
    """API function that sends predictions file to frontend.

    """
    try:
       return flask.send_file(session['modelDirectory']+"/predictions/"+session.get('currentFile', None),
                               attachment_filename=session.get('currentFile', None),as_attachment=True)
    except Exception as e:
        return str(e)

@app.route('/')
def hello():
    return render_template('home.html')

@app.route('/')
def index():
    return "Index API"

# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally.
    app.run(host='localhost',port=8080, debug=True)