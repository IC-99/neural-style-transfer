import flask
from flask import request, render_template, redirect, flash, url_for
from neural_style_transfer import transfer

NEURAL_STYLE_TRANSFER = flask.Flask(__name__)
NEURAL_STYLE_TRANSFER.secret_key = 'chiave segreta'
NEURAL_STYLE_TRANSFER.config['SESSION_TYPE'] = 'filesystem'

@NEURAL_STYLE_TRANSFER.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET': return render_template('index.html')
    
    content_image = request.files.get('content_image', '')
    style_image = request.files.get('style_image', '')
    epochs = request.form.get('epochs', type=int)
    steps = request.form.get('steps', type=int)
    mode = request.form.get('mode', type=int)

    transfer(content_image, style_image, epochs, steps, mode)
    return render_template('result.html')

@NEURAL_STYLE_TRANSFER.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'GET': return render_template('result.html')
    return render_template('result.html')

if __name__ == '__main__':
    NEURAL_STYLE_TRANSFER.run(debug=True, port=80, use_reloader=True)