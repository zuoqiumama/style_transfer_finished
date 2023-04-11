"""
@FileName：upload_img.py
@Description：a server that show a web pag to let user upload content image
@Author：zuoqiumama
@Time：2023/4/6 15:07
"""

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '../content_img'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('res', result='success'))

    return redirect(url_for('res', result='false'))


@app.route('/res/<result>')
def res(result):
    if result == 'success':
        message = 'Upload success!'
    else:
        message = 'Upload failed, please try again.'

    return render_template('res.html', message=message)


app.config['DOWNLOAD_FOLDER'] = 'res_img'


@app.route("/download/<img_name>")
def download_form(img_name):
    app.config['DOWNLOAD_IMG'] = img_name
    image_url = url_for('static', filename='res_img/'+img_name+'.jpg')
    return render_template('download.html', img_path=image_url)


@app.route('/d', methods=['POST'])
def d():
    image_path = os.path.join(app.static_folder, 'res_img/'+app.config['DOWNLOAD_IMG']+'.jpg')
    return send_from_directory(app.static_folder, 'res_img/'+app.config['DOWNLOAD_IMG']+'.jpg', as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
