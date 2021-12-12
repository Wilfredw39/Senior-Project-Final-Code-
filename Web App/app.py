import os
from datetime import datetime
from flask import Flask, request, render_template, send_from_directory
from index import predict


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))  # project abs path


@app.route("/")
def index():
    return render_template("Home.html")


@app.route("/upload_page", methods=["GET"])
def upload_page():
    return render_template("upload.html")


@app.route("/Home", methods=["POST"])
def upload():
    # folder_name = request.form['uploads']
    target = os.path.join(APP_ROOT, 'static/Uploaded_imgs')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    print(request.files.getlist("file"))
    option = request.form.get('optionsPrediction')
    print("Selected Option:: {}".format(option))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        # This is to verify files are supported
        ext = os.path.splitext(filename)[1]
        if (ext == ".jpg") or (ext == ".png"):
            print("File supported moving on...")
        else:
            render_template(
                "Error.html", message="Files uploaded are not supported...")
        savefname = datetime.now().strftime('%Y-%m-%d_%H_%M_%S') + "_MRI"+ext
        # destination = "/".join([target, savefname])
        destination = os.path.join(target, savefname)
        print("Accept incoming file:", filename)
        print("Save it to:", destination)
        upload.save(destination)
        prediction = get_prediction(destination, option)
        print("Prediction: ", prediction["label"])
    return render_template("Results.html", Label=prediction["label"], pre_img_name=prediction["prediction_path"], Description=prediction["description"])


def get_prediction(path, type):
    return predict(path)


if __name__ == "__main__":
    app.run(port=4555, debug=True)
