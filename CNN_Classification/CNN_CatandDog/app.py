from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import os

app = Flask(__name__)

dic = {0 : 'Cat', 1 : 'Dog'}

UPLOAT_FOLDER = "uploads"
if not os.path.exists(UPLOAT_FOLDER):
    os.makedirs(UPLOAT_FOLDER)

model = load_model('model.h5')

model.make_predict_function()

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(100,100))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 100,100,3)
	p = model.predict_classes(i)
	return dic[p[0]]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/submit", methods=["GET", "POST"])
def get_output():
    if request.method == "POST":
        if "my_image" not in request.files:
            return render_template("index.html", prediction="No file selected")

        img = request.files["my_image"]
        if img.filename == "":
            return render_template("index.html", prediction="No file selected")

        img_path = os.path.join(UPLOAT_FOLDER, img.filename)
        img.save(img_path)
        p = predict_label(img_path)

        os.remove(img_path)

        return render_template("index.html", prediction=p, img_filename=img_path)

    return render_template("index.html")


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)