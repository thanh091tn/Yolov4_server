from flask import Flask, request,flash,redirect
from flask_restful import Api, Resource
import base64
from test import Yolo4
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
CORS(app)
api = Api(app)

if __name__ == '__main__':
    app.run(debug=True)

model_path = 'yolo4_weight.h5'
# File anchors cua YOLO
anchors_path = 'yolo4_anchors.txt'
# File danh sach cac class
classes_path = 'yolo.names'
score = 0.5
iou = 0.5
model_image_size = (608, 608)
yolo4_model = Yolo4(score, iou, anchors_path, classes_path, model_path)

class UploadImage(Resource):
    def post(self):
        # json_data = request.json
        # file_upload = json_data['image']
        if 'file' not in request.files:
            flash('No file part')
            #return redirect(request.url)
            #print('allo')
        img = request.files['file']    
        if img:
            # image_decoded_bytes = base64.b64decode(file_upload)
            # buff_image = io.BytesIO(image_decoded_bytes)
            image = Image.open(img)
            result = yolo4_model.detect_image(image, model_image_size=model_image_size)
            return result
    def get(self):
        return {'hello': "from python"}

api.add_resource(UploadImage, "/detect_image")