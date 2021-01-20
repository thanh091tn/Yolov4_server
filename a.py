from test import Yolo4


model_path = 'lic.h5'
# File anchors cua YOLO
anchors_path = 'yolo4_anchors.txt'
# File danh sach cac class
classes_path = 'custom.names'
score = 0.5
iou = 0.5
model_image_size = (608, 608)
yolo4_model = Yolo4(score, iou, anchors_path, classes_path, model_path)
def gg():
	yolo4_model.detect_licence()
	

if __name__ == '__main__':
	gg()