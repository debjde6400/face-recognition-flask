import cv2, io, base64
import numpy as np
from PIL import Image
import json

def convert_to_cv2_img(base64img):
  b64ImgPrefix = "data:image/png;base64,"
  decodedB64ImgStr = base64.b64decode(base64img[len(b64ImgPrefix):])
  decodedB64Img = Image.open(io.BytesIO(decodedB64ImgStr))
  opencv_img = cv2.cvtColor(np.array(decodedB64Img), cv2.COLOR_BGR2RGB)
  return opencv_img

def detect_unauthorized_objects(base64ImgStr):
   base64img = convert_to_cv2_img(base64ImgStr)

   with open('./models/object_detection_classes_coco.txt', 'r') as f:
      class_names = f.read().split('\n')

   model = cv2.dnn.readNet(model='./models/frozen_inference_graph.pb',
                           config='./models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt',
                           framework='TensorFlow')

   blob = cv2.dnn.blobFromImage(image=base64img, size=(300, 300), mean=(104, 117, 123),
                              swapRB=True)
   model.setInput(blob)
   output = model.forward()

   detected_objects_dict = {}

   for detection in output[0, 0, :, :]:
      confidence = detection[2]

      if confidence > 0.6:
      # get the class id
         class_id = detection[1]
         # map the class id to the class
         class_name = class_names[int(class_id)-1]

         if class_name in detected_objects_dict.keys():
            detected_objects_dict[class_name] += 1

         else:
            detected_objects_dict[class_name] = 1


   detected_objects = detected_objects_dict.keys()
   violations_dict = { "unauthorized_objects": [] }

   for unauthorized_object in ['cell phone', 'laptop']:
      if unauthorized_object in detected_objects:
         violations_dict["unauthorized_objects"].append(unauthorized_object)

   if 'person' in detected_objects:
      violations_dict['person_count'] = detected_objects_dict['person']
   else:
      violations_dict['person_count'] = 0

   violations_response = json.dumps(violations_dict)
   return violations_response
