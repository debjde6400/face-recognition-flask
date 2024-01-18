import cv2, io, base64
import numpy as np
import mediapipe as mp
from PIL import Image
import json
import tensorflow as tf

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

   print("tui tui: ", violations_dict)

   return violations_dict

def get_face_orientation(input):
   mp_face_mesh = mp.solutions.face_mesh
   face_mesh = mp_face_mesh.FaceMesh(
      min_detection_confidence=0.5, min_tracking_confidence=0.5
   )

   image = cv2.cvtColor(cv2.flip(input, 1), cv2.COLOR_BGR2RGB)

   # To improve performance
   image.flags.writeable = False

   # Get the result
   results = face_mesh.process(image)

   # To improve performance
   image.flags.writeable = True

   # Convert the color space from RGB to BGR
   image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

   img_h, img_w, _ = image.shape
   face_3d = []
   face_2d = []

   if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
         for idx, lm in enumerate(face_landmarks.landmark):
               if (
                  idx == 33
                  or idx == 263
                  or idx == 1
                  or idx == 61
                  or idx == 291
                  or idx == 199
               ):
                  x, y = int(lm.x * img_w), int(lm.y * img_h)

                  # Get the 2D Coordinates
                  face_2d.append([x, y])

                  # Get the 3D Coordinates
                  face_3d.append([x, y, lm.z])

         face_2d = np.array(face_2d, dtype=np.float64)
         face_3d = np.array(face_3d, dtype=np.float64)

         # The camera matrix
         focal_length = 1 * img_w
         cam_matrix = np.array(
               [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
         )

         # The distortion parameters
         dist_matrix = np.zeros((4, 1), dtype=np.float64)

         # Solve PnP
         _, rot_vec, _ = cv2.solvePnP(
               face_3d, face_2d, cam_matrix, dist_matrix
         )

         # Get rotational matrix
         rmat, _ = cv2.Rodrigues(rot_vec)

         # Get angles
         angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

         # Get the y rotation degree
         x = angles[0] * 360
         y = angles[1] * 360

         return { 'x': x, 'y': y }

def get_face_embeddings(face):
   model = tf.keras.applications.ResNet50(weights='imagenet')

   face = cv2.resize(face, (224, 224))
   face = face.astype("float32") / 255.0
   face = np.expand_dims(face, axis=0)

   facenet_embeddings = model.predict(tf.keras.applications.imagenet_utils.preprocess_input(face))
   return facenet_embeddings[0].tolist()

def get_euclidean_distance(vectorA, vectorB):
   l2_normal_A = vectorA / np.sqrt(np.sum(np.multiply(vectorA, vectorA)))
   l2_normal_B = vectorB / np.sqrt(np.sum(np.multiply(vectorB, vectorB)))
   vector_diff = l2_normal_A - l2_normal_B
   euclidean_sqdist = np.sum(
      np.multiply(vector_diff, vector_diff)
   )

   euclidean_distance = np.sqrt(euclidean_sqdist)
   return euclidean_distance

def detect_face(base64ImgStr):
   base64img = convert_to_cv2_img(base64ImgStr)
   face_detection_resdict = { "face_present": "none" }

   net = cv2.dnn.readNetFromCaffe(
    "./models/deploy.prototxt", "./models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
   )

   h, w = base64img.shape[:2]
   blob = cv2.dnn.blobFromImage(
      cv2.resize(base64img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
   )
   net.setInput(blob)
   ref_image_detections = net.forward()

   prominent_faces = ref_image_detections[:, :, np.where(ref_image_detections[0, 0, :, 2] > 0.8), :]

   if prominent_faces.shape[3] > 1:
      face_detection_resdict['face_present'] = "multiple"

   elif prominent_faces.shape[3] == 1:
      face_detection_resdict['face_present'] = "single"
      prominent_faces = prominent_faces.ravel()
      confidence = prominent_faces[2]
      face_detection_resdict['face_confidence'] = float(confidence)

      if confidence >= 0.8:
         box = prominent_faces[3:7] * np.array([w, h, w, h])
         (startX, startY, endX, endY) = box.astype("int")
         face = base64img[startY:endY, startX:endX]

         face_direction = get_face_orientation(base64img)
         print("fc : ", abs(face_direction["x"]) >= 5 and abs(face_direction["x"]) < 20 and abs(face_direction["y"]) < 10)

         if abs(face_direction["x"]) >= 5 and abs(face_direction["x"]) < 20 and abs(face_direction["y"]) < 10:
            face_detection_resdict['face_direction'] = "straight"
            face_detection_resdict['face_embedding'] = get_face_embeddings(face)
         else:
            face_detection_resdict['face_direction'] = "tilted"
            face_detection_resdict['face_embedding'] = None

   return face_detection_resdict

def recognize_face(base64ImgStr, face_embedding):
   unauth_object_violation = detect_unauthorized_objects(base64ImgStr)
   if len(unauth_object_violation["unauthorized_objects"]) > 0 or unauth_object_violation["person_count"] != 1:
      return unauth_object_violation

   base64img = convert_to_cv2_img(base64ImgStr)
   net = cv2.dnn.readNetFromCaffe(
    "./models/deploy.prototxt", "./models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
   )

   h, w = base64img.shape[:2]
   blob = cv2.dnn.blobFromImage(
      cv2.resize(base64img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
   )
   net.setInput(blob)
   preview_image_detections = net.forward()

   prominent_faces = preview_image_detections[:, :, np.where(preview_image_detections[0, 0, :, 2] > 0.8), :]

   if prominent_faces.shape[3] > 1:
      unauth_object_violation["person_count"] = prominent_faces.shape[3]
      return unauth_object_violation

   elif prominent_faces.shape[3] == 1:
      prominent_faces = prominent_faces.ravel()
      print("prominent faces : ", prominent_faces)
      confidence = prominent_faces[2]

      if confidence >= 0.8:
         face_recognition_response = unauth_object_violation
         box = prominent_faces[3:7] * np.array([w, h, w, h])
         (startX, startY, endX, endY) = box.astype("int")
         face = base64img[startY:endY, startX:endX]

         face_direction = get_face_orientation(base64img)
         print("fc : ", face_direction)

         if face_direction and abs(face_direction["x"]) < 25 and abs(face_direction["y"]) < 25:
            face_recognition_response['face_direction'] = 'straight'

            preview_window_embeddings = np.array(get_face_embeddings(face))
            ref_image_embeddings = np.array([ float(x) for x in face_embedding.split(',') ])

            face_recognition_response['euclidean_distance'] = get_euclidean_distance(
               preview_window_embeddings, ref_image_embeddings
            )

         else:
            face_recognition_response['face_direction'] = 'tilted'

         return face_recognition_response

   unauth_object_violation["person_count"] = 0
   return unauth_object_violation
