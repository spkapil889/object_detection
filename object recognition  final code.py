#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import os
import tensorflow as tf
import cv2

from utils import label_map_util
from utils import visualization_utils as vis_util

from object_detection.protos import string_int_label_map_pb2
from google.protobuf import text_format

from gtts import gTTS 
from tensorflow.python.platform import gfile

#import pytesseract	  
#import pyttsx3		  
#from googletrans import Translator	  

cap = cv2.VideoCapture(0)  

PATH_TO_CKPT = 'C:/Users/spkapil/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('C:/Users/spkapil/AppData/Local/Programs/Python/Python37/Lib/site-packages/tensorflow/models/research/object_detection/data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def =tf.compat.v1.GraphDef()   # -> instead of tf.GraphDef()

    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

def load_labelmap(path):   # -> instead of tf.gfile.GFile()

 
 with tf.compat.v2.io.gfile.GFile(path, 'r') as fid:
    label_map_string = fid.read()
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    try:
        text_format.Merge(label_map_string, label_map)
    except text_format.ParseError:
        label_map.ParseFromString(label_map_string)
        _validate_label_map(label_map)
    return label_map


def convert_label_map_to_categories(label_map,
                                    max_num_classes,
                                    use_display_name=True):
    categories = []
    list_of_ids_already_added = []
    if not label_map:
        label_id_offset = 1
    for item in label_map.item:  
        categories.append({
          'id': item.id,
          'name': item.display_name
      }) 

    return categories
    for item in label_map.item:
        if not 0 < item.id <= max_num_classes:
            logging.info(
                'Ignore item %d since it falls outside of requested '
                'label range.', item.id)
            continue
        if use_display_name and item.HasField('display_name'):
            name = item.display_name
        else:
            name = item.name
        if item.id not in list_of_ids_already_added:
            list_of_ids_already_added.append(item.id)
            categories.append({'id': item.id, 'name': name})
    return categories

def create_category_index(categories):
    category_index = {}
    for cat in categories:
        category_index[cat['id']] = cat
    return category_index


label_map = load_labelmap(PATH_TO_LABELS)
categories = convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = create_category_index(categories)

# Detection
list_of_persons=[]
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:

            ret, image_np = cap.read()
            
            image_np_expanded = np.expand_dims(image_np, axis=0)
           
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            
            objects=[category_index.get(value).get('name') for index,value in enumerate(classes[0]) if scores[0,index] > 0.5]
            threshold = 0.5 
            print(objects)
            for i in range(len(objects)):
                print (objects[i])
                mytext = objects[i]
                p = Translator()
                k = p.translate(mytext,dest='german')
                engine = pyttsx3.init()  # an audio will be played which speaks the test if pyttsx3 recognizes it 
                engine.say(k.pronunciation)							 
                engine.runAndWait() 
                
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True)    

            cv2.imshow('object detection', cv2.resize(image_np, (900,500)))
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            
 
            

                            
            


# In[ ]:


pip install torch


# In[ ]:




