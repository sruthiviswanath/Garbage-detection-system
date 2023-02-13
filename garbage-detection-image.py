import cv2
import numpy as np

def post_process_stage(iimage, outputs):
      class_ids = []
      confidences_scores = []
      boxes = []
      rows = outputs[0].shape[1]
      image_h, image_w = iimage.shape[:2]
      # Resizing factor.
      x_factor = image_w/ 640
      y_factor =  image_h / 640
      # Iterate through detections.
      ## o/p is a 85-length 1D array having detection. First 4= xywh coordinates of bb. 5th=confidence level. The 6th up to 85th elements are the scores of each class.
      for r in range(rows):
            row = outputs[0][0][r]
            confidences_score =row[4]
            
            # Remove unvalid predictions
            if confidences_score >= 0.45:
                  classes_scores = row[5:]
                  # Get the index of max class score.
                  class_id = np.argmax(classes_scores)
                  #  Continue if the class score is above threshold.
                  if (classes_scores[class_id] > 0.5):
                        confidences_scores.append(confidences_score)
                        class_ids.append(class_id)
                        cx, cy, w, h = row[0], row[1], row[2], row[3]
                        left = int((cx - w/2) * x_factor)
                        top = int((cy - h/2) * y_factor)
                        width = int(w * x_factor)
                        height = int(h * y_factor)
                        box = np.array([left, top, width, height])
                        boxes.append(box)

      indices = cv2.dnn.NMSBoxes(boxes, confidences_scores, 0.45, 0.45)
      for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]             
            # Draw bounding box.             
            cv2.rectangle(iimage, (left, top), (left + width, top + height), (0,255,255), 3)
            # Class label.                      
            label = "{}:{:.2f}".format(classes[class_ids[i]], confidences_scores[i])             
            # Draw label.    
            print(label)

            #read label and give font style,size, etc
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,  0.7, 1)
            dim_size, baseline_dim = text_size[0], text_size[1]
            # give a white background for text
            cv2.rectangle(iimage, (left,top), (left + dim_size[0], top + dim_size[1] + baseline_dim), (255,255,255), cv2.FILLED);
            # write the tabel inside white background
            cv2.putText(iimage, label, (left, top + dim_size[1]),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
      return iimage

if __name__ == '__main__':
      
      classesFile = "classes.txt"
      
      with open("classes.txt" ,'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
      
      inputimg = cv2.imread("testset/test_image4.jpg")
      net = cv2.dnn.readNet("best.onnx")


      iimage_cv = cv2.dnn.blobFromImage(inputimg, 1/255,  (640, 640), [0,0,0], 1, crop=False)
      net.setInput(iimage_cv)
      detections = net.forward(net.getUnconnectedOutLayersNames())
      
      outputimg = post_process_stage(inputimg.copy(), detections)
     
      t, _ = net.getPerfProfile()
      label = 'Inference time: %.2f ms' % (t * 1000.0 /  cv2.getTickFrequency())
      
      cv2.putText(outputimg, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,  (0, 0, 255), 1, cv2.LINE_AA)
      cv2.imshow('Output', outputimg)
      cv2.waitKey(0)