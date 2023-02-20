# Image segmentation
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Load the model
net = cv2.dnn.readNetFromTensorflow("asset/frozen_inference_graph_coco.pb",
                                    "asset/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

# Store Coco Names in a list
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')

st.set_page_config(page_title='Test Image Segementation')
st.markdown('Mencoba image segmentation menggunakan Masking.')
st.markdown('Masukkan gambar yang disegmentasi:')
uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    height, width, _ = img.shape

    # Create black image
    blank_mask = np.zeros((height, width, 3), np.uint8)
    blank_mask[:] = (0, 0, 0)

    # Create blob from the image
    blob = cv2.dnn.blobFromImage(img, swapRB=True)

    # Detect objects
    net.setInput(blob)
    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    detection_count = boxes.shape[2]
    count = 0
    for i in range(detection_count):
        # Extract information from detection
        box = boxes[0, 0, i]
        class_id = int(box[1])
        score = box[2]
        if score < 0.6:
            continue

        # Extract class name and bounding box coordinates
        class_name = (classNames[class_id])
        x = int(box[3] * width)
        y = int(box[4] * height)
        x2 = int(box[5] * width)
        y2 = int(box[6] * height)

        # Extract the region of interest (ROI) from the black mask
        roi = blank_mask[y: y2, x: x2]
        roi_height, roi_width, _ = roi.shape

        # Get the mask
        mask = masks[i, int(class_id)]
        mask = cv2.resize(mask, (roi_width, roi_height))
        _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

        # Find contours of the mask and fill them with a random color
        contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = np.random.randint(0, 255, 3, dtype='uint8')
        cv2.fillPoly(roi, contours, color.tolist())

        # Draw bounding box and class label
        cv2.rectangle(img, (x, y), (x2, y2), color.tolist(), 2)
        cv2.putText(img, class_name + " " + str(score), (x + 10, y - 6), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (255, 100, 120), thickness=2)

    # Add mask to image with alpha blending
    # alpha is the transparency of the first picture, beta is the transparency of the second picture
    alpha, beta = 1, 0.8
    mask_img = cv2.addWeighted(img, alpha, blank_mask, beta, 0)

    # Display the final image in a matplotlib window
    plt.imshow(cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB))
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
