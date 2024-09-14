import pytesseract
from pytesseract import Output
import cv2
import os


pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Define input and output folders
input_folder = './images'
output_folder = './annotations'

# Create the annotations folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define a mapping for entity_name to class_id
class_mapping = {
    'weight': 0,
    'height': 1,
    'width': 2,
    # Add more entity classes as needed
}

# Function to write YOLO format annotations
def write_yolo_annotation(image_path, img_shape, boxes, texts, output_path):
    height, width, _ = img_shape

    with open(output_path, 'w') as file:
        for (x, y, w, h), entity_name in zip(boxes, texts):
            if entity_name not in class_mapping:
                continue  # Skip unrecognized entities

            # Normalize coordinates for YOLO
            class_id = class_mapping[entity_name]
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            norm_width = w / width
            norm_height = h / height

            # Ensure bounding box has valid dimensions
            if norm_width <= 0 or norm_height <= 0:
                print(f"Invalid bounding box for {entity_name}: {x}, {y}, {w}, {h}")
                continue

            # Write the annotation in YOLO format
            file.write(f'{class_id} {x_center} {y_center} {norm_width} {norm_height}\n')
            print(f"Written: {class_id}, {x_center}, {y_center}, {norm_width}, {norm_height}")

# Function to annotate images with Tesseract OCR
def annotate_image(image_path, output_path):
    # Read the image
    img = cv2.imread(image_path)
    h, w, d = img.shape

    # Perform OCR and extract text and bounding box data
    d = pytesseract.image_to_data(img, output_type=Output.DICT)

    boxes = []
    texts = []

    # Filter for confident text detections
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:  # Confidence threshold to filter weak detections
            (x, y, width, height) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            entity_name = d['text'][i].strip().lower()  # Clean and normalize text to lowercase
            
            if entity_name in class_mapping:
                boxes.append((x, y, width, height))
                texts.append(entity_name)
                print(f"Detected: {entity_name} at {x}, {y}, {width}, {height} (Confidence: {d['conf'][i]})")

    if not boxes:
        print(f"No relevant text detected in {image_path}")

    # Save the annotation in YOLO format
    annotation_filename = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
    annotation_path = os.path.join(output_folder, annotation_filename)
    write_yolo_annotation(image_path, (h, w, d), boxes, texts, annotation_path)

# Iterate over all images in the input folder and annotate
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        print(f"Processing {image_path}")
        annotate_image(image_path, output_folder)
