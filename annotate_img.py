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

# Function to write Pascal VOC annotation
def write_voc_annotation(image_path, img_shape, boxes, texts, output_path):
    with open(output_path, 'w') as file:
        file.write('<annotation>\n')
        file.write(f'\t<folder>{os.path.basename(os.path.dirname(image_path))}</folder>\n')
        file.write(f'\t<filename>{os.path.basename(image_path)}</filename>\n')
        file.write(f'\t<path>{os.path.abspath(image_path)}</path>\n')
        file.write('\t<source>\n\t\t<database>Unknown</database>\n\t</source>\n')

        # Write image size
        height, width, depth = img_shape
        file.write(f'\t<size>\n\t\t<width>{width}</width>\n\t\t<height>{height}</height>\n\t\t<depth>{depth}</depth>\n\t</size>\n')
        file.write('\t<segmented>0</segmented>\n')

        # Write each bounding box and corresponding label
        for (x, y, w, h), entity_name in zip(boxes, texts):
            file.write('\t<object>\n')
            file.write(f'\t\t<name>{entity_name}</name>\n')
            file.write('\t\t<pose>Unspecified</pose>\n')
            file.write('\t\t<truncated>0</truncated>\n')
            file.write('\t\t<difficult>0</difficult>\n')
            file.write(f'\t\t<bndbox>\n\t\t\t<xmin>{x}</xmin>\n\t\t\t<ymin>{y}</ymin>\n')
            file.write(f'\t\t\t<xmax>{x + w}</xmax>\n\t\t\t<ymax>{y + h}</ymax>\n')
            file.write('\t\t</bndbox>\n')
            file.write('\t</object>\n')

        file.write('</annotation>\n')

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
            entity_name = d['text'][i].strip()  # Clean up text
            if entity_name:  # Avoid empty labels
                boxes.append((x, y, width, height))
                texts.append(entity_name)

    # Save the annotation in Pascal VOC format
    annotation_filename = os.path.splitext(os.path.basename(image_path))[0] + '.xml'
    annotation_path = os.path.join(output_folder, annotation_filename)
    write_voc_annotation(image_path, (h, w, d), boxes, texts, annotation_path)

# Iterate over all images in the input folder and annotate
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        annotate_image(image_path, output_folder)
