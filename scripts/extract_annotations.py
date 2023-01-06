import os
import numpy as np


root = os.path.dirname(__file__)
data_dir = os.path.join(root, '../data')

DATASET = {'data':[],'labels':{}}

# content - ['35 0.19453125 0.48671875 0.075 0.075','35 0.19453125 0.48671875 0.075 0.075'] list[str]
#  id [x_center, y_center, width, height]
# '35 0.19453125 0.48671875 0.075 0.075'
# filename - name of the annotation file (001_png.rf.d103eb57fd9f9ac246b602a8e3393dc2.txt)
# path - full file path
def _handle_annotations(content, filename, path, ext = ".jpg"):
  filename_split = filename.split(".")

  # handle image info
  d = ['', []]
  img_path = path.replace(".txt", ext)

  d[0] = img_path
  d[1] = []

  # handle annotations
  for line in content:
    line_split = [float(n) for n in line.split(" ")]
    label_id = int(line_split[0])
    annotation = line_split[1:]
    if DATASET["labels"] != None:
      label = DATASET["labels"][label_id]
      annotation.append(label)
    d[1].append(annotation)

  return d


def _handle_labels(content):
  label = {}
  for idx, item in enumerate(content):
    label[idx] = item.strip()
  return label

def _extract_labels():
  labels_path = os.path.join(data_dir, "_darknet.labels")
  labels_content = _read_file(labels_path)
  labels = _handle_labels(labels_content)
  DATASET["labels"] = labels
  return

def _read_file(file_path):
  with open(file_path) as f:
    return f.readlines()

def _extract_annotations():
  for file in os.listdir(data_dir):
    file_path = os.path.join(data_dir, file)
    if file.endswith(".txt"):
      # Handle Annotations
      content = _read_file(file_path)
      data = _handle_annotations(content, file, file_path)
      DATASET["data"].append(data)

def extract():
  _extract_labels()
  _extract_annotations()
  return DATASET
  

# [[image_path, [[bb],[bb],...]], [image_path, [[bb],[bb],...]]]
def main():
  list = extract()
  print(list["data"])

if __name__ == "__main__":
    main()