import json
import shutil
import os
import uuid
from PIL import Image
import argparse

root = os.path.dirname(__file__)
data_dir = os.path.join(root, "../data/t100k")
out_dir = os.path.join(root, "../generated/darknet_2")


def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox["xmax"] + bbox["xmin"]) / 2) / w
    y_center = ((bbox["ymax"] + bbox["ymin"]) / 2) / h
    width = (bbox["xmax"] - bbox["xmin"]) / w
    height = (bbox["ymax"] - bbox["ymin"]) / h
    return [x_center, y_center, width, height]


def make_labels(labels):
    f = open(os.path.join(out_dir, "_darknet.labels"), "w")
    for label in labels:
        f.write("{}\n".format(label))
    f.close()


def make_annotation(content, dest):
    f = open(dest, "w")
    f.write(content)
    f.close()


def move_img(source, dest, size):
    # shutil.copy(source, dest)
    image = Image.open(source)
    new_image = image.resize((size, size))
    new_image.save(dest)


def main(PYTORCH=False):
    global img_out_path, anno_out_path
    f = open(os.path.join(data_dir, 'annotations_all.json'))
    data = json.load(f)

    lables = data["types"]
    images = data["imgs"]

    make_labels(lables)
    print("labels: {}".format(lables))
    for img in images:
        value = images[img]
        path = value["path"]
        id = value["id"]
        objects = value["objects"]

        print("path: {}, id: {}".format(path, id))

        if "other" not in path:
            _uuid = uuid.uuid4().hex
            paths = path.split("/")

            img_in_path = os.path.join(data_dir, path)
            if PYTORCH:
                img_out_path = os.path.join(out_dir, "{}/images/{}-{}.jpg".format(paths[0], paths[1], _uuid))
                anno_out_path = os.path.join(out_dir, "{}/labels/{}-{}.txt".format(paths[0], paths[1], _uuid))
            else:
                p = path.replace(".", "_")
                img_out_path = os.path.join(out_dir, "{}-{}.jpg".format(p, _uuid))
                anno_out_path = os.path.join(out_dir, "{}-{}.txt".format(p, _uuid))

            move_img(img_in_path, img_out_path, 416)

            yolo_content = ""

            for object in objects:
                bbox = object["bbox"]
                category = object["category"]
                category_id = lables.index(category)
                print(
                    "category: {}, category_id: {}, bbox: {}".format(
                        category, category_id, bbox
                    )
                )
                # <object-class> <x> <y> <width> <height>
                yolo = xml_to_yolo_bbox(bbox, 2048, 2048)
                yolo_str = "{} {} {} {} {}\n".format(category_id, yolo[0], yolo[1], yolo[2], yolo[3])
                yolo_content = yolo_content + yolo_str
            print("YOLO {}".format(yolo_content))
            make_annotation(yolo_content, anno_out_path)
        else:
            print("skipping")


if __name__ == "__main__":
    main(False)
