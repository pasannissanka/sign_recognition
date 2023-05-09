from datetime import datetime, timezone
import cv2
import shutil
import os


def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]


def save_to_files(transformed_arr, path, LABELS):
    for _transformed in transformed_arr:
        _id = datetime.now(timezone.utc).timestamp()
        _img = _transformed["image"]
        _img = cv2.cvtColor(_img, cv2.COLOR_RGB2BGR)

        _labels = _transformed["bboxes"]
        _content = ""
        with open(path + "/" + str(_id) + ".txt", 'x') as f:
            for _label in _labels:
                f.write(
                    "{} {} {} {} {}".format(get_keys_from_value(LABELS, _label[4])[0], _label[0], _label[1], _label[2],
                                            _label[3]))
                f.write("\n")
        cv2.imwrite(path + "/" + str(_id) + ".jpg", _img)


def move_files(data, output_dir, datatype):
    for d in data.values:
        shutil.copy(d[1], os.path.join(output_dir, datatype))
        shutil.copy(d[2], os.path.join(output_dir, datatype))
