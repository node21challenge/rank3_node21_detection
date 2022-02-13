import os
from medpy.io import load, header, save
import logging
import shutil
import json
import png
import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm
from shutil import copyfile
import glob
import pandas as pd
import SimpleITK as sitk
import h5py

def clean_path(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    else:
        shutil.rmtree(filepath)
        os.makedirs(filepath)


class NODE_2021_TEST():
    """X-RAY Pulmonary nodules processed to COCO Json format"""

    def __init__(self, input_image, output_path, dataset_name="node2021_test"):
        self.info = {"year": "2021",
                     "version": "1.0",
                     "description": "A Dataset pulmonary nodules' detection on x-ray imaging",
                     "contributor": "Flute XU",
                     "url": "https://node21.grand-challenge.org/Home/",
                     "date_creatd": "2021/11/29"}
        self.licenses = [{"id": 1,
                          "name": "Attribution-4.0-International",
                          "url": "https://creativecommons.org/licenses/by/4.0/"
                          }]
        self.type = "instances"
        self.input_image = input_image
        self.output_path = os.path.join(output_path, dataset_name)
        clean_path(self.output_path)
        self.imId = 0
        self.categories = [{"id": 1, "name": "node", "supercategory": 'Pulmonary_Nodules'}, ]

        images = self.get_image_set(self.input_image)
        json_data = {"info": self.info,
                     "images": images,
                     "licenses": self.licenses,
                     "type": self.type,
                     "annotations": [],
                     "categories": self.categories}

        ann_out_dir = os.path.join(self.output_path, "annotations")
        if not os.path.exists(ann_out_dir): os.makedirs(ann_out_dir)

        with open(os.path.join(ann_out_dir, 'annotations.json'), "w") as jsonfile:
            json.dump(json_data, jsonfile, sort_keys=True, indent=4)

    def get_image_set(self, input_image):
        images = []
        for i in tqdm(range(input_image.shape[0])):
            current_slice = input_image[i, :, :]
            img_output_path = os.path.join(self.output_path, "images")
            if not os.path.exists(img_output_path): os.makedirs(img_output_path)
            filename = 'n' + str(i) + '.h5'
            # hf = h5py.File(os.path.join(img_output_path, filename), 'w')
            # hf.create_dataset('slice', data=current_slice)
            with h5py.File(os.path.join(img_output_path, filename), 'w') as hf:
                hf.create_dataset('slice', data=current_slice)

            images.append({"date_captured": "2021",
                           "file_name": filename,
                           "id": self.imId,
                           "license": 1,
                           "url": "",
                           "height": int(current_slice.shape[0]),
                           "width": int(current_slice.shape[1])
                           })
            self.imId += 1
        return images