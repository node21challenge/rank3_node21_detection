import os
import SimpleITK as sitk
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

def clean_path(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    else:
        shutil.rmtree(filepath)
        os.makedirs(filepath)


class NODE_2021():
    """X-RAY Pulmonary nodules processed to COCO Json format"""

    def __init__(self, input_path, output_path, dataset_name="node2021"):
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
        self.input_path = os.path.join(input_path, 'Images')
        self.output_path = os.path.join(output_path, dataset_name)
        clean_path(self.output_path)
        self.general_meta = pd.read_csv(os.path.join(input_path, 'metadata.csv'),
                                        index_col=0)  # general annotation records
        self.imId = 0
        self.annId = 0
        self.categories = [{"id": 1, "name": "node", "supercategory": 'Pulmonary_Nodules'}, ]

        imlist = sorted(glob.glob(os.path.join(self.input_path, "*.mha")))
        images, annotations = self.get_image_annotation_set(imlist)
        json_data = {"info": self.info,
                     "images": images,
                     "licenses": self.licenses,
                     "type": self.type,
                     "annotations": annotations,
                     "categories": self.categories}

        ann_out_dir = os.path.join(self.output_path, "annotations")
        if not os.path.exists(ann_out_dir): os.makedirs(ann_out_dir)

        with open(os.path.join(ann_out_dir, 'annotations.json'), "w") as jsonfile:
            json.dump(json_data, jsonfile, sort_keys=True, indent=4)

    def get_image_annotation_set(self, image_set):
        images = []
        annotations = []
        #         clean_path(os.path.join(self.output_path, split))
        for patient in tqdm(image_set):
            image_data = sitk.GetArrayFromImage(sitk.ReadImage(patient))
            pixel_spacing = header.get_pixel_spacing(header_meta)
            meta_anns = self.general_meta[self.general_meta.img_name == patient.split('/')[-1]].copy()
            images, annotations = self.get_image_annotation_pairs(image_data, pixel_spacing, meta_anns, annotations, images,
                                                                  patient)
        return images, annotations

    def get_image_annotation_pairs(self, image_data, pixel_spacing, meta_anns, annotations, images, patient):
        ### coco json annotation generation block
        img_output_path = os.path.join(self.output_path, patient)

        # write out imgs
        #         output_path = os.path.join(img_output_dir, patient.replace('mha', 'png'))
        #         with open(output_path, 'wb') as f:
        #             writer = png.Writer(width=image_data.shape[1], height=image_data.shape[0], bitdepth=16, greyscale=True)
        #             zgray2list = image_data.tolist()
        #             writer.write(f, zgray2list)
        img_output_path = os.path.join(self.output_path, "images")
        if not os.path.exists(img_output_path): os.makedirs(img_output_path)
        copyfile(patient, os.path.join(img_output_path, patient.split('/')[-1]))
        # img annotation
        self.imId += 1
        images.append({"date_captured": "2021",
                       "file_name": patient.split('/')[-1],
                       "id": self.imId,
                       "license": 1,
                       "url": "",
                       "height": int(image_data.shape[0]),
                       "width": int(image_data.shape[1]),
                       "spacing": list(pixel_spacing)
                       })

        num_ann = meta_anns.shape[0]
        for i in np.arange(num_ann):
            ann = meta_anns.iloc[i, :]
            if ann.label != 0:
                bbox = float(ann.x), float(ann.y), float(ann.width), float(ann.height)
                area = bbox[-2] * bbox[-1]
                catId = 1
                self.annId += 1
                annotations.append({"area": float(area),
                                    "iscrowd": 0,
                                    "image_id": self.imId,
                                    "bbox": bbox,
                                    "category_id": catId,
                                    "id": self.annId})
        return images, annotations