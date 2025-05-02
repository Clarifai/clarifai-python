import glob
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from typing import DefaultDict, Dict, List

from PIL import Image
from tqdm import tqdm

from clarifai.datasets.upload.base import ClarifaiDataLoader

from ..features import VisualDetectionFeatures


class xviewDetectionDataLoader(ClarifaiDataLoader):
    """xview Image Detection Dataset"""

    xview_concepts = [
        'Fixed-wing Aircraft',
        'Small Aircraft',
        'Cargo Plane',
        'Helicopter',
        'Passenger Vehicle',
        'Small Car',
        'Bus',
        'Pickup Truck',
        'Utility Truck',
        'Truck',
        'Cargo Truck',
        'Truck w-Box',
        'Truck Tractor',
        'Trailer',
        'Truck w-Flatbed',
        'Truck w-Liquid',
        'Crane Truck',
        'Railway Vehicle',
        'Passenger Car',
        'Cargo Car',
        'Flat Car',
        'Tank car',
        'Locomotive',
        'Maritime Vessel',
        'Motorboat',
        'Sailboat',
        'Tugboat',
        'Barge',
        'Fishing Vessel',
        'Ferry',
        'Yacht',
        'Container Ship',
        'Oil Tanker',
        'Engineering Vehicle',
        'Tower crane',
        'Container Crane',
        'Reach Stacker',
        'Straddle Carrier',
        'Mobile Crane',
        'Dump Truck',
        'Haul Truck',
        'Scraper-Tractor',
        'Front loader-Bulldozer',
        'Excavator',
        'Cement Mixer',
        'Ground Grader',
        'Hut-Tent',
        'Shed',
        'Building',
        'Aircraft Hangar',
        'Damaged Building',
        'Facility',
        'Construction Site',
        'Vehicle Lot',
        'Helipad',
        'Storage Tank',
        'Shipping container lot',
        'Shipping Container',
        'Pylon',
        'Tower',
    ]

    def __init__(self, data_dir) -> None:
        """Initialize and Compress xview dataset.
        Args:
        data_dir: the local dataset directory.
        """

        self.data_dir = data_dir
        self.img_dir = os.path.join(self.data_dir, "train_images")
        self.img_comp_dir = os.path.join(self.data_dir, "train_images_comp")
        self.label_file = os.path.join(self.data_dir, "xview_train.geojson")

        self.preprocess()
        self.all_data = self.xview_data_parser()

        self.load_data()

    @property
    def task(self):
        return "visual_detection"

    def compress_tiff(self, img_path: str) -> None:
        """Compress tiff image"""
        img_comp_path = os.path.join(self.img_comp_dir, os.path.basename(img_path))
        img_arr = Image.open(img_path)
        img_arr.save(img_comp_path, 'TIFF', compression='tiff_deflate')

    def preprocess(self):
        """Compress the tiff images to comply with clarifai grpc image encoding limit(<20MB) Uses ADOBE_DEFLATE compression algorithm"""
        all_img_ids = glob.glob(os.path.join(self.img_dir, "*.tif"))

        if not os.path.exists(self.img_comp_dir):
            os.mkdir(self.img_comp_dir)

        num_workers = cpu_count()
        futures = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(total=len(all_img_ids), desc="compressing ...") as progress:
                for img_path in all_img_ids:
                    future = executor.submit(self.compress_tiff, img_path)
                    future.add_done_callback(lambda _: progress.update())
                    futures.append(future)

                results = []
                for future in futures:
                    result = future.result()
                    results.append(result)

    def xview_classes2indices(self, classes: List) -> List:
        """remap xview classes 11-94 to 0-59"""
        indices = [
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            0,
            1,
            2,
            -1,
            3,
            -1,
            4,
            5,
            6,
            7,
            8,
            -1,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            -1,
            -1,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            -1,
            23,
            24,
            25,
            -1,
            26,
            27,
            -1,
            28,
            -1,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            -1,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            -1,
            -1,
            -1,
            -1,
            46,
            47,
            48,
            49,
            -1,
            50,
            51,
            -1,
            52,
            -1,
            -1,
            -1,
            53,
            54,
            -1,
            55,
            -1,
            -1,
            56,
            -1,
            57,
            -1,
            58,
            59,
        ]
        return [indices[int(c)] for c in classes]

    def xview_indices2concepts(self, indices: List) -> List:
        """remap classes to concept names"""
        return [self.xview_concepts[i] for i in indices]

    def xview_data_parser(self) -> DefaultDict[str, Dict[List, List]]:
        """Parse geojson data into nested dict of imageid w.r.t bounding boxes, concepts"""
        all_data = defaultdict(lambda: dict(bboxes=[], concepts=[]))

        with open(self.label_file) as f:
            geojson_data = json.loads(f.read())

        for feature in tqdm(
            geojson_data['features'],
            total=len(geojson_data['features']),
            desc="Parsing geojson data",
        ):
            image_id = feature['properties']['image_id'].split(".")[0]
            xview_classid = feature['properties']['type_id']
            bbox = list(map(int, feature['properties']['bounds_imcoords'].split(",")))
            concept = self.xview_indices2concepts(self.xview_classes2indices([xview_classid]))

            all_data[image_id]['bboxes'].append(bbox)
            all_data[image_id]['concepts'].append(concept[0])

        return all_data

    def load_data(self):
        """Load image paths"""
        self.image_paths = []
        all_img_ids = glob.glob(os.path.join(self.img_comp_dir, "*.tif"))
        self.image_paths = all_img_ids

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """Get dataset for a given index.
        Returns:
            VisualDetectionFeature type.
        """
        _id = os.path.splitext(os.path.basename(self.image_paths[index]))[0]
        image_path = self.image_paths[index]

        image = Image.open(image_path)
        image_width, image_height = image.size
        annots = []
        class_names = []
        for bbox, concept in zip(self.all_data[_id]['bboxes'], self.all_data[_id]['concepts']):
            x_min = max(min(bbox[0] / image_width, 1.0), 0.0)  # left_col
            y_min = max(min(bbox[1] / image_height, 1.0), 0.0)  # top_row
            x_max = max(min(bbox[2] / image_width, 1.0), 0.0)  # right_col
            y_max = max(min(bbox[3] / image_height, 1.0), 0.0)  # bottom_row
            if (x_min >= x_max) or (y_min >= y_max):
                continue
            annots.append([x for x in [x_min, y_min, x_max, y_max]])
            class_names.append(concept)

        assert len(class_names) == len(annots), (
            f"Num classes must match num bbox annotations\
        for a single image. Found {len(class_names)} classes and {len(annots)} bboxes."
        )

        return VisualDetectionFeatures(image_path, class_names, annots, id=_id)
