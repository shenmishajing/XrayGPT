import json
import os

from PIL import Image

from xraygpt.datasets.datasets.caption_datasets import CaptionDataset


class IUXrayDataset(CaptionDataset):
    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            for line in open(ann_path).readlines():
                self.annotation.append(json.loads(line))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def __getitem__(self, index):
        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = ann["question"].replace("<image>", "").strip()

        return {
            "image": image,
            "caption": caption,
            "image_id": ann["image"],
        }
