import os
import shutil

import fitz

from mlutils.data.filter import filter_images_by_dimension
from mlutils.image.cv_helper import apply_bbox_padding

target_20 = './data/target_20'
source_20 = './data/source_20'
pdf_file = './data/pdf/01ArtisanBilingual(01_10_2022).pdf'

class TestDataSplitDataset():
    def test_filter_dataset_by_dimension_valid(self):
        filter_images_by_dimension(source_20, target_20, 150, 150)
        assert sum(len(files) for _, _, files in os.walk(target_20)) == 6
        if os.path.exists(target_20):
            shutil.rmtree(target_20)

    def test_filter_dataset_by_dimension_valid_2(self):
        filter_images_by_dimension(source_20, target_20, 0, 0)
        assert sum(len(files) for _, _, files in os.walk(target_20)) == 10
        if os.path.exists(target_20):
            shutil.rmtree(target_20)

    def test_apply_bbox_padding(self):
        doc = fitz.open(pdf_file)
        page_dime = doc[0].cropbox
        x0, y0, x1, y1 = (0.01, 0.01, 0.01, 0.01)
        x = apply_bbox_padding(page_dim=page_dime,
                               input_bbox=(0.25, 0.5, 0.5, 0.75),
                               x0_pad=x0, y0_pad=y0, x1_pad=x1, y1_pad=y1)
        result = (6.37, 8.42, 6.62, 8.67)
        assert x == result
