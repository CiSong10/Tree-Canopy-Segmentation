"""
Author: Charles Song
Date: 12/2024
Python Version: 3.11
"""


from segmentation import *


run_segmentation(
    "data_name.tif",
    data_dir = "data",
    output_dir="output",
    min_distance=2
    )