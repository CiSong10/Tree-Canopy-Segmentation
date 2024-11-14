import segmentation

segmentation.run_segmentation(
    "Dominion_CHM0.tif",
    output_path="output_Dominion0",
    compactness=0.01,
    min_circularity=0.1
)