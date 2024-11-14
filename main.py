import segmentation

segmentation.run_segmentation(
    "NEON_D17_SJER_DP3_256000_4106000_CHM.tif",
    output_path="output",
    compactness=0.01,
    min_circularity=0.1
)