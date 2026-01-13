import tifffile
import numpy as np
import tifffile
import numpy as np

# --- First pair (cropped images) ---
image_path_crop = "c01_t1707_gt_62.tif"
#com_image_path_crop = "c01_t1707_MU-Lux-CZ_62.tif"
com_image_path_crop = "data/qa_data/BF-C2DL-HSC_with_fused/c01_t1707_MU-Lux-CZ_62.tif"
img_crop = tifffile.imread(image_path_crop)
img2_crop = tifffile.imread(com_image_path_crop)[1] # Segmentation is the second layer

print("--- Processing Cropped Images ---")
print(f"GT unique values: {np.unique(img_crop)}")
print(f"Segmentation unique values: {np.unique(img2_crop)}")

label = 62
mask1_crop = (img_crop == label).astype(np.uint8) * 255
mask2_crop = (img2_crop > 0).astype(np.uint8) * 255 # Assuming any non-zero value is the mask

# Save the masks as images
tifffile.imwrite("gt_label_62_mask_crop.tif", mask1_crop)
tifffile.imwrite("seg_label_62_mask_crop.tif", mask2_crop)

intersection_crop = np.logical_and(mask1_crop, mask2_crop)
union_crop = np.logical_or(mask1_crop, mask2_crop)
jaccard_crop = np.sum(intersection_crop) / np.sum(union_crop) if np.sum(union_crop) > 0 else 0.0
print(f"Jaccard score for label {label} (crop): {jaccard_crop}")
print("Saved gt_label_62_mask_crop.tif and seg_label_62_mask_crop.tif\n")


# --- Second pair (full images) ---
image_path_full = "data/synchronized_data/BF-C2DL-HSC/01_GT/SEG/man_seg1707.tif"
com_image_path_full = "data/synchronized_data/BF-C2DL-HSC/MU-Lux-CZ/01_RES/mask1707.tif"
img_full = tifffile.imread(image_path_full)
img2_full = tifffile.imread(com_image_path_full) # Single layer segmentation mask

print("--- Processing Full Images ---")
print(f"GT unique values: {np.unique(img_full)}")
print(f"Segmentation unique values: {np.unique(img2_full)}")

mask1_full = (img_full == label).astype(np.uint8) * 255
mask2_full = (img2_full == label).astype(np.uint8) * 255

# Save the masks as images
tifffile.imwrite("gt_label_62_mask_full.tif", mask1_full)
tifffile.imwrite("seg_label_62_mask_full.tif", mask2_full)

intersection_full = np.logical_and(mask1_full, mask2_full)
union_full = np.logical_or(mask1_full, mask2_full)
jaccard_full = np.sum(intersection_full) / np.sum(union_full) if np.sum(union_full) > 0 else 0.0
print(f"Jaccard score for label {label} (full): {jaccard_full}")
print("Saved gt_label_62_mask_full.tif and seg_label_62_mask_full.tif")

