from deap import creator, base, tools, algorithms
import numpy as np
import os

# Example folder path
example_folder = "./examples/"

# Example id
ex = 616

# Plot intermediate results
plot_intermediate_results = False

# Random seed
seed = 8910

# Layout indexing
wall_type = 1
floor_type = 2
ceil_type = 3
layout_type_to_string_dict = {}
layout_type_to_string_dict[1] = "wall"
layout_type_to_string_dict[2] = "floor"
layout_type_to_string_dict[3] = "ceil"
floor_wall_ceil = [2, 1,3]

valid_layout_ind_dict = {}
valid_layout_ind_dict[floor_type] = [2]
valid_layout_ind_dict[wall_type] = [1]
valid_layout_ind_dict[ceil_type] = [3]

# MSEG segmentation indices (you might want to consider additional classes (e.g. windows, doors)
floor_indices = [43]
ceiling_indices = [36]
wall_indices = [191]

# Max depth (Too high is not good, due to depth measurement errors at larger depths)
max_depth_value = 8.

# Segmentation confidence thresholds (MSEG confidence for ceiling is low)
seg_confidence_wall_thresh = 0.7
seg_confidence_floor_thresh = 0.7
seg_confidence_ceil_thresh = 0.5

# percentage of pixels that should belong to layout semantic category across the layout plane
val_threshold = 0.4

# Ignore nearby planes ( 0. will work as well)
min_depth = 1.3 # Do not consider planes that are closer than this threshold value

# Plane fitting
ransac_thresh = 0.005
min_plane_size = 2000

#Merging (PlaneRCNN sometimes outputs multiple planes in place of one. Merge them)
merge_parallel_threshold = 3e-1
merge_offset_threshold = .3
merge_line_depth_thresh = 0.2

# Boundary line thickness
boundary_mask_thickness = 0

# Invisible layout planes
camera_height = 1.3

# Small value that determines whether two planes are parallel
par_thresh = 1e-2

# Max. / Min. depth of room corners
max_depth_thresh = 10.
min_depth_thresh = 0.

# Optimization cost term costs
depth_discr_w = 1.
poly_iou_w = 1.
other_poly_iou_w = 1.
empty_poly_cost = 10.

# Iterative adjustment params
min_discr_error = 0.01
plane_depth_inc_thresh = -1e-1
plane_depth_alignment_thresh = 2e-1

# LINE FITTING
ransac_line_thresh = .01
ransac_line_points_n = 20

# Refinement variables
# Max. Number of refinements
refinement_max_iter = 3
discr_threshold = 0.3
