import matplotlib.pyplot as plt
import numpy as np
import cv2
from os.path import join as ospj

analysis_results_path = 'analysis_results_fcos'

colormap = plt.get_cmap('GnBu')

for image_idx in range(10):
    image_path = ospj(analysis_results_path, f"image_{image_idx}.png")

    image = cv2.imread(image_path)
    features = []
    pos_anchors = []

    for scale in [1.0, 2.0]:
        for fid in range(4):
            feature_map_path = ospj(analysis_results_path, f"image_{image_idx}_feature_{fid}_norm_scale_{scale}.png")
            feat = cv2.imread(feature_map_path, 0)
            # feat_heatmap = (colormap(feat) * 2**8).astype(np.uint16)[:,:,:3]
            # feat_heatmap = cv2.cvtColor(feat_heatmap, cv2.COLOR_RGB2BGR)
            features.append(feat)

            pos_anchor_flag_path = ospj(analysis_results_path, f"image_{image_idx}_feature_{fid}_flatten_anchor_flags_scale_{scale}.png")
            pos_anchor = cv2.imread(pos_anchor_flag_path, 0)
            # pos_anchor = (colormap(pos_anchor) * 2**8).astype(np.uint16)[:,:,:3]
            # pos_anchor_heatmap = cv2.cvtColor(pos_anchor_heatmap, cv2.COLOR_RGB2BGR)
            pos_anchors.append(pos_anchor)


    fig = plt.figure(figsize=(8, 8))
    # fig = plt.figure(figsize=(8, 4))

    columns = 4
    rows = 4

    for i, feat in enumerate(features):
        p = i if i < 4 else 4 + i
        # p = i
        fig.add_subplot(rows, columns, p+1)
        plt.imshow(feat)

    for i, anchor in enumerate(pos_anchors):
        p = 4 + i if i < 4 else 8 + i
        fig.add_subplot(rows, columns, p+1)
        plt.imshow(anchor)

    plt.tight_layout()
    plt.savefig(ospj(analysis_results_path, f"image_{image_idx}_features_anchors.png"))
