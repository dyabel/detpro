import matplotlib.pyplot as plt
import numpy as np
import cv2
from os.path import join as ospj
import torch
import torch.nn.functional as F


analysis_results_path = 'analysis_results_fcos'

# feature_type = 'feature' # fpn
# feature_type = 'cls_feature' # cls
# feature_type = 'reg_feature' # reg
# feature_type = 'cls' # cls
feature_type = 'reg' # reg

colormap = plt.get_cmap('GnBu')

for image_idx in range(10):

    features = []
    features_similarites = []

    for fid in range(3):
        feature_map_scale1_path = ospj(analysis_results_path, f"image_{image_idx}_{feature_type}_{fid}_scale_1.0.pt")
        feat_scale1 = torch.load(feature_map_scale1_path)
        # feat_scale1 = torch.nn.functional.normalize(feat_scale1, dim=1, p=2)

        feature_map_scale2_path = ospj(analysis_results_path, f"image_{image_idx}_{feature_type}_{fid+1}_scale_2.0.pt")
        feat_scale2 = torch.load(feature_map_scale2_path)
        # feat_scale2 = torch.nn.functional.normalize(feat_scale2, dim=1, p=2)

        try:
            feat_diff = feat_scale1 - feat_scale2
            feat_diff_square = feat_diff * feat_diff
            feat_diff_sum = torch.sum(feat_diff_square, dim=1, keepdim=True)
            # feat_diff_norm = torch.norm(feat_diftf, dim=1, keepdim=True)
            feat_diff_norm = feat_diff_sum / torch.max(feat_diff_sum)
            
            feat_diff_norm = feat_diff_norm.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

            feat_sim = F.cosine_similarity(feat_scale1, feat_scale2, dim=1)
            feat_sim = feat_sim.unsqueeze(1).squeeze(0)
            feat_sim = feat_sim.permute(1, 2, 0).cpu().numpy()

            # feat_heatmap = (colormap(feat_diff_norm) * 2**8).astype(np.uint16)[:,:,:3]
            # feat_heatmap = cv2.cvtColor(feat_heatmap, cv2.COLOR_RGB2BGR)
            features.append(feat_diff_norm)
            features_similarites.append(feat_sim)

        except Exception:
            break

    if len(features) == 3 and len(features_similarites) == 3:
        fig = plt.figure(figsize=(8, 4))

        columns = 3
        rows = 2

        for i, feat in enumerate(features):
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(feat)
        
        for i, sim in enumerate(features_similarites):
            fig.add_subplot(rows, columns, i+1+3)
            plt.imshow(sim)

        plt.tight_layout()
        plt.savefig(ospj(analysis_results_path, f"image_{image_idx}_{feature_type}_diff.png"))
        print(f"image {image_idx} processed.")