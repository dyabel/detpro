import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base_analysis import BaseDetectorAnalysis


@DETECTORS.register_module()
class SingleStageDetectorAnalysis(BaseDetectorAnalysis):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetectorAnalysis, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(SingleStageDetectorAnalysis, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      rescale=False,
                      batch_idx=0,
                      analysis_scale=1.0):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetectorAnalysis, self).forward_train(img, img_metas)

        save_image(img, f"analysis_results_fcos/image_{batch_idx}.png", normalize=True)
        torch.save(gt_bboxes, f"analysis_results_fcos/image_{batch_idx}_gt_bboxes.pt")
        torch.save(gt_labels, f"analysis_results_fcos/image_{batch_idx}_gt_labels.pt")
        # FCOS, not rgb
        x = self.extract_feat(img)

        for i, xi in enumerate(x):
            # xi_normalize = F.normalize(xi, p=2, dim=1)
            xi_norm = torch.norm(xi, dim=1, keepdim=True)
            # xi_sum = torch.sum(xi_norm, dim=1, keepdim=True)
            xi_norm_scale = xi_norm / torch.max(xi_norm)
            save_image(xi_norm_scale, f"analysis_results_fcos/image_{batch_idx}_feature_{i}_norm_scale_{analysis_scale}.png")
            torch.save(xi, f"analysis_results_fcos/image_{batch_idx}_feature_{i}_scale_{analysis_scale}.pt")

        bbox_head_outputs = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                                         gt_labels, gt_bboxes_ignore, batch_idx=batch_idx, analysis_scale=analysis_scale)
        cls_score_list, bbox_pred_list, cls_feat_list, reg_feat_list = bbox_head_outputs[-4:]
        losses = bbox_head_outputs[0]

        for i, (cls, reg, cls_feat, reg_feat) in enumerate(zip(cls_score_list, bbox_pred_list, cls_feat_list, reg_feat_list)):
            torch.save(cls, f"analysis_results_fcos/image_{batch_idx}_cls_{i}_scale_{analysis_scale}.pt")
            torch.save(reg, f"analysis_results_fcos/image_{batch_idx}_reg_{i}_scale_{analysis_scale}.pt")
            torch.save(cls_feat, f"analysis_results_fcos/image_{batch_idx}_cls_feature_{i}_scale_{analysis_scale}.pt")
            torch.save(reg_feat, f"analysis_results_fcos/image_{batch_idx}_reg_feature_{i}_scale_{analysis_scale}.pt")

        test_result = self.simple_test(img, img_metas, rescale, batch_idx)
        # return losses
        # For analysis, we return the test_result to prevent bugs in tools/test_analysis.py
        return test_result

    def simple_test(self, img, img_metas, rescale=False, batch_idx=0):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outputs = self.bbox_head(x)
        # outs = outputs[:2]  # Retina
        outs = outputs[:3]  # FCOS
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas=img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        return [self.bbox_head.aug_test(feats, img_metas, rescale=rescale)]
