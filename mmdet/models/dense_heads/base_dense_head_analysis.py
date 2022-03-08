from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseDenseHeadAnalysis(nn.Module, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self):
        super(BaseDenseHeadAnalysis, self).__init__()

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    @abstractmethod
    def get_bboxes(self, **kwargs):
        """Transform network output for a batch into bbox predictions."""
        pass

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      batch_idx=0,
                      analysis_scale=1.0,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        # outputs_with_features = self(x)
        # outs = outputs_with_features[:2]
        # cls_score_list, bbox_pred_list = outs
        # cls_feat_list, reg_feat_list = outputs_with_features[2:]
        outputs_with_features = self(x)
        outs = outputs_with_features[:3]
        cls_score_list, bbox_pred_list, _ = outs
        cls_feat_list, reg_feat_list = outputs_with_features[3:]
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, batch_idx=batch_idx, analysis_scale=analysis_scale)
        if proposal_cfg is None:
            return losses, cls_score_list, bbox_pred_list, cls_feat_list, reg_feat_list
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list, cls_score_list, bbox_pred_list, cls_feat_list, reg_feat_list
