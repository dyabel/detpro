from ..builder import DETECTORS
from .single_stage_analysis import SingleStageDetectorAnalysis


@DETECTORS.register_module()
class RetinaNetAnalysis(SingleStageDetectorAnalysis):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNetAnalysis, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)
