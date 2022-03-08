import torch
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
import torch.nn as nn
import torch
import clip
import time
import mmcv
from mmcv.ops.roi_align import roi_align
# from pytorch_memlab import profile,MemReporter
import os
from PIL import Image
import numpy as np
# from mmcv.runner import auto_fp16
from .class_name import *
import time
import torch.nn.functional as F
from torch import distributed as dist
from .visualize import visualize_oam_boxes

@HEADS.register_module()
class StandardRoIHeadSaveProposal(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 load_feature=False,
                 text_embedding='lvis_text_embedding.pt',
                 save_feature_dir='./data/lvis_clip_image_proposal_embedding/train',
                 ):
        super(StandardRoIHeadSaveProposal, self).__init__(bbox_roi_extractor=bbox_roi_extractor,
                                              bbox_head=bbox_head,
                                              mask_roi_extractor=mask_roi_extractor,
                                              mask_head=mask_head,
                                              shared_head=shared_head,
                                              train_cfg=train_cfg,
                                              test_cfg=test_cfg,
                                              )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if bbox_head.num_classes == 80:
            self.CLASSES = COCO_CLASSES
            dataset = 'coco'
        elif bbox_head.num_classes == 20:
            self.CLASSES = VOC_CLASSES
            dataset = 'voc'
        else:
            self.CLASSES = LVIS_CLASSES
            dataset = 'lvis'
        self.num_classes = len(self.CLASSES)
        print('num_classes:',self.num_classes)
        if self.num_classes == 1203:
            self.novel_label_ids = lvis_novel_label_ids
            self.base_label_ids = lvis_base_label_ids
            self.novel_label_ids = torch.tensor(self.novel_label_ids, device=device)
            self.novel_index = torch.cat([torch.bincount(self.novel_label_ids),self.novel_label_ids.new_zeros(
                self.num_classes - self.novel_label_ids.max(dim=0)[0])], dim=0).bool()
        elif self.num_classes == 20:
            self.novel_label_ids = voc_novel_label_ids
            self.novel_label_ids = torch.tensor(self.novel_label_ids, device=device)
            self.novel_index = torch.cat(
                [torch.bincount(self.novel_label_ids), self.novel_label_ids.new_zeros(self.num_classes - self.novel_label_ids.max(dim=0)[0])],dim=0).bool()
        self.clip_model, self.preprocess = clip.load('ViT-B/32', device, jit=False)
        self.clip_model.eval()
        self.rank = dist.get_rank()
        # self.reporter = MemReporter(self.clip_model)
        for child in self.clip_model.children():
            for param in child.parameters():
                param.requires_grad = False
        self.text_features_for_classes = []
        self.iters = 0
        self.ensemble = bbox_head.ensemble
        self.load_feature = load_feature
        self.save_feature_dir = save_feature_dir
        print('ensemble:{}'.format(self.ensemble))
        save_path = dataset + '_text_embedding.pt'
        time_start = time.time()
        if os.path.exists(save_path):
            # if False:
            print('load text feature')
            self.text_features_for_classes = torch.load(save_path).to(device)
            # self.text_features_for_classes = torch.load(save_path, map_location={'cuda:0': 'cuda:0', 'cuda:2': 'cuda:0', 'cuda:1': 'cuda:1', 'cuda:3': 'cuda:1'}).to(device)
            # self.text_features_for_classes = torch.load(save_path, map_location={'cuda:0': 'cuda:0', 'cuda:2': 'cuda:0', 'cuda:1': 'cuda:0', 'cuda:3': 'cuda:0'}).to(device)
        else:
            for template in template_list:
                print(template)
                text_features_for_classes = torch.cat([self.clip_model.encode_text(clip.tokenize(template.format(c.split('_(')[0])).to(device)).clone().detach() for c in self.CLASSES])
                self.text_features_for_classes.append(text_features_for_classes)

            self.text_features_for_classes = torch.stack(self.text_features_for_classes).mean(dim=0,keepdim=True)
            torch.save(self.text_features_for_classes,save_path)
        self.text_features_for_classes = self.text_features_for_classes.float()
        self.text_features_for_classes = torch.nn.functional.normalize(self.text_features_for_classes,p=2,dim=-1).squeeze(0)

        self.text_features_for_classes_learned = torch.load(text_embedding).to(self.device)
        # self.text_features_for_classes_learned = torch.load("text_embedding/pos56789_ens.pth").to(self.device)
        self.text_features_for_classes_learned = torch.nn.functional.normalize(
            self.text_features_for_classes_learned,p=2,dim=-1).squeeze(0)
        # reporter.report()
        print('text embedding finished, {} passed'.format(time.time()-time_start))
        self.bg_embedding = nn.Linear(1,512)
        self.projection = nn.Linear(1024,512)
        self.temperature = 0.01
        # self.bg_classifier = nn.Sequential(
        #     nn.Linear(512,256),
        #     nn.ReLU(),
        #     nn.Linear(256,2)
        # )
        if self.ensemble:
            self.projection_for_image = nn.Linear(1024,512)
            nn.init.xavier_uniform_(self.projection_for_image.weight)
            nn.init.constant_(self.projection_for_image.bias, 0)

        nn.init.xavier_uniform_(self.bg_embedding.weight)
        nn.init.constant_(self.bg_embedding.bias, 0)

        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.constant_(self.projection.bias, 0)


    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def bbox_jitter(self, gt, gt_labels, img_meta, N):
        origin = gt.repeat(N, 1)
        delta = torch.rand_like(origin)
        a = torch.Tensor([0.6, 0.6, 0.2, 0.2]).to(delta.device)
        b = torch.Tensor([-.3, -.3, -.1, -.1]).to(delta.device)
        delta = delta * a + b
        
        means = delta.new_tensor(self.bbox_head.bbox_coder.means).view(1, -1)
        stds = delta.new_tensor(self.bbox_head.bbox_coder.stds).view(1, -1)
        delta = (delta - means) / stds

        jitter = self.bbox_head.bbox_coder.decode(origin, delta, max_shape=img_meta['img_shape'])
        
        return jitter, gt_labels.repeat(N)

    def forward_train(self,
                      x,
                      img,
                      img_metas,
                      proposal_list,
                      proposals_pre_computed,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      img_no_normalize=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        # if self.rank == 0:
        #     viz_box, viz_label = self.bbox_jitter(gt_bboxes[0], gt_labels[0], img_metas[0], 4)
        #     visualize_oam_boxes(viz_box, viz_label, img[0], img_metas[0],
        #                         win_name='visualize box base', show=False,
        #                         out_dir='/home/common/zzh/mmdetection/workdirs/viz/', show_score_thr=0)

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                # print("------assign-------")
                # print(gt_bboxes_ignore)
                # print(gt_labels)
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x,img,sampling_results,proposals_pre_computed,
                                                    gt_bboxes, gt_labels,
                                                    img_metas, img_no_normalize)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses


    # @auto_fp16()
    def clip_image_forward(self,img,bboxes):
        # bboxes = bboxes.half()
        #----------------------------------------
        #using roi_align
        cropped_images = roi_align(img,bboxes,(224,224))
        print("clip_image_forward,shape=", cropped_images.shape)
        #--------------------------------------
        image_features = self.clip_model.encode_image(cropped_images)
        return image_features



    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        rois = rois.float()
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        bbox_pred = self.bbox_head(bbox_feats)
        bbox_results = dict(
            bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        region_embeddings = self.bbox_head.forward_embedding(bbox_feats)
        return bbox_results, region_embeddings

    def _bbox_forward_for_image(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        rois = rois.float()
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        region_embeddings = self.bbox_head.forward_embedding_for_image(bbox_feats)

        return None, region_embeddings

    def img2pil2feat(self, img, boxs, name):
        img = np.array(img.detach().cpu()).astype(np.uint8)
        img = Image.fromarray(img)
        
        boxs = torch.dstack([torch.floor(boxs[:,0]),torch.floor(boxs[:,1]),torch.ceil(boxs[:,2]),torch.ceil(boxs[:,3])]).squeeze(0)
        boxs = boxs.detach().cpu().numpy()
        # print(boxs)
        preprocessed = []
        i = 0
        for box in boxs:
            croped = img.crop(box)
            # croped.save(name+f'_pil_{i}.jpg')
            i += 1
            croped = self.preprocess(croped)
            preprocessed.append(croped)
            
        preprocessed = torch.stack(preprocessed).to(self.device)
        features = self.clip_model.encode_image(preprocessed)
        return features
    
    def boxto15(self, bboxes):
        bboxes15 = torch.dstack(
            [1.25 * bboxes[:, 0] - 0.25 * bboxes[:, 2], 1.25 * bboxes[:, 1] - 0.25 * bboxes[:, 3],
            1.25 * bboxes[:, 2] - 0.25 * bboxes[:, 0], 1.25 * bboxes[:, 3] - 0.25 * bboxes[:, 1]]).squeeze(0)
        return bboxes15

    def _bbox_forward_train(self, x, img, sampling_results, proposals_pre_computed, gt_bboxes, gt_labels,
                            img_metas, img_no_normalize):
        """Run forward function and calculate loss for box head in training."""

        rois = bbox2roi([res.bboxes for res in sampling_results])
        input_one = x[0].new_ones(1)
        bg_class_embedding = self.bg_embedding(input_one).reshape(1, 512)
        bg_class_embedding = torch.nn.functional.normalize(bg_class_embedding, p=2, dim=1)
        bbox_results, region_embeddings = self._bbox_forward(x, rois)
        # ----------------------------------------------------------
        # if self.ensemble:
        # num_proposals_per_img = tuple(len(proposal) for proposal in proposals_pre_computed)
        # rois_image = torch.cat(proposals_pre_computed, dim=0)
        # batch_index = torch.cat([x[0].new_full((num_proposals_per_img[i],1),i) for i in range(len(num_proposals_per_img))],0)
        # rois_image = torch.cat([batch_index, rois_image[..., :4]], dim=-1)
        # bboxes = rois_image
        # ------------------------------------------------------------
        # not using precomputed proposals
        # bboxes = rois
        # -------------------------------------------------------------
        # print("bboxes\n", gt_bboxes[0])
        # print("img\n", img.shape)
        # print("img no norm\n", img_no_normalize.shape)
        
        
        assert self.clip_model.visual.input_resolution==224

        for i in range(len(img_metas)):
            res = self.bbox_assigner.assign(proposals_pre_computed[i], gt_bboxes[i], gt_labels=gt_labels[i])
            valid = res.max_overlaps >= 0.1
            proposal = proposals_pre_computed[i][valid]
            label = res.labels[valid]
            iou = res.max_overlaps[valid]

            # proposal, label, iou = proposals_pre_computed[i], res.labels, res.max_overlaps
            

            proposal = torch.cat([proposal, gt_bboxes[i]])
            label = torch.cat([label, gt_labels[i]])
            iou = torch.cat([iou, iou.new_ones(len(gt_bboxes[i]))])

            if len(proposal) > 0:
                proposal15 = self.boxto15(proposal)
                save_path = os.path.join('./testbed/', img_metas[i]['ori_filename'].split('.')[0] + "_crop")
                feats = self.img2pil2feat(img_no_normalize[i], proposal, save_path)
                feats15 = self.img2pil2feat(img_no_normalize[i], proposal15, save_path+"15")

                clip_image_features_ensemble = feats + feats15
                clip_image_features_ensemble = clip_image_features_ensemble.float()
                clip_image_features_ensemble = torch.nn.functional.normalize(clip_image_features_ensemble, p=2, dim=1)

                save_path = os.path.join(self.save_feature_dir, img_metas[0]['ori_filename'].split('.')[0] + '.pth')
                os.makedirs(os.path.split(save_path)[0], exist_ok=True)
                torch.save((clip_image_features_ensemble.cpu(), label.cpu(), iou.cpu()), save_path)
                print("save ", save_path)

        # if self.load_feature:
        #     clip_image_features_ensemble = []
        #     for i in range(len(img_metas)):
        #         save_path = os.path.join('./data/lvis_clip_image_embedding', img_metas[i]['ori_filename'].split('.')[0] + '.pth')
        #         clip_image_features_ensemble.append(torch.load(save_path).to(self.device))
        #     clip_image_features_ensemble = torch.cat(clip_image_features_ensemble, dim=0)
        # else:
        #     clip_image_features = self.clip_image_forward(img, bboxes)
        #     bboxes15 = torch.dstack(
        #         [bboxes[:, 0], 1.25 * bboxes[:, 1] - 0.25 * bboxes[:, 3], 1.25 * bboxes[:, 2] - 0.25 * bboxes[:, 4],
        #         1.25 * bboxes[:, 3] - 0.25 *
        #         bboxes[:, 1], 1.25 * bboxes[:, 4] - 0.25 * bboxes[:, 2]]).squeeze()
        #     clip_image_features15 = self.clip_image_forward(img, bboxes15)
        #     clip_image_features_ensemble = clip_image_features + clip_image_features15

        #     clip_image_features_ensemble = clip_image_features_ensemble.float()
        #     clip_image_features_ensemble = torch.nn.functional.normalize(clip_image_features_ensemble, p=2, dim=1)

        #     for i in range(len(img_metas)):
        #         save_path = os.path.join('./data/lvis_clip_image_proposal_embedding', img_metas[i]['ori_filename'].split('.')[0] + '.pth')
        #         torch.save((clip_image_features_ensemble, gt_labels), save_path)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)

        loss_bbox = self.bbox_head.loss(
            bbox_results['bbox_pred'], rois,
            *bbox_targets)
        # loss_bbox.update(text_cls_loss=text_cls_loss)
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results



    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.bool))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.bool))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results


    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                # proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test_bboxes(self,
                           x,
                           img,
                           img_metas,
                           proposals,
                           # proposals_pre_computed,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """
        # get origin input shape to support onnx dynamic input shape
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        rois = bbox2roi(proposals)

        bbox_results,region_embeddings = self._bbox_forward(x,rois)
        region_embeddings = self.projection(region_embeddings)
        region_embeddings = torch.nn.functional.normalize(region_embeddings,p=2,dim=1)
        input_one = x[0].new_ones(1)
        bg_class_embedding = self.bg_embedding(input_one).unsqueeze(0)
        bg_class_embedding = torch.nn.functional.normalize(bg_class_embedding,p=2,dim=1)
        text_features = torch.cat([self.text_features_for_classes,bg_class_embedding],dim=0)
        # text_features_learned =  torch.cat([self.text_features_for_classes_learned,bg_class_embedding],dim=0)
        text_features_learned =  self.text_features_for_classes_learned
        # text_features_learned = text_features
        #-----------------------------------------------------
        # """
        cls_score_text = region_embeddings@text_features.T
        # cls_score_text = cls_score_text/cls_score_text.std(dim=1).mean()*4.7
        #0.009#0.008#0.007
        cls_score_text = cls_score_text/0.007
        cls_score_text = cls_score_text.softmax(dim=1)
        #--------------------------------------------
        # """
        assert self.ensemble
        if self.ensemble:
            _,region_embeddings_image = self._bbox_forward_for_image(x,rois)
            region_embeddings_image = self.projection_for_image(region_embeddings_image)
            region_embeddings_image = torch.nn.functional.normalize(region_embeddings_image,p=2,dim=1)

            cls_score_image = region_embeddings_image@text_features_learned.T
            # tmp = self.clip_image_forward(img,bboxes)
            # self.reporter.report()
        #0.3#0.65
            # cls_score_image[:,:-1] = cls_score_image[:,:-1]/cls_score_image[:,:-1].std(dim=1).mean()*0.65
        #0.006#0.007
            cls_score_image = cls_score_image/0.007
            cls_score_image[:,-1] = -1e11
            cls_score_image = cls_score_image.softmax(dim=1)
        # """
        #------------------------------------------------
        """
        #using clip to inference
        clip_image_features = self.clip_image_forward(img,rois)
        clip_image_features = torch.nn.functional.normalize(clip_image_features,p=2,dim=1)
        clip_image_features = clip_image_features.type(torch.float)
        cls_score_clip = (clip_image_features @ text_features.T).reshape(clip_image_features.size(0), -1,self.num_classes + 1)
        cls_score_clip = cls_score_clip.max(dim=1)[0]
        cls_score_clip[:,self.novel_label_ids] -= 0.001
        # print('#'*100)
        # print(cls_score_clip[:,self.base_label_ids].mean())
        # print(cls_score_clip[:,self.novel_label_ids].mean())
        #0.2261
        #0.2270
        cls_score_clip[:,:-1] = cls_score_clip[:,:-1]/cls_score_clip[:,:-1].std(dim=1,keepdim=True)*2.6
        cls_score_clip[:,-1] = -1e2
        # cls_score_clip = torch.exp(cls_score_clip-1)
        # cls_score_clip[:,-1] = 1 - cls_score_clip[:,:-1].max(dim=1)[0]
        # cls_score_clip = cls_score_clip/self.temperature
        #0.007#0.006
        # cls_score_clip = cls_score_clip/0.006
        cls_score_clip = cls_score_clip.softmax(dim=1)
        cls_score_image = cls_score_clip
        """
        #--------------------------------------------------
        # """
        a = 0.4
        # if self.ensemble:
        if False:
            # cls_score= torch.where(self.novel_index,cls_score_image**(1-a)*cls_score_text**a,
            #                   cls_score_text**(1-a)*cls_score_image**a)
            cls_score = cls_score_image**(1-a)*cls_score_text**a
        else:
            # cls_score = cls_score_text
            cls_score = cls_score_image
        
        # """
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None,) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels

    def simple_test(self,
                    x,
                    img,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False,
                    **kwargs):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x,img, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        if torch.onnx.is_in_onnx_export():
            if self.with_mask:
                segm_results = self.simple_test_mask(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale)
                return det_bboxes, det_labels, segm_results
            else:
                return det_bboxes, det_labels

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False,**kwargs):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]
