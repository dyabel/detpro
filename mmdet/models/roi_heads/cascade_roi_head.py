from re import S
import torch
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
import torch.nn as nn
import torch
import clip
import time
from mmcv.ops.roi_align import roi_align
# from pytorch_memlab import profile,MemReporter
import os
# from PIL import Image
# from mmcv.runner import auto_fp16
from .class_name import *
import time
import torch.nn.functional as F
from torch import distributed as dist
from .visualize import visualize_oam_boxes
from .zip import ZipBackend
import io
import mmcv
from torchvision.transforms import ToPILImage
import numpy as np
import os.path as osp
from PIL import Image
import random
from lvis import LVIS
from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)
from multiprocessing import Process
from tqdm import tqdm

@HEADS.register_module()
class CascadeRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 load_feature=True,
                 use_clip_inference=False,
                 kd_weight = 256,
                 fixed_lambda=None,
                 prompt_path=None,
                 coco_setting=False,
                 fix_bg=False,
                 feature_path='data/lvis_clip_image_embedding.zip'
                 ):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, \
            'Shared head is not supported in Cascade RCNN anymore'
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        super(CascadeRoIHead, self).__init__(bbox_roi_extractor=bbox_roi_extractor,
                                              bbox_head=bbox_head,
                                              mask_roi_extractor=mask_roi_extractor,
                                              mask_head=mask_head,
                                              shared_head=shared_head,
                                              train_cfg=train_cfg,
                                              test_cfg=test_cfg,
                                              )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if bbox_head[0].num_classes == 80:
            self.CLASSES = COCO_CLASSES
            dataset = 'coco'
        elif bbox_head[0].num_classes == 20:
            self.CLASSES = VOC_CLASSES
            dataset = 'voc'
        elif bbox_head[0].num_classes == 1203:
            self.CLASSES = LVIS_CLASSES
            dataset = 'lvis'
        elif bbox_head[0].num_classes == 365:
            self.CLASSES = Object365_CLASSES
            dataset = 'objects365'
        self.num_classes = len(self.CLASSES)
        print('num_classes:',self.num_classes)
        if self.num_classes == 1203:
            self.base_label_ids = torch.tensor(lvis_base_label_ids, device=device)
            self.novel_label_ids = torch.tensor(lvis_novel_label_ids, device=device)
            self.novel_index = F.pad(torch.bincount(self.novel_label_ids),(0,self.num_classes-self.novel_label_ids.max())).bool()
        elif self.num_classes == 20:
            self.novel_label_ids = torch.tensor(voc_novel_label_ids, device=device)
            self.novel_index = F.pad(torch.bincount(self.novel_label_ids),(0,self.num_classes-self.novel_label_ids.max())).bool()
        elif self.num_classes == 80:
            # self.base_label_ids = torch.tensor(coco_base_label_ids, device=device)
            self.novel_label_ids = torch.tensor(coco_unseen_ids_train, device=device)
            self.unseen_label_ids_test = torch.tensor(coco_unseen_ids_test, device=device)

            self.novel_index = F.pad(torch.bincount(self.novel_label_ids),(0,self.num_classes-self.novel_label_ids.max())).bool()
        self.rank = dist.get_rank()
        self.load_feature = load_feature
        self.use_clip_inference = use_clip_inference
        self.kd_weight = kd_weight
        self.fixed_lambda = fixed_lambda
        self.coco_setting = coco_setting
        self.fix_bg = fix_bg
        print('load_feature',load_feature)
        print('use_clip_inference',use_clip_inference)
        print('fixed_lambda',fixed_lambda)
        print('prompt path',prompt_path)
        self.coco_setting = coco_setting
        # self.reporter = MemReporter(self.clip_model)
        self.clip_model, self.preprocess = clip.load('ViT-B/32', device)
        self.clip_model.eval()
        for child in self.clip_model.children():
            for param in child.parameters():
                param.requires_grad = False
        if not self.load_feature:
            self.clip_model, self.preprocess = clip.load('ViT-B/32', device)
            self.clip_model.eval()
            for child in self.clip_model.children():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            time_start = time.time()
            self.zipfile = ZipBackend(feature_path)
            print('load zip:',time.time()-time_start)
        self.text_features_for_classes = []
        self.iters = 0
        self.ensemble = bbox_head[0].ensemble
        print('ensemble:{}'.format(self.ensemble))
        if prompt_path is not None:
            save_path = prompt_path
        print('load:',save_path)
        time_start = time.time()
        if osp.exists(save_path):
        # if False:
            if not self.fix_bg:
                self.text_features_for_classes = torch.load(save_path).to(device).squeeze()[:self.num_classes]
            else:
                self.text_features_for_classes = torch.load(save_path).to(device).squeeze()
                print(self.text_features_for_classes.shape)
        else:
            self.clip_model, self.preprocess = clip.load('ViT-B/32', device)
            self.clip_model.eval()
            for child in self.clip_model.children():
                for param in child.parameters():
                    param.requires_grad = False
            for template in tqdm(template_list):
                print(template)
                text_features_for_classes = torch.cat([self.clip_model.encode_text(clip.tokenize(template.format(c)).to(device)).detach() for c in self.CLASSES])
                self.text_features_for_classes.append(F.normalize(text_features_for_classes,dim=-1))

            self.text_features_for_classes = torch.stack(self.text_features_for_classes).mean(dim=0)
            torch.save(self.text_features_for_classes.detach().cpu(),save_path)
        self.text_features_for_classes = self.text_features_for_classes.float()
        self.text_features_for_classes = F.normalize(self.text_features_for_classes,dim=-1)
        print('text embedding finished, {} passed'.format(time.time()-time_start))
        print(self.text_features_for_classes.shape)
        # reporter.report()
        self.proposals = mmcv.load('data/lvis_v1/proposals/rpn_r101_fpn_lvis_train.pkl')
        coco = LVIS('data/lvis_v1/annotations/lvis_v1_train.json')
        img_ids = coco.get_img_ids()
        self.file_idxs = dict()
        for i,id in enumerate(img_ids):
            info = coco.load_imgs([id])[0]
            filename = info['coco_url'].replace(
            'http://images.cocodataset.org/', '')
            self.file_idxs[filename] = i
        if not self.fix_bg:
            self.bg_embedding = nn.Linear(1,512)
            nn.init.xavier_uniform_(self.bg_embedding.weight)
            nn.init.constant_(self.bg_embedding.bias, 0)
        # self.projection = nn.Linear(1024,512)
        self.temperature = 0.01
        self.accuracy_align = []
        self.accuracy = []
        # self.trans_to_pil = ToPILImage()
        self.color_type = 'color'
        self.file_client = mmcv.FileClient(backend='disk')
            # nn.init.xavier_uniform_(self.projection_for_image.weight)
            # nn.init.constant_(self.projection_for_image.bias, 0)



        # nn.init.xavier_uniform_(self.projection.weight)
        # nn.init.constant_(self.projection.bias, 0)


    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)
    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict): Config of box roi extractor.
            bbox_head (dict): Config of box in box head.
        """
        self.bbox_roi_extractor = nn.ModuleList()
        self.bbox_head = nn.ModuleList()
        self.projection = nn.ModuleList()
        if bbox_head[0].ensemble:
            self.projection_for_image = nn.ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            self.bbox_head.append(build_head(head))
            self.projection.append(nn.Linear(1024,512))
            if bbox_head[0].ensemble:
                self.projection_for_image.append(nn.Linear(1024,512))

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            mask_head (dict): Config of mask in mask head.
        """
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(build_head(head))
        if mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.mask_roi_extractor = nn.ModuleList()
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [
                    mask_roi_extractor for _ in range(self.num_stages)
                ]
            assert len(mask_roi_extractor) == self.num_stages
            for roi_extractor in mask_roi_extractor:
                self.mask_roi_extractor.append(
                    build_roi_extractor(roi_extractor))
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor

    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage."""
        self.bbox_assigner = []
        self.bbox_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.bbox_assigner.append(
                    build_assigner(rcnn_train_cfg.assigner))
                self.current_stage = idx
                self.bbox_sampler.append(
                    build_sampler(rcnn_train_cfg.sampler, context=self))


    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        for i in range(self.num_stages):
            if self.with_bbox:
                self.bbox_roi_extractor[i].init_weights()
                self.bbox_head[i].init_weights()
            if self.with_mask:
                if not self.share_roi_extractor:
                    self.mask_roi_extractor[i].init_weights()
                self.mask_head[i].init_weights()

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                outs = outs + (bbox_results['cls_score'],
                               bbox_results['bbox_pred'])
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            for i in range(self.num_stages):
                mask_results = self._mask_forward(i, x, mask_rois)
                outs = outs + (mask_results['mask_pred'], )
        return outs

    
    def is_main_process(self): 
        return self.rank == 0

    def forward_train(self,
                      x,
                      img,
                      img_no_normalize,
                      img_metas,
                      proposal_list,
                      proposals_pre_computed,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
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
        # assign gts and sample proposals
        losses = dict()
        for j in range(self.num_stages):
            self.current_stage = j
            rcnn_train_cfg = self.train_cfg[j]
            lw = self.stage_loss_weights[j]

            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[j]
                bbox_sampler = self.bbox_sampler[j]
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]
                sampling_results = []
                for i in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                        gt_labels[i])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[i],
                        gt_bboxes[i],
                        gt_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = self._bbox_forward_train(j,x,img,img_no_normalize,sampling_results,proposals_pre_computed,
                                                    gt_bboxes, gt_labels,
                                                    img_metas,bbox_assigner,rcnn_train_cfg)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{j}.{name}'] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                mask_results = self._mask_forward_train(
                j, x, sampling_results, gt_masks, rcnn_train_cfg,
                    bbox_results['bbox_feats'])
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{j}.{name}'] = (
                        value * lw if 'loss' in name else value)

            # refine bboxes
            if j < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                # bbox_targets is a tuple
                roi_labels = bbox_results['bbox_targets'][0]
                with torch.no_grad():
                    roi_labels = torch.where(
                        roi_labels == self.bbox_head[j].num_classes,
                        bbox_results['cls_score'][:, :-1].argmax(1),
                        roi_labels)
                    proposal_list = self.bbox_head[j].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)
                losses.update(bbox_results['loss_bbox'])


        return losses

    def clip_image_forward_align(self,img,bboxes,num_proposals_per_img,flag=False):
        cropped_images = roi_align(img,bboxes,(224,224))
        image_features = self.clip_model.encode_image(cropped_images)
        return image_features

    # @auto_fp16()
    def clip_image_forward(self,img_metas,bboxes,num_proposals_per_img,flag=False):
        imgs = []
        bboxes = list(bboxes.clone().split(num_proposals_per_img))
        scale_factors = tuple(img_meta['scale_factor'] for img_meta in img_metas)
        for i,img_meta in enumerate(img_metas):
            img_bytes = self.file_client.get(img_meta['filename'])
            buff = io.BytesIO(img_bytes)
            im = Image.open(buff)
            imgs.append(im)
            if img_meta['flip']:
                w = img_meta['img_shape'][1]
                flipped = bboxes[i].clone()
                flipped[..., 1::4] = w - bboxes[i][..., 3::4]
                flipped[..., 3::4] = w - bboxes[i][..., 1::4]
                bboxes[i] = flipped
        # if self.rank == 0:
        #     print(img_metas[0],imgs[0].size)
        #     print(bboxes[0][:,1:]/bboxes[0].new_tensor(scale_factors[0]),self.proposals[self.file_idxs[img_metas[0]['ori_filename']]][0])
        cropped_images = []
        for img_id,bbox in enumerate(bboxes):
            bbox_raw = bbox[:,1:]
            bbox_raw /= bbox_raw.new_tensor(scale_factors[img_id])
            img_shape = imgs[img_id].size
            # bbox = bbox_raw
            # bbox = torch.dstack([torch.floor(bbox_raw[:,0]),torch.floor(bbox_raw[:,1]),torch.ceil(bbox_raw[:,2]),torch.ceil(bbox_raw[:,3])]).squeeze(0)
            bbox = torch.dstack([torch.floor(bbox_raw[:,0]-0.001),torch.floor(bbox_raw[:,1]-0.001),torch.ceil(bbox_raw[:,2]+0.001),torch.ceil(bbox_raw[:,3]+0.001)]).squeeze(0)
            bbox[:,[0,2]].clamp_(min=0,max=img_shape[0])
            bbox[:,[1,3]].clamp_(min=0,max=img_shape[1])
            bbox = bbox.detach().cpu().numpy()
            # bbox = np.dstack([np.floor(bbox_raw[:,0]),np.floor(bbox_raw[:,1]),np.ceil(bbox_raw[:,2]),np.ceil(bbox_raw[:,3])]).squeeze(0)
            cnt = -1
            for box in bbox:
                cnt += 1
                cropped_image = imgs[img_id].crop(box)
                # if flag:
                    # cropped_image.save('workdirs/output_proposals_15/' + str(cnt) + '_' + img_metas[img_id]['filename'].split('/')[-1])
                try:
                    cropped_image = self.preprocess(cropped_image).to(self.device)
                except:
                    print(img_metas[img_id]['flip'],flag)
                    print(box)
                    raise RuntimeError
                cropped_images.append(cropped_image)
        cropped_images = torch.stack(cropped_images)
        image_features = self.clip_model.encode_image(cropped_images)
        return image_features

    def boxto15(self, bboxes):
        if bboxes.shape[1] == 5:
            bboxes15 = torch.dstack([
                        bboxes[:,0],
                        1.25 * bboxes[:, 1] - 0.25 * bboxes[:, 3], 
                        1.25 * bboxes[:, 2] - 0.25 * bboxes[:, 4],
                        1.25 * bboxes[:, 3] - 0.25 * bboxes[:, 1], 
                        1.25 * bboxes[:, 4] - 0.25 * bboxes[:, 2]
                        ]).squeeze(0)
        else:
            bboxes15 = torch.dstack([
                        1.25 * bboxes[:, 0] - 0.25 * bboxes[:, 2], 
                        1.25 * bboxes[:, 1] - 0.25 * bboxes[:, 3],
                        1.25 * bboxes[:, 2] - 0.25 * bboxes[:, 0], 
                        1.25 * bboxes[:, 3] - 0.25 * bboxes[:, 1]
                        ]).squeeze(0)
        return bboxes15

    def checkdir(self, path): 
        path_prefix = osp.dirname(path)
        if not osp.exists(path_prefix):
            os.makedirs(path_prefix)

    def _bbox_forward(self, stage, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        rois = rois.float()
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(
            x[:bbox_roi_extractor.num_inputs], rois)
        region_embeddings = bbox_head.forward_embedding(bbox_feats)
        bbox_pred = bbox_head(region_embeddings)
        bbox_results = dict(
            bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results, region_embeddings

    def _bbox_forward_for_image(self, stage, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        rois = rois.float()
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(
            x[:bbox_roi_extractor.num_inputs], rois)

        region_embeddings = bbox_head.forward_embedding_for_image(bbox_feats)

        return None, region_embeddings

    def img2pil2feat(self, img, boxs, name=None):
        img = np.array(img.detach().cpu()).astype(np.uint8)
        img = Image.fromarray(img.transpose(1,2,0))
        img_shape = img.size
        # print(img.mode)
        # print(img.size)
        # print(boxs)
        boxs = torch.dstack([torch.floor(boxs[:,0]-0.001),torch.floor(boxs[:,1]-0.001),torch.ceil(boxs[:,2]+0.001),torch.ceil(boxs[:,3]+0.001)]).squeeze(0)
        # boxs = torch.dstack([torch.floor(boxs[:,0]),torch.floor(boxs[:,1]),torch.ceil(boxs[:,2]),torch.ceil(boxs[:,3])]).squeeze(0)
        boxs[:,[0,2]].clamp_(min=0,max=img_shape[0])
        boxs[:,[1,3]].clamp_(min=0,max=img_shape[1])
        boxs = boxs.detach().cpu().numpy()
        # print(boxs)
        preprocessed = []
        i = 0
        for box in boxs:
            try:
                croped = img.crop(box)
            except:
                print(box)
            # croped.save(name+f'_pil_{i}.jpg')
            i += 1
            croped = self.preprocess(croped)
            preprocessed.append(croped)
            
        preprocessed = torch.stack(preprocessed).to(self.device)
        features = self.clip_model.encode_image(preprocessed)
        return features

    
    def _bbox_forward_train(self, stage, x, img, img_no_normalize, sampling_results, proposals_pre_computed, gt_bboxes, gt_labels,
                            img_metas,bbox_assigner,train_cfg):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        if not self.fix_bg:
            input_one = x[0].new_ones(1)
            bg_class_embedding = self.bg_embedding(input_one).reshape(1, 512)
            bg_class_embedding = torch.nn.functional.normalize(bg_class_embedding, p=2, dim=1)
        # ----------------------------------------------------------
        num_proposals_per_img = tuple(len(proposal) for proposal in proposals_pre_computed)
        rois_image = torch.cat(proposals_pre_computed, dim=0)
        batch_index = torch.cat([x[0].new_full((num_proposals_per_img[i],1),i) for i in range(len(num_proposals_per_img))],0)
        rois_image = torch.cat([batch_index, rois_image[..., :4]], dim=-1)
        bboxes = rois_image
        # bboxes = rois
        # bboxes = bbox2roi(gt_bboxes)
        # ------------------------------------------------------------
        # not using precomputed proposals
        # num_proposals_per_img = tuple(len(gt_bbox) for gt_bbox in gt_bboxes)
        # num_proposals_per_img = tuple(len(res.bboxes) for res in sampling_results)
        bbox_results, region_embeddings = self._bbox_forward(stage, x, rois)
        # if len(gt_bboxes[0])==0: 
            # bboxes = rois
            # num_proposals_per_img = tuple(len(res.bboxes) for res in sampling_results)
        # bboxes = rois
        # -------------------------------------------------------------
        if self.ensemble:
            _, region_embeddings_image = self._bbox_forward_for_image(stage, x, bboxes)
            region_embeddings_image = self.projection_for_image[stage](region_embeddings_image)
            region_embeddings_image = torch.nn.functional.normalize(region_embeddings_image, p=2, dim=1)
        else:
            _, region_embeddings_image = self._bbox_forward(stage, x, bboxes)
            region_embeddings_image = self.projection[stage](region_embeddings_image)
            region_embeddings_image = torch.nn.functional.normalize(region_embeddings_image, p=2, dim=1)
        if stage == 0:
            if self.load_feature:
                clip_image_features_ensemble = []
                bboxes_all = bboxes.split(num_proposals_per_img)
                for i in range(len(img_metas)):
                    if self.num_classes == 1203:
                        save_path = os.path.join('lvis_clip_image_embedding.zip/data/lvis_clip_image_embedding', img_metas[i]['ori_filename'].split('.')[0] + '.pth')
                    elif self.num_classes == 80:
                        save_path = os.path.join('coco_clip_image_embedding.zip/data/coco_clip_image_embedding', img_metas[i]['ori_filename'].split('.')[0] + '.pth')
                    try:
                        f = self.zipfile.get(save_path)
                        stream = io.BytesIO(f)
                        tmp = torch.load(stream)
                        clip_image_features_ensemble.append(tmp.to(self.device))
                    except:
                        bboxes_single_image = bboxes_all[i]
                        bboxes15 = self.boxto15(bboxes_single_image)
                        if self.num_classes == 1203:
                            save_path = os.path.join('./data/lvis_clip_image_embedding', img_metas[i]['ori_filename'].split('.')[0] + '.pth')
                        elif self.num_classes == 80:
                            save_path = os.path.join('./data/coco_clip_image_embedding', img_metas[i]['ori_filename'].split('.')[0] + '.pth')
                        # save_path = osp.join('./data/lvis_clip_image_embedding', img_metas[i]['ori_filename'].split('.')[0] + '.pth')
                        self.checkdir(save_path)
                        # clip_image_features = self.clip_image_forward((img_metas[i],), bboxes[:,1:],(num_proposals_per_img[i],))
                        # clip_image_features15 = self.clip_image_forward((img_metas[i],), bboxes15[:,1:],(num_proposals_per_img[i],))
                        clip_image_features = self.img2pil2feat(img_no_normalize[i], bboxes_single_image[:,1:])
                        clip_image_features15 = self.img2pil2feat(img_no_normalize[i], bboxes15[:,1:])
                        clip_image_features_single = clip_image_features + clip_image_features15
                        clip_image_features_single = clip_image_features_single.float()
                        clip_image_features_single = torch.nn.functional.normalize(clip_image_features_single, p=2, dim=1)
                        torch.save(clip_image_features_single.cpu(), save_path)
                        clip_image_features_ensemble.append(clip_image_features_single)
                clip_image_features_ensemble = torch.cat(clip_image_features_ensemble, dim=0)
            else:
                clip_image_features_ensemble = []
                clip_image_features_ensemble_align = []
                bboxes_all = bboxes.split(num_proposals_per_img)
                for i in range(len(img_metas)):
                    bboxes_single_image = bboxes_all[i]
                    bboxes15 = self.boxto15(bboxes_single_image)
                    if self.num_classes == 1203:
                        save_path = os.path.join('./data/lvis_clip_image_embedding', img_metas[i]['ori_filename'].split('.')[0] + '.pth')
                    elif self.num_classes == 80:
                        save_path = os.path.join('./data/coco_clip_image_embedding', img_metas[i]['ori_filename'].split('.')[0] + '.pth')
                    self.checkdir(save_path)
                    clip_image_features = self.img2pil2feat(img_no_normalize[i], bboxes_single_image[:,1:])
                    clip_image_features15 = self.img2pil2feat(img_no_normalize[i], bboxes15[:,1:])

                    # clip_image_features = self.clip_image_forward((img_metas[i],), bboxes_single_image, (num_proposals_per_img[i],))
                    # clip_image_features15 = self.clip_image_forward((img_metas[i],), bboxes15,(num_proposals_per_img[i],),True)

                    # clip_image_features_align = self.clip_image_forward_align(img, bboxes,(num_proposals_per_img[i],))
                    # clip_image_features15_align = self.clip_image_forward_align(img, bboxes15,(num_proposals_per_img[i],))
                    # clip_image_features_single_align = clip_image_features_align + clip_image_features15_align
                    # clip_image_features_single_align = clip_image_features_single_align.float()
                    # clip_image_features_single_align = torch.nn.functional.normalize(clip_image_features_single_align, p=2, dim=1)
                    # clip_image_features_ensemble_align.append(clip_image_features_single_align)

                    clip_image_features_single = clip_image_features + clip_image_features15
                    clip_image_features_single = clip_image_features_single.float()
                    clip_image_features_single = torch.nn.functional.normalize(clip_image_features_single, p=2, dim=1)

                    clip_image_features_ensemble.append(clip_image_features_single)
                    torch.save(clip_image_features_single.cpu(), save_path)
                clip_image_features_ensemble = torch.cat(clip_image_features_ensemble, dim=0)
            # clip_image_features_ensemble_align = torch.cat(clip_image_features_ensemble_align, dim=0)
            self.clip_image_features_ensemble = clip_image_features_ensemble
        else:
            clip_image_features_ensemble = self.clip_image_features_ensemble

        bbox_targets = self.bbox_head[stage].get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, train_cfg)
        labels, _, _, _ = bbox_targets
        
        region_embeddings = self.projection[stage](region_embeddings)
        region_embeddings = torch.nn.functional.normalize(region_embeddings, p=2, dim=1)
        if not self.fix_bg:
            text_features = torch.cat([self.text_features_for_classes, bg_class_embedding], dim=0)
        else:
            text_features = self.text_features_for_classes

        # clip_logits_align = clip_image_features_ensemble_align @ text_features.T
        # clip_logits_align[:,-1] = -1e11
        self.iters += 1
        if self.iters<200:
            clip_logits = clip_image_features_ensemble @ text_features.T
            clip_logits[:,-1] = -1e11
            num_imgs = len(img_metas)
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
            labels_image = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposals_pre_computed[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                labels_image += assign_result.labels
            labels_image = torch.tensor(labels_image,device=self.device)

            fg_index = labels_image.ge(0)
            self.accuracy = self.accuracy[-1000:]
            self.accuracy += (clip_logits.argmax(dim=1).eq(labels_image))[fg_index].detach().cpu().tolist()
            print(np.mean(self.accuracy))
        # if self.is_main_process():
        #     print('#'*100)
        #     print(rois[:10,:])
        #     print(clip_logits.argmax(dim=1)[:10],clip_logits_align.argmax(dim=1)[:10])
        # print(fg_index.sum())
        # if len(gt_bboxes[0])>0: 
        # self.accuracy_align = self.accuracy_align[-1000:]
        # self.accuracy_align += (clip_logits_align.argmax(dim=1).eq(labels))[fg_index].detach().cpu().tolist()
        # print('align:{} Image crop:{}'.format(np.mean(self.accuracy_align),np.mean(self.accuracy)))

        cls_score_text = region_embeddings @ text_features.T
        # self.iters += 1
        cls_score_text[:,self.novel_label_ids] = -1e11
        text_cls_loss = F.cross_entropy(cls_score_text / self.temperature, labels, reduction='mean')
        kd_loss = F.l1_loss(region_embeddings_image,clip_image_features_ensemble)
        loss_bbox = self.bbox_head[stage].loss(
            bbox_results['bbox_pred'], rois,
            *bbox_targets)
        loss_bbox.update(text_cls_loss=text_cls_loss, kd_loss=kd_loss * self.kd_weight)
        bbox_results.update(loss_bbox=loss_bbox,cls_score=cls_score_text,rois=rois,bbox_targets=bbox_targets)
        return bbox_results


    def _mask_forward(self, stage, x, rois):
        """Mask head forward function used in both training and testing."""
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        mask_pred = mask_head(mask_feats)

        mask_results = dict(mask_pred=mask_pred)
        return mask_results

    def _mask_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_masks,
                            rcnn_train_cfg,
                            bbox_feats=None):
        """Run forward function and calculate loss for mask head in
        training."""
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        mask_results = self._mask_forward(stage, x, pos_rois)

        mask_targets = self.mask_head[stage].get_targets(
            sampling_results, gt_masks, rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head[stage].loss(mask_results['mask_pred'],
                                               mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask)
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
                           img_no_normalize,
                           img_metas,
                           proposals,
                           proposals_pre_computed,
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
        # if self.use_clip_inference:
            # proposals = proposals_pre_computed
        rois = bbox2roi(proposals)
        num_proposals_per_img = tuple(len(proposal) for proposal in proposals)
        # rois_image = torch.cat(proposals_pre_computed, dim=0)
        # batch_index = torch.cat([x[0].new_full((num_proposals_per_img[i],1),i) for i in range(len(num_proposals_per_img))],0)
        # rois = torch.cat([batch_index, rois_image[..., :4]], dim=-1)

        bbox_results,region_embeddings = self._bbox_forward(x,rois)
        region_embeddings = self.projection(region_embeddings)
        region_embeddings = torch.nn.functional.normalize(region_embeddings,p=2,dim=1)
        if not self.fix_bg:
            input_one = x[0].new_ones(1)
            bg_class_embedding = self.bg_embedding(input_one).unsqueeze(0)
            bg_class_embedding = torch.nn.functional.normalize(bg_class_embedding,p=2,dim=1)
            text_features = torch.cat([self.text_features_for_classes,bg_class_embedding],dim=0)
        else:
            text_features = self.text_features_for_classes
        #-----------------------------------------------------
        # """
        cls_score_text = region_embeddings@text_features.T
        
        if self.num_classes == 80 and self.coco_setting:
            cls_score_text[:,self.unseen_label_ids_test] = -1e11
        cls_score_text = cls_score_text/0.007
        # cls_score_text = cls_score_text/cls_score_text.std(dim=1,keepdim=True)*4
        cls_score_text = cls_score_text.softmax(dim=1)
        #--------------------------------------------
        if self.ensemble and not self.use_clip_inference:
            # """
            # bbox_pred = bbox_results['bbox_pred']
            # num_proposals_per_img = tuple(len(p) for p in proposals)
            # rois = rois.split(num_proposals_per_img, 0)
            # # some detector with_reg is False, bbox_pred will be None
            # if bbox_pred is not None:
            #     # the bbox prediction of some detectors like SABL is not Tensor
            #     if isinstance(bbox_pred, torch.Tensor):
            #         bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            #     else:
            #         bbox_pred = self.bbox_head.bbox_pred_split(
            #             bbox_pred, num_proposals_per_img)
            # bboxes = []
            # for i in range(len(proposals)):
            #     bbox = self.bbox_head.compute_bboxes(
            #     rois[i],
            #     bbox_pred[i],
            #     img_shapes[i],
            #     scale_factors[i],
            #     rescale=rescale,
            #     cfg=None)
            #     bboxes.append(bbox)
            # bboxes = torch.cat(bboxes,0)
            # """
            # rois_image = bbox2roi(bboxes[:,:4])
            _,region_embeddings_image = self._bbox_forward_for_image(x,rois)
            region_embeddings_image = self.projection_for_image(region_embeddings_image)
            region_embeddings_image = torch.nn.functional.normalize(region_embeddings_image,p=2,dim=1)
            cls_score_image = region_embeddings_image@text_features.T
            cls_score_image = cls_score_image/0.007
            if self.num_classes == 80 and self.coco_setting:
                cls_score_image[:,self.unseen_label_ids_test] = -1e11
            # cls_score_image[:,:-1] = cls_score_image[:,:-1]/cls_score_image[:,:-1].std(dim=1,keepdim=True)*4
            cls_score_image[:,-1] = -1e11
            cls_score_image = cls_score_image.softmax(dim=1)
        #------------------------------------------------
        #using clip to inference
        if self.ensemble and self.use_clip_inference:
            bboxes = rois
            save_path = os.path.join('./data/lvis_clip_image_embedding_test_offline', img_metas[0]['ori_filename'].split('.')[0] + '.pth')
            # save_path = os.path.join('./data/lvis_clip_image_embedding_test_offline_img2pil', img_metas[0]['ori_filename'].split('.')[0] + '.pth')
            if not osp.exists(save_path):
                # if True:
                bboxes15 = self.boxto15(bboxes)

                clip_image_features_img2pil = self.img2pil2feat(img_no_normalize[0], bboxes[:,1:])
                clip_image_features15_img2pil = self.img2pil2feat(img_no_normalize[0], bboxes15[:,1:])
                clip_image_features_ensemble_img2pil = clip_image_features_img2pil + clip_image_features15_img2pil
                clip_image_features_ensemble_img2pil = clip_image_features_ensemble_img2pil.float()
                clip_image_features_ensemble_img2pil = F.normalize(clip_image_features_ensemble_img2pil,p=2,dim=1)

                # clip_image_features = self.clip_image_forward(img_metas,bboxes,num_proposals_per_img)
                # clip_image_features15 = self.clip_image_forward(img_metas, bboxes15, num_proposals_per_img)
                # clip_image_features_ensemble = clip_image_features + clip_image_features15
                # clip_image_features_ensemble = clip_image_features_ensemble.float()
                # clip_image_features_ensemble = F.normalize(clip_image_features_ensemble,p=2,dim=1)

                # clip_image_features_align = self.clip_image_forward_align(img,bboxes,num_proposals_per_img)
                # clip_image_features15_align = self.clip_image_forward_align(img, bboxes15, num_proposals_per_img)
                # clip_image_features_ensemble_align = clip_image_features_align + clip_image_features15_align
                # clip_image_features_ensemble_align = clip_image_features_ensemble_align.float()
                # clip_image_features_ensemble_align = F.normalize(clip_image_features_ensemble_align,p=2,dim=1)

                # torch.save(clip_image_features_ensemble_img2pil.cpu(), save_path)
                self.checkdir(save_path)
                # torch.save(clip_image_features_ensemble.cpu(), save_path)
            else:
                clip_image_features_ensemble = torch.load(save_path).to(self.device)
                # clip_image_features_ensemble_img2pil = torch.load(save_path).to(self.device)
            # cls_score_clip[:,:-1] = cls_score_clip[:,:-1]/cls_score_clip[:,:-1].std(dim=1,keepdim=True)*0.006
            # print(cls_score_clip.std(dim=1).mean())
            cls_score_clip = clip_image_features_ensemble_img2pil @ text_features.T
            cls_score_clip[:,:-1] = cls_score_clip[:,:-1]/cls_score_clip[:,:-1].std(dim=1,keepdim=True)*4
            # cls_score_clip = torch.exp(cls_score_clip-1)
            # cls_score_clip = cls_score_clip/0.007
            cls_score_clip[:,-1] = -1e11
            cls_score_clip = cls_score_clip.softmax(dim=1)

            # cls_score_clip_img2pil = clip_image_features_ensemble_img2pil @ text_features.T
            # cls_score_clip_img2pil = torch.exp(cls_score_clip_img2pil-1)
            # cls_score_clip_img2pil = cls_score_clip_img2pil/0.007
            # cls_score_clip_img2pil[:,-1] = -1e11
            # cls_score_clip_img2pil = cls_score_clip_img2pil.softmax(dim=1)

            # cls_score_clip_align = clip_image_features_ensemble_align @ text_features.T
            # cls_score_clip_align = torch.exp(cls_score_clip_align-1)
            # cls_score_clip_align = cls_score_clip_align/0.007
            # cls_score_clip_align[:,-1] = -1e11
            # cls_score_clip_align = cls_score_clip_align.softmax(dim=1)
            cls_score_image = cls_score_clip
        #--------------------------------------------------
        # """
        a = 1/3
        if self.ensemble:
            if self.fixed_lambda is not None:
                cls_score = cls_score_image**(1-self.fixed_lambda)*cls_score_text**self.fixed_lambda
            else:
                cls_score= torch.where(self.novel_index,cls_score_image**(1-a)*cls_score_text**a,
                               cls_score_text**(1-a)*cls_score_image**a)
                # print(11)

            # cls_score_align= torch.where(self.novel_index,cls_score_clip_align**(1-a)*cls_score_text**a,
                            #    cls_score_text**(1-a)*cls_score_clip_align**a)
            # cls_score = cls_score_image**(1-a)*cls_score_text**a
            # cls_score = cls_score_image
        else:
            cls_score = cls_score_text
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
            # if self.use_clip_inference:
            #     # proposal_label = self.novel_label_ids[cls_score[i][:,self.novel_label_ids].argmax(dim=1)]
                # for j,label in enumerate(proposal_label):
                #     box = proposals[i][j].detach().cpu().numpy().tolist()
                #     print('{} {} {} {} {} {}'.format(img_metas[0]['ori_filename'],cls_score[i].max(dim=1)[0][j],box[0],box[1],box[2],box[3]),file=open('/home/dy20/mmdetection27/workdirs/det_result/train_novel_det/{}_det_{}.txt'.format(self.rank,label),'a'))
        return det_bboxes, det_labels

    def simple_test(self,
                    x,
                    img,
                    img_no_normalize,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False,
                    **kwargs):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg
        if not self.fix_bg:
            input_one = x[0].new_ones(1)
            bg_class_embedding = self.bg_embedding(input_one).unsqueeze(0)
            bg_class_embedding = torch.nn.functional.normalize(bg_class_embedding,p=2,dim=1)
            text_features = torch.cat([self.text_features_for_classes,bg_class_embedding],dim=0)
        else:
            text_features = self.text_features_for_classes

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_results,region_embeddings = self._bbox_forward(i,x,rois)
            region_embeddings = self.projection[i](region_embeddings)
            region_embeddings = torch.nn.functional.normalize(region_embeddings,p=2,dim=1)

            cls_score_text = region_embeddings@text_features.T
        
            if self.num_classes == 80 and self.coco_setting:
                cls_score_text[:,self.unseen_label_ids_test] = -1e11
            cls_score_text = cls_score_text/0.007
            # cls_score_text = cls_score_text/cls_score_text.std(dim=1,keepdim=True)*4
            cls_score_text = cls_score_text.softmax(dim=1)

            _,region_embeddings_image = self._bbox_forward_for_image(i,x,rois)
            region_embeddings_image = self.projection_for_image[i](region_embeddings_image)
            region_embeddings_image = torch.nn.functional.normalize(region_embeddings_image,p=2,dim=1)
            cls_score_image = region_embeddings_image@text_features.T
            cls_score_image = cls_score_image/0.007
            if self.num_classes == 80 and self.coco_setting:
                cls_score_image[:,self.unseen_label_ids_test] = -1e11
            # cls_score_image[:,:-1] = cls_score_image[:,:-1]/cls_score_image[:,:-1].std(dim=1,keepdim=True)*4
            cls_score_image[:,-1] = -1e11
            cls_score_image = cls_score_image.softmax(dim=1)

            a = 1/3
            if self.ensemble:
                if self.fixed_lambda is not None:
                    cls_score = cls_score_image**(1-self.fixed_lambda)*cls_score_text**self.fixed_lambda
                else:
                    cls_score= torch.where(self.novel_index,cls_score_image**(1-a)*cls_score_text**a,
                                   cls_score_text**(1-a)*cls_score_image**a)

            # split batch bbox prediction back to each image
            # cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                bbox_label = [s[:, :-1].argmax(dim=1) for s in cls_score]
                rois = torch.cat([
                    self.bbox_head[i].regress_by_class(rois[j], bbox_label[j],
                                                       bbox_pred[j],
                                                       img_metas[j])
                    for j in range(num_imgs)
                ])

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            # print((cls_score[i]>0.001).sum())
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        if torch.onnx.is_in_onnx_export():
            return det_bboxes, det_labels
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_results

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i][:, :4]
                    for i in range(len(det_bboxes))
                ]
                mask_rois = bbox2roi(_bboxes)
                num_mask_rois_per_img = tuple(
                    _bbox.size(0) for _bbox in _bboxes)
                aug_masks = []
                for i in range(self.num_stages):
                    mask_results = self._mask_forward(i, x, mask_rois)
                    mask_pred = mask_results['mask_pred']
                    # split batch mask prediction back to each image
                    mask_pred = mask_pred.split(num_mask_rois_per_img, 0)
                    aug_masks.append(
                        [m.sigmoid().cpu().numpy() for m in mask_pred])

                # apply mask post-processing to each image individually
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [[]
                             for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_masks = merge_aug_masks(
                            aug_mask, [[img_metas[i]]] * self.num_stages,
                            rcnn_test_cfg)
                        segm_result = self.mask_head[-1].get_seg_masks(
                            merged_masks, _bboxes[i], det_labels[i],
                            rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                            rescale)
                        segm_results.append(segm_result)
            ms_segm_result['ensemble'] = segm_results

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return results
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

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        """Test det bboxes with test time augmentation."""
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']
            # TODO more flexible
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            rois = bbox2roi([proposals])
            bbox_results,region_embeddings = self._bbox_forward(x,rois)
            region_embeddings = self.projection(region_embeddings)
            region_embeddings = torch.nn.functional.normalize(region_embeddings,p=2,dim=1)
            input_one = x[0].new_ones(1)
            bg_class_embedding = self.bg_embedding(input_one).unsqueeze(0)
            bg_class_embedding = torch.nn.functional.normalize(bg_class_embedding,p=2,dim=1)
            text_features = torch.cat([self.text_features_for_classes,bg_class_embedding],dim=0)
            cls_score_text = region_embeddings@text_features.T
            cls_score_text = cls_score_text/0.007
            #0.009#0.008#0.007
            cls_score_text = cls_score_text.softmax(dim=1)
            cls_score = cls_score_text
            bboxes, scores = self.bbox_head.get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels
