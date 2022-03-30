CURPATH=$(cd "$(dirname "$0")"; pwd)
echo $CURPATH
cd $CURPATH
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements/build.txt
pip install -e .
# pip install git+https://github.com/openai/CLIP.git
pip install -e CLIP
pip uninstall pycocotools -y
pip uninstall mmpycocotools -y
pip install mmpycocotools
pip install -e lvis-api
pip install mmcv-full==1.2.5 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_test.sh configs/lvis/cascade_mask_rcnn_r50_fpn_sample1e-3_mstrain_20e_lvis_v1_pretrain_ens.py workdirs/models/cascade/epoch_20.pth 2 --eval bbox segm --cfg-options model.roi_head.prompt_path=iou_neg5_ens.pth  model.roi_head.load_feature=False
