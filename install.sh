pip install -r requirements/build.txt
pip install  -v -e .
##
pip uninstall pycocotools -y
pip uninstall mmpycocotools -y
pip install mmpycocotools
pip install -e CLIP
pip install -e lvis-api
pip install mmcv-full==1.2.5 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html