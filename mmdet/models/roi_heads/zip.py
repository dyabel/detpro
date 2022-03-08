import os 
from mmcv import FileClient, BaseStorageBackend 
from zipfile import ZipFile
import torch
import io
class ZipBackend(BaseStorageBackend): 
    """ 
    Only single image directory is supported 
    """ 
    def __init__(self, zip_file_name=None): 
        if zip_file_name is not None: 
            self.zip_file = ZipFile(zip_file_name, mode="r") 
            self.root_prefix = self.zip_file.namelist()[0] 
            print(self.root_prefix)
        else: 
            self.zip_file = None 
            self.root_prefix = None 
            print("Use Zip Backends") 
    def get(self, filepath):
        file_name = None
        zip_name = None
        if ".zip" in filepath:
            zip_name, file_name = filepath.split(".zip/")
            zip_name = zip_name + ".zip"
        if self.zip_file is None:
            if zip_name is None:
                zip_name = os.path.dirname(filepath) + ".zip"
            if not os.path.exists(zip_name):
                raise FileNotFoundError(f"There is no zip file in {zip_name}")
            else:
                print(f"Load Zip File{zip_name}")
                self.zip_file = ZipFile(zip_name, mode="r")
                # print(self.zip_file.namelist()[:10])
                self.root_prefix = self.zip_file.namelist()[0]
        else:
            assert isinstance(self.zip_file, ZipFile), "Error: no such zip file."
        if file_name is None:
            file_name = self.root_prefix + os.path.basename(filepath)
        value_buf = self.zip_file.read(file_name)
        return value_buf 
    def get_text(self, filepath): 
        raise NotImplementedError 
        FileClient.register_backend("zip", ZipBackend)
# zipfile = ZipBackend('data/test12.zip')
# zipfile = ZipBackend('data/lvis_clip_image_embedding.zip')
# # with open('data/test12.txt','r') as f: 
# f = zipfile.get('lvis_clip_image_embedding.zip/lvis_clip_image_embedding/train2017/000000123509.pth')
# stream = io.BytesIO(f)
# tmp = torch.load(stream)
# print(tmp)
    # for i in f.readlines():
        # print(i.strip())