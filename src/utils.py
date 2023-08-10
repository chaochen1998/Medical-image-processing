import numpy as np
import skimage.io as io

# Normalize CT images as needed
class Normalization_np(object):
    def __init__(self, windowMin, windowMax):
        self.name = 'ManualNormalization'
        assert isinstance(windowMax, (int,float))
        assert isinstance(windowMin, (int,float))
        self.windowMax = windowMax
        self.windowMin = windowMin
    
    def __call__(self, img_3d):
        img_3d_norm = np.clip(img_3d, self.windowMin, self.windowMax)
        img_3d_norm-=np.min(img_3d_norm)
        max_99_val=np.percentile(img_3d_norm, 99)
        if max_99_val>0:
            img_3d_norm = img_3d_norm/max_99_val*255
        
        return img_3d_norm
    
def get_CT_image(image_path, windowMin=-1000, windowMax=600, need_norm=True):
    raw_img = io.imread(image_path, plugin='simpleitk')
    raw_img = np.array(raw_img, dtype=np.float)
    
    if need_norm:
        normalization=Normalization_np(windowMin=windowMin, windowMax=windowMax)
        return normalization(raw_img)
    else:
        return raw_img
