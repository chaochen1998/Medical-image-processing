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

# according to the image spacing to resample the image
def resample_image(loadpath, savepath, out_spacing=[0.357,0.357,0.500], ratio=False):
    """
    input:
        loadpath: the path of the CT image u want to resample, with format 'xxx.nii.gz'
        savepath: where u want to save the image after resample, with format 'xxx.nii.gz'
        out_spacing: your target spacing for the image
        ratio: type: bool, default to false, if True the out_spacing should be the ratio you want to change.
    output:
        the image after resample
    example:
        original_spacing = [0.5,0.5,0.5]
        image_after_resamlpe = resample_image(path1, path2, spacing=[1,1,2], ratio=False)
        This means you want to get the image with spacing=[1,1,2]
        while if u set the ratio equals to True and keep other the same, you will get the iamge
        with spacing=[0.5,0.5,0.25]
    """
    itk_image = sitk.ReadImage(loadpath)
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
 
    # 根据输出out_spacing设置新的size
    if ratio:
        out_size = [
            int(np.round(original_size[0] * out_spacing[0])),
            int(np.round(original_size[1] * out_spacing[1])),
            int(np.round(original_size[2] * out_spacing[2]))
        ]
        for i,(c,r) in enumerate(zip(original_spacing, out_spacing)):
            out_spacing[i] = c / r
    else:
        out_size = [
            int(np.round(original_size[0] * original_spacing[0] / out_spacing[0])),
            int(np.round(original_size[1] * original_spacing[1] / out_spacing[1])),
            int(np.round(original_size[2] * original_spacing[2] / out_spacing[2]))
        ]
 
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
 
    resample.SetInterpolator(sitk.sitkBSpline)

    sitk.WriteImage(resample.Execute(itk_image),savepath)
 
    return resample.Execute(itk_image)

def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx,maxzidx], [minxidx,maxxidx], [minyidx,maxyidx]]
