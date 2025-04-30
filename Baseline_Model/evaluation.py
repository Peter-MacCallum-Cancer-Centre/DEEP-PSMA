import SimpleITK as sitk
import numpy as np

### TO BE UPDATED...
### From AutoPET Evaluation Method https://github.com/lab-midas/autoPET/blob/master/val_script.py
import cc3d 
def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 18
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    return conn_comp
def false_pos_pix(gt_array,pred_array):
    # compute number of voxels of false positive connected components in prediction mask
    pred_conn_comp = con_comp(pred_array)
    false_pos = 0
    for idx in range(1,pred_conn_comp.max()+1):
        comp_mask = np.isin(pred_conn_comp, idx)
        if (comp_mask*gt_array).sum() == 0:
            false_pos = false_pos+comp_mask.sum()
    return false_pos
def false_neg_pix(gt_array,pred_array):
    # compute number of voxels of false negative connected components (of the ground truth mask) in the prediction mask
    gt_conn_comp = con_comp(gt_array)
    false_neg = 0
    for idx in range(1,gt_conn_comp.max()+1):
        comp_mask = np.isin(gt_conn_comp, idx)
        if (comp_mask*pred_array).sum() == 0:
            false_neg = false_neg+comp_mask.sum()
    return false_neg
def dice_score(mask1,mask2):
    # compute foreground Dice coefficient
    overlap = (mask1*mask2).sum()
    sum = mask1.sum()+mask2.sum()
    dice_score = 2*overlap/sum
    return dice_score


#updated methods for DEEP-PSMA grand challenge

#surface dice analysis
from SimpleITK import GetArrayViewFromImage as ArrayView
from functools import partial
distance_map = partial(sitk.SignedMaurerDistanceMap, squaredDistance=False, useImageSpacing=True)
def get_surface_dice(gold,prediction):
    gold_surface = sitk.LabelContour(gold == 1, False)
    distance_map = partial(sitk.SignedMaurerDistanceMap, squaredDistance=False, useImageSpacing=True)
    prediction_surface = sitk.LabelContour(prediction == 1, False)
    prediction_distance_map = sitk.Abs(distance_map(prediction_surface))
    gold_distance_map = sitk.Abs(distance_map(gold_surface))
    gold_to_prediction = ArrayView(prediction_distance_map)[ArrayView(gold_surface) == 1]
    matching_surface_voxels=(gold_to_prediction==0).sum()
##    gold_surface_voxels=gold_to_prediction.size
    gold_surface_voxels=(ArrayView(gold_surface) == 1).sum()
    prediction_surface_voxels=(ArrayView(prediction_surface) == 1).sum()
    surface_dice=(2.*matching_surface_voxels)/(gold_surface_voxels+prediction_surface_voxels)
    return surface_dice

def read_label(label_path):
    label=sitk.ReadImage(label_path)
    ar=sitk.GetArrayFromImage(label)
    voxel_volume=np.prod(np.array(label.GetSpacing()))/1000.
    return ar,voxel_volume


def score_labels(gt_path,pred_path,pt_path):
    gt_ar,voxel_volume=read_label(gt_path)
    pred_ar,voxel_volume=read_label(pred_path) #could input check for matching resolution...
    pt_ar,voxel_volume=read_label(pt_path)
    dice=dice_score(gt_ar,pred_ar)
    false_positive_volume=false_pos_pix(gt_ar,pred_ar)*voxel_volume
    false_negative_volume=false_neg_pix(gt_ar,pred_ar)*voxel_volume
    surface_dice=get_surface_dice(sitk.ReadImage(gt_path),sitk.ReadImage(pred_path))
    suv_mean_ratio=pt_ar[pred_ar>0].mean()/pt_ar[gt_ar>0].mean()
    ttb_volume_ratio=pred_ar.sum()/gt_ar.sum()
    
    return dice, false_positive_volume, false_negative_volume, surface_dice, suv_mean_ratio, ttb_volume_ratio
