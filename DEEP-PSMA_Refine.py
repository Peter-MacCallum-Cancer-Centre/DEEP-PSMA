import SimpleITK as sitk
import numpy as np

def expand_contract_label(label,distance=5.):
    """expand or contract sitk label image by distance indicated.
    negative values will contract, positive values expand.
    returns sitk image of adjusted label""" 
    distance_filter = sitk.SignedMaurerDistanceMapImageFilter() #filter to calculate distance map
    distance_filter.SetUseImageSpacing(True)  #confirm use of image spacing mm
    distance_filter.SquaredDistanceOff() #default distances computed as d^2, set to linear distance
    dmap=distance_filter.Execute(label) #execute on label, returns SITK image in same space with voxels as distance from surface
    dmap_ar=sitk.GetArrayFromImage(dmap) #array of above SITK image
    new_label_ar=(dmap_ar<=distance).astype('int16') #binary array of values less than or equal to selected d
    new_label=sitk.GetImageFromArray(new_label_ar) #create new SITK image and copy spatial information
    new_label.CopyInformation(label)
    return new_label



def refine_my_ttb_label(ttb_image,pet_image,totseg_multilabel,expansion_radius_mm=5.,
                        pet_threshold_value=3.0,totseg_non_expand_values=[5],
                        normal_image=None):
    """ refine ttb label by expanding and rethresholding the original prediction
    set expansion radius mm to control distance that inferred TTB boundary is initially grown
    set pet_threshold_value to match designated value for ground truth contouring workflow
    (eg PSMA PET SUV=3).
    Includes option to avoid growing the label in certain
    tissue types in the total segmentator label (ex PSMA avoid expanding into liver, could
    include [2,3,5,21] to also avoid kidneys and urinary bladder)
    For other organ values see "total" class map from:
    https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py
    Lastly, possible to include the "normal" tissue inferred label from the baseline example
    algorithm and will similarly avoid expanding into this region"""
    ttb_original_array=sitk.GetArrayFromImage(ttb_image) #original TTB array for inpainting back voxels in certain tissues
    ttb_expanded_image=expand_contract_label(ttb_image,expansion_radius_mm) #expand inferred label with function above
    ttb_expanded_array=sitk.GetArrayFromImage(ttb_expanded_image) #convert to numpy array
    pet_threshold_array=(sitk.GetArrayFromImage(pet_image)>=pet_threshold_value) #get numpy array of PET image voxels above threshold value
    ttb_rethresholded_array=np.logical_and(ttb_expanded_array,pet_threshold_array) #remove expanded TTB voxels below PET threshold

    #loop through total segmentator tissue #s and use original TTB prediction in those labels
    totseg_multilabel=sitk.Resample(totseg_multilabel,ttb_image,sitk.TranslationTransform(3),sitk.sitkNearestNeighbor,0)
    totseg_multilabel_array=sitk.GetArrayFromImage(totseg_multilabel)
    for totseg_value in totseg_non_expand_values:
        #paint the original TTB prediction into the totseg tissue regions - probably of most relevance for PSMA liver VOI
        ttb_rethresholded_array[totseg_multilabel_array==totseg_value]=ttb_original_array[totseg_multilabel_array==totseg_value] 

    #check if inferred normal array is included and if so set TTB voxels to background
    if normal_image is not None:
        normal_array=sitk.GetArrayFromImage(normal_image)
        ttb_rethresholded_array[normal_array>0]=0
    ttb_rethresholded_image=sitk.GetImageFromArray(ttb_rethresholded_array.astype('int16')) #create output image & copy information
    ttb_rethresholded_image.CopyInformation(ttb_image)
    return ttb_rethresholded_image
    
    
