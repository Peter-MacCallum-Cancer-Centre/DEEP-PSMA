import os
from os.path import join,isdir
import SimpleITK as sitk
import shutil
import pandas as pd
import json
from DEEP_PSMA_Infer import run_inference
import numpy as np
from evaluation import score_labels


##Code to infer nnU-Net on all cases and apply baseline refinement
##A little slow since adapted from inference function running one case at a time
##


def get_dice(gt,seg): #dice calculation if doing final label analysis
    dice = np.sum(seg[gt==1]==1)*2.0 / (np.sum(seg[seg==1]==1) + np.sum(gt[gt==1]==1))
    return dice

input_dataset_folder='data' #location of input training data

force_infer_fold_number=False  #if set False will use the fold designated by in nnunet validation splits
##force_infer_fold_number='0'   #for quick test, just train fold 0 and set here

for tracer in ['PSMA','FDG']:
    df=pd.DataFrame(columns=['case','fold','nn_dice','baseline_dice']) #for quick accuracy evaluation
    nn_inferred_dir=join(input_dataset_folder,tracer,'nn_direct_inferred') #where to save predicted ttb direct from nnunet
    baseline_inferred_dir=join(input_dataset_folder,tracer,'baseline_inferred') #where to save refined labels
    os.makedirs(nn_inferred_dir,exist_ok=True)
    os.makedirs(baseline_inferred_dir,exist_ok=True)
    df_val=pd.read_csv(join(input_dataset_folder,tracer,'validation_folds.csv'),index_col=0)
    for case in os.listdir(join(input_dataset_folder,tracer,'CT')): #loop through all cases in training directory
        print(tracer,case)
        ct_fname=join(input_dataset_folder,tracer,'CT',case)
        pt_fname=join(input_dataset_folder,tracer,'PET',case)
        gt_fname=join(input_dataset_folder,tracer,'TTB',case)
        gt_ttb=sitk.ReadImage(gt_fname)
        

        #output filenames                      
        baseline_fname=join(baseline_inferred_dir,case)
        nn_fname=join(nn_inferred_dir,case)

        #read threshold value
        with open(join(input_dataset_folder,tracer,'thresholds',case.replace('.nii.gz','.json')),'r') as f:
            suv_threshold=json.load(f)['suv_threshold']
            print('Detected contouring SUV threshold',suv_threshold)


        fold=str(df_val[df_val.case==case.replace('.nii.gz','')].val_fold.values[0]) #get validation fold to use for case
        print('fold:',fold)
        if force_infer_fold_number: #sets which fold to use for nnunet inference. If switched from False at start of script will occur here
            infer_fold=force_infer_fold_number
        else:
            infer_fold=fold

        baseline_ttb=run_inference(pt_fname,ct_fname,tracer=tracer,output_fname=r"inference_test.nii.gz",
                           return_ttb_sitk=True,temp_dir='temp',
                           fold=infer_fold,suv_threshold=suv_threshold)            
        
        nn_ttb=sitk.ReadImage(join('temp','nn_output','deep-psma.nii.gz'))  ##nnunet output from temporary folder
        nn_ttb_single=sitk.GetImageFromArray((sitk.GetArrayFromImage(nn_ttb)==1).astype('int8')) # nn TTB inferred 1=disease 2=normal
        nn_ttb_single.CopyInformation(nn_ttb)
        sitk.WriteImage(nn_ttb_single,nn_fname) 
        sitk.WriteImage(baseline_ttb,baseline_fname)
        
        nn_dice=get_dice(sitk.GetArrayFromImage(gt_ttb),sitk.GetArrayFromImage(nn_ttb_single)) #quick dice calculation and save as csv, to replace with full eval script...
        baseline_dice=get_dice(sitk.GetArrayFromImage(gt_ttb),sitk.GetArrayFromImage(baseline_ttb))
        print('Dice scores (nnU-Net Direct/baseline-refined):',nn_dice,baseline_dice)
        df.loc[len(df)]=[case.replace('.nii.gz',''),fold,nn_dice,baseline_dice]
        print('nnU-Net Direct evaluation:')
        print(score_labels(gt_fname,nn_fname,pt_fname))
        print('Baseline Model evaluation:')
        print(score_labels(gt_fname,baseline_fname,pt_fname)              )
    df.to_csv(join(input_dataset_folder,tracer,'inferred_dice_scores.csv'))
