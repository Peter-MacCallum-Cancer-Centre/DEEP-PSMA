Baseline segmentation model using nnU-Net v2 with some pre- and post-processing refinements suited to the theshold segmentation task in PET/CT

Simple pre-processing code for converting native training data to input for nnU-Net is outlined in scripts 00 & 01. Note: Output Tumour Burden segmentation is always defined in the PET image spacing so this example takes the convention to downsample the CT input to match the PET resolution.

This method provides simple optimisations which improve the performance of nnU-Net for this use case. First, a label for "normal/physiological" tracer uptake is derived based on voxels which exceed the designated threshold for the image-of-interest but are excluded from the manually annotated total tumor burden (TTB) VOI. This sets a the training output to predict a multi-class label (Background=0, TTB=1, Physiological=2). Second, the intensity values of the PET image are normalised to the threshold value (threshold=1.0). So for PSMA PET images where SUV=3 is the designated threshold, this results in all values divided by a factor of 3. In FDG where the liver-based threshold varies by case, this provides a more meaningful normalisation. Subsequently, in the nnU-Net dataset.json file, the PET image is flagged "noNorm" to prevent additional rescaling and the CT utilises the standard CT normalisation parameter:

{ "channel_names": { "0": "noNorm", "1": "CT" }, "labels": { "background": 0, "ttb": 1, "norm": 2 }, "numTraining": 100, "file_ending": ".nii.gz" }

![ttb_and_normal_example](https://github.com/user-attachments/assets/163c3c6a-91f2-413d-ac3d-fb636aa74ead)

Example case with manual Total Tumour Burden (TTB) label provided (red) with derived physiological/normal VOI (blue)

It is then straightforward to train an nnU-Net model for each tracer and use some simple post-processing refinements to improve the agreement with the known threshold boundaries in the vicinity of inferred tumor voxels.

This method does not account for matched PSMA & FDG images from the same patient. The data in this challenge is provided such that the paired PSMA and FDG PET/CT images may be used to improve the discrimination of equivocal lesions. This is one of the research questions we hope to see explored by participants in DEEP-PSMA.
