# debiasing-skin

# Custom Data
## Normalized Background images

To generate the background normalized images, use the code on the folder 'norm-background'.
Example of usage:
python bg_norm_pix_crossdataset.py --train_image_path isic-rgb-299/ --train_mask_path isic-seg-299/ --train_csv_path isic-csv/isic2018-part2-all.csv --test_image_path atlas-rgb-299/ --test_mask_path atlas-seg-dermato-299/ test_csv_path atlas-csv/atlas-dermato-all.csv --output_path isic2018-norm-pix-alldata/

Notice that the images and masks must have the same size, and contain the same names.
The normalized background image is learned on the training set, and applied to both training and test sets.

## Artifact annotation
The artifact annotation for both ISIC 2018 - Task 1/2 (2594 images) and Atlas dermoscopy (872 images) is available at the folder 'artefacts-annotation'.
The tool used for this annotation is available at: https://github.com/phillipecardenuto/VisualizationLib

## Trap sets 
The trap sets are available in the folder 'isic-trap-csv', already divided in train, validation and test.

## Learning not to Learn
We followed the original implementation, available at https://github.com/feidfoe/learning-not-to-learn.

## Classification Networks
To train the classification networks for all the experiments in the paper, we used the implemenation available at: https://github.com/alceubissoto/deconstructing-bias-skin-lesion

