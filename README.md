# Debiasing Skin Lesion Datasets and Models? Not So Fast
Code to reproduce the results for the paper "Debiasing Skin Lesion Datasets and Models? Not So Fast" in ISIC Skin Image Analysis Workshop @ CVPR 2020.  [Link to the paper.](http://openaccess.thecvf.com/content_CVPRW_2020/papers/w42/Bissoto_Debiasing_Skin_Lesion_Datasets_and_Models_Not_So_Fast_CVPRW_2020_paper.pdf)

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

## Citation
```
@inproceedings{bissoto19deconstructing,
 author    = {Alceu Bissoto and Eduardo Valle and Sandra Avila},
 title     = {Debiasing Skin Lesion Datasets and Models? Not So Fast},
 booktitle = {ISIC Skin Image Anaylsis Workshop, 2020 {IEEE} Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
 year      = {2020},
}
```
## Acknowledgments
A. Bissoto is partially funded by CAPES (88887. 388163/2019-00). S. Avila is partially funded by FAPESP (2017/16246-0, 2013/08293-7). A. Bissoto and S. Avila are also partially funded by Google LARA 2019. E. Valle is partially funded by a CNPq PQ-2 grant (311905/2017-0), and by a FAPESP grant (2019/05018-1). This project is partially funded by a CNPq Universal grant (424958/2016-3). RECODLab. is supported by projects from FAPESP, CNPq, and CAPES.
We acknowledge the donation of GPUs by NVIDIA. We thank João Cardenuto for providing the annotation tool.
