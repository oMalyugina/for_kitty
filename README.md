You can find a solution for first task here. I choose the image classification task, because my private laptop is 
old and don't have a lot of power. Second and third task I couldn't do in one week 
(i think to training on cpu top layers for pretrained voxelnet takes one day minimun). And I also don't have a nessecary 
software (pcl, ros) for visualisation. Installation of all this staff can take from 2 hours to 8 hours.

So. There is solution for object classification here.

firstly, what you can find in my code:
1. visualisation of input data you can find in notebook "visualisation_input_data.ipynb". You have to define paths to
 left color images and training labels from http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d 
2. cutting and resizing objects from image for training you can find in "data_generator.py"
3. there are training, computing errors ans errors visualisation in "train.ipynd"
4. and there are models, trained by me, in models.py

You can find comments in notebooks and files. Welcome:) 