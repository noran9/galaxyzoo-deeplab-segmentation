## Galaxy Zoo semantic segmentation with deeplab

Semantic segmentation of GalaxyZoo images using deeplab (state-of-the-art tensorflow model)

Folder structure:
- In GenerateDeepLabData.ipynb is the code for calculating and assigning a class to each image and creating the segmentation dataset using the images and training solutions
- small-training contains 100 images of the original dataset for demo purposes
- deeplab and slim are the tensorflow models used
- the image dataset is placed in deeplab/datasets/Galaxy zoo:
   - JPEGImages (original images)
   - SegmentationClass (segmentation ground truth images)
   - SegmentationClassRaw (index value images)
   - ImageSets (training and validation lists of images used)

  The images used in this project are part of the Galaxy zoo project for classification of space images:    
  https://data.galaxyzoo.org/
	
Preview of the images:

![images](/images/galaxy-zoo-images.png)

The dataset is processed in GenerateDeeplabData.ipynb in order to train the segmentation model. We have chosen 5 classes which correspond to the first question in the process of human classification in the galaxy zoo project. In the dataset these correspond to the morphological characteristics of class 1.1, 1.2, 1.3, 6.1, and 6.2 Our goal is to train the model to classify the galaxy in one of these classes using the image as input. After extracting 100 images for demo purposes, the ground truth images are created using opencv thresholding: 

![segmentedImages](/images/segmented-images.png)

These images are converted to index value images instead of RGB, to match the desired deeplab input which improves performance. 
Afterwards, tfrecord files are created by building the dataset to create the input to a tensorflow model. As an initialization model the pascal VOC pre-trained model is used. By running the training script we start the training process: 

![training](/images/training.png)

The evaluation script can be run afterwards.
