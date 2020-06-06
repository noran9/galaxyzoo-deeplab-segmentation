## Galaxy Zoo semantic segmentation with deeplab

Semantic segmentation of GalaxyZoo images using deeplab (state-of-the-art tensorflow model)

Folder structure:
- In GenerateAndAnalyseDeepLabData.ipynb is the code for calculating and assigning a class to each image and creating the segmentation dataset using the images and training solutions
- In Galaxy_ZooDeeplabColabTraining.ipynb is the code used for training on Google Colab with varying parametes
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

# Segmentation Evaluation

12 different experiments were conducted with varying parameters in order to evaluate the segmentation accuracy, by using a smaller training dataset of 1400 images with ground truth generated in one class color. In addition to the model evaluation function, a Dice Coefficient calculating function is added in the jupyter notebook (GenerateAndAnalyseDeepLabData). The Dice Coefficient value highlights the difference between the predicted segmentation and the calculated ground truth for the testing image on a pixel level.

The training experiments were conducted using a seperate notebook on Google Colab (Galaxy_ZooDeeplabColabTraining).


|    | Batch size | Crop-size | Iterations | Fine - tuning | Initial checkpoint    | Training Time     | Time in minutes | Tensorflow Evaluation | Dice coefficient |
|----|------------|------------|------------|---------------|-----------------------|-------------------|-----------------|-----------------------|------------------|
| 1  | 4          | 424 x 424  | 9000       | TRUE          | PascalVOC             | 4 hours           | 240             | 0.389497876           | 0.5125081108     |
| 2  | 4          | 424 x 424  | 14000      | TRUE          | PascalVOC             | 7 hours           | 420             | 0.512435              | 0.556208577      |
| 3  | 8          | 212 x 212  | 3000       | TRUE          | PascalVOC             | 20 minutes        | 20              | 0.489129335           | 0.3145089413     |
| 4  | 8          | 424 x 424  | 3000       | TRUE          | PascalVOC             | 1 hour            | 60              | 0.780309916           | 0.4000274228     |
| 5  | 16         | 212 x 212  | 1500       | TRUE          | PascalVOC             | 30 minutes        | 30              | 0.517493546           | 0.03604324644    |
| 6  | 16         | 212 x 212  | 5000       | TRUE          | PascalVOC             | 2.5 hours         | 150             | 0.325162381           | 0.5012619152     |
| 7  | 16         | 212 x 212  | 5000       | TRUE          | Random initialization | 50 minutes        | 50              | 0.489129335           | 0                |
| 8  | 8          | 212 x 212  | 7000       | FALSE         | PascalVOC             | 2 hours           | 120             | 0.844646156           | 0.7304488889     |
| 9  | 4          | 424 x 424  | 6000       | FALSE         | PascalVOC             | 2 hours           | 120             | 0.863921046           | 0.5577976091     |
| 10 | 32         | 212 x 212  | 1200       | TRUE          | PascalVOC             | 20 minutes        | 20              | 0.56085               | 0.07685808697    |
| 11 | 32         | 212 x 212  | 5500       | TRUE          | PascalVOC             | 1 hour 45 minutes | 105             | 0.195511907           | 0.4863402111     |
| 12 | 4          | 212 x 212  | 7000       | FALSE         | PascalVOC             | 1 hour 10 minutes | 70              | 0.835105538           | 0.7059139809     |

Visualisation of the results:

![evaluation](/images/TrainingEvaluation.png)

The experiments aim at locating parameters that have short training time and high accuracy. The best experiments in the 1st quadrant of the chart are the 8th and 12th entry.

In the 8th entry a batch size of 8 was used and 7000 iterations, resulting in experiment with the highest accuracy.
In the last experiment, a batch size of 4 was used with the same parameters, resulting in similar accuracy value, but decreasing the training time for 50 minutes compared to the most accurate entry.

In one of the experiments, random initialisation of the weights is used, instead of using the frozen PascalVOC graph for transfer learning. In this case the learning rate was very slow and the number of iterations were not enough for training, resulting with a Dice Coefficient of 0. 

In some experiments a smaller image crop size was used for training in order to decrease the strain on memory.
