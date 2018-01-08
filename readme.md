# README - WikiArt_Style

## Description
This data set contains around 80.000 Western fine-art paintings labeled by style (total 27). These images are publicly visible on wikiart.org, where you can also read more about the background of the data. Tan et al. (2016), performed a research in which they tried to predict the style of a painting based on the paintings as input. They used a ConvNet inspired by AlexNet and managed to get an accuracy of 54.50% on a very similar dataset to this one. The researchers explain that recognizing a style of art is more difficult than recognizing object, because you have to take the ‘meaning’ of a painting into account.

This repository was used as a source of images and labels for this dataset. First the original images were scaled to 272 x 272 images. Then, for the training data, one random crop of 256 x 256 was taken from each image. Half of the training images were randomly flipped horizontally to prevent overfitting. For the validation data, a centered crop of 256 x 256 was taken of each original image. Finally the dataset was converted to .tfrecords files for use in TensorFlow. 

## Files
- style_train_1.tfrecords - style_train_10.tfrecords: the training dataset containing 57025 images
- style_val_1.tfrecords - style_val_10.tfrecords: the testing dataset containing 24421 images

The reason of splitting up the .tfrecords is for convenience. It is important that all files are used for their respective purpose since we did not divide the classes evenly over the files. The total size is 14.9 GB where the maximum size of a file is 1 GB.

## Features
- height [int64]: height of the image (256)
- width [int64]: width of the image (256)
- depth [int64]: depth of the image (3)
- label [int64]: style of the image (0-26)
- image_raw [byte/uint8 string]: tensor of height x width x depth with the color values (0-255) in RGB

## Style labels
It is important to note that some style classes (e.g. Impressionism: 9142 train images) are much better represented than others (e.g. Synthetic Cubism: 152 train images).
0. Abstract_Expressionism
1. Action_painting
2. Analytical_Cubism
3. Art_Nouveau
4. Baroque
5. Color_Field_Painting
6. Contemporary_Realism
7. Cubism
8. Early_Renaissance
9. Expressionism
10. Fauvism
11. High_Renaissance
12. Impressionism
13. Mannerism_Late_Renaissance
14. Minimalism
15. Naive_Art_Primitivism
16. New_Realism
17. Northern_Renaissance
18. Pointillism
19. Pop_Art
20. Post_Impressionism
21. Realism
22. Rococo
23. Romanticism
24. Symbolism
25. Synthetic_Cubism
26. Ukiyo_e

## Authors
Martin ter Haak (martth@kth.se)

Jelle van Miltenburg (jellevm@kth.se)

> ## Wikiart disclaimer
> Your use of the Service is at your sole risk. The Service is provided on an "AS IS" and "AS AVAILABLE" basis. The Service is provided without warranties of any kind, whether express or implied, including, but not limited to, implied warranties of merchantability, fitness for a particular purpose, non-infringement or course of performance.
> WikiArt its subsidiaries, affiliates, and its licensors do not warrant that a) the Service will function uninterrupted, secure or available at any particular time or location; b) any errors or defects will be corrected; c) the Service is free of viruses or other harmful components; or d) the results of using the Service will meet your requirements.

