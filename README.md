# Object detection


This python program can be run from the command line and takes a video file as an argument.

The program displayes a four panel window with the original footage in the top left, the estimated background image in the top right, the moving objects in binary mask in the bottom left, and the isolated moving objects in original colour with noise suppression techniques applied in the bottom right.

Connected component analysis is also employed to estimate the number of objects per frame and whether these objects are people, cars or other based on their dimensions. This information is displayed in the console.

The method employed for detecting the moving objects is background modelling using a gaussian mixture model. Morphological operations are used to suppress noise.
