### CNN(Convolutional Neural Network) Cat-Dog Classfier

An algorithm to distinguish dogs from cats using Pytorch



**1. Dataset**

Data from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats)

The training archive contains 25,000 images of dogs and cats. 



**2. Model**

Convolutional Neural Networks (CNN)



**3. Architecture**

    1. INPUT [256x256x3]: the raw pixel values of the image
       (width 256, height 256, and with three color channels R,G,B.)
    2. CONV layer 
    3. RELU layer 
    4. POOL layer 
    5. FC (i.e. fully-connected) layer 
    6. stride = 1
    7. padding = 1 
    
    

**4. Conclusion** 

   The ConvNet architecture and parameters used in this Convolutional Neural Network are capable of producing accuracy of 94% on Validation Data. It is possible to achieve more accuracy on this dataset using deeper network layers and fine tuning of hyper parameters for training. 
