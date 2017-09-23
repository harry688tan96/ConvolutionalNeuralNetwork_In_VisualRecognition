# Convolutional Neural Networks for Visual Recognition


## This course is offered by the [Stanford University](http://cs231n.stanford.edu/) at Spring 2017.

Recent developments in neural network (aka “**deep learning**”) approaches have greatly advanced the performance of these state-of-the-art visual recognition systems. This course is a deep dive into details of the deep learning architectures with a focus on learning end-to-end models for these tasks, particularly image classification. 

## What did I learn in this course?
> * **Convolutional Neural Nets** (Deep Learning) in classifying images with high accuracy.
> * **Hyperparameters tuning** for better optimization and convergence rate during training phase.
> * **Activation functions** (ie. RELU, tanh) to introduce non-linearity in neural networks.<sup>[1]</sup>
> * Pros and cons of different **optimization algorithms in Neural Networks** (ie. *SGD with momentum*, *RMSProp*, *AdaGrad*, *Adam* etc.), and implemented them from scratch.<sup>[1]</sup>
> *  **Backpropagation** in Neural Networks using **computational graph**.<sup>[1]</sup>
> * **BatchNormalization** and **Dropout** which improves optimization.<sup>[1]</sup>
> * Combined **Convolutional Neural Nets** and **LSTM Recurrent Nets** to implement an image captioning system.
> * **Style Transfer**, **Generative Adversarial Network(GAN)** in generating realistic fake images.
> * **Visualizing Image Classification** to understand what Convolutional Neural Nets understands about images.
> * **Tensorflow** framework.
> * **Transfer Learning**.

<sub>*[1] Implemented from scratch without Tensorflow.*</sub>


## Cool projects completed in this course:
### :ghost: [Style Transfer](https://github.com/harry688tan96/VisualRecognition_DeepLearning/blob/master/assignment3/StyleTransfer-TensorFlow.ipynb)
* Implemented Style Transfer technique on images using pretrained [SqueezeNet](https://arxiv.org/abs/1602.07360).

Styles from various paintings are added to a photo of the M3 Math Building in University of Waterloo. Click on thumbnails to see full applied style images.
<p align="center">
  <img height="200px" src='assignment3/styles/Waterloo_Math_Faculty.jpg'>
</p>
<p align = 'center'>
<a href = 'assignment3/styles/muse.jpg'> <img src = 'assignment3/styles/muse.jpg' height = '200px' width='162'></a>
<img src = 'styleExamples/Math_TheMuse.jpeg' height = '200px'>
<img src = 'styleExamples/Math_TheScream.jpeg' height = '200px'>
<a href = 'assignment3/styles/the_scream.jpg'><img src = 'assignment3/styles/the_scream.jpg' height = '200px' width='161'></a>
<br>
<a href = 'assignment3/styles/nebula_planets.jpg'><img src = 'assignment3/styles/nebula_planets.jpg' height = '200px' width='163'></a>
<img src = 'styleExamples/Math_NebulaPlanets.jpeg' height = '200px'>
<img src = 'styleExamples/Math_Sailboat.jpeg' height = '200px'>
<a href = 'assignment3/styles/Sailboat_OilPaint.jpg'><img src = 'assignment3/styles/Sailboat_OilPaint.jpg' height = '200px' width = '161'></a>
<br>
<a href = 'assignment3/styles/PrettyNight_OilPaint.jpg'><img src = 'assignment3/styles/PrettyNight_OilPaint.jpg' height = '200px'></a>
<img src = 'styleExamples/Math_PrettyNight.jpeg' height = '200px'>
</p>

### :ghost: [Class Visualization](https://github.com/harry688tan96/VisualRecognition_DeepLearning/blob/master/assignment3/NetworkVisualization-TensorFlow.ipynb)
* Produced images that the Deep Neural Network (DNN) will recognize as the target class. These images which were shown to individual neurons in the DNN maximally activated these neurons. Thus, the neurons learned how to "differentiate" images that were fed to them.
##### Results:
How the neurons in DNN see **Tarantula**:
<p align = 'center'>
<img src = 'classVisualize/tarantula_100.png' height = '200px'>
<img src = 'classVisualize/tarantula_300.png' height = '200px'>
<img src = 'classVisualize/tarantula_1000.png' height = '200px'>
</p>
How the neurons in DNN see **Gorilla**:
<p align = 'center'>
<img src = 'classVisualize/Gorilla_100.png' height = '200px'>
<img src = 'classVisualize/Gorilla_300.png' height = '200px'>
<img src = 'classVisualize/Gorilla_1000.png' height = '200px'>
</p>

### :ghost: [Generative Adversarial Networks (GANs)](https://github.com/harry688tan96/VisualRecognition_DeepLearning/blob/master/assignment3/GANs-TensorFlow.ipynb)
* Generated realistic fake MNIST images using the following Generative Adversarial Networks technique:
  * Vanilla GANs
  * Least Square GANs
  * Deep Convolutional GANs
##### Results:
<table align="center">
  <caption></caption>
  <tr>
    <th>Vanilla GANs</th>
    <th>Least Squares GANs</th>
    <th>DC GANs</th>
  </tr>
  <tr>
    <td><img src = 'GANs_examples/Vanilla GANs.png' height = '200px'></td>
    <td><img src = 'GANs_examples/LeastSquares_GANs.png' height = '200px'></td>
    <td><img src = 'GANs_examples/DC_GANs.png' height = '200px'></td>
  </tr>
</table>
