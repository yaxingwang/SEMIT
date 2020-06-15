# Semi-supervised Learning for Few-shot Image-to-Image Translation 
### [[paper]](https://arxiv.org/abs/2003.13853)

# Abstract: 
In the last few years, unpaired image-to-image translation has witnessed remarkable progress. Although the latest methods are able to generate realistic images, they crucially rely on a large number of labeled images. Recently, some methods have tackled the challenging setting of few-shot image-to-image translation, reducing the labeled data requirements for the target domain during inference. In this work, we go one step further and reduce the amount of required labeled data also from the source domain during training. To do so, we propose applying semi-supervised learning via a noise-tolerant pseudo-labeling procedure. We also apply a cycle consistency constraint to further exploit the information from unlabeled images, either from the same dataset or external. Additionally, we propose several structural modifications to facilitate the image translation task under these circumstances. Our semi-supervised method for few-shot image translation, called SEMIT, achieves excellent results on four different datasets using as little as 10% of the source labels, and matches the performance of the main fully-supervised competitor using only 20% labeled data.
# Overview 
- [Dependences](#dependences)
- [Installation](#installtion)
- [Instructions](#instructions)
- [Framework](#Framework)
- [Results](#results)
- [References](#references)
- [Contact](#contact)
# Dependences 
- Python3.7, NumPy, SciPy,NVIDIA DGX1(8-V100, 32GB) 
- **Pytorch:** the version is 1.2 (https://pytorch.org/)
- **Dataset:** [Animal Face Dataset](https://github.com/NVlabs/FUNIT), [NABirds](https://dl.allaboutbirds.org/nabirds), [flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html), [UECFOOD256](http://foodcam.mobi/dataset256.html)  

# Installation 
- Install pytorch 
# Instructions
- Using 'https://github.com/yaxingwang/SEMIT.git'

    You will get new folder whose name is 'SEMIT' in your current path, then  use 'cd SEMIT' to enter the downloaded new folder
    
- Download dataset or use your dataset.

    Downloading the specific dataset (e.g.,'animals' ) and put  target path ('datasets/animals/'). Please check 'configs/animals.yaml' to learn the path information. 

- Run: 
    python train.py  --config configs/animals.yaml --output_path results/animals --multigpus

# Framework 
<br>
<p align="center"><img width="100%" height='60%'src="img/framework/framework.pdf" /></p>

# Results 
<br>
<p align="center"><img width="100%" height='60%'src="img/smaples/animals.png" /></p>


# References 
Our code  heavily rely on the following projects: 
- \[1\] '[Few-Shot Unsupervised Image-to-Image Translation](https://arxiv.org/pdf/1905.01723.pdf)' by Liu et. al, [code](https://github.com/NVlabs/FUNIT) 
- \[2\] '[Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks With Octave Convolution](https://arxiv.org/abs/1804.04732)' by Chen  et. al, [code](https://github.com/facebookresearch/OctConv) 

It would be helpful to understand this project if you are familiar with the above projects.
# Contact

If you run into any problems with this code, please submit a bug report on the Github site of the project. For another inquries pleace contact with me: yaxing@cvc.uab.es
