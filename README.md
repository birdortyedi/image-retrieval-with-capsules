# image-retrieval-with-capsules
Fashion Image Retrieval with Capsule Networks

*Accepted to the International Conference on Computer Vision, ICCV 2019, Workshop on Computer Vision for Fashion, Art and Design*

![architecture][arch]

[arch]: ./assets/model_arc.png

## TODO LIST

- [x] Literature search for clothing retrieval tasks
- [x] Getting permission for data set
- [x] Triplet directory iterator implementation
- [x] Triplet-based capsule network implementation
- [x] Train/Gallery/Query partitioning
- [x] Class partitioning
- [x] Triplet loss with Euclidean
- [x] Triplet loss with Cosine (not used)
- [x] Retrieval implementation
- [x] Architecture design
- [x] Starting the training process
- [x] Reconstruction or Regularization (not used)

## Project Details

Keras (Backend: TF) implementation of SCCapsNet and RCCapsNet

Base code for Capsule architecture: [XifengGuo](https://github.com/XifengGuo/CapsNet-Keras), [birdortyedi](https://github.com/birdortyedi/fashion-caps-net)

Dataset: 

Large-scale Fashion Recognition and Retrieval (DeepFashion) Dataset. ~50K In-Shop image pairs with 23 fine-grained categories. 

Environment:

* Intel Core i7-8700K CPU with 3.70GHz
* 32 GB RAM 
* 2 MSI GTX 1080 Ti Armor OC 11GB GPUs

## Installation

``` git clone git@github.com:birdortyedi/image-retrieval-with-capsules.git ```

## Prerequisites

Tested on Ubuntu 18.04 with:
* Python 3.5
* Tensorflow-gpu >= 1.14
* Keras = 2.2.4
* Keras-Applications = 1.0.8
* Keras-Preprocessing = 1.1.0
* numpy = 1.16.4
* h5py = 2.8.0
* colorama = 0.4.1
* tqdm = 4.32.1

## Run

``` python main.py --model_type <rc or sc> --filepath <dataset_folder> --save_dir <results_folder>```

## Implementation details
#### Hyper-parameter settings

| Hyper-parameter        | Value         |
| -------------          |:-------------:|
| Optimizer              | Adam          |
| Learning Rate          | 0.001         |
| Decay Rate             | 0.0005        |
| Batch Size             | 32            |
| Routings               | 3             |
| Normalization          | Pixel-wise    |

#### Augmentation

| Methods                | Range         |
| -------------          |:-------------:|
| Rotation               | [0-30]        |
| Width Shifting         | [0-0.1]       |
| Height Shifting        | [0-0.1]       |
| Brightness             | [0.5-1.5]     |
| Shearing               | [0-0.1]       |
| Zoom                   | [0-0.1]       |
| Flipping               | Horizontal    |

## Qualitative Results

![q][qualitative]

[qualitative]: ./assets/qualitative.png

## Quantitative Results

#### Comparison with the baseline study

* Recall@K performance of the variants of the baseline study and our proposed model. A: Number of attributes, L: Fashion landmarks, J: Human joints, P: Poselets

| Models                 | Top-20 (%)         | Top-50 (%)         |
| -------------          |:------------------:|:------------------:|
| FashionNet+100A+L      | 57.3               | 62.5               |
| FashionNet+500A+L      | 64.6               | 69.5               |
| FashionNet+1000A+J     | 68.0               | 73.5               |
| FashionNet+1000A+P     | 70.0               | 75.0               |
| FashionNet+1000A+L     | 76.4               | 80.0               |
| SCCapsNet *(ours)*     | 81.8               | 90.9               |
| RCCapsNet *(ours)*     | **84.6**           | **92.6**           |

#### Comparison with the SOTA methods

* Experimental results of in-shop clothing retrieval task on DeepFashion data set. "-": not reported

| Models                 | Top-1 (%)          | Top-10 (%)         | Top-20 (%)         | Top-30 (%)         | Top-40 (%)         | Top-50 (%)         |
| -------------          |:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
| WTBI                   | 35.0               | 47.0               | 50.6               | 51.5               | 53.0               | 54.5               |
| DARN                   | 38.0               | 56.0               | 67.5               | 70.0               | 72.0               | 72.5               |
| FashionNet             | 53.2               | 72.5               | 76.4               | 77.0               | 79.0               | 80.0               |
| Corbiere *et al.*      | 39.0               | 71.8               | 78.1               | 81.6               | 83.8               | 85.6               |
| SCCapsNet *(ours)*     | 32.1               | 72.4               | 81.8               | 86.3               | 89.2               | 90.9               |
| RCCapsNet *(ours)*     | 33.9               | 75.2               | 84.6               | 88.6               | 91.0               | 92.6               |
| HDC                    | 62.1               | 84.9               | 89.0               | 91.2               | 92.3               | 93.1               |
| VAM                    | 66.6               | 88.7               | 92.3               | -                  | -                  | -                  |
| BIER                   | 76.9               | 92.8               | 95.2               | 96.2               | 96.7               | 97.1               |
| HTL                    | 80.9               | 94.3               | 95.8               | 97.2               | 97.4               | 97.8               |
| A-BIER                 | 83.1               | 95.1               | 96.9               | 97.5               | 97.8               | 98.0               |
| ABE                    | 87.3               | 96.7               | 97.9               | 98.2               | 98.5               | 98.7               |

## Contacts

Please feel free to open an issue or to send an e-mail to `furkan.kinli@ozyegin.edu.tr`
