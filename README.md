# AEPT
This is an official implemention for "Adaptive ensemble perception tracking".


## Dependencies
* python 3.7
* pytorch 1.1
* numpy
* CUDA 10
* skimage
* matplotlib
* python-opencv
* jeph4py
* scipy
* yacs
* tensorboardX

## Models and Raw Results
We provide the [pre-trained models and raw results](https://drive.google.com/drive/folders/1r-25gTFA4deCpykgIMoqixes1fJCSMRz?usp=sharing)

## Testing:
```
python test_vot18.py
python test_vot16.py
```

## Acknowledgment
We use the source codes from [Pytracking](https://github.com/visionml/pytracking) and [FCOS](https://github.com/tianzhi0549/FCOS). We sincerely thank the authors.

### Citation
If you're using this code in a publication, please cite our paper.

  @article{zhou2021adaptive,  
    &emsp;title={Adaptive ensemble perception tracking},  
    &emsp;author={Zhou, Zikun and Fan, Nana and Yang, Kai and Wang, Hongpeng and He, Zhenyu},  
    &emsp;journal={Neural Networks},  
    &emsp;volume={142},  
    &emsp;pages={316--328},  
    &emsp;year={2021},  
    &emsp;publisher={Elsevier}  
  }
