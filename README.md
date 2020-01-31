
# MIRROR


Xiaochuan Zhang,
Wenjie Sun, 
Jianmin Pang,
Fudong Liu
and Zhen Ma

### Table of Contents
1. [Introduction](#introduction)
2. [Citation](#citation)
3. [Installation](#installation)
4. [Requirements and Dependencies](#requirements-and-dependencies)
5. [Data Preprocessing](#data_preprocessing)
6. [Pretrain](#pretrain)
7. [Train](#prain) 

### Introduction
This is the the prototype system of paper Similarity Metric Method for Binary Basic Blocks of Cross-Instruction Set Architecture.

### Citation
If you find the code and datasets useful in your research, please cite:

    @inproceedings{mirror,
	  title={Similarity Metric Method for Binary Basic Blocks of Cross-Instruction Set Architecture},
	  author={Xiaochuan, Zhang and Wenjie, Sun and Jianmin, Pang and Fudong, Liu and Zhen, Ma},
	  booktitle={Proceedings of the NDSS Workshop on Binary Analysis Research},
	  year={2020}
	}

### Requirements and Dependencies
- Ubuntu (We test with Ubuntu = 18.04 LTS)
- Python (We test with Python = 3.7.4)
- CUDA & cuDNN (We test with CUDA = 10.2 and cuDNN = 7.6.5)
- PyTorch （We test with PyTorch = 1.0.0）
- NVIDIA GPU(s) (We use 4 RTX 2080Ti)

### Installation

Download repository:

```
$ git clone https://github.com/zhangxiaochuan/MIRROR.git
$ cd MIRROR
```

The dataset **MISA** (Multi-ISAs basic block dataset) is available at [link](https://drive.google.com/open?id=1krJbsfu6EsLhF86QAUVxVRQjbkfWx7ZF). Please download and uncompress ``MISA.zip``, and place MISA in the root directory of the project.


### Data Preprocessing



```
$ python data_manager.py
```

### Pretrain

```
$ python pretrain.py
```

### train

```
$ python pretrain.py
```

### Contact
[Xiaochuan Zhang](mailto:zhangxiaochuan@outlook.com)

### License
See [MIT License](https://github.com/baowenbo/DAIN/blob/master/LICENSE)