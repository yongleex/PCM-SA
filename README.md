# PCM-SA: Projection Concentration Maximization with Smooth Annealing for EBIV
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code for the submitted paper *[Smooth Annealing Projection Concentration Maximization for Event-based Velocimetry](https://github.com/yongleex/PCM-SA)* and an accepted conference abstract *[A Projection Concentration Maximization Method for Event-based Imaging Velocimetry](https://github.com/yongleex/PCM-SA/blob/main/data/pcm.pdf)*.  In this work, we propose an event-based velocimetry framework that models the event projection distribution using continuous mixture of Gaussians, and formulates velocity estimation as a **concentration maximization problem**. The solution is obtained via Newton–Raphson iterative updates combined with **smooth annealing**.  


## Motivation
![demo](https://github.com/yongleex/PCM-SA/blob/main/data/motivation.png)

Different from conventional frame-based EBIV, PCM-SA directly optimizes the sharpness of projected event distributions, achieving accurate and robust velocity measurements under challenging recording conditions.  

More details are available in the paper ([link to be updated upon publication](https://github.com/yongleex/PCM-SA)).


## Experiments

Example Jupyter notebooks are provided to reproduce all results in the paper:

- **[Fig4_TIM.ipynb](https://github.com/yongleex/PCM-SA/blob/main/Fig4_TIM.ipynb)**  
Parameter sensitivity analysis and convergence performance with respect to the number of iterations.

- **[Fig8_TIM.ipynb](https://github.com/yongleex/PCM-SA/blob/main/Fig8_TIM.ipynb)**  
Tests on four synthetic flows (e.g., Lamb–Oseen vortex, sinusoidal wave).

- **[Fig12_TIM.ipynb](https://github.com/yongleex/PCM-SA/blob/main/Fig12_TIM.ipynb)**  
Investigating the influence of various recording parameters on measurement accuracy, such as particle diameter and motion velocity.

- **[Fig14_TIM.ipynb](https://github.com/yongleex/PCM-SA/blob/main/Fig14_TIM.ipynb)**  
Comparison of different baseline methods on uniform flow.

- **[Fig15_TIM.ipynb](https://github.com/yongleex/PCM-SA/blob/main/Fig15_TIM.ipynb)**  
Evaluation on real event-based velocimetry datasets.


## Install dependencies
We recommend using `conda`:

```bash
conda env create -f environment.yaml
```


### BibTeX

```
@inproceedings{ai2025projection,
  title = {A Projection Concentration Maximization Method for Event-Based Imaging Velocimetry},
  author = {Ai, Jia and Chen, Zuobing and Ning, Wenbin and Lee, Yong},
  booktitle = {16th International Symposium on Particle Image Velocimetry (ISPIV 2025)},
  year = {2025},
  address = {Tokyo, Japan}
}
```

```
@article{ai2025smooth,
  title={Smooth Annealing Projection Concentration Maximization for Event-based Velocimetr},
  author = {Ai, Jia and Chen, Zuobing and Ning, Wenbin and Lee, Yong},
  journal={Submitted},  
  year={2025},
  volume={},
  number={},
  pages={},
  doi={}
}

```

### Questions?
For any questions regarding this work, please email the first author at [aijia@whut.edu.com](mailto:aijia@whut.edu.com) or the correspoingding author at [yongli.cv@gmail.com](mailto:yongli.cv@gmail.com).

#### Acknowledgements
Parts of the code/deep net in this repository have been adapted from the following repos:

* [cewdlr/ebiv](https://github.com/cewdlr/ebiv)
* [TimoStoff/events_contrast_maximization](https://github.com/TimoStoff/events_contrast_maximization)
* [opencv/opencv](https://github.com/opencv/opencv)
* [pytorch/pytorch](https://github.com/pytorch/pytorch)
