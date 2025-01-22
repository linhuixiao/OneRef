[//]: # (<br />)
<p align="center"> <h1 align="center">OneRef: Unified One-tower Expression Grounding and Segmentation with Mask Referring Modeling</h1>
  <p align="center">
    <b> NeurIPS 2024 </b>
    <br />
    <a href="https://scholar.google.com.hk/citations?user=4rTE4ogAAAAJ&hl=zh-CN&oi=sra"><strong> Linhui Xiao </strong></a>
    ·
    <a href="https://yangxs.ac.cn/home"><strong>Xiaoshan Yang </strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=HBZ9plsAAAAJ&hl=zh-CN"><strong>Fang Peng </strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=o_DllmIAAAAJ&hl=zh-CN"><strong>Yaowei Wang </strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=hI9NRDkAAAAJ&hl=zh-CN"><strong>Changsheng Xu</strong></a>
  </p>

  <p align="center">
    <a href='https://openreview.net/pdf?id=siPdcro6uD'>
      <img src='https://img.shields.io/badge/NeurIPS PDF-purple' alt='arXiv PDF'>
    </a>
    <a href='https://neurips.cc/virtual/2024/poster/93378'>
      <img src='https://img.shields.io/badge/NeurIPS Paper Homepage-blue' alt='arXiv PDF'>
    </a>
    <a href='https://neurips.cc/media/PosterPDFs/NeurIPS%202024/93378.png?t=1729402553.3015864'>
      <img src='https://img.shields.io/badge/NeurIPS Poster-lightblue' alt='arXiv PDF'>
    </a>
    <a href='https://neurips.cc/media/neurips-2024/Slides/93378_ROahXfO.pdf'>
      <img src='https://img.shields.io/badge/NeurIPS Slides-lightgreen' alt='arXiv PDF'>
    </a>
    <a href='https://arxiv.org/pdf/2410.08021'>
      <img src='https://img.shields.io/badge/arXiv-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'>
    </a>
<br />


<p align="center"> <img src='docs/fig1.jpg' align="center" width="95%"> </p>

**<p align="center"> A Comparison between OneRef model and the mainstream REC/RES architectures. </p>**

This repository is the official Pytorch implementation for the paper [**OneRef: Unified One-tower Expression Grounding 
and Segmentation with Mask Referring Modeling**](https://openreview.net/pdf?id=siPdcro6uD), which is an advanced version
of our preliminary work **HiVG** ([Publication](https://dl.acm.org/doi/abs/10.1145/3664647.3681071), [Paper](https://openreview.net/pdf?id=NMMyGy1kKZ), 
[Code](https://github.com/linhuixiao/HiVG)) and **CLIP-VG** ([Publication](https://ieeexplore.ieee.org/abstract/document/10269126),
[Paper](https://arxiv.org/pdf/2305.08685), [Code](https://github.com/linhuixiao/CLIP-VG)). 

If you have any questions, please feel free to open an issue or contact me with emails: <xiaolinhui16@mails.ucas.ac.cn>.
Any kind discussions are welcomed!

**Please leave a <font color='orange'>STAR ⭐</font> if you like this project!**

## News

- **All of the code and models will be released soon!**
- :fire: **Update on 2024/12/28: We conducted a survey of Visual Grounding over the past decade, entitled "Towards Visual Grounding: A Survey" ([Paper](https://arxiv.org/pdf/2412.20206), [Project](https://github.com/linhuixiao/Awesome-Visual-Grounding)), Comments are welcome !!!**
- :fire: **Update on 2024/10/10: Our new work **OneRef** has been accepted by the top conference NeurIPS 2024 ! ([paper](https://arxiv.org/abs/2410.08021), [github](https://github.com/linhuixiao/OneRef))**
- **Update on 2024/07/16:** Our grounding work HiVG ([Publication](https://dl.acm.org/doi/abs/10.1145/3664647.3681071), [Paper](https://openreview.net/pdf?id=NMMyGy1kKZ), [Code](https://github.com/linhuixiao/HiVG)) has been accepted by the top conference ACM MM 2024 !
- **Update on 2023/9/25:** Our grounding work CLIP-VG has been accepted by the top journal IEEE Transaction on Multimedia (2023)! ([paper](https://ieeexplore.ieee.org/abstract/document/10269126), [github](https://github.com/linhuixiao/CLIP-VG))


## Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.   

```bibtex
@inproceedings{xiao2024oneref,
  title={OneRef: Unified One-tower Expression Grounding and Segmentation with Mask Referring Modeling},
  author={Xiao, Linhui and Yang, Xiaoshan and Peng, Fang and Wang, Yaowei and Xu, Changsheng},
  booktitle={Proceedings of the 38th International Conference on Neural Information Processing Systems},
  year={2024}
}
```

<h3 align="left">
Links: 
<a href="https://arxiv.org/abs/2410.08021">ArXiv</a>, 
<a href="https://neurips.cc/virtual/2024/poster/93378">NeurIPS 2024</a>
</h3>


## TODO

The code is currently being tidied up, and both the code and model will be made publicly available soon!

- [ ] Release all the checkpoints.
- [ ] Release the full model code, training and inference code.



## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Contacts](#contacts)
5. [Acknowledgments](#acknowledgments)


## Highlight

- **(i) We pioneer the application of mask modeling to referring tasks by introducing a novel paradigm called mask referring modeling.** This paradigm
effectively models the referential relation between visual and language. 
- **(ii) Diverging from previous works, we propose a remarkably concise one-tower framework for grounding and referring 
segmentation in a unified modality-shared feature space.** Our model eliminates the commonly used modality 
interaction modules, modality fusion en-/decoders, and special grounding tokens. 
- **(iii) We extensively validate the effectiveness of OneRef in three referring tasks on five datasets.** Our method consistently
surpasses existing approaches and achieves SoTA performance across several settings, providing a
valuable new insights for future grounding and referring segmentation research.


## Introduction

Constrained by the separate encoding of vision and language, existing grounding
and referring segmentation works heavily rely on bulky Transformer-based fusion
en-/decoders and a variety of early-stage interaction technologies. Simultaneously,
the current mask visual language modeling (MVLM) fails to capture the nuanced
referential relationship between image-text in referring tasks. In this paper, we
propose **OneRef, a minimalist referring framework built on the modality-shared
one-tower transformer that unifies the visual and linguistic feature spaces**. To
modeling the referential relationship, we introduce a novel **MVLM paradigm** called
**Mask Referring Modeling (MRefM)**, which encompasses both referring-aware
mask image modeling and referring-aware mask language modeling. Both modules not 
only reconstruct modality-related content but also cross-modal referring
content. Within MRefM, we propose a referring-aware dynamic image masking
strategy that is aware of the referred region rather than relying on fixed ratios
or generic random masking schemes. By leveraging the unified visual language
feature space and incorporating MRefM’s ability to model the referential relations,
our approach enables direct regression of the referring results without resorting
to various complex techniques. Our method consistently surpasses existing approaches
and achieves SoTA performance on both grounding and segmentation
tasks, providing valuable insights for future research.

For more details, please refer to [our paper](https://openreview.net/pdf?id=siPdcro6uD).


## Usage
### Dependencies
- Python 3.9.10
- PyTorch 1.9.0 + cu111 + cp39
- Check [requirements.txt](requirements.txt) for other dependencies. 

Our model is **easy to deploy** in a variety of environments and **has been successfully tested** on multiple pytorch versions.


### Image Data Preparation
1.You can download the images from the original source and place them in your disk folder, such as `$/path_to_image_data`:
- [MS COCO 2014](download_mscoco2014.sh) (for RefCOCO, RefCOCO+, RefCOCOg dataset, almost 13.0GB) 
- [ReferItGame](https://drive.google.com/drive/folders/1D4shieeoKly6FswpdjSpaOrxJQNKTyTv)
- [Flickr30K Entities](http://shannon.cs.illinois.edu/DenotationGraph/#:~:text=make%20face-,Downloads,-Please%20fill%20in)

   We provide a script to download the mscoco2014 dataset, you just need to run the script in terminal with the following command:
   ```
   bash download_mscoco2014.sh
   ```
   Or you can also follow the data preparation of TransVG, which can be found in [GETTING_STARTED.md](https://github.com/djiajunustc/TransVG/blob/main/docs/GETTING_STARTED.md).

Only the image data in these datasets is used, and these image data is easily find in similar repositories of visual grounding work, such as [TransVG](https://github.com/linhuixiao/TransVG) etc. 
Finally, the `$/path_to_image_data` folder will have the following structure:

```angular2html
|-- image_data
   |-- Flickr30k
      |-- flickr30k-images
   |-- other
      |-- images
        |-- mscoco
            |-- images
                |-- train2014
   |-- referit
      |-- images
```
- ```$/path_to_image_data/image_data/Flickr30k/flickr30k-images/```: Image data for the Flickr30K dataset, please download from this [link](http://shannon.cs.illinois.edu/DenotationGraph/#:~:text=make%20face-,Downloads,-Please%20fill%20in). Fill the form and download the images.
- ```$/path_to_image_data/image_data/other/images/```: Image data for RefCOCO/RefCOCO+/RefCOCOg, i.e., mscoco2014. 
- ```$/path_to_image_data/image_data/referit/images/```: Image data for ReferItGame.

## Text-Box Anotations 
The labels in the fully supervised scenario is consistent with previous works such as [TransVG](https://github.com/linhuixiao/TransVG).


### Fully supervised setting
<table>
    <tr> <!-- line 3 -->
    <th style="text-align:center" > Datasets </th>
    <th style="text-align:center" > RefCOCO </th>
    <th style="text-align:center" > RefCOCO+ </th>
    <th style="text-align:center" > RefCOCOg-g </th>
    <th style="text-align:center" > RefCOCOg-u </th>
    <th style="text-align:center" > ReferIt </th>
    <th style="text-align:center" > Flickr </th>
    <th style="text-align:center" > Mixup1 </th>
    <th style="text-align:center" > Mixup2 </th>
    </tr>
    <tr> <!-- line 2 -->
        <th style="text-align:center" rowspan="1"> url, size </th> <!-- table head -->
        <th style="text-align:center" colspan="8"> <a href="https://drive.google.com/file/d/1ituKSxWU5aXsGnXePd7twv7ImJoFiATc/view?usp=drive_link">All of six datasets</a>,  89.0MB </th>  <!-- table head -->
</tr>
</table>

\* The mixup1 denotes the mixup of the training data from RefCOCO/+/g-umd (without use gref), which used in RES task. The mixup2 denotes the 
mixup of the training data from RefCOCO/+/g (without use gref) and ReferIt Game, which used in REC task. The val and test split of both Mixup1
and Mixup2 are used the val and testA file from RefCOCOg. The training data in RefCOCOg-g (i.e., gref) exist data leakage.


Download the above annotations to a disk directory such as `$/path_to_split`; then will have the following similar directory structure:

```angular2html
|-- /full_sup_data
    ├── flickr
    │   ├── flickr_test.pth
    │   ├── flickr_train.pth
    │   └── flickr_val.pth
    ├── gref
    │   ├── gref_train.pth
    │   └── gref_val.pth
    ├── gref_umd
    │   ├── gref_umd_test.pth
    │   ├── gref_umd_train.pth
    │   └── gref_umd_val.pth
    ├── mixup1
    │   ├── mixup1_test.pth
    │   ├── mixup1_train.pth
    │   └── mixup1_val.pth
    ├── mixup2
    │   ├── mixup2_test.pth
    │   ├── mixup2_train.pth
    │   └── mixup2_val.pth
    ├── referit
    │   ├── referit_test.pth
    │   ├── referit_train.pth
    │   └── referit_val.pth
    ├── unc
    │   ├── unc_testA.pth
    │   ├── unc_testB.pth
    │   ├── unc_train.pth
    │   └── unc_val.pth
    └── unc+
        ├── unc+_testA.pth
        ├── unc+_testB.pth
        ├── unc+_train.pth
        └── unc+_val.pth
```


## Pre-trained Checkpoints

### Fully supervised setting

<table>
    <tr> <!-- line 3 -->
    <th style="text-align:center" > Datasets </th>
    <th style="text-align:center" > RefCOCO </th>
    <th style="text-align:center" > RefCOCO+ </th>
    <th style="text-align:center" > RefCOCOg-g </th>
    <th style="text-align:center" > RefCOCOg-u </th>
    <th style="text-align:center" > ReferIt </th>
    <th style="text-align:center" > Flickr </th>
    </tr>
    <tr> <!-- line 3 -->
    <th style="text-align:center" > separate </th>
    <th style="text-align:center" > <a href="todo">model</a> </th>
    <th style="text-align:center" > <a href="todo">model</a> </th>
    <th style="text-align:center" > <a href="todo">model</a> </th>
    <th style="text-align:center" > <a href="todo">model</a> </th>
    <th style="text-align:center" > <a href="todo">model</a> </th>
    <th style="text-align:center" > <a href="todo">model</a> </th>
    </tr>
    <tr> <!-- line 2 -->
        <th style="text-align:center" rowspan="1"> url, size </th> <!-- table head -->
        <th style="text-align:center" colspan="6"> <a href="todo">All of six models (All have not ready)</a>  </th>  <!-- table head -->
    </tr>
</table>

The checkpoints include the Base model and Large mode under the fine-tuning setting and dataset-mixed pretraining setting. 


## Training and Evaluation

You just only need to change ```$/path_to_split```, ``` $/path_to_image_data```, ``` $/path_to_output``` to your own file directory to execute the following command.
The first time we run the command below, it will take some time for the repository to download the CLIP model.

1. Training on RefCOCO with fully supervised setting. 
    The only difference is an additional control flag: ```--sup_type full```
    ```
    CUDA_VISIBLE_DEVICES=3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=5 --master_port 28887 --use_env train_clip_vg.py --num_workers 32 --epochs 120 --batch_size 64 --lr 0.00025  --lr_scheduler cosine --aug_crop --aug_scale --aug_translate    --imsize 224 --max_query_len 77  --sup_type full --dataset unc      --data_root $/path_to_image_data --split_root $/path_to_split --output_dir $/path_to_output/output_v01/unc;
    ```
    Please refer to [train_and_eval_script/train_and_eval_full_sup.sh](train_and_eval_script/train_and_eval_full_sup.sh) for training commands on other datasets.

2. Evaluation on RefCOCO. The instructions are the same for the fully supervised Settings.
    ```
    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port 28888 --use_env eval.py --num_workers 2 --batch_size 128    --dataset unc      --imsize 224 --max_query_len 77 --data_root $/path_to_image_data --split_root $/path_to_split --eval_model $/path_to_output/output_v01/unc/best_checkpoint.pth      --eval_set val    --output_dir $/path_to_output/output_v01/unc;
    ```
    Please refer to [train_and_eval_script/train_and_eval_unsup.sh](train_and_eval_script/train_and_eval_unsup.sh) for evaluation commands on other splits or datasets.
    
3. We strongly recommend to use the following commands to training or testing with different datasets and splits, 
    which will significant reduce the training workforce.
    ```
    bash train_and_eval_script/train_and_eval_full_sup.sh
    ```



## Results

### 1. REC task
<details open>
<summary><font size="4">
REC Single-dataset Fine-tuning SoTA Result Table
</font></summary>
<img src="docs/tab1.jpg" alt="COCO" width="100%">
</details>

<details open>
<summary><font size="4">
REC Dataset-mixed Pretraining SoTA Result Table
</font></summary>
<img src="docs/tab2.jpg" alt="COCO" width="100%">
</details>

### 2. RES task

<details open>
<summary><font size="4">
RES Single-dataset Fine-tuning and Dataset-mixed Pretraining SoTA Result Table (mIoU)
</font></summary>
<img src="docs/tab3.jpg" alt="COCO" width="100%">
</details>


<details open>
<summary><font size="4">
RES Single-dataset Fine-tuning and Dataset-mixed Pretraining SoTA Result Table (oIoU)
</font></summary>
<img src="docs/tab4.jpg" alt="COCO" width="100%">
</details>

### 3. Our model also has significant energy efficiency advantages.

<details open>
<summary><font size="4">
Comparison of the computational cost in REC task.
</font></summary>
<div align=center>
<img src="docs/tab5.jpg" alt="COCO" width="70%"></div>
</details>



## Methods 

<p align="center"> <img src='docs/fig2.jpg' align="center" width="100%"> </p>

**<p align="center"> An Illustration of our multimodal Mask Referring Modeling (MRefM) paradigm, which
includes Referring-aware mask image modeling and Referring-aware mask language modeling. </p>**


<p align="center"> <img src='docs/fig3.jpg' align="center" width="100%"> </p>

**<p align="center">  An Illustration of the referring-based grounding and segmentation transfer. </p>**

<p align="center"> <img src='docs/fig4.jpg' align="center" width="100%"> </p>

**<p align="center"> Illustrations of random masking (MAE) [27], block-wise masking (BEiT) [4], and our
referring-aware dynamic masking. α denotes the entire masking ratio, while β and γ denote the
masking ratio beyond and within the referred region. </p>**


## Visualization
<p align="center"> <img src='docs/fig6.jpg' align="center" width="80%"> </p>
  
**<p align="center">  Qualitative results on the RefCOCO-val dataset. </p>**

<p align="center"> <img src='docs/fig7.jpg' align="center" width="80%"> </p>
  
**<p align="center">  Qualitative results on the RefCOCO+-val dataset. </p>**

<p align="center"> <img src='docs/fig8.jpg' align="center" width="80%"> </p>
  
**<p align="center">  Qualitative results on the RefCOCOg-val dataset. </p>**

Each example shows two different query texts. From left to right: the original input image, the ground truth with
box and segmentation mask (in green), the RES prediction of OneRef (in cyan), the REC prediction
of OneRef (in cyan), and the cross-modal feature.


## Contacts
Email: <xiaolinhui16@mails.ucas.ac.cn>.
Any kind discussions are welcomed!

## Acknowledgement

Our model is related to [BEiT-3](https://github.com/microsoft/unilm/tree/master/beit3) and 
[MAE](https://github.com/facebookresearch/mae). Thanks for their great work!

We also thank the great previous work including [TransVG](https://github.com/linhuixiao/TransVG), 
[DETR](https://github.com/facebookresearch/detr), [CLIP](https://github.com/openai/CLIP), 
[CLIP-VG](https://github.com/linhuixiao/CLIP-VG), etc. 

Thanks [Microsoft](https://github.com/microsoft/unilm) for their awesome models.



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=linhuixiao/OneRef&type=Date)](https://star-history.com/#linhuixiao/OneRef&Date)









