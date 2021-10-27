# [CMA-Net: A Cascaded Mutual Attention Network for Light Field Salient Object Detection](https://arxiv.org/abs/2105.00949)

Authors: [*Yi Zhang*](https://scholar.google.com/citations?user=NeHBHVUAAAAJ&hl=en), [*Lu Zhang*](https://luzhang.perso.insa-rennes.fr/), [*Wassim Hamidouche*](https://scholar.google.com/citations?user=ywBnUIAAAAAJ&hl=en), [*Olivier Deforges*](https://scholar.google.com/citations?user=c5DiiBUAAAAJ&hl=en)

# Introduction

<p align="center">
    <img src="./figures/fig_main.jpg" width="90%"/> <br />
    <em> 
    Figure 1: An overview of our CMA-Net. RGB-D high level features extracted from duel-branch encoder are fed into two proposed cascaded mutual attention
modules, followed by a group of (de-)convolutional layers used in BBSNet. The abbreviations in the figure are detailed as follows: AiF Image = all-in-focus
image. GT = ground truth. Resi = the ith ResNet layer. (De)Conv = (de-)convolutional layer. MAi = the ith mutual attention module. CMA = cascaded
mutual attention module. CW = column-wise normalization. RW = row-wise normalization.
    </em>
</p>

we propose CMA-Net, which consists of two novel cascaded mutual attention modules aiming at fusing the high level features from the modalities of all-in-focus and depth. Our proposed CMANet outperforms 30 state-of-the-art SOD methods on two widely applied light field benchmark datasets. Besides, the proposed CMA-Net is able to inference at a speed of 53 fps, thus being much faster than the top-ranked light field SOD methods. 
