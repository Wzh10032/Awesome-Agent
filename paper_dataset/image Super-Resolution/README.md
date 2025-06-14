# Awesome image Super-Resolution Research

> 自动生成的论文列表，更新于: 2025-06-03

## 目录
- [Image Super-Resolution](#image-super-resolution)

---

## Image Super-Resolution

### [Super-Resolution through StyleGAN Regularized Latent Search: A Realism-Fidelity Trade-off](http://arxiv.org/pdf/2311.16923v1)

**Authors**: Marzieh Gheisari, Auguste Genovesio

**Published**: Nov 28, 2023 | **Updated**: Nov 28, 2023 | **Arxiv ID**: [2311.16923v1](http://arxiv.org/pdf/2311.16923v1)

**Abstract**:
> This paper addresses the problem of super-resolution: constructing a highly
> resolved (HR) image from a low resolved (LR) one. Recent unsupervised
> approaches search the latent space of a StyleGAN pre-trained on HR images, for
> the image that best downscales to the input LR image. However, they tend to
> produce out-of-domain images and fail to accurately reconstruct HR images that
> are far from the original domain. Our contribution is twofold. Firstly, we
> introduce a new regularizer to constrain the search in the latent space,
> ensuring that the inverted code lies in the original image manifold. Secondly,
> we further enhanced the reconstruction through expanding the image prior around
> the optimal latent code. Our results show that the proposed approach recovers
> realistic high-quality images for large magnification factors. Furthermore, for
> low magnification factors, it can still reconstruct details that the generator
> could not have produced otherwise. Altogether, our approach achieves a good
> trade-off between fidelity and realism for the super-resolution task.

---

### [Towards Arbitrary-scale Histopathology Image Super-resolution: An Efficient Dual-branch Framework based on Implicit Self-texture Enhancement](http://arxiv.org/pdf/2304.04238v1)

**Authors**: Linhao Qu, Minghong Duan, Zhiwei Yang, Manning Wang, Zhijian Song

**Published**: Apr 09, 2023 | **Updated**: Apr 09, 2023 | **Arxiv ID**: [2304.04238v1](http://arxiv.org/pdf/2304.04238v1)

**Abstract**:
> Existing super-resolution models for pathology images can only work in fixed
> integer magnifications and have limited performance. Though implicit neural
> network-based methods have shown promising results in arbitrary-scale
> super-resolution of natural images, it is not effective to directly apply them
> in pathology images, because pathology images have special fine-grained image
> textures different from natural images. To address this challenge, we propose a
> dual-branch framework with an efficient self-texture enhancement mechanism for
> arbitrary-scale super-resolution of pathology images. Extensive experiments on
> two public datasets show that our method outperforms both existing fixed-scale
> and arbitrary-scale algorithms. To the best of our knowledge, this is the first
> work to achieve arbitrary-scale super-resolution in the field of pathology    
> images. Codes will be available.

---

### [Combination of Single and Multi-frame Image Super-resolution: An Analytical Perspective](http://arxiv.org/pdf/2303.03212v1)

**Authors**: Mohammad Mahdi Afrasiabi, Reshad Hosseini, Aliazam Abbasfar

**Published**: Mar 06, 2023 | **Updated**: Mar 06, 2023 | **Arxiv ID**: [2303.03212v1](http://arxiv.org/pdf/2303.03212v1)

**Abstract**:
> Super-resolution is the process of obtaining a high-resolution image from one
> or more low-resolution images. Single image super-resolution (SISR) and
> multi-frame super-resolution (MFSR) methods have been evolved almost
> independently for years. A neglected study in this field is the theoretical
> analysis of finding the optimum combination of SISR and MFSR. To fill this gap,
> we propose a novel theoretical analysis based on the iterative shrinkage and
> thresholding algorithm. We implement and compare several approaches for
> combining SISR and MFSR, and simulation results support the finding of our
> theoretical analysis, both quantitatively and qualitatively.

---

### [Infrared Image Super-Resolution: Systematic Review, and Future Trends](http://arxiv.org/pdf/2212.12322v4)

**Authors**: Yongsong Huang, Tomo Miyazaki, Xiaofeng Liu, Shinichiro Omachi

**Published**: Dec 22, 2022 | **Updated**: Feb 20, 2025 | **Arxiv ID**: [2212.12322v4](http://arxiv.org/pdf/2212.12322v4)

**Abstract**:
> Image Super-Resolution (SR) is essential for a wide range of computer vision
> and image processing tasks. Investigating infrared (IR) image (or thermal
> images) super-resolution is a continuing concern within the development of deep
> learning. This survey aims to provide a comprehensive perspective of IR image
> super-resolution, including its applications, hardware imaging system dilemmas,
> and taxonomy of image processing methodologies. In addition, the datasets and
> evaluation metrics in IR image super-resolution tasks are also discussed.
> Furthermore, the deficiencies in current technologies and possible promising
> directions for the community to explore are highlighted. To cope with the rapid
> development in this field, we intend to regularly update the relevant excellent
> work at \url{https://github.com/yongsongH/Infrared_Image_SR_Survey

---

### [Multi-Reference Image Super-Resolution: A Posterior Fusion Approach](http://arxiv.org/pdf/2212.09988v1)

**Authors**: Ke Zhao, Haining Tan, Tsz Fung Yau

**Published**: Dec 20, 2022 | **Updated**: Dec 20, 2022 | **Arxiv ID**: [2212.09988v1](http://arxiv.org/pdf/2212.09988v1)

**Abstract**:
> Reference-based Super-resolution (RefSR) approaches have recently been
> proposed to overcome the ill-posed problem of image super-resolution by
> providing additional information from a high-resolution image. Multi-reference
> super-resolution extends this approach by allowing more information to be
> incorporated. This paper proposes a 2-step-weighting posterior fusion approach
> to combine the outputs of RefSR models with multiple references. Extensive
> experiments on the CUFED5 dataset demonstrate that the proposed methods can be
> applied to various state-of-the-art RefSR models to get a consistent
> improvement in image quality.

---

### [NTIRE 2022 Challenge on Stereo Image Super-Resolution: Methods and Results](http://arxiv.org/pdf/2204.09197v1)

**Authors**: Longguang Wang, Yulan Guo, Yingqian Wang, Juncheng Li, Shuhang Gu, Radu Timofte

**Published**: Apr 20, 2022 | **Updated**: Apr 20, 2022 | **Arxiv ID**: [2204.09197v1](http://arxiv.org/pdf/2204.09197v1)

**Abstract**:
> In this paper, we summarize the 1st NTIRE challenge on stereo image
> super-resolution (restoration of rich details in a pair of low-resolution
> stereo images) with a focus on new solutions and results. This challenge has 1
> track aiming at the stereo image super-resolution problem under a standard
> bicubic degradation. In total, 238 participants were successfully registered,
> and 21 teams competed in the final testing phase. Among those participants, 20
> teams successfully submitted results with PSNR (RGB) scores better than the
> baseline. This challenge establishes a new benchmark for stereo image SR.

---

### [Blind Motion Deblurring Super-Resolution: When Dynamic Spatio-Temporal Learning Meets Static Image Understanding](http://arxiv.org/pdf/2105.13077v2)

**Authors**: Wenjia Niu, Kaihao Zhang, Wenhan Luo, Yiran Zhong

**Published**: May 27, 2021 | **Updated**: Oct 19, 2021 | **Arxiv ID**: [2105.13077v2](http://arxiv.org/pdf/2105.13077v2)

**Abstract**:
> Single-image super-resolution (SR) and multi-frame SR are two ways to super
> resolve low-resolution images. Single-Image SR generally handles each image
> independently, but ignores the temporal information implied in continuing
> frames. Multi-frame SR is able to model the temporal dependency via capturing
> motion information. However, it relies on neighbouring frames which are not
> always available in the real world. Meanwhile, slight camera shake easily
> causes heavy motion blur on long-distance-shot low-resolution images. To
> address these problems, a Blind Motion Deblurring Super-Reslution Networks,
> BMDSRNet, is proposed to learn dynamic spatio-temporal information from single
> static motion-blurred images. Motion-blurred images are the accumulation over
> time during the exposure of cameras, while the proposed BMDSRNet learns the
> reverse process and uses three-streams to learn Bidirectional spatio-temporal
> information based on well designed reconstruction loss functions to recover
> clean high-resolution images. Extensive experiments demonstrate that the
> proposed BMDSRNet outperforms recent state-of-the-art methods, and has the
> ability to simultaneously deal with image deblurring and SR.

---

### [Real-World Single Image Super-Resolution: A Brief Review](http://arxiv.org/pdf/2103.02368v1)

**Authors**: Honggang Chen, Xiaohai He, Linbo Qing, Yuanyuan Wu, Chao Ren, Ce Zhu

**Published**: Mar 03, 2021 | **Updated**: Mar 03, 2021 | **Arxiv ID**: [2103.02368v1](http://arxiv.org/pdf/2103.02368v1)

**Abstract**:
> Single image super-resolution (SISR), which aims to reconstruct a
> high-resolution (HR) image from a low-resolution (LR) observation, has been an
> active research topic in the area of image processing in recent decades.
> Particularly, deep learning-based super-resolution (SR) approaches have drawn
> much attention and have greatly improved the reconstruction performance on
> synthetic data. Recent studies show that simulation results on synthetic data
> usually overestimate the capacity to super-resolve real-world images. In this
> context, more and more researchers devote themselves to develop SR approaches
> for realistic images. This article aims to make a comprehensive review on
> real-world single image super-resolution (RSISR). More specifically, this
> review covers the critical publically available datasets and assessment metrics
> for RSISR, and four major categories of RSISR methods, namely the degradation
> modeling-based RSISR, image pairs-based RSISR, domain translation-based RSISR,
> and self-learning-based RSISR. Comparisons are also made among representative
> RSISR methods on benchmark datasets, in terms of both reconstruction quality
> and computational efficiency. Besides, we discuss challenges and promising
> research topics on RSISR.

---

### [Unsupervised Super-Resolution: Creating High-Resolution Medical Images from Low-Resolution Anisotropic Examples](http://arxiv.org/pdf/2010.13172v1)

**Authors**: Jörg Sander, Bob D. de Vos, Ivana Išgum

**Published**: Oct 25, 2020 | **Updated**: Oct 25, 2020 | **Arxiv ID**: [2010.13172v1](http://arxiv.org/pdf/2010.13172v1)

**Abstract**:
> Although high resolution isotropic 3D medical images are desired in clinical
> practice, their acquisition is not always feasible. Instead, lower resolution
> images are upsampled to higher resolution using conventional interpolation
> methods. Sophisticated learning-based super-resolution approaches are
> frequently unavailable in clinical setting, because such methods require
> training with high-resolution isotropic examples. To address this issue, we
> propose a learning-based super-resolution approach that can be trained using
> solely anisotropic images, i.e. without high-resolution ground truth data. The
> method exploits the latent space, generated by autoencoders trained on
> anisotropic images, to increase spatial resolution in low-resolution images.
> The method was trained and evaluated using 100 publicly available cardiac cine
> MR scans from the Automated Cardiac Diagnosis Challenge (ACDC). The
> quantitative results show that the proposed method performs better than
> conventional interpolation methods. Furthermore, the qualitative results
> indicate that especially finer cardiac structures are synthesized with high
> quality. The method has the potential to be applied to other anatomies and
> modalities and can be easily applied to any 3D anisotropic medical image
> dataset.

---

### [PIRM2018 Challenge on Spectral Image Super-Resolution: Dataset and Study](http://arxiv.org/pdf/1904.00540v2)

**Authors**: Mehrdad Shoeiby, Antonio Robles-Kelly, Ran Wei, Radu Timofte

**Published**: Apr 01, 2019 | **Updated**: May 01, 2019 | **Arxiv ID**: [1904.00540v2](http://arxiv.org/pdf/1904.00540v2)

**Abstract**:
> This paper introduces a newly collected and novel dataset (StereoMSI) for
> example-based single and colour-guided spectral image super-resolution. The
> dataset was first released and promoted during the PIRM2018 spectral image
> super-resolution challenge. To the best of our knowledge, the dataset is the
> first of its kind, comprising 350 registered colour-spectral image pairs. The
> dataset has been used for the two tracks of the challenge and, for each of
> these, we have provided a split into training, validation and testing. This
> arrangement is a result of the challenge structure and phases, with the first
> track focusing on example-based spectral image super-resolution and the second
> one aiming at exploiting the registered stereo colour imagery to improve the
> resolution of the spectral images. Each of the tracks and splits has been
> selected to be consistent across a number of image quality metrics. The dataset
> is quite general in nature and can be used for a wide variety of applications
> in addition to the development of spectral image super-resolution methods.

---

