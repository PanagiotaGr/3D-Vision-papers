# Neural Rendering & View Synthesis

_Updated: 2026-03-01 07:02 UTC_

Total papers shown: **1**


---

- **DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis**  
  Xinglong Luo, Ao Luo, Zhengning Wang, Yueqi Yang, Chaoyu Feng, Lei Lei, Bing Zeng, Shuaicheng Liu  
  _2026-02-26_ · https://arxiv.org/abs/2602.23022v1  
  <details><summary>Abstract</summary>

  Image alignment is a fundamental task in computer vision with broad applications. Existing methods predominantly employ optical flow-based image warping. However, this technique is susceptible to common challenges such as occlusions and illumination variations, leading to degraded alignment visual quality and compromised accuracy in downstream tasks. In this paper, we present DMAligner, a diffusion-based framework for image alignment through alignment-oriented view synthesis. DMAligner is crafted to tackle the challenges in image alignment from a new perspective, employing a generation-based solution that showcases strong capabilities and avoids the problems associated with flow-based image warping. Specifically, we propose a Dynamics-aware Diffusion Training approach for learning conditional image generation, synthesizing a novel view for image alignment. This incorporates a Dynamics-aware Mask Producing (DMP) module to adaptively distinguish dynamic foreground regions from static backgrounds, enabling the diffusion model to more effectively handle challenges that classical methods struggle to solve. Furthermore, we develop the Dynamic Scene Image Alignment (DSIA) dataset using Blender, which includes 1,033 indoor and outdoor scenes with over 30K image pairs tailored for image alignment. Extensive experimental results demonstrate the superiority of the proposed approach on DSIA benchmarks, as well as on a series of widely-used video datasets for qualitative comparisons. Our code is available at https://github.com/boomluo02/DMAligner.

  </details>


