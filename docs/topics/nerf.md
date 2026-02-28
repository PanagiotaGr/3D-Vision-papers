# NeRF & Neural Radiance Fields

_Updated: 2026-02-28 06:55 UTC_

Total papers shown: **6**


---

- **DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis**  
  Xinglong Luo, Ao Luo, Zhengning Wang, Yueqi Yang, Chaoyu Feng, Lei Lei, Bing Zeng, Shuaicheng Liu  
  _2026-02-26_ · https://arxiv.org/abs/2602.23022v1  
  <details><summary>Abstract</summary>

  Image alignment is a fundamental task in computer vision with broad applications. Existing methods predominantly employ optical flow-based image warping. However, this technique is susceptible to common challenges such as occlusions and illumination variations, leading to degraded alignment visual quality and compromised accuracy in downstream tasks. In this paper, we present DMAligner, a diffusion-based framework for image alignment through alignment-oriented view synthesis. DMAligner is crafted to tackle the challenges in image alignment from a new perspective, employing a generation-based solution that showcases strong capabilities and avoids the problems associated with flow-based image warping. Specifically, we propose a Dynamics-aware Diffusion Training approach for learning conditional image generation, synthesizing a novel view for image alignment. This incorporates a Dynamics-aware Mask Producing (DMP) module to adaptively distinguish dynamic foreground regions from static backgrounds, enabling the diffusion model to more effectively handle challenges that classical methods struggle to solve. Furthermore, we develop the Dynamic Scene Image Alignment (DSIA) dataset using Blender, which includes 1,033 indoor and outdoor scenes with over 30K image pairs tailored for image alignment. Extensive experimental results demonstrate the superiority of the proposed approach on DSIA benchmarks, as well as on a series of widely-used video datasets for qualitative comparisons. Our code is available at https://github.com/boomluo02/DMAligner.

  </details>



- **Sapling-NeRF: Geo-Localised Sapling Reconstruction in Forests for Ecological Monitoring**  
  Miguel Ángel Muñoz-Bañón, Nived Chebrolu, Sruthi M. Krishna Moorthy, Yifu Tao, Fernando Torres, Roberto Salguero-Gómez, Maurice Fallon  
  _2026-02-26_ · https://arxiv.org/abs/2602.22731v1  
  <details><summary>Abstract</summary>

  Saplings are key indicators of forest regeneration and overall forest health. However, their fine-scale architectural traits are difficult to capture with existing 3D sensing methods, which make quantitative evaluation difficult. Terrestrial Laser Scanners (TLS), Mobile Laser Scanners (MLS), or traditional photogrammetry approaches poorly reconstruct thin branches, dense foliage, and lack the scale consistency needed for long-term monitoring. Implicit 3D reconstruction methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) are promising alternatives, but cannot recover the true scale of a scene and lack any means to be accurately geo-localised. In this paper, we present a pipeline which fuses NeRF, LiDAR SLAM, and GNSS to enable repeatable, geo-localised ecological monitoring of saplings. Our system proposes a three-level representation: (i) coarse Earth-frame localisation using GNSS, (ii) LiDAR-based SLAM for centimetre-accurate localisation and reconstruction, and (iii) NeRF-derived object-centric dense reconstruction of individual saplings. This approach enables repeatable quantitative evaluation and long-term monitoring of sapling traits. Our experiments in forest plots in Wytham Woods (Oxford, UK) and Evo (Finland) show that stem height, branching patterns, and leaf-to-wood ratios can be captured with increased accuracy as compared to TLS. We demonstrate that accurate stem skeletons and leaf distributions can be measured for saplings with heights between 0.5m and 2m in situ, giving ecologists access to richer structural and quantitative data for analysing forest dynamics.

  </details>



- **Interactive Medical-SAM2 GUI: A Napari-based semi-automatic annotation tool for medical images**  
  Woojae Hong, Jong Ha Hwang, Jiyong Chung, Joongyeon Choi, Hyunngun Kim, Yong Hwy Kim  
  _2026-02-26_ · https://arxiv.org/abs/2602.22649v1  
  <details><summary>Abstract</summary>

  Interactive Medical-SAM2 GUI is an open-source desktop application for semi-automatic annotation of 2D and 3D medical images. Built on the Napari multi-dimensional viewer, box/point prompting is integrated with SAM2-style propagation by treating a 3D volume as a slice sequence, enabling mask propagation from sparse prompts using Medical-SAM2 on top of SAM2. Voxel-level annotation remains essential for developing and validating medical imaging algorithms, yet manual labeling is slow and expensive for 3D scans, and existing integrations frequently emphasize per-slice interaction without providing a unified, cohort-oriented workflow for navigation, propagation, interactive correction, and quantitative export in a single local pipeline. To address this practical limitation, a local-first Napari workflow is provided for efficient 3D annotation across multiple studies using standard DICOM series and/or NIfTI volumes. Users can annotate cases sequentially under a single root folder with explicit proceed/skip actions, initialize objects via box-first prompting (including first/last-slice initialization for single-object propagation), refine predictions with point prompts, and finalize labels through prompt-first correction prior to saving. During export, per-object volumetry and 3D volume rendering are supported, and image geometry is preserved via SimpleITK. The GUI is implemented in Python using Napari and PyTorch, with optional N4 bias-field correction, and is intended exclusively for research annotation workflows. The code is released on the project page: https://github.com/SKKU-IBE/Medical-SAM2GUI/.

  </details>



- **BetterScene: 3D Scene Synthesis with Representation-Aligned Generative Model**  
  Yuci Han, Charles Toth, John E. Anderson, William J. Shuart, Alper Yilmaz  
  _2026-02-26_ · https://arxiv.org/abs/2602.22596v1  
  <details><summary>Abstract</summary>

  We present BetterScene, an approach to enhance novel view synthesis (NVS) quality for diverse real-world scenes using extremely sparse, unconstrained photos. BetterScene leverages the production-ready Stable Video Diffusion (SVD) model pretrained on billions of frames as a strong backbone, aiming to mitigate artifacts and recover view-consistent details at inference time. Conventional methods have developed similar diffusion-based solutions to address these challenges of novel view synthesis. Despite significant improvements, these methods typically rely on off-the-shelf pretrained diffusion priors and fine-tune only the UNet module while keeping other components frozen, which still leads to inconsistent details and artifacts even when incorporating geometry-aware regularizations like depth or semantic conditions. To address this, we investigate the latent space of the diffusion model and introduce two components: (1) temporal equivariance regularization and (2) vision foundation model-aligned representation, both applied to the variational autoencoder (VAE) module within the SVD pipeline. BetterScene integrates a feed-forward 3D Gaussian Splatting (3DGS) model to render features as inputs for the SVD enhancer and generate continuous, artifact-free, consistent novel views. We evaluate on the challenging DL3DV-10K dataset and demonstrate superior performance compared to state-of-the-art methods.

  </details>



- **SwiftNDC: Fast Neural Depth Correction for High-Fidelity 3D Reconstruction**  
  Kang Han, Wei Xiang, Lu Yu, Mathew Wyatt, Gaowen Liu, Ramana Rao Kompella  
  _2026-02-26_ · https://arxiv.org/abs/2602.22565v1  
  <details><summary>Abstract</summary>

  Depth-guided 3D reconstruction has gained popularity as a fast alternative to optimization-heavy approaches, yet existing methods still suffer from scale drift, multi-view inconsistencies, and the need for substantial refinement to achieve high-fidelity geometry. Here, we propose SwiftNDC, a fast and general framework built around a Neural Depth Correction field that produces cross-view consistent depth maps. From these refined depths, we generate a dense point cloud through back-projection and robust reprojection-error filtering, obtaining a clean and uniformly distributed geometric initialization for downstream reconstruction. This reliable dense geometry substantially accelerates 3D Gaussian Splatting (3DGS) for mesh reconstruction, enabling high-quality surfaces with significantly fewer optimization iterations. For novel-view synthesis, SwiftNDC can also improve 3DGS rendering quality, highlighting the benefits of strong geometric initialization. We conduct a comprehensive study across five datasets, including two for mesh reconstruction, as well as three for novel-view synthesis. SwiftNDC consistently reduces running time for accurate mesh reconstruction and boosts rendering fidelity for view synthesis, demonstrating the effectiveness of combining neural depth refinement with robust geometric initialization for high-fidelity and efficient 3D reconstruction.

  </details>



- **Lie Flow: Video Dynamic Fields Modeling and Predicting with Lie Algebra as Geometric Physics Principle**  
  Weidong Qiao, Wangmeng Zuo, Hui Li  
  _2026-02-25_ · https://arxiv.org/abs/2602.21645v1  
  <details><summary>Abstract</summary>

  Modeling 4D scenes requires capturing both spatial structure and temporal motion, which is challenging due to the need for physically consistent representations of complex rigid and non-rigid motions. Existing approaches mainly rely on translational displacements, which struggle to represent rotations, articulated transformations, often leading to spatial inconsistency and physically implausible motion. LieFlow, a dynamic radiance representation framework that explicitly models motion within the SE(3) Lie group, enabling coherent learning of translation and rotation in a unified geometric space. The SE(3) transformation field enforces physically inspired constraints to maintain motion continuity and geometric consistency. The evaluation includes a synthetic dataset with rigid-body trajectories and two real-world datasets capturing complex motion under natural lighting and occlusions. Across all datasets, LieFlow consistently improves view-synthesis fidelity, temporal coherence, and physical realism over NeRF-based baselines. These results confirm that SE(3)-based motion modeling offers a robust and physically grounded framework for representing dynamic 4D scenes.

  </details>


