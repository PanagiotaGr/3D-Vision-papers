# 3D Reconstruction

_Updated: 2026-04-07 07:47 UTC_

Total papers shown: **14**


---

- **PointTPA: Dynamic Network Parameter Adaptation for 3D Scene Understanding**  
  Siyuan Liu, Chaoqun Zheng, Xin Zhou, Tianrui Feng, Dingkang Liang, Xiang Bai  
  _2026-04-06_ · https://arxiv.org/abs/2604.04933v1  
  <details><summary>Abstract</summary>

  Scene-level point cloud understanding remains challenging due to diverse geometries, imbalanced category distributions, and highly varied spatial layouts. Existing methods improve object-level performance but rely on static network parameters during inference, limiting their adaptability to dynamic scene data. We propose PointTPA, a Test-time Parameter Adaptation framework that generates input-aware network parameters for scene-level point clouds. PointTPA adopts a Serialization-based Neighborhood Grouping (SNG) to form locally coherent patches and a Dynamic Parameter Projector (DPP) to produce patch-wise adaptive weights, enabling the backbone to adjust its behavior according to scene-specific variations while maintaining a low parameter overhead. Integrated into the PTv3 structure, PointTPA demonstrates strong parameter efficiency by introducing two lightweight modules of less than 2% of the backbone's parameters. Despite this minimal parameter overhead, PointTPA achieves 78.4% mIoU on ScanNet validation, surpassing existing parameter-efficient fine-tuning (PEFT) methods across multiple benchmarks, highlighting the efficacy of our test-time dynamic network parameter adaptation mechanism in enhancing 3D scene understanding. The code is available at https://github.com/H-EmbodVis/PointTPA.

  </details>



- **LoMa: Local Feature Matching Revisited**  
  David Nordström, Johan Edstedt, Georg Bökman, Jonathan Astermark, Anders Heyden, Viktor Larsson, Mårten Wadenbäck, Michael Felsberg, Fredrik Kahl  
  _2026-04-06_ · https://arxiv.org/abs/2604.04931v1  
  <details><summary>Abstract</summary>

  Local feature matching has long been a fundamental component of 3D vision systems such as Structure-from-Motion (SfM), yet progress has lagged behind the rapid advances of modern data-driven approaches. The newer approaches, such as feed-forward reconstruction models, have benefited extensively from scaling dataset sizes, whereas local feature matching models are still only trained on a few mid-sized datasets. In this paper, we revisit local feature matching from a data-driven perspective. In our approach, which we call LoMa, we combine large and diverse data mixtures, modern training recipes, scaled model capacity, and scaled compute, resulting in remarkable gains in performance. Since current standard benchmarks mainly rely on collecting sparse views from successful 3D reconstructions, the evaluation of progress in feature matching has been limited to relatively easy image pairs. To address the resulting saturation of benchmarks, we collect 1000 highly challenging image pairs from internet data into a new dataset called HardMatch. Ground truth correspondences for HardMatch are obtained via manual annotation by the authors. In our extensive benchmarking suite, we find that LoMa makes outstanding progress across the board, outperforming the state-of-the-art method ALIKED+LightGlue by +18.6 mAA on HardMatch, +29.5 mAA on WxBS, +21.4 (1m, 10$^\circ$) on InLoc, +24.2 AUC on RUBIK, and +12.4 mAA on IMC 2022. We release our code and models publicly at https://github.com/davnords/LoMa.

  </details>



- **Fully Procedural Synthetic Data from Simple Rules for Multi-View Stereo**  
  Zeyu Ma, Alexander Raistrick, Jia Deng  
  _2026-04-06_ · https://arxiv.org/abs/2604.04925v1  
  <details><summary>Abstract</summary>

  In this paper, we explore the design space of procedural rules for multi-view stereo (MVS). We demonstrate that we can generate effective training data using SimpleProc: a new, fully procedural generator driven by a very small set of rules using Non-Uniform Rational Basis Splines (NURBS), as well as basic displacement and texture patterns. At a modest scale of 8,000 images, our approach achieves superior results compared to manually curated images (at the same scale) sourced from games and real-world objects. When scaled to 352,000 images, our method yields performance comparable to--and in several benchmarks, exceeding--models trained on over 692,000 manually curated images. The source code and the data are available at https://github.com/princeton-vl/SimpleProc.

  </details>



- **AvatarPointillist: AutoRegressive 4D Gaussian Avatarization**  
  Hongyu Liu, Xuan Wang, Yating Wang, Zijian Wu, Ziyu Wan, Yue Ma, Runtao Liu, Boyao Zhou, Yujun Shen, Qifeng Chen  
  _2026-04-06_ · https://arxiv.org/abs/2604.04787v1  
  <details><summary>Abstract</summary>

  We introduce AvatarPointillist, a novel framework for generating dynamic 4D Gaussian avatars from a single portrait image. At the core of our method is a decoder-only Transformer that autoregressively generates a point cloud for 3D Gaussian Splatting. This sequential approach allows for precise, adaptive construction, dynamically adjusting point density and the total number of points based on the subject's complexity. During point generation, the AR model also jointly predicts per-point binding information, enabling realistic animation. After generation, a dedicated Gaussian decoder converts the points into complete, renderable Gaussian attributes. We demonstrate that conditioning the decoder on the latent features from the AR generator enables effective interaction between stages and markedly improves fidelity. Extensive experiments validate that AvatarPointillist produces high-quality, photorealistic, and controllable avatars. We believe this autoregressive formulation represents a new paradigm for avatar generation, and we will release our code inspire future research.

  </details>



- **3D Gaussian Splatting for Annular Dark Field Scanning Transmission Electron Microscopy Tomography Reconstruction**  
  Beiyuan Zhang, Hesong Li, Ruiwen Shao, Ying Fu  
  _2026-04-06_ · https://arxiv.org/abs/2604.04693v1  
  <details><summary>Abstract</summary>

  Analytical Dark Field Scanning Transmission Electron Microscopy (ADF-STEM) tomography reconstructs nanoscale materials in 3D by integrating multi-view tilt-series images, enabling precise analysis of their structural and compositional features. Although integrating more tilt views improves 3D reconstruction, it requires extended electron exposure that risks damaging dose-sensitive materials and introduces drift and misalignment, making it difficult to balance reconstruction fidelity with sample preservation. In practice, sparse-view acquisition is frequently required, yet conventional ADF-STEM methods degrade under limited views, exhibiting artifacts and reduced structural fidelity. To resolve these issues, in this paper, we adapt 3D GS to this domain with three key components. We first model the local scattering strength as a learnable scalar field, denza, to address the mismatch between 3DGS and ADF-STEM imaging physics. Then we introduce a coefficient $γ$ to stabilize scattering across tilt angles, ensuring consistent denza via scattering view normalization. Finally, We incorporate a loss function that includes a 2D Fourier amplitude term to suppress missing wedge artifacts in sparse-view reconstruction. Experiments on 45-view and 15-view tilt series show that DenZa-Gaussian produces high-fidelity reconstructions and 2D projections that align more closely with original tilts, demonstrating superior robustness under sparse-view conditions.

  </details>



- **ZeD-MAP: Bundle Adjustment Guided Zero-Shot Depth Maps for Real-Time Aerial Imaging**  
  Selim Ahmet Iz, Francesco Nex, Norman Kerle, Henry Meissner, Ralf Berger  
  _2026-04-06_ · https://arxiv.org/abs/2604.04667v1  
  <details><summary>Abstract</summary>

  Real-time depth reconstruction from ultra-high-resolution UAV imagery is essential for time-critical geospatial tasks such as disaster response, yet remains challenging due to wide-baseline parallax, large image sizes, low-texture or specular surfaces, occlusions, and strict computational constraints. Recent zero-shot diffusion models offer fast per-image dense predictions without task-specific retraining, and require fewer labelled datasets than transformer-based predictors while avoiding the rigid capture geometry requirement of classical multi-view stereo. However, their probabilistic inference prevents reliable metric accuracy and temporal consistency across sequential frames and overlapping tiles. We present ZeD-MAP, a cluster-level framework that converts a test-time diffusion depth model into a metrically consistent, SLAM-like mapping pipeline by integrating incremental cluster-based bundle adjustment (BA). Streamed UAV frames are grouped into overlapping clusters; periodic BA produces metrically consistent poses and sparse 3D tie-points, which are reprojected into selected frames and used as metric guidance for diffusion-based depth estimation. Validation on ground-marker flights captured at approximately 50 m altitude (GSD is approximately 0.85 cm/px, corresponding to 2,650 square meters ground coverage per frame) with the DLR Modular Aerial Camera System (MACS) shows that our method achieves sub-meter accuracy, with approximately 0.87 m error in the horizontal (XY) plane and 0.12 m in the vertical (Z) direction, while maintaining per-image runtimes between 1.47 and 4.91 seconds. Results are subject to minor noise from manual point-cloud annotation. These findings show that BA-based metric guidance provides consistency comparable to classical photogrammetric methods while significantly accelerating processing, enabling real-time 3D map generation.

  </details>



- **Synthesis4AD: Synthetic Anomalies are All You Need for 3D Anomaly Detection**  
  Yihan Sun, Yuqi Cheng, Junjie Zu, Yuxiang Tan, Guoyang Xie, Yucheng Wang, Yunkang Cao, Weiming Shen  
  _2026-04-06_ · https://arxiv.org/abs/2604.04658v1  
  <details><summary>Abstract</summary>

  Industrial 3D anomaly detection performance is fundamentally constrained by the scarcity and long-tailed distribution of abnormal samples. To address this challenge, we propose Synthesis4AD, an end-to-end paradigm that leverages large-scale, high-fidelity synthetic anomalies to learn more discriminative representations for 3D anomaly detection. At the core of Synthesis4AD is 3D-DefectStudio, a software platform built upon the controllable synthesis engine MPAS, which injects geometrically realistic defects guided by higher-dimensional support primitives while simultaneously generating accurate point-wise anomaly masks. Furthermore, Synthesis4AD incorporates a multimodal large language model (MLLM) to interpret product design information and automatically translate it into executable anomaly synthesis instructions, enabling scalable and knowledge-driven anomalous data generation. To improve the robustness and generalization of the downstream detector on unstructured point clouds, Synthesis4AD further introduces a training pipeline based on spatial-distribution normalization and geometry-faithful data augmentations, which alleviates the sensitivity of Point Transformer architectures to absolute coordinates and improves feature learning under realistic data variations. Extensive experiments demonstrate state-of-the-art performance on Real3D-AD, MulSen-AD, and a real-world industrial parts dataset. The proposed synthesis method MPAS and the interactive system 3D-DefectStudio will be publicly released at https://github.com/hustCYQ/Synthesis4AD.

  </details>



- **PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis**  
  Inseong Choi, Siwoo Lee, Seung-Hun Nam, Soohwan Song  
  _2026-04-06_ · https://arxiv.org/abs/2604.04576v1  
  <details><summary>Abstract</summary>

  Diffusion models are promising for sparse-view novel view synthesis (NVS), as they can generate pseudo-ground-truth views to aid 3D reconstruction pipelines like 3D Gaussian Splatting (3DGS). However, these synthesized images often contain photometric and geometric inconsistencies, and their direct use for supervision can impair reconstruction. To address this, we propose Partial-Reference Image Quality Assessment (PR-IQA), a framework that evaluates diffusion-generated views using reference images from different poses, eliminating the need for ground truth. PR-IQA first computes a geometrically consistent partial quality map in overlapping regions. It then performs quality completion to inpaint this partial map into a dense, full-image map. This completion is achieved via a cross-attention mechanism that incorporates reference-view context, ensuring cross-view consistency and enabling thorough quality assessment. When integrated into a diffusion-augmented 3DGS pipeline, PR-IQA restricts supervision to high-confidence regions identified by its quality maps. Experiments demonstrate that PR-IQA outperforms existing IQA methods, achieving full-reference-level accuracy without ground-truth supervision. Thus, our quality-aware 3DGS approach more effectively filters inconsistencies, producing superior 3D reconstructions and NVS results.The project page is available at https://kakaomacao.github.io/pr-iqa-project-page/.

  </details>



- **NAIMA: Semantics Aware RGB Guided Depth Super-Resolution**  
  Tayyab Nasir, Daochang Liu, Ajmal Mian  
  _2026-04-06_ · https://arxiv.org/abs/2604.04407v1  
  <details><summary>Abstract</summary>

  Guided depth super-resolution (GDSR) is a multi-modal approach for depth map super-resolution that relies on a low-resolution depth map and a high-resolution RGB image to restore finer structural details. However, the misleading color and texture cues indicating depth discontinuities in RGB images often lead to artifacts and blurred depth boundaries in the generated depth map. We propose a solution that introduces global contextual semantic priors, generated from pretrained vision transformer token embeddings. Our approach to distilling semantic knowledge from pretrained token embeddings is motivated by their demonstrated effectiveness in related monocular depth estimation tasks. We introduce a Guided Token Attention (GTA) module, which iteratively aligns encoded RGB spatial features with depth encodings, using cross-attention for selectively injecting global semantic context extracted from different layers of a pretrained vision transformer. Additionally, we present an architecture called Neural Attention for Implicit Multi-token Alignment (NAIMA), which integrates DINOv2 with GTA blocks for a semantics-aware GDSR. Our proposed architecture, with its ability to distill semantic knowledge, achieves significant improvements over existing methods across multiple scaling factors and datasets.

  </details>



- **3D-Fixer: Coarse-to-Fine In-place Completion for 3D Scenes from a Single Image**  
  Ze-Xin Yin, Liu Liu, Xinjie Wang, Wei Sui, Zhizhong Su, Jian Yang, Jin Xie  
  _2026-04-06_ · https://arxiv.org/abs/2604.04406v1  
  <details><summary>Abstract</summary>

  Compositional 3D scene generation from a single view requires the simultaneous recovery of scene layout and 3D assets. Existing approaches mainly fall into two categories: feed-forward generation methods and per-instance generation methods. The former directly predict 3D assets with explicit 6DoF poses through efficient network inference, but they generalize poorly to complex scenes. The latter improve generalization through a divide-and-conquer strategy, but suffer from time-consuming pose optimization. To bridge this gap, we introduce 3D-Fixer, a novel in-place completion paradigm. Specifically, 3D-Fixer extends 3D object generative priors to generate complete 3D assets conditioned on the partially visible point cloud at the original locations, which are cropped from the fragmented geometry obtained from the geometry estimation methods. Unlike prior works that require explicit pose alignment, 3D-Fixer uses fragmented geometry as a spatial anchor to preserve layout fidelity. At its core, we propose a coarse-to-fine generation scheme to resolve boundary ambiguity under occlusion, supported by a dual-branch conditioning network and an Occlusion-Robust Feature Alignment (ORFA) strategy for stable training. Furthermore, to address the data scarcity bottleneck, we present ARSG-110K, the largest scene-level dataset to date, comprising over 110K diverse scenes and 3M annotated images with high-fidelity 3D ground truth. Extensive experiments show that 3D-Fixer achieves state-of-the-art geometric accuracy, which significantly outperforms baselines such as MIDI and Gen3DSR, while maintaining the efficiency of the diffusion process. Code and data will be publicly available at https://zx-yin.github.io/3dfixer.

  </details>



- **A Persistent Homology Design Space for 3D Point Cloud Deep Learning**  
  Prachi Kudeshia, Jiju Poovvancheri, Amr Ghoneim, Dong Chen  
  _2026-04-05_ · https://arxiv.org/abs/2604.04299v1  
  <details><summary>Abstract</summary>

  Persistent Homology (PH) offers stable, multi-scale descriptors of intrinsic shape structure by capturing connected components, loops, and voids that persist across scales, providing invariants that complement purely geometric representations of 3D data. Yet, despite strong theoretical guarantees and increasing empirical adoption, its integration into deep learning for point clouds remains largely ad hoc and architecturally peripheral. In this work, we introduce a unified design space for Persistent-Homology driven learning in 3D point clouds (3DPHDL), formalizing the interplay between complex construction, filtration strategy, persistence representation, neural backbone, and prediction task. Beyond the canonical pipeline of diagram computation and vectorization, we identify six principled injection points through which topology can act as a structural inductive bias reshaping sampling, neighborhood graphs, optimization dynamics, self-supervision, output calibration, and even internal network regularization. We instantiate this framework through a controlled empirical study on ModelNet40 classification and ShapeNetPart segmentation, systematically augmenting representative backbones (PointNet, DGCNN, and Point Transformer) with persistence diagrams, images, and landscapes, and analyzing their impact on accuracy, robustness to noise and sampling variation, and computational scalability. Our results demonstrate consistent improvements in topology-sensitive discrimination and part consistency, while revealing meaningful trade-offs between representational expressiveness and combinatorial complexity. By viewing persistent homology not merely as an auxiliary feature but as a structured component within the learning pipeline, this work provides a systematic framework for incorporating topological reasoning into 3D point cloud learning.

  </details>



- **NTIRE 2026 3D Restoration and Reconstruction in Real-world Adverse Conditions: RealX3D Challenge Results**  
  Shuhong Liu, Chenyu Bao, Ziteng Cui, Xuangeng Chu, Bin Ren, Lin Gu, Xiang Chen, Mingrui Li, Long Ma, Marcos V. Conde, et al.  
  _2026-04-05_ · https://arxiv.org/abs/2604.04135v1  
  <details><summary>Abstract</summary>

  This paper presents a comprehensive review of the NTIRE 2026 3D Restoration and Reconstruction (3DRR) Challenge, detailing the proposed methods and results. The challenge seeks to identify robust reconstruction pipelines that are robust under real-world adverse conditions, specifically extreme low-light and smoke-degraded environments, as captured by our RealX3D benchmark. A total of 279 participants registered for the competition, of whom 33 teams submitted valid results. We thoroughly evaluate the submitted approaches against state-of-the-art baselines, revealing significant progress in 3D reconstruction under adverse conditions. Our analysis highlights shared design principles among top-performing methods and provides insights into effective strategies for handling 3D scene degradation.

  </details>



- **Learning 3D Reconstruction with Priors in Test Time**  
  Lei Zhou, Haoyu Wu, Akshat Dave, Dimitris Samaras  
  _2026-04-04_ · https://arxiv.org/abs/2604.03878v1  
  <details><summary>Abstract</summary>

  We introduce a test-time framework for multiview Transformers (MVTs) that incorporates priors (e.g., camera poses, intrinsics, and depth) to improve 3D tasks without retraining or modifying pre-trained image-only networks. Rather than feeding priors into the architecture, we cast them as constraints on the predictions and optimize the network at inference time. The optimization loss consists of a self-supervised objective and prior penalty terms. The self-supervised objective captures the compatibility among multi-view predictions and is implemented using photometric or geometric loss between renderings from other views and each view itself. Any available priors are converted into penalty terms on the corresponding output modalities. Across a series of 3D vision benchmarks, including point map estimation and camera pose estimation, our method consistently improves performance over base MVTs by a large margin. On the ETH3D, 7-Scenes, and NRGBD datasets, our method reduces the point-map distance error by more than half compared with the base image-only models. Our method also outperforms retrained prior-aware feed-forward methods, demonstrating the effectiveness of our test-time constrained optimization (TCO) framework for incorporating priors into 3D vision tasks.

  </details>



- **Confidence-Driven Facade Refinement of 3D Building Models Using MLS Point Clouds**  
  Xiaoyu Huang  
  _2026-04-04_ · https://arxiv.org/abs/2604.03797v1  
  <details><summary>Abstract</summary>

  Digital twins require continuous maintenance to meet the increasing demand for high-precision geospatial data. However, traditional coarse CityGML building models, typically derived from Airborne Laser Scanning (ALS), often exhibit significant geometric deficiencies, particularly regarding facade accuracy due to the nadir perspective of airborne sensors. Integrating these coarse models with high-precision Mobile Laser Scanning (MLS) data is essential to recover detailed facade geometry. Unlike reconstruction-from-scratch approaches that discard existing semantic information and rely heavily on complete data coverage, this work presents an automated refinement framework that utilizes the coarse model as a geometric prior. This method enables targeted updates to facade geometry even in complex urban environments. It integrates surface matching to identify outdated surfaces and employs a binary integer optimization to select optimal faces from candidate data. Crucially, hard constraints are enforced within the optimization to ensure the topological validity of the refined output. Experimental results demonstrate that the proposed approach effectively corrects facade misalignments, reducing the Cloud-to-Mesh RMSE by approximately 36% and achieving centimeter-level alignment. Furthermore, the framework guarantees strictly watertight and manifold geometry, providing a robust solution for upgrading ALS-derived city models.

  </details>


