# 3D Reconstruction

_Updated: 2026-04-08 07:51 UTC_

Total papers shown: **23**


---

- **EfficientMonoHair: Fast Strand-Level Reconstruction from Monocular Video via Multi-View Direction Fusion**  
  Da Li, Dominik Engel, Deng Luo, Ivan Viola  
  _2026-04-07_ · https://arxiv.org/abs/2604.05794v1  
  <details><summary>Abstract</summary>

  Strand-level hair geometry reconstruction is a fundamental problem in virtual human modeling and the digitization of hairstyles. However, existing methods still suffer from a significant trade-off between accuracy and efficiency. Implicit neural representations can capture the global hair shape but often fail to preserve fine-grained strand details, while explicit optimization-based approaches achieve high-fidelity reconstructions at the cost of heavy computation and poor scalability. To address this issue, we propose EfficientMonoHair, a fast and accurate framework that combines the implicit neural network with multi-view geometric fusion for strand-level reconstruction from monocular video. Our method introduces a fusion-patch-based multi-view optimization that reduces the number of optimization iterations for point cloud direction, as well as a novel parallel hair-growing strategy that relaxes voxel occupancy constraints, allowing large-scale strand tracing to remain stable and robust even under inaccurate or noisy orientation fields. Extensive experiments on representative real-world hairstyles demonstrate that our method can robustly reconstruct high-fidelity strand geometries with accuracy. On synthetic benchmarks, our method achieves reconstruction quality comparable to state-of-the-art methods, while improving runtime efficiency by nearly an order of magnitude.

  </details>



- **GaussianGrow: Geometry-aware Gaussian Growing from 3D Point Clouds with Text Guidance**  
  Weiqi Zhang, Junsheng Zhou, Haotian Geng, Kanle Shi, Shenkun Xu, Yi Fang, Yu-Shen Liu  
  _2026-04-07_ · https://arxiv.org/abs/2604.05721v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting has demonstrated superior performance in rendering efficiency and quality, yet the generation of 3D Gaussians still remains a challenge without proper geometric priors. Existing methods have explored predicting point maps as geometric references for inferring Gaussian primitives, while the unreliable estimated geometries may lead to poor generations. In this work, we introduce GaussianGrow, a novel approach that generates 3D Gaussians by learning to grow them from easily accessible 3D point clouds, naturally enforcing geometric accuracy in Gaussian generation. Specifically, we design a text-guided Gaussian growing scheme that leverages a multi-view diffusion model to synthesize consistent appearances from input point clouds for supervision. To mitigate artifacts caused by fusing neighboring views, we constrain novel views generated at non-preset camera poses identified in overlapping regions across different views. For completing the hard-to-observe regions, we propose to iteratively detect the camera pose by observing the largest un-grown regions in point clouds and inpainting them by inpainting the rendered view with a pretrained 2D diffusion model. The process continues until complete Gaussians are generated. We extensively evaluate GaussianGrow on text-guided Gaussian generation from synthetic and even real-scanned point clouds. Project Page: https://weiqi-zhang.github.io/GaussianGrow

  </details>



- **In Depth We Trust: Reliable Monocular Depth Supervision for Gaussian Splatting**  
  Wenhui Xiao, Ethan Goan, Rodrigo Santa Cruz, David Ahmedt-Aristizabal, Olivier Salvado, Clinton Fookes, Leo Lebrat  
  _2026-04-07_ · https://arxiv.org/abs/2604.05715v1  
  <details><summary>Abstract</summary>

  Using accurate depth priors in 3D Gaussian Splatting helps mitigate artifacts caused by sparse training data and textureless surfaces. However, acquiring accurate depth maps requires specialized acquisition systems. Foundation monocular depth estimation models offer a cost-effective alternative, but they suffer from scale ambiguity, multi-view inconsistency, and local geometric inaccuracies, which can degrade rendering performance when applied naively. This paper addresses the challenge of reliably leveraging monocular depth priors for Gaussian Splatting (GS) rendering enhancement. To this end, we introduce a training framework integrating scale-ambiguous and noisy depth priors into geometric supervision. We highlight the importance of learning from weakly aligned depth variations. We introduce a method to isolate ill-posed geometry for selective monocular depth regularization, restricting the propagation of depth inaccuracies into well-reconstructed 3D structures. Extensive experiments across diverse datasets show consistent improvements in geometric accuracy, leading to more faithful depth estimation and higher rendering quality across different GS variants and monocular depth backbones tested.

  </details>



- **Human Interaction-Aware 3D Reconstruction from a Single Image**  
  Gwanghyun Kim, Junghun James Kim, Suh Yoon Jeon, Jason Park, Se Young Chun  
  _2026-04-07_ · https://arxiv.org/abs/2604.05436v1  
  <details><summary>Abstract</summary>

  Reconstructing textured 3D human models from a single image is fundamental for AR/VR and digital human applications. However, existing methods mostly focus on single individuals and thus fail in multi-human scenes, where naive composition of individual reconstructions often leads to artifacts such as unrealistic overlaps, missing geometry in occluded regions, and distorted interactions. These limitations highlight the need for approaches that incorporate group-level context and interaction priors. We introduce a holistic method that explicitly models both group- and instance-level information. To mitigate perspective-induced geometric distortions, we first transform the input into a canonical orthographic space. Our primary component, Human Group-Instance Multi-View Diffusion (HUG-MVD), then generates complete multi-view normals and images by jointly modeling individuals and group context to resolve occlusions and proximity. Subsequently, the Human Group-Instance Geometric Reconstruction (HUG-GR) module optimizes the geometry by leveraging explicit, physics-based interaction priors to enforce physical plausibility and accurately model inter-human contact. Finally, the multi-view images are fused into a high-fidelity texture. Together, these components form our complete framework, HUG3D. Extensive experiments show that HUG3D significantly outperforms both single-human and existing multi-human methods, producing physically plausible, high-fidelity 3D reconstructions of interacting people from a single image. Project page: https://jongheean11.github.io/HUG3D_project

  </details>



- **3DTurboQuant: Training-Free Near-Optimal Quantization for 3D Reconstruction Models**  
  Jae Joong Lee  
  _2026-04-07_ · https://arxiv.org/abs/2604.05366v1  
  <details><summary>Abstract</summary>

  Every existing method for compressing 3D Gaussian Splatting, NeRF, or transformer-based 3D reconstructors requires learning a data-dependent codebook through per-scene fine-tuning. We show this is unnecessary. The parameter vectors that dominate storage in these models, 45-dimensional spherical harmonics in 3DGS and 1024-dimensional key-value vectors in DUSt3R, fall in a dimension range where a single random rotation transforms any input into coordinates with a known Beta distribution. This makes precomputed, data-independent Lloyd-Max quantization near-optimal, within a factor of 2.7 of the information-theoretic lower bound. We develop 3D, deriving (1) a dimension-dependent criterion that predicts which parameters can be quantized and at what bit-width before running any experiment, (2) norm-separation bounds connecting quantization MSE to rendering PSNR per scene, (3) an entry-grouping strategy extending rotation-based quantization to 2-dimensional hash grid features, and (4) a composable pruning-quantization pipeline with a closed-form compression ratio. On NeRF Synthetic, 3DTurboQuant compresses 3DGS by 3.5x with 0.02dB PSNR loss and DUSt3R KV caches by 7.9x with 39.7dB pointmap fidelity. No training, no codebook learning, no calibration data. Compression takes seconds. The code will be released (https://github.com/JaeLee18/3DTurboQuant)

  </details>



- **Unsupervised Multi-agent and Single-agent Perception from Cooperative Views**  
  Haochen Yang, Baolu Li, Lei Li, Delin Ren, Jiacheng Guo, Minghai Qin, Tianyun Zhang, Hongkai Yu  
  _2026-04-07_ · https://arxiv.org/abs/2604.05354v1  
  <details><summary>Abstract</summary>

  The LiDAR-based multi-agent and single-agent perception has shown promising performance in environmental understanding for robots and automated vehicles. However, there is no existing method that simultaneously solves both multi-agent and single-agent perception in an unsupervised way. By sharing sensor data between multiple agents via communication, this paper discovers two key insights: 1) Improved point cloud density after the data sharing from cooperative views could benefit unsupervised object classification, 2) Cooperative view of multiple agents can be used as unsupervised guidance for the 3D object detection in the single view. Based on these two discovered insights, we propose an Unsupervised Multi-agent and Single-agent (UMS) perception framework that leverages multi-agent cooperation without human annotations to simultaneously solve multi-agent and single-agent perception. UMS combines a learning-based Proposal Purifying Filter to better classify the candidate proposals after multi-agent point cloud density cooperation, followed by a Progressive Proposal Stabilizing module to yield reliable pseudo labels by the easy-to-hard curriculum learning. Furthermore, we design a Cross-View Consensus Learning to use multi-agent cooperative view to guide detection in single-agent view. Experimental results on two public datasets V2V4Real and OPV2V show that our UMS method achieved significantly higher 3D detection performance than the state-of-the-art methods on both multi-agent and single-agent perception tasks in an unsupervised setting.

  </details>



- **SmokeGS-R: Physics-Guided Pseudo-Clean 3DGS for Real-World Multi-View Smoke Restoration**  
  Xueming Fu, Lixia Han  
  _2026-04-07_ · https://arxiv.org/abs/2604.05301v1  
  <details><summary>Abstract</summary>

  Real-world smoke simultaneously attenuates scene radiance, adds airlight, and destabilizes multi-view appearance consistency, making robust 3D reconstruction particularly difficult. We present \textbf{SmokeGS-R}, a practical pipeline developed for the NTIRE 2026 3D Restoration and Reconstruction Track 2 challenge. The key idea is to decouple geometry recovery from appearance correction: we generate physics-guided pseudo-clean supervision with a refined dark channel prior and guided filtering, train a sharp clean-only 3D Gaussian Splatting source model, and then harmonize its renderings with a donor ensemble using geometric-mean reference aggregation, LAB-space Reinhard transfer, and light Gaussian smoothing. On the official challenge testing leaderboard, the final submission achieved \mbox{PSNR $=15.217$} and \mbox{SSIM $=0.666$}. After the public release of RealX3D, we re-evaluated the same frozen result on the seven released challenge scenes without retraining and obtained \mbox{PSNR $=15.209$}, \mbox{SSIM $=0.644$}, and \mbox{LPIPS $=0.551$}, outperforming the strongest official baseline average on the same scenes by $+3.68$ dB PSNR. These results suggest that a geometry-first reconstruction strategy combined with stable post-render appearance harmonization is an effective recipe for real-world multi-view smoke restoration. The code is available at https://github.com/windrise/3drr_Track2_SmokeGS-R.

  </details>



- **Coverage Optimization for Camera View Selection**  
  Timothy Chen, Adam Dai, Maximilian Adang, Grace Gao, Mac Schwager  
  _2026-04-06_ · https://arxiv.org/abs/2604.05259v1  
  <details><summary>Abstract</summary>

  What makes a good viewpoint? The quality of the data used to learn 3D reconstructions is crucial for enabling efficient and accurate scene modeling. We study the active view selection problem and develop a principled analysis that yields a simple and interpretable criterion for selecting informative camera poses. Our key insight is that informative views can be obtained by minimizing a tractable approximation of the Fisher Information Gain, which reduces to favoring viewpoints that cover geometry that has been insufficiently observed by past cameras. This leads to a lightweight coverage-based view selection metric that avoids expensive transmittance estimation and is robust to noise and training dynamics. We call this metric COVER (Camera Optimization for View Exploration and Reconstruction). We integrate our method into the Nerfstudio framework and evaluate it on real datasets within fixed and embodied data acquisition scenarios. Across multiple datasets and radiance-field baselines, our method consistently improves reconstruction quality compared to state-of-the-art active view selection methods. Additional visualizations and our Nerfstudio package can be found at https://chengine.github.io/nbv_gym/.

  </details>



- **Boxer: Robust Lifting of Open-World 2D Bounding Boxes to 3D**  
  Daniel DeTone, Tianwei Shen, Fan Zhang, Lingni Ma, Julian Straub, Richard Newcombe, Jakob Engel  
  _2026-04-06_ · https://arxiv.org/abs/2604.05212v1  
  <details><summary>Abstract</summary>

  Detecting and localizing objects in space is a fundamental computer vision problem. While much progress has been made to solve 2D object detection, 3D object localization is much less explored and far from solved, especially for open-world categories. To address this research challenge, we propose Boxer, an algorithm to estimate static 3D bounding boxes (3DBBs) from 2D open-vocabulary object detections, posed images and optional depth either represented as a sparse point cloud or dense depth. At its core is BoxerNet, a transformer-based network which lifts 2D bounding box (2DBB) proposals into 3D, followed by multi-view fusion and geometric filtering to produce globally consistent de-duplicated 3DBBs in metric world space. Boxer leverages the power of existing 2DBB detection algorithms (e.g. DETIC, OWLv2, SAM3) to localize objects in 2D. This allows the main BoxerNet model to focus on lifting to 3D rather than detecting, ultimately reducing the demand for costly annotated 3DBB training data. Extending the CuTR formulation, we incorporate an aleatoric uncertainty for robust regression, a median depth patch encoding to support sparse depth inputs, and large-scale training with over 1.2 million unique 3DBBs. BoxerNet outperforms state-of-the-art baselines in open-world 3DBB lifting, including CuTR in egocentric settings without dense depth (0.532 vs. 0.010 mAP) and on CA-1M with dense depth available (0.412 vs. 0.250 mAP).

  </details>



- **LSRM: High-Fidelity Object-Centric Reconstruction via Scaled Context Windows**  
  Zhengqin Li, Cheng Zhang, Jakob Engel, Zhao Dong  
  _2026-04-06_ · https://arxiv.org/abs/2604.05182v1  
  <details><summary>Abstract</summary>

  We introduce the Large Sparse Reconstruction Model to study how scaling transformer context windows impacts feed-forward 3D reconstruction. Although recent object-centric feed-forward methods deliver robust, high-quality reconstruction, they still lag behind dense-view optimization in recovering fine-grained texture and appearance. We show that expanding the context window -- by substantially increasing the number of active object and image tokens -- remarkably narrows this gap and enables high-fidelity 3D object reconstruction and inverse rendering. To scale effectively, we adapt native sparse attention in our architecture design, unlocking its capacity for 3D reconstruction with three key contributions: (1) an efficient coarse-to-fine pipeline that focuses computation on informative regions by predicting sparse high-resolution residuals; (2) a 3D-aware spatial routing mechanism that establishes accurate 2D-3D correspondences using explicit geometric distances rather than standard attention scores; and (3) a custom block-aware sequence parallelism strategy utilizing an All-gather-KV protocol to balance dynamic, sparse workloads across GPUs. As a result, LSRM handles 20x more object tokens and >2x more image tokens than prior state-of-the-art (SOTA) methods. Extensive evaluations on standard novel-view synthesis benchmarks show substantial gains over the current SOTA, yielding 2.5 dB higher PSNR and 40% lower LPIPS. Furthermore, when extending LSRM to inverse rendering tasks, qualitative and quantitative evaluations on widely-used benchmarks demonstrate consistent improvements in texture and geometry details, achieving an LPIPS that matches or exceeds that of SOTA dense-view optimization methods. Code and model will be released on our project page.

  </details>



- **R3PM-Net: Real-time, Robust, Real-world Point Matching Network**  
  Yasaman Kashefbahrami, Erkut Akdag, Panagiotis Meletis, Evgeniya Balmashnova, Dip Goswami, Egor Bondarau  
  _2026-04-06_ · https://arxiv.org/abs/2604.05060v1  
  <details><summary>Abstract</summary>

  Accurate Point Cloud Registration (PCR) is an important task in 3D data processing, involving the estimation of a rigid transformation between two point clouds. While deep-learning methods have addressed key limitations of traditional non-learning approaches, such as sensitivity to noise, outliers, occlusion, and initialization, they are developed and evaluated on clean, dense, synthetic datasets (limiting their generalizability to real-world industrial scenarios). This paper introduces R3PM-Net, a lightweight, global-aware, object-level point matching network designed to bridge this gap by prioritizing both generalizability and real-time efficiency. To support this transition, two datasets, Sioux-Cranfield and Sioux-Scans, are proposed. They provide an evaluation ground for registering imperfect photogrammetric and event-camera scans to digital CAD models, and have been made publicly available. Extensive experiments demonstrate that R3PM-Net achieves competitive accuracy with unmatched speed. On ModelNet40, it reaches a perfect fitness score of $1$ and inlier RMSE of $0.029$ cm in only $0.007$s, approximately 7 times faster than the state-of-the-art method RegTR. This performance carries over to the Sioux-Cranfield dataset, maintaining a fitness of $1$ and inlier RMSE of $0.030$ cm with similarly low latency. Furthermore, on the highly challenging Sioux-Scans dataset, R3PM-Net successfully resolves edge cases in under 50 ms. These results confirm that R3PM-Net offers a robust, high-speed solution for critical industrial applications, where precision and real-time performance are indispensable. The code and datasets are available at https://github.com/YasiiKB/R3PM-Net.

  </details>



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
  _2026-04-06_ · https://arxiv.org/abs/2604.04576v2  
  <details><summary>Abstract</summary>

  Diffusion models are promising for sparse-view novel view synthesis (NVS), as they can generate pseudo-ground-truth views to aid 3D reconstruction pipelines like 3D Gaussian Splatting (3DGS). However, these synthesized images often contain photometric and geometric inconsistencies, and their direct use for supervision can impair reconstruction. To address this, we propose Partial-Reference Image Quality Assessment (PR-IQA), a framework that evaluates diffusion-generated views using reference images from different poses, eliminating the need for ground truth. PR-IQA first computes a geometrically consistent partial quality map in overlapping regions. It then performs quality completion to inpaint this partial map into a dense, full-image map. This completion is achieved via a cross-attention mechanism that incorporates reference-view context, ensuring cross-view consistency and enabling thorough quality assessment. When integrated into a diffusion-augmented 3DGS pipeline, PR-IQA restricts supervision to high-confidence regions identified by its quality maps. Experiments demonstrate that PR-IQA outperforms existing IQA methods, achieving full-reference-level accuracy without ground-truth supervision. Thus, our quality-aware 3DGS approach more effectively filters inconsistencies, producing superior 3D reconstructions and NVS results. The project page is available at https://kakaomacao.github.io/pr-iqa-project-page/.

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


