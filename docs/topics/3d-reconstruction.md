# 3D Reconstruction

_Updated: 2026-03-13 07:10 UTC_

Total papers shown: **21**


---

- **DVD: Deterministic Video Depth Estimation with Generative Priors**  
  Hongfei Zhang, Harold Haodong Chen, Chenfei Liao, Jing He, Zixin Zhang, Haodong Li, Yihao Liang, Kanghao Chen, Bin Ren, Xu Zheng, et al.  
  _2026-03-12_ · https://arxiv.org/abs/2603.12250v1  
  <details><summary>Abstract</summary>

  Existing video depth estimation faces a fundamental trade-off: generative models suffer from stochastic geometric hallucinations and scale drift, while discriminative models demand massive labeled datasets to resolve semantic ambiguities. To break this impasse, we present DVD, the first framework to deterministically adapt pre-trained video diffusion models into single-pass depth regressors. Specifically, DVD features three core designs: (i) repurposing the diffusion timestep as a structural anchor to balance global stability with high-frequency details; (ii) latent manifold rectification (LMR) to mitigate regression-induced over-smoothing, enforcing differential constraints to restore sharp boundaries and coherent motion; and (iii) global affine coherence, an inherent property bounding inter-window divergence, which enables seamless long-video inference without requiring complex temporal alignment. Extensive experiments demonstrate that DVD achieves state-of-the-art zero-shot performance across benchmarks. Furthermore, DVD successfully unlocks the profound geometric priors implicit in video foundation models using 163x less task-specific data than leading baselines. Notably, we fully release our pipeline, providing the whole training suite for SOTA video depth estimation to benefit the open-source community.

  </details>



- **Ada3Drift: Adaptive Training-Time Drifting for One-Step 3D Visuomotor Robotic Manipulation**  
  Chongyang Xu, Yixian Zou, Ziliang Feng, Fanman Meng, Shuaicheng Liu  
  _2026-03-12_ · https://arxiv.org/abs/2603.11984v1  
  <details><summary>Abstract</summary>

  Diffusion-based visuomotor policies effectively capture multimodal action distributions through iterative denoising, but their high inference latency limits real-time robotic control. Recent flow matching and consistency-based methods achieve single-step generation, yet sacrifice the ability to preserve distinct action modes, collapsing multimodal behaviors into averaged, often physically infeasible trajectories. We observe that the compute budget asymmetry in robotics (offline training vs.\ real-time inference) naturally motivates recovering this multimodal fidelity by shifting iterative refinement from inference time to training time. Building on this insight, we propose Ada3Drift, which learns a training-time drifting field that attracts predicted actions toward expert demonstration modes while repelling them from other generated samples, enabling high-fidelity single-step generation (1 NFE) from 3D point cloud observations. To handle the few-shot robotic regime, Ada3Drift further introduces a sigmoid-scheduled loss transition from coarse distribution learning to mode-sharpening refinement, and multi-scale field aggregation that captures action modes at varying spatial granularities. Experiments on three simulation benchmarks (Adroit, Meta-World, and RoboTwin) and real-world robotic manipulation tasks demonstrate that Ada3Drift achieves state-of-the-art performance while requiring $10\times$ fewer function evaluations than diffusion-based alternatives.

  </details>



- **AstroSplat: Physics-Based Gaussian Splatting for Rendering and Reconstruction of Small Celestial Bodies**  
  Jennifer Nolan, Travis Driver, John Christian  
  _2026-03-12_ · https://arxiv.org/abs/2603.11969v1  
  <details><summary>Abstract</summary>

  Image-based surface reconstruction and characterization are crucial for missions to small celestial bodies (e.g., asteroids), as it informs mission planning, navigation, and scientific analysis. Recent advances in Gaussian splatting enable high-fidelity neural scene representations but typically rely on a spherical harmonic intensity parameterization that is strictly appearance-based and does not explicitly model material properties or light-surface interactions. We introduce AstroSplat, a physics-based Gaussian splatting framework that integrates planetary reflectance models to improve the autonomous reconstruction and photometric characterization of small-body surfaces from in-situ imagery. The proposed framework is validated on real imagery taken by NASA's Dawn mission, where we demonstrate superior rendering performance and surface reconstruction accuracy compared to the typical spherical harmonic parameterization.

  </details>



- **Single-View Rolling-Shutter SfM**  
  Sofía Errázuriz Muñoz, Kim Kiehn, Petr Hruby, Kathlén Kohn  
  _2026-03-12_ · https://arxiv.org/abs/2603.11888v1  
  <details><summary>Abstract</summary>

  Rolling-shutter (RS) cameras are ubiquitous, but RS SfM (structure-from-motion) has not been fully solved yet. This work suggests an approach to remedy this: We characterize RS single-view geometry of observed world points or lines. Exploiting this geometry, we describe which motion and scene parameters can be recovered from a single RS image and systematically derive minimal reconstruction problems. We evaluate several representative cases with proof-of-concept solvers, highlighting both feasibility and practical limitations.

  </details>



- **CEI-3D: Collaborative Explicit-Implicit 3D Reconstruction for Realistic and Fine-Grained Object Editing**  
  Yue Shi, Rui Shi, Yuxuan Xiong, Bingbing Ni, Wenjun Zhang  
  _2026-03-12_ · https://arxiv.org/abs/2603.11810v1  
  <details><summary>Abstract</summary>

  Existing 3D editing methods often produce unrealistic and unrefined results due to the deeply integrated nature of their reconstruction networks. To address the challenge, this paper introduces CEI-3D, an editing-oriented reconstruction pipeline designed to facilitate realistic and fine-grained editing. Specifically, we propose a collaborative explicit-implicit reconstruction approach, which represents the target object using an implicit SDF network and a differentially sampled, locally controllable set of handler points. The implicit network provides a smooth and continuous geometry prior, while the explicit handler points offer localized control, enabling mutual guidance between the global 3D structure and user-specified local editing regions. To independently control each attribute of the handler points, we design a physical properties disentangling module to decouple the color of the handler points into separate physical properties. We also propose a dual-diffuse-albedo network in this module to process the edited and non-edited regions through separate branches, thereby preventing undesired interference from editing operations. Building on the reconstructed collaborative explicit-implicit representation with disentangled properties, we introduce a spatial-aware editing module that enables part-wise adjustment of relevant handler points. This module employs a cross-view propagation-based 3D segmentation strategy, which helps users to edit the specified physical attributes of a target part efficiently. Extensive experiments on both real and synthetic datasets demonstrate that our approach achieves more realistic and fine-grained editing results than the state-of-the-art (SOTA) methods while requiring less editing time. Our code is available on https://github.com/shiyue001/CEI-3D.

  </details>



- **R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection**  
  Zhongyu Xia, Yousen Tang, Yongtao Wang, Zhifeng Wang, Weijun Qin  
  _2026-03-12_ · https://arxiv.org/abs/2603.11566v1  
  <details><summary>Abstract</summary>

  4D radar-camera sensing configuration has gained increasing importance in autonomous driving. However, existing 3D object detection methods that fuse 4D Radar and camera data confront several challenges. First, their absolute depth estimation module is not robust and accurate enough, leading to inaccurate 3D localization. Second, the performance of their temporal fusion module will degrade dramatically or even fail when the ego vehicle's pose is missing or inaccurate. Third, for some small objects, the sparse radar point clouds may completely fail to reflect from their surfaces. In such cases, detection must rely solely on visual unimodal priors. To address these limitations, we propose R4Det, which enhances depth estimation quality via the Panoramic Depth Fusion module, enabling mutual reinforcement between absolute and relative depth. For temporal fusion, we design a Deformable Gated Temporal Fusion module that does not rely on the ego vehicle's pose. In addition, we built an Instance-Guided Dynamic Refinement module that extracts semantic prototypes from 2D instance guidance. Experiments show that R4Det achieves state-of-the-art 3D object detection results on the TJ4DRadSet and VoD datasets.

  </details>



- **High-Precision 6DOF Pose Estimation via Global Phase Retrieval in Fringe Projection Profilometry for 3D Mapping**  
  Sehoon Tak, Keunhee Cho, Sangpil Kim, Jae-Sang Hyun  
  _2026-03-12_ · https://arxiv.org/abs/2603.11389v1  
  <details><summary>Abstract</summary>

  Digital fringe projection (DFP) enables micrometer-level 3D reconstruction, yet extending it to large-scale mapping remains challenging because six-degree-of-freedom pose estimation often cannot match the reconstruction's precision. Conventional iterative closest point (ICP) registration becomes inefficient on multi-million-point clouds and typically relies on downsampling or feature-based selection, which can reduce local detail and degrade pose precision. Drift-correction methods improve long-term consistency but do not resolve sampling sensitivity in dense DFP point clouds.We propose a high-precision pose estimation method that augments a moving DFP system with a fixed, intrinsically calibrated global projector. Using the global projector's phase-derived pixel constraints and a PnP-style reprojection objective, the method estimates the DFP system pose in a fixed reference frame without relying on deterministic feature extraction, and we experimentally demonstrate sampling invariance under coordinate-preserving subsampling. Experiments demonstrate sub-millimeter pose accuracy against a reference with quantified uncertainty bounds, high repeatability under aggressive subsampling, robust operation on homogeneous surfaces and low-overlap views, and reduced error accumulation when used to correct ICP-based trajectories. The method extends DFP toward accurate 3D mapping in quasi-static scenarios such as inspection and metrology, with the trade-off of time-multiplexed acquisition for the additional projector measurements.

  </details>



- **InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction**  
  Dingqiang Ye, Jiacong Xu, Jianglu Ping, Yuxiang Guo, Chao Fan, Vishal M. Patel  
  _2026-03-11_ · https://arxiv.org/abs/2603.11298v1  
  <details><summary>Abstract</summary>

  High dynamic range (HDR) novel view synthesis (NVS) aims to reconstruct HDR scenes from multi-exposure low dynamic range (LDR) images. Existing HDR pipelines heavily rely on known camera poses, well-initialized dense point clouds, and time-consuming per-scene optimization. Current feed-forward alternatives overlook the HDR problem by assuming exposure-invariant appearance. To bridge this gap, we propose InstantHDR, a feed-forward network that reconstructs 3D HDR scenes from uncalibrated multi-exposure LDR collections in a single forward pass. Specifically, we design a geometry-guided appearance modeling for multi-exposure fusion, and a meta-network for generalizable scene-specific tone mapping. Due to the lack of HDR scene data, we build a pre-training dataset, called HDR-Pretrain, for generalizable feed-forward HDR models, featuring 168 Blender-rendered scenes, diverse lighting types, and multiple camera response functions. Comprehensive experiments show that our InstantHDR delivers comparable synthesis performance to the state-of-the-art optimization-based HDR methods while enjoying $\sim700\times$ and $\sim20\times$ reconstruction speed improvement with our single-forward and post-optimization settings. All code, models, and datasets will be released after the review process.

  </details>



- **GGPT: Geometry Grounded Point Transformer**  
  Yutong Chen, Yiming Wang, Xucong Zhang, Sergey Prokudin, Siyu Tang  
  _2026-03-11_ · https://arxiv.org/abs/2603.11174v1  
  <details><summary>Abstract</summary>

  Recent feed-forward networks have achieved remarkable progress in sparse-view 3D reconstruction by predicting dense point maps directly from RGB images. However, they often suffer from geometric inconsistencies and limited fine-grained accuracy due to the absence of explicit multi-view constraints. We introduce the Geometry-Grounded Point Transformer (GGPT), a framework that augments feed-forward reconstruction with reliable sparse geometric guidance. We first propose an improved Structure-from-Motion pipeline based on dense feature matching and lightweight geometric optimisation to efficiently estimate accurate camera poses and partial 3D point clouds from sparse input views. Building on this foundation, we propose a geometry-guided 3D point transformer that refines dense point maps under explicit partial-geometry supervision using an optimised guidance encoding. Extensive experiments demonstrate that our method provides a principled mechanism for integrating geometric priors with dense feed-forward predictions, producing reconstructions that are both geometrically consistent and spatially complete, recovering fine structures and filling gaps in textureless areas. Trained solely on ScanNet++ with VGGT predictions, GGPT generalises across architectures and datasets, substantially outperforming state-of-the-art feed-forward 3D reconstruction models in both in-domain and out-of-domain settings.

  </details>



- **Neural Field Thermal Tomography: A Differentiable Physics Framework for Non-Destructive Evaluation**  
  Tao Zhong, Yixun Hu, Dongzhe Zheng, Aditya Sood, Christine Allen-Blanchette  
  _2026-03-11_ · https://arxiv.org/abs/2603.11045v1  
  <details><summary>Abstract</summary>

  We propose Neural Field Thermal Tomography (NeFTY), a differentiable physics framework for the quantitative 3D reconstruction of material properties from transient surface temperature measurements. While traditional thermography relies on pixel-wise 1D approximations that neglect lateral diffusion, and soft-constrained Physics-Informed Neural Networks (PINNs) often fail in transient diffusion scenarios due to gradient stiffness, NeFTY parameterizes the 3D diffusivity field as a continuous neural field optimized through a rigorous numerical solver. By leveraging a differentiable physics solver, our approach enforces thermodynamic laws as hard constraints while maintaining the memory efficiency required for high-resolution 3D tomography. Our discretize-then-optimize paradigm effectively mitigates the spectral bias and ill-posedness inherent in inverse heat conduction, enabling the recovery of subsurface defects at arbitrary scales. Experimental validation on synthetic data demonstrates that NeFTY significantly improves the accuracy of subsurface defect localization over baselines. Additional details at https://cab-lab-princeton.github.io/nefty/

  </details>



- **TreeON: Reconstructing 3D Tree Point Clouds from Orthophotos and Heightmaps**  
  Angeliki Grammatikaki, Johannes Eschner, Pedro Hermosilla, Oscar Argudo, Manuela Waldner  
  _2026-03-11_ · https://arxiv.org/abs/2603.10996v1  
  <details><summary>Abstract</summary>

  We present TreeON, a novel neural-based framework for reconstructing detailed 3D tree point clouds from sparse top-down geodata, using only a single orthophoto and its corresponding Digital Surface Model (DSM). Our method introduces a new training supervision strategy that combines both geometric supervision and differentiable shadow and silhouette losses to learn point cloud representations of trees without requiring species labels, procedural rules, terrestrial reconstruction data, or ground laser scans. To address the lack of ground truth data, we generate a synthetic dataset of point clouds from procedurally modeled trees and train our network on it. Quantitative and qualitative experiments demonstrate better reconstruction quality and coverage compared to existing methods, as well as strong generalization to real-world data, producing visually appealing and structurally plausible tree point cloud representations suitable for integration into interactive digital 3D maps. The codebase, synthetic dataset, and pretrained model are publicly available at https://angelikigram.github.io/treeON/.

  </details>



- **Pointy - A Lightweight Transformer for Point Cloud Foundation Models**  
  Konrad Szafer, Marek Kraft, Dominik Belter  
  _2026-03-11_ · https://arxiv.org/abs/2603.10963v1  
  <details><summary>Abstract</summary>

  Foundation models for point cloud data have recently grown in capability, often leveraging extensive representation learning from language or vision. In this work, we take a more controlled approach by introducing a lightweight transformer-based point cloud architecture. In contrast to the heavy reliance on cross-modal supervision, our model is trained only on 39k point clouds - yet it outperforms several larger foundation models trained on over 200k training samples. Interestingly, our method approaches state-of-the-art results from models that have seen over a million point clouds, images, and text samples, demonstrating the value of a carefully curated training setup and architecture. To ensure rigorous evaluation, we conduct a comprehensive replication study that standardizes the training regime and benchmarks across multiple point cloud architectures. This unified experimental framework isolates the impact of architectural choices, allowing for transparent comparisons and highlighting the benefits of our design and other tokenizer-free architectures. Our results show that simple backbones can deliver competitive results to more complex or data-rich strategies. The implementation, including code, pre-trained models, and training protocols, is available at https://github.com/KonradSzafer/Pointy.

  </details>



- **S2D: Sparse to Dense Lifting for 3D Reconstruction with Minimal Inputs**  
  Yuzhou Ji, Qijian Tian, He Zhu, Xiaoqi Jiang, Guangzhi Cao, Lizhuang Ma, Yuan Xie, Xin Tan  
  _2026-03-11_ · https://arxiv.org/abs/2603.10893v1  
  <details><summary>Abstract</summary>

  Explicit 3D representations have already become an essential medium for 3D simulation and understanding. However, the most commonly used point cloud and 3D Gaussian Splatting (3DGS) each suffer from non-photorealistic rendering and significant degradation under sparse inputs. In this paper, we introduce Sparse to Dense lifting (S2D), a novel pipeline that bridges the two representations and achieves high-quality 3DGS reconstruction with minimal inputs. Specifically, the S2D lifting is two-fold. We first present an efficient one-step diffusion model that lifts sparse point cloud for high-fidelity image artifact fixing. Meanwhile, to reconstruct 3D consistent scenes, we also design a corresponding reconstruction strategy with random sample drop and weighted gradient for robust model fitting from sparse input views to dense novel views. Extensive experiments show that S2D achieves the best consistency in generating novel view guidance and first-tier sparse view reconstruction quality under different input sparsity. By reconstructing stable scenes with the least possible captures among existing methods, S2D enables minimal input requirements for 3DGS applications.

  </details>



- **PolGS++: Physically-Guided Polarimetric Gaussian Splatting for Fast Reflective Surface Reconstruction**  
  Yufei Han, Chu Zhou, Youwei Lyu, Qi Chen, Si Li, Boxin Shi, Yunpeng Jia, Heng Guo, Zhanyu Ma  
  _2026-03-11_ · https://arxiv.org/abs/2603.10801v1  
  <details><summary>Abstract</summary>

  Accurate reconstruction of reflective surfaces remains a fundamental challenge in computer vision, with broad applications in real-time virtual reality and digital content creation. Although 3D Gaussian Splatting (3DGS) enables efficient novel-view rendering with explicit representations, its performance on reflective surfaces still lags behind implicit neural methods, especially in recovering fine geometry and surface normals. To address this gap, we propose PolGS++, a physically-guided polarimetric Gaussian Splatting framework for fast reflective surface reconstruction. Specifically, we integrate a polarized BRDF (pBRDF) model into 3DGS to explicitly decouple diffuse and specular components, providing physically grounded reflectance modeling and stronger geometric cues for reflective surface recovery. Furthermore, we introduce a depth-guided visibility mask acquisition mechanism that enables angle-of-polarization (AoP)-based tangent-space consistency constraints in Gaussian Splatting without costly ray-tracing intersections. This physically guided design improves reconstruction quality and efficiency, requiring only about 10 minutes of training. Extensive experiments on both synthetic and real-world datasets validate the effectiveness of our method.

  </details>



- **WalkGPT: Grounded Vision-Language Conversation with Depth-Aware Segmentation for Pedestrian Navigation**  
  Rafi Ibn Sultan, Hui Zhu, Xiangyu Zhou, Chengyin Li, Prashant Khanduri, Marco Brocanelli, Dongxiao Zhu  
  _2026-03-11_ · https://arxiv.org/abs/2603.10703v1  
  <details><summary>Abstract</summary>

  Ensuring accessible pedestrian navigation requires reasoning about both semantic and spatial aspects of complex urban scenes, a challenge that existing Large Vision-Language Models (LVLMs) struggle to meet. Although these models can describe visual content, their lack of explicit grounding leads to object hallucinations and unreliable depth reasoning, limiting their usefulness for accessibility guidance. We introduce WalkGPT, a pixel-grounded LVLM for the new task of Grounded Navigation Guide, unifying language reasoning and segmentation within a single architecture for depth-aware accessibility guidance. Given a pedestrian-view image and a navigation query, WalkGPT generates a conversational response with segmentation masks that delineate accessible and harmful features, along with relative depth estimation. The model incorporates a Multi-Scale Query Projector (MSQP) that shapes the final image tokens by aggregating them along text tokens across spatial hierarchies, and a Calibrated Text Projector (CTP), guided by a proposed Region Alignment Loss, that maps language embeddings into segmentation-aware representations. These components enable fine-grained grounding and depth inference without user-provided cues or anchor points, allowing the model to generate complete and realistic navigation guidance. We also introduce PAVE, a large-scale benchmark of 41k pedestrian-view images paired with accessibility-aware questions and depth-grounded answers. Experiments show that WalkGPT achieves strong grounded reasoning and segmentation performance. The source code and dataset are available on the \href{https://sites.google.com/view/walkgpt-26/home}{project website}.

  </details>



- **TopGen: Learning Structural Layouts and Cross-Fields for Quadrilateral Mesh Generation**  
  Yuguang Chen, Xinhai Liu, Xiangyu Zhu, Yiling Zhu, Zhuo Chen, Dongyu Zhang, Chunchao Guo  
  _2026-03-11_ · https://arxiv.org/abs/2603.10606v1  
  <details><summary>Abstract</summary>

  High-quality quadrilateral mesh generation is a fundamental challenge in computer graphics. Traditional optimization-based methods are often constrained by the topological quality of input meshes and suffer from severe efficiency bottlenecks, frequently becoming computationally prohibitive when handling high-resolution models. While emerging learning-based approaches offer greater flexibility, they primarily focus on cross-field prediction, often resulting in the loss of critical structural layouts and a lack of editability. In this paper, we propose TopGen, a robust and efficient learning-based framework that mimics professional manual modeling workflows by simultaneously predicting structural layouts and cross-fields. By processing input triangular meshes through point cloud sampling and a shape encoder, TopGen is inherently robust to non-manifold geometries and low-quality initial topologies. We introduce a dual-query decoder using edge-based and face-based sampling points as queries to perform structural line classification and cross-field regression in parallel. This integrated approach explicitly extracts the geometric skeleton while concurrently capturing orientation fields. Such synergy ensures the preservation of geometric integrity and provides an intuitive, editable foundation for subsequent quadrilateral remeshing. To support this framework, we also introduce a large-scale quadrilateral mesh dataset, TopGen-220K, featuring high-quality paired data comprising raw triangular meshes, structural layouts, cross-fields, and their corresponding quad meshes. Experimental results demonstrate that TopGen significantly outperforms existing state-of-the-art methods in both geometric fidelity and topological edge flow rationality.

  </details>



- **LCAMV: High-Accuracy 3D Reconstruction of Color-Varying Objects Using LCA Correction and Minimum-Variance Fusion in Structured Light**  
  Wonbeen Oh, Jae-Sang Hyun  
  _2026-03-11_ · https://arxiv.org/abs/2603.10456v1  
  <details><summary>Abstract</summary>

  Accurate 3D reconstruction of colored objects with structured light (SL) is hindered by lateral chromatic aberration (LCA) in optical components and uneven noise characteristics across RGB channels. This paper introduces lateral chromatic aberration correction and minimum-variance fusion (LCAMV), a robust 3D reconstruction method that operates with a single projector-camera pair without additional hardware or acquisition constraints. LCAMV analytically models and pixel-wise compensates LCA in both the projector and camera, then adaptively fuses multi-channel phase data using a Poisson-Gaussian noise model and minimum-variance estimation. Unlike existing methods that require extra hardware or multiple exposures, LCAMV enables fast acquisition. Experiments on planar and non-planar colored surfaces show that LCAMV outperforms grayscale conversion and conventional channel-weighting, reducing depth error by up to 43.6\%. These results establish LCAMV as an effective solution for high-precision 3D reconstruction of nonuniformly colored objects.

  </details>



- **AsyncMDE: Real-Time Monocular Depth Estimation via Asynchronous Spatial Memory**  
  Lianjie Ma, Yuquan Li, Bingzheng Jiang, Ziming Zhong, Han Ding, Lijun Zhu  
  _2026-03-11_ · https://arxiv.org/abs/2603.10438v1  
  <details><summary>Abstract</summary>

  Foundation-model-based monocular depth estimation offers a viable alternative to active sensors for robot perception, yet its computational cost often prohibits deployment on edge platforms. Existing methods perform independent per-frame inference, wasting the substantial computational redundancy between adjacent viewpoints in continuous robot operation. This paper presents AsyncMDE, an asynchronous depth perception system consisting of a foundation model and a lightweight model that amortizes the foundation model's computational cost over time. The foundation model produces high-quality spatial features in the background, while the lightweight model runs asynchronously in the foreground, fusing cached memory with current observations through complementary fusion, outputting depth estimates, and autoregressively updating the memory. This enables cross-frame feature reuse with bounded accuracy degradation. At a mere 3.83M parameters, it operates at 237 FPS on an RTX 4090, recovering 77% of the accuracy gap to the foundation model while achieving a 25X parameter reduction. Validated across indoor static, dynamic, and synthetic extreme-motion benchmarks, AsyncMDE degrades gracefully between refreshes and achieves 161FPS on a Jetson AGX Orin with TensorRT, clearly demonstrating its feasibility for real-time edge deployment.

  </details>



- **On the Structural Failure of Chamfer Distance in 3D Shape Optimization**  
  Chang-Yong Song, David Hyde  
  _2026-03-10_ · https://arxiv.org/abs/2603.09925v1  
  <details><summary>Abstract</summary>

  Chamfer distance is the standard training loss for point cloud reconstruction, completion, and generation, yet directly optimizing it can produce worse Chamfer values than not optimizing it at all. We show that this paradoxical failure is gradient-structural. The per-point Chamfer gradient creates a many-to-one collapse that is the unique attractor of the forward term and cannot be resolved by any local regularizer, including repulsion, smoothness, and density-aware re-weighting. We derive a necessary condition for collapse suppression: coupling must propagate beyond local neighborhoods. In a controlled 2D setting, shared-basis deformation suppresses collapse by providing global coupling; in 3D shape morphing, a differentiable MPM prior instantiates the same principle, consistently reducing the Chamfer gap across 20 directed pairs with a 2.5$\times$ improvement on the topologically complex dragon. The presence or absence of non-local coupling determines whether Chamfer optimization succeeds or collapses. This provides a practical design criterion for any pipeline that optimizes point-level distance metrics.

  </details>



- **VLM-Loc: Localization in Point Cloud Maps via Vision-Language Models**  
  Shuhao Kang, Youqi Liao, Peijie Wang, Wenlong Liao, Qilin Zhang, Benjamin Busam, Xieyuanli Chen, Yun Liu  
  _2026-03-10_ · https://arxiv.org/abs/2603.09826v1  
  <details><summary>Abstract</summary>

  Text-to-point-cloud (T2P) localization aims to infer precise spatial positions within 3D point cloud maps from natural language descriptions, reflecting how humans perceive and communicate spatial layouts through language. However, existing methods largely rely on shallow text-point cloud correspondence without effective spatial reasoning, limiting their accuracy in complex environments. To address this limitation, we propose VLM-Loc, a framework that leverages the spatial reasoning capability of large vision-language models (VLMs) for T2P localization. Specifically, we transform point clouds into bird's-eye-view (BEV) images and scene graphs that jointly encode geometric and semantic context, providing structured inputs for the VLM to learn cross-modal representations bridging linguistic and spatial semantics. On top of these representations, we introduce a partial node assignment mechanism that explicitly associates textual cues with scene graph nodes, enabling interpretable spatial reasoning for accurate localization. To facilitate systematic evaluation across diverse scenes, we present CityLoc, a benchmark built from multi-source point clouds for fine-grained T2P localization. Experiments on CityLoc demonstrate VLM-Loc achieves superior accuracy and robustness compared to state-of-the-art methods. Our code, model, and dataset are available at \href{https://github.com/MCG-NKU/nku-3d-vision}{repository}.

  </details>



- **ConfCtrl: Enabling Precise Camera Control in Video Diffusion via Confidence-Aware Interpolation**  
  Liudi Yang, George Eskandar, Fengyi Shen, Mohammad Altillawi, Yang Bai, Chi Zhang, Ziyuan Liu, Abhinav Valada  
  _2026-03-10_ · https://arxiv.org/abs/2603.09819v1  
  <details><summary>Abstract</summary>

  We address the challenge of novel view synthesis from only two input images under large viewpoint changes. Existing regression-based methods lack the capacity to reconstruct unseen regions, while camera-guided diffusion models often deviate from intended trajectories due to noisy point cloud projections or insufficient conditioning from camera poses. To address these issues, we propose ConfCtrl, a confidence-aware video interpolation framework that enables diffusion models to follow prescribed camera poses while completing unseen regions. ConfCtrl initializes the diffusion process by combining a confidence-weighted projected point cloud latent with noise as the conditioning input. It then applies a Kalman-inspired predict-update mechanism, treating the projected point cloud as a noisy measurement and using learned residual corrections to balance pose-driven predictions with noisy geometric observations. This allows the model to rely on reliable projections while down-weighting uncertain regions, yielding stable, geometry-aware generation. Experiments on multiple datasets show that ConfCtrl produces geometrically consistent and visually plausible novel views, effectively reconstructing occluded regions under large viewpoint changes.

  </details>


