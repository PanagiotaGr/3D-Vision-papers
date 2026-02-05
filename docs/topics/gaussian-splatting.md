# Gaussian Splatting & 3DGS

_Updated: 2026-02-05 07:14 UTC_

Total papers shown: **11**


---

- **Nix and Fix: Targeting 1000x Compression of 3D Gaussian Splatting with Diffusion Models**  
  Cem Eteke, Enzo Tartaglione  
  _2026-02-04_ · https://arxiv.org/abs/2602.04549v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) revolutionized novel view rendering. Instead of inferring from dense spatial points, as implicit representations do, 3DGS uses sparse Gaussians. This enables real-time performance but increases space requirements, hindering applications such as immersive communication. 3DGS compression emerged as a field aimed at alleviating this issue. While impressive progress has been made, at low rates, compression introduces artifacts that degrade visual quality significantly. We introduce NiFi, a method for extreme 3DGS compression through restoration via artifact-aware, diffusion-based one-step distillation. We show that our method achieves state-of-the-art perceptual quality at extremely low rates, down to 0.1 MB, and towards 1000x rate improvement over 3DGS at comparable perceptual performance. The code will be open-sourced upon acceptance.

  </details>



- **VecSet-Edit: Unleashing Pre-trained LRM for Mesh Editing from Single Image**  
  Teng-Fang Hsiao, Bo-Kai Ruan, Yu-Lun Liu, Hong-Han Shuai  
  _2026-02-04_ · https://arxiv.org/abs/2602.04349v1  
  <details><summary>Abstract</summary>

  3D editing has emerged as a critical research area to provide users with flexible control over 3D assets. While current editing approaches predominantly focus on 3D Gaussian Splatting or multi-view images, the direct editing of 3D meshes remains underexplored. Prior attempts, such as VoxHammer, rely on voxel-based representations that suffer from limited resolution and necessitate labor-intensive 3D mask. To address these limitations, we propose \textbf{VecSet-Edit}, the first pipeline that leverages the high-fidelity VecSet Large Reconstruction Model (LRM) as a backbone for mesh editing. Our approach is grounded on a analysis of the spatial properties in VecSet tokens, revealing that token subsets govern distinct geometric regions. Based on this insight, we introduce Mask-guided Token Seeding and Attention-aligned Token Gating strategies to precisely localize target regions using only 2D image conditions. Also, considering the difference between VecSet diffusion process versus voxel we design a Drift-aware Token Pruning to reject geometric outliers during the denoising process. Finally, our Detail-preserving Texture Baking module ensures that we not only preserve the geometric details of original mesh but also the textural information. More details can be found in our project page: https://github.com/BlueDyee/VecSet-Edit/tree/main

  </details>



- **JOintGS: Joint Optimization of Cameras, Bodies and 3D Gaussians for In-the-Wild Monocular Reconstruction**  
  Zihan Lou, Jinlong Fan, Sihan Ma, Yuxiang Yang, Jing Zhang  
  _2026-02-04_ · https://arxiv.org/abs/2602.04317v1  
  <details><summary>Abstract</summary>

  Reconstructing high-fidelity animatable 3D human avatars from monocular RGB videos remains challenging, particularly in unconstrained in-the-wild scenarios where camera parameters and human poses from off-the-shelf methods (e.g., COLMAP, HMR2.0) are often inaccurate. Splatting (3DGS) advances demonstrate impressive rendering quality and real-time performance, they critically depend on precise camera calibration and pose annotations, limiting their applicability in real-world settings. We present JOintGS, a unified framework that jointly optimizes camera extrinsics, human poses, and 3D Gaussian representations from coarse initialization through a synergistic refinement mechanism. Our key insight is that explicit foreground-background disentanglement enables mutual reinforcement: static background Gaussians anchor camera estimation via multi-view consistency; refined cameras improve human body alignment through accurate temporal correspondence; optimized human poses enhance scene reconstruction by removing dynamic artifacts from static constraints. We further introduce a temporal dynamics module to capture fine-grained pose-dependent deformations and a residual color field to model illumination variations. Extensive experiments on NeuMan and EMDB datasets demonstrate that JOintGS achieves superior reconstruction quality, with 2.1~dB PSNR improvement over state-of-the-art methods on NeuMan dataset, while maintaining real-time rendering. Notably, our method shows significantly enhanced robustness to noisy initialization compared to the baseline.Our source code is available at https://github.com/MiliLab/JOintGS.

  </details>



- **SkeletonGaussian: Editable 4D Generation through Gaussian Skeletonization**  
  Lifan Wu, Ruijie Zhu, Yubo Ai, Tianzhu Zhang  
  _2026-02-04_ · https://arxiv.org/abs/2602.04271v1  
  <details><summary>Abstract</summary>

  4D generation has made remarkable progress in synthesizing dynamic 3D objects from input text, images, or videos. However, existing methods often represent motion as an implicit deformation field, which limits direct control and editability. To address this issue, we propose SkeletonGaussian, a novel framework for generating editable dynamic 3D Gaussians from monocular video input. Our approach introduces a hierarchical articulated representation that decomposes motion into sparse rigid motion explicitly driven by a skeleton and fine-grained non-rigid motion. Concretely, we extract a robust skeleton and drive rigid motion via linear blend skinning, followed by a hexplane-based refinement for non-rigid deformations, enhancing interpretability and editability. Experimental results demonstrate that SkeletonGaussian surpasses existing methods in generation quality while enabling intuitive motion editing, establishing a new paradigm for editable 4D generation. Project page: https://wusar.github.io/projects/skeletongaussian/

  </details>



- **AnyStyle: Single-Pass Multimodal Stylization for 3D Gaussian Splatting**  
  Joanna Kaleta, Bartosz Świrta, Kacper Kania, Przemysław Spurek, Marek Kowalski  
  _2026-02-03_ · https://arxiv.org/abs/2602.04043v1  
  <details><summary>Abstract</summary>

  The growing demand for rapid and scalable 3D asset creation has driven interest in feed-forward 3D reconstruction methods, with 3D Gaussian Splatting (3DGS) emerging as an effective scene representation. While recent approaches have demonstrated pose-free reconstruction from unposed image collections, integrating stylization or appearance control into such pipelines remains underexplored. Existing attempts largely rely on image-based conditioning, which limits both controllability and flexibility. In this work, we introduce AnyStyle, a feed-forward 3D reconstruction and stylization framework that enables pose-free, zero-shot stylization through multimodal conditioning. Our method supports both textual and visual style inputs, allowing users to control the scene appearance using natural language descriptions or reference images. We propose a modular stylization architecture that requires only minimal architectural modifications and can be integrated into existing feed-forward 3D reconstruction backbones. Experiments demonstrate that AnyStyle improves style controllability over prior feed-forward stylization methods while preserving high-quality geometric reconstruction. A user study further confirms that AnyStyle achieves superior stylization quality compared to an existing state-of-the-art approach. Repository: https://github.com/joaxkal/AnyStyle.

  </details>



- **Constrained Dynamic Gaussian Splatting**  
  Zihan Zheng, Zhenglong Wu, Xuanxuan Wang, Houqiang Zhong, Xiaoyun Zhang, Qiang Hu, Guangtao Zhai, Wenjun Zhang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03538v1  
  <details><summary>Abstract</summary>

  While Dynamic Gaussian Splatting enables high-fidelity 4D reconstruction, its deployment is severely hindered by a fundamental dilemma: unconstrained densification leads to excessive memory consumption incompatible with edge devices, whereas heuristic pruning fails to achieve optimal rendering quality under preset Gaussian budgets. In this work, we propose Constrained Dynamic Gaussian Splatting (CDGS), a novel framework that formulates dynamic scene reconstruction as a budget-constrained optimization problem to enforce a strict, user-defined Gaussian budget during training. Our key insight is to introduce a differentiable budget controller as the core optimization driver. Guided by a multi-modal unified importance score, this controller fuses geometric, motion, and perceptual cues for precise capacity regulation. To maximize the utility of this fixed budget, we further decouple the optimization of static and dynamic elements, employing an adaptive allocation mechanism that dynamically distributes capacity based on motion complexity. Furthermore, we implement a three-phase training strategy to seamlessly integrate these constraints, ensuring precise adherence to the target count. Coupled with a dual-mode hybrid compression scheme, CDGS not only strictly adheres to hardware constraints (error < 2%}) but also pushes the Pareto frontier of rate-distortion performance. Extensive experiments demonstrate that CDGS delivers optimal rendering quality under varying capacity limits, achieving over 3x compression compared to state-of-the-art methods.

  </details>



- **Pi-GS: Sparse-View Gaussian Splatting with Dense π^3 Initialization**  
  Manuel Hofer, Markus Steinberger, Thomas Köhler  
  _2026-02-03_ · https://arxiv.org/abs/2602.03327v1  
  <details><summary>Abstract</summary>

  Novel view synthesis has evolved rapidly, advancing from Neural Radiance Fields to 3D Gaussian Splatting (3DGS), which offers real-time rendering and rapid training without compromising visual fidelity. However, 3DGS relies heavily on accurate camera poses and high-quality point cloud initialization, which are difficult to obtain in sparse-view scenarios. While traditional Structure from Motion (SfM) pipelines often fail in these settings, existing learning-based point estimation alternatives typically require reliable reference views and remain sensitive to pose or depth errors. In this work, we propose a robust method utilizing π^3, a reference-free point cloud estimation network. We integrate dense initialization from π^3 with a regularization scheme designed to mitigate geometric inaccuracies. Specifically, we employ uncertainty-guided depth supervision, normal consistency loss, and depth warping. Experimental results demonstrate that our approach achieves state-of-the-art performance on the Tanks and Temples, LLFF, DTU, and MipNeRF360 datasets.

  </details>



- **WebSplatter: Enabling Cross-Device Efficient Gaussian Splatting in Web Browsers via WebGPU**  
  Yudong Han, Chao Xu, Xiaodan Ye, Weichen Bi, Zilong Dong, Yun Ma  
  _2026-02-03_ · https://arxiv.org/abs/2602.03207v1  
  <details><summary>Abstract</summary>

  We present WebSplatter, an end-to-end GPU rendering pipeline for the heterogeneous web ecosystem. Unlike naive ports, WebSplatter introduces a wait-free hierarchical radix sort that circumvents the lack of global atomics in WebGPU, ensuring deterministic execution across diverse hardware. Furthermore, we propose an opacity-aware geometry culling stage that dynamically prunes splats before rasterization, significantly reducing overdraw and peak memory footprint. Evaluation demonstrates that WebSplatter consistently achieves 1.2$\times$ to 4.5$\times$ speedups over state-of-the-art web viewers.

  </details>



- **SharpTimeGS: Sharp and Stable Dynamic Gaussian Splatting via Lifespan Modulation**  
  Zhanfeng Liao, Jiajun Zhang, Hanzhang Tu, Zhixi Wang, Yunqi Gao, Hongwen Zhang, Yebin Liu  
  _2026-02-03_ · https://arxiv.org/abs/2602.02989v1  
  <details><summary>Abstract</summary>

  Novel view synthesis of dynamic scenes is fundamental to achieving photorealistic 4D reconstruction and immersive visual experiences. Recent progress in Gaussian-based representations has significantly improved real-time rendering quality, yet existing methods still struggle to maintain a balance between long-term static and short-term dynamic regions in both representation and optimization. To address this, we present SharpTimeGS, a lifespan-aware 4D Gaussian framework that achieves temporally adaptive modeling of both static and dynamic regions under a unified representation. Specifically, we introduce a learnable lifespan parameter that reformulates temporal visibility from a Gaussian-shaped decay into a flat-top profile, allowing primitives to remain consistently active over their intended duration and avoiding redundant densification. In addition, the learned lifespan modulates each primitives' motion, reducing drift in long-lived static points while retaining unrestricted motion for short-lived dynamic ones. This effectively decouples motion magnitude from temporal duration, improving long-term stability without compromising dynamic fidelity. Moreover, we design a lifespan-velocity-aware densification strategy that mitigates optimization imbalance between static and dynamic regions by allocating more capacity to regions with pronounced motion while keeping static areas compact and stable. Extensive experiments on multiple benchmarks demonstrate that our method achieves state-of-the-art performance while supporting real-time rendering up to 4K resolution at 100 FPS on one RTX 4090.

  </details>



- **SoMA: A Real-to-Sim Neural Simulator for Robotic Soft-body Manipulation**  
  Mu Huang, Hui Wang, Kerui Ren, Linning Xu, Yunsong Zhou, Mulin Yu, Bo Dai, Jiangmiao Pang  
  _2026-02-02_ · https://arxiv.org/abs/2602.02402v1  
  <details><summary>Abstract</summary>

  Simulating deformable objects under rich interactions remains a fundamental challenge for real-to-sim robot manipulation, with dynamics jointly driven by environmental effects and robot actions. Existing simulators rely on predefined physics or data-driven dynamics without robot-conditioned control, limiting accuracy, stability, and generalization. This paper presents SoMA, a 3D Gaussian Splat simulator for soft-body manipulation. SoMA couples deformable dynamics, environmental forces, and robot joint actions in a unified latent neural space for end-to-end real-to-sim simulation. Modeling interactions over learned Gaussian splats enables controllable, stable long-horizon manipulation and generalization beyond observed trajectories without predefined physical models. SoMA improves resimulation accuracy and generalization on real-world robot manipulation by 20%, enabling stable simulation of complex tasks such as long-horizon cloth folding.

  </details>



- **Uncertainty-Aware Image Classification In Biomedical Imaging Using Spectral-normalized Neural Gaussian Processes**  
  Uma Meleti, Jeffrey J. Nirschl  
  _2026-02-02_ · https://arxiv.org/abs/2602.02370v1  
  <details><summary>Abstract</summary>

  Accurate histopathologic interpretation is key for clinical decision-making; however, current deep learning models for digital pathology are often overconfident and poorly calibrated in out-of-distribution (OOD) settings, which limit trust and clinical adoption. Safety-critical medical imaging workflows benefit from intrinsic uncertainty-aware properties that can accurately reject OOD input. We implement the Spectral-normalized Neural Gaussian Process (SNGP), a set of lightweight modifications that apply spectral normalization and replace the final dense layer with a Gaussian process layer to improve single-model uncertainty estimation and OOD detection. We evaluate SNGP vs. deterministic and MonteCarlo dropout on six datasets across three biomedical classification tasks: white blood cells, amyloid plaques, and colorectal histopathology. SNGP has comparable in-distribution performance while significantly improving uncertainty estimation and OOD detection. Thus, SNGP or related models offer a useful framework for uncertainty-aware classification in digital pathology, supporting safe deployment and building trust with pathologists.

  </details>


