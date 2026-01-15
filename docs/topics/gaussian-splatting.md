# Gaussian Splatting

_Updated: 2026-01-15 07:17 UTC_

Total papers shown: **17**


---

- **Variable Basis Mapping for Real-Time Volumetric Visualization**  
  Qibiao Li, Yuxuan Wang, Youcheng Cai, Huangsheng Du, Ligang Liu  
  _2026-01-14_ · https://arxiv.org/abs/2601.09417v1  
  <details><summary>Abstract</summary>

  Real-time visualization of large-scale volumetric data remains challenging, as direct volume rendering and voxel-based methods suffer from prohibitively high computational cost. We propose Variable Basis Mapping (VBM), a framework that transforms volumetric fields into 3D Gaussian Splatting (3DGS) representations through wavelet-domain analysis. First, we precompute a compact Wavelet-to-Gaussian Transition Bank that provides optimal Gaussian surrogates for canonical wavelet atoms across multiple scales. Second, we perform analytical Gaussian construction that maps discrete wavelet coefficients directly to 3DGS parameters using a closed-form, mathematically principled rule. Finally, a lightweight image-space fine-tuning stage further refines the representation to improve rendering fidelity. Experiments on diverse datasets demonstrate that VBM significantly accelerates convergence and enhances rendering quality, enabling real-time volumetric visualization.

  </details>



- **TIDI-GS: Floater Suppression in 3D Gaussian Splatting for Enhanced Indoor Scene Fidelity**  
  Sooyeun Yang, Cheyul Im, Jee Won Lee, Jongseong Brad Choi  
  _2026-01-14_ · https://arxiv.org/abs/2601.09291v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) is a technique to create high-quality, real-time 3D scenes from images. This method often produces visual artifacts known as floaters--nearly transparent, disconnected elements that drift in space away from the actual surface. This geometric inaccuracy undermines the reliability of these models for practical applications, which is critical. To address this issue, we introduce TIDI-GS, a new training framework designed to eliminate these floaters. A key benefit of our approach is that it functions as a lightweight plugin for the standard 3DGS pipeline, requiring no major architectural changes and adding minimal overhead to the training process. The core of our method is a floater pruning algorithm--TIDI--that identifies and removes floaters based on several criteria: their consistency across multiple viewpoints, their spatial relationship to other elements, and an importance score learned during training. The framework includes a mechanism to preserve fine details, ensuring that important high-frequency elements are not mistakenly removed. This targeted cleanup is supported by a monocular depth-based loss function that helps improve the overall geometric structure of the scene. Our experiments demonstrate that TIDI-GS improves both the perceptual quality and geometric integrity of reconstructions, transforming them into robust digital assets, suitable for high-fidelity applications.

  </details>



- **GaussianFluent: Gaussian Simulation for Dynamic Scenes with Mixed Materials**  
  Bei Huang, Yixin Chen, Ruijie Lu, Gang Zeng, Hongbin Zha, Yuru Pei, Siyuan Huang  
  _2026-01-14_ · https://arxiv.org/abs/2601.09265v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has emerged as a prominent 3D representation for high-fidelity and real-time rendering. Prior work has coupled physics simulation with Gaussians, but predominantly targets soft, deformable materials, leaving brittle fracture largely unresolved. This stems from two key obstacles: the lack of volumetric interiors with coherent textures in GS representation, and the absence of fracture-aware simulation methods for Gaussians. To address these challenges, we introduce GaussianFluent, a unified framework for realistic simulation and rendering of dynamic object states. First, it synthesizes photorealistic interiors by densifying internal Gaussians guided by generative models. Second, it integrates an optimized Continuum Damage Material Point Method (CD-MPM) to enable brittle fracture simulation at remarkably high speed. Our approach handles complex scenarios including mixed-material objects and multi-stage fracture propagation, achieving results infeasible with previous methods. Experiments clearly demonstrate GaussianFluent's capability for photo-realistic, real-time rendering with structurally consistent interiors, highlighting its potential for downstream application, such as VR and Robotics.

  </details>



- **3DGS-Drag: Dragging Gaussians for Intuitive Point-Based 3D Editing**  
  Jiahua Dong, Yu-Xiong Wang  
  _2026-01-12_ · https://arxiv.org/abs/2601.07963v1  
  <details><summary>Abstract</summary>

  The transformative potential of 3D content creation has been progressively unlocked through advancements in generative models. Recently, intuitive drag editing with geometric changes has attracted significant attention in 2D editing yet remains challenging for 3D scenes. In this paper, we introduce 3DGS-Drag -- a point-based 3D editing framework that provides efficient, intuitive drag manipulation of real 3D scenes. Our approach bridges the gap between deformation-based and 2D-editing-based 3D editing methods, addressing their limitations to geometry-related content editing. We leverage two key innovations: deformation guidance utilizing 3D Gaussian Splatting for consistent geometric modifications and diffusion guidance for content correction and visual quality enhancement. A progressive editing strategy further supports aggressive 3D drag edits. Our method enables a wide range of edits, including motion change, shape adjustment, inpainting, and content extension. Experimental results demonstrate the effectiveness of 3DGS-Drag in various scenes, achieving state-of-the-art performance in geometry-related 3D content editing. Notably, the editing is efficient, taking 10 to 20 minutes on a single RTX 4090 GPU.

  </details>



- **ViewMorpher3D: A 3D-aware Diffusion Framework for Multi-Camera Novel View Synthesis in Autonomous Driving**  
  Farhad G. Zanjani, Hong Cai, Amirhossein Habibian  
  _2026-01-12_ · https://arxiv.org/abs/2601.07540v2  
  <details><summary>Abstract</summary>

  Autonomous driving systems rely heavily on multi-view images to ensure accurate perception and robust decision-making. To effectively develop and evaluate perception stacks and planning algorithms, realistic closed-loop simulators are indispensable. While 3D reconstruction techniques such as Gaussian Splatting offer promising avenues for simulator construction, the rendered novel views often exhibit artifacts, particularly in extrapolated perspectives or when available observations are sparse. We introduce ViewMorpher3D, a multi-view image enhancement framework based on image diffusion models, designed to elevate photorealism and multi-view coherence in driving scenes. Unlike single-view approaches, ViewMorpher3D jointly processes a set of rendered views conditioned on camera poses, 3D geometric priors, and temporally adjacent or spatially overlapping reference views. This enables the model to infer missing details, suppress rendering artifacts, and enforce cross-view consistency. Our framework accommodates variable numbers of cameras and flexible reference/target view configurations, making it adaptable to diverse sensor setups. Experiments on real-world driving datasets demonstrate substantial improvements in image quality metrics, effectively reducing artifacts while preserving geometric fidelity.

  </details>



- **Mon3tr: Monocular 3D Telepresence with Pre-built Gaussian Avatars as Amortization**  
  Fangyu Lin, Yingdong Hu, Zhening Liu, Yufan Zhuang, Zehong Lin, Jun Zhang  
  _2026-01-12_ · https://arxiv.org/abs/2601.07518v1  
  <details><summary>Abstract</summary>

  Immersive telepresence aims to transform human interaction in AR/VR applications by enabling lifelike full-body holographic representations for enhanced remote collaboration. However, existing systems rely on hardware-intensive multi-camera setups and demand high bandwidth for volumetric streaming, limiting their real-time performance on mobile devices. To overcome these challenges, we propose Mon3tr, a novel Monocular 3D telepresence framework that integrates 3D Gaussian splatting (3DGS) based parametric human modeling into telepresence for the first time. Mon3tr adopts an amortized computation strategy, dividing the process into a one-time offline multi-view reconstruction phase to build a user-specific avatar and a monocular online inference phase during live telepresence sessions. A single monocular RGB camera is used to capture body motions and facial expressions in real time to drive the 3DGS-based parametric human model, significantly reducing system complexity and cost. The extracted motion and appearance features are transmitted at < 0.2 Mbps over WebRTC's data channel, allowing robust adaptation to network fluctuations. On the receiver side, e.g., Meta Quest 3, we develop a lightweight 3DGS attribute deformation network to dynamically generate corrective 3DGS attribute adjustments on the pre-built avatar, synthesizing photorealistic motion and appearance at ~ 60 FPS. Extensive experiments demonstrate the state-of-the-art performance of our method, achieving a PSNR of > 28 dB for novel poses, an end-to-end latency of ~ 80 ms, and > 1000x bandwidth reduction compared to point-cloud streaming, while supporting real-time operation from monocular inputs across diverse scenarios. Our demos can be found at https://mon3tr3d.github.io.

  </details>



- **R3-RECON: Radiance-Field-Free Active Reconstruction via Renderability**  
  Xiaofeng Jin, Matteo Frosi, Yiran Guo, Matteo Matteucci  
  _2026-01-12_ · https://arxiv.org/abs/2601.07484v1  
  <details><summary>Abstract</summary>

  In active reconstruction, an embodied agent must decide where to look next to efficiently acquire views that support high-quality novel-view rendering. Recent work on active view planning for neural rendering largely derives next-best-view (NBV) criteria by backpropagating through radiance fields or estimating information entropy over 3D Gaussian primitives. While effective, these strategies tightly couple view selection to heavy, representation-specific mechanisms and fail to account for the computational and resource constraints required for lightweight online deployment. In this paper, we revisit active reconstruction from a renderability-centric perspective. We propose $\mathbb{R}^{3}$-RECON, a radiance-fields-free active reconstruction framework that induces an implicit, pose-conditioned renderability field over SE(3) from a lightweight voxel map. Our formulation aggregates per-voxel online observation statistics into a unified scalar renderability score that is cheap to update and can be queried in closed form at arbitrary candidate viewpoints in milliseconds, without requiring gradients or radiance-field training. This renderability field is strongly correlated with image-space reconstruction error, naturally guiding NBV selection. We further introduce a panoramic extension that estimates omnidirectional (360$^\circ$) view utility to accelerate candidate evaluation. In the standard indoor Replica dataset, $\mathbb{R}^{3}$-RECON achieves more uniform novel-view quality and higher 3D Gaussian splatting (3DGS) reconstruction accuracy than recent active GS baselines with matched view and time budgets.

  </details>



- **SARA: Scene-Aware Reconstruction Accelerator**  
  Jee Won Lee, Hansol Lim, Minhyeok Im, Dohyeon Lee, Jongseong Brad Choi  
  _2026-01-11_ · https://arxiv.org/abs/2601.06831v1  
  <details><summary>Abstract</summary>

  We present SARA (Scene-Aware Reconstruction Accelerator), a geometry-driven pair selection module for Structure-from-Motion (SfM). Unlike conventional pipelines that select pairs based on visual similarity alone, SARA introduces geometry-first pair selection by scoring reconstruction informativeness - the product of overlap and parallax - before expensive matching. A lightweight pre-matching stage uses mutual nearest neighbors and RANSAC to estimate these cues, then constructs an Information-Weighted Spanning Tree (IWST) augmented with targeted edges for loop closure, long-baseline anchors, and weak-view reinforcement. Compared to exhaustive matching, SARA reduces rotation errors by 46.5+-5.5% and translation errors by 12.5+-6.5% across modern learned detectors, while achieving at most 50x speedup through 98% pair reduction (from 30,848 to 580 pairs). This reduces matching complexity from quadratic to quasi-linear, maintaining within +-3% of baseline reconstruction metrics for 3D Gaussian Splatting and SVRaster.

  </details>



- **SRFlow: A Dataset and Regularization Model for High-Resolution Facial Optical Flow via Splatting Rasterization**  
  JiaLin Zhang, Dong Li  
  _2026-01-10_ · https://arxiv.org/abs/2601.06479v1  
  <details><summary>Abstract</summary>

  Facial optical flow supports a wide range of tasks in facial motion analysis. However, the lack of high-resolution facial optical flow datasets has hindered progress in this area. In this paper, we introduce Splatting Rasterization Flow (SRFlow), a high-resolution facial optical flow dataset, and Splatting Rasterization Guided FlowNet (SRFlowNet), a facial optical flow model with tailored regularization losses. These losses constrain flow predictions using masks and gradients computed via difference or Sobel operator. This effectively suppresses high-frequency noise and large-scale errors in texture-less or repetitive-pattern regions, enabling SRFlowNet to be the first model explicitly capable of capturing high-resolution skin motion guided by Gaussian splatting rasterization. Experiments show that training with the SRFlow dataset improves facial optical flow estimation across various optical flow models, reducing end-point error (EPE) by up to 42% (from 0.5081 to 0.2953). Furthermore, when coupled with the SRFlow dataset, SRFlowNet achieves up to a 48% improvement in F1-score (from 0.4733 to 0.6947) on a composite of three micro-expression datasets. These results demonstrate the value of advancing both facial optical flow estimation and micro-expression recognition.

  </details>



- **NAS-GS: Noise-Aware Sonar Gaussian Splatting**  
  Shida Xu, Jingqi Jiang, Jonatan Scharff Willners, Sen Wang  
  _2026-01-09_ · https://arxiv.org/abs/2601.06285v1  
  <details><summary>Abstract</summary>

  Underwater sonar imaging plays a crucial role in various applications, including autonomous navigation in murky water, marine archaeology, and environmental monitoring. However, the unique characteristics of sonar images, such as complex noise patterns and the lack of elevation information, pose significant challenges for 3D reconstruction and novel view synthesis. In this paper, we present NAS-GS, a novel Noise-Aware Sonar Gaussian Splatting framework specifically designed to address these challenges. Our approach introduces a Two-Ways Splatting technique that accurately models the dual directions for intensity accumulation and transmittance calculation inherent in sonar imaging, significantly improving rendering speed without sacrificing quality. Moreover, we propose a Gaussian Mixture Model (GMM) based noise model that captures complex sonar noise patterns, including side-lobes, speckle, and multi-path noise. This model enhances the realism of synthesized images while preventing 3D Gaussian overfitting to noise, thereby improving reconstruction accuracy. We demonstrate state-of-the-art performance on both simulated and real-world large-scale offshore sonar scenarios, achieving superior results in novel view synthesis and 3D reconstruction.

  </details>



- **GS-DMSR: Dynamic Sensitive Multi-scale Manifold Enhancement for Accelerated High-Quality 3D Gaussian Splatting**  
  Nengbo Lu, Minghua Pan, Shaohua Sun, Yizhou Liang  
  _2026-01-09_ · https://arxiv.org/abs/2601.05584v1  
  <details><summary>Abstract</summary>

  In the field of 3D dynamic scene reconstruction, how to balance model convergence rate and rendering quality has long been a critical challenge that urgently needs to be addressed, particularly in high-precision modeling of scenes with complex dynamic motions. To tackle this issue, this study proposes the GS-DMSR method. By quantitatively analyzing the dynamic evolution process of Gaussian attributes, this mechanism achieves adaptive gradient focusing, enabling it to dynamically identify significant differences in the motion states of Gaussian models. It then applies differentiated optimization strategies to Gaussian models with varying degrees of significance, thereby significantly improving the model convergence rate. Additionally, this research integrates a multi-scale manifold enhancement module, which leverages the collaborative optimization of an implicit nonlinear decoder and an explicit deformation field to enhance the modeling efficiency for complex deformation scenes. Experimental results demonstrate that this method achieves a frame rate of up to 96 FPS on synthetic datasets, while effectively reducing both storage overhead and training time.Our code and data are available at https://anonymous.4open.science/r/GS-DMSR-2212.

  </details>



- **GaussianSwap: Animatable Video Face Swapping with 3D Gaussian Splatting**  
  Xuan Cheng, Jiahao Rao, Chengyang Li, Wenhao Wang, Weilin Chen, Lvqing Yang  
  _2026-01-09_ · https://arxiv.org/abs/2601.05511v1  
  <details><summary>Abstract</summary>

  We introduce GaussianSwap, a novel video face swapping framework that constructs a 3D Gaussian Splatting based face avatar from a target video while transferring identity from a source image to the avatar. Conventional video swapping frameworks are limited to generating facial representations in pixel-based formats. The resulting swapped faces exist merely as a set of unstructured pixels without any capacity for animation or interactive manipulation. Our work introduces a paradigm shift from conventional pixel-based video generation to the creation of high-fidelity avatar with swapped faces. The framework first preprocesses target video to extract FLAME parameters, camera poses and segmentation masks, and then rigs 3D Gaussian splats to the FLAME model across frames, enabling dynamic facial control. To ensure identity preserving, we propose an compound identity embedding constructed from three state-of-the-art face recognition models for avatar finetuning. Finally, we render the face-swapped avatar on the background frames to obtain the face-swapped video. Experimental results demonstrate that GaussianSwap achieves superior identity preservation, visual clarity and temporal consistency, while enabling previously unattainable interactive applications.

  </details>



- **Sketch&Patch++: Efficient Structure-Aware 3D Gaussian Representation**  
  Yuang Shi, Géraldine Morin, Simone Gasparini, Wei Tsang Ooi  
  _2026-01-08_ · https://arxiv.org/abs/2601.05394v2  
  <details><summary>Abstract</summary>

  We observe that Gaussians exhibit distinct roles and characteristics analogous to traditional artistic techniques -- like how artists first sketch outlines before filling in broader areas with color, some Gaussians capture high-frequency features such as edges and contours, while others represent broader, smoother regions analogous to brush strokes that add volume and depth. Based on this observation, we propose a hybrid representation that categorizes Gaussians into (i) Sketch Gaussians, which represent high-frequency, boundary-defining features, and (ii) Patch Gaussians, which cover low-frequency, smooth regions. This semantic separation naturally enables layered progressive streaming, where the compact Sketch Gaussians establish the structural skeleton before Patch Gaussians incrementally refine volumetric detail. In this work, we extend our previous method to arbitrary 3D scenes by proposing a novel hierarchical adaptive categorization framework that operates directly on the 3DGS representation. Our approach employs multi-criteria density-based clustering, combined with adaptive quality-driven refinement. This method eliminates dependency on external 3D line primitives while ensuring optimal parametric encoding effectiveness. Our comprehensive evaluation across diverse scenes, including both man-made and natural environments, demonstrates that our method achieves up to 1.74 dB improvement in PSNR, 6.7% in SSIM, and 41.4% in LPIPS at equivalent model sizes compared to uniform pruning baselines. For indoor scenes, our method can maintain visual quality with only 0.5\% of the original model size. This structure-aware representation enables efficient storage, adaptive streaming, and rendering of high-fidelity 3D content across bandwidth-constrained networks and resource-limited devices.

  </details>



- **Akasha 2: Hamiltonian State Space Duality and Visual-Language Joint Embedding Predictive Architectur**  
  Yani Meziani  
  _2026-01-08_ · https://arxiv.org/abs/2601.06212v1  
  <details><summary>Abstract</summary>

  We present Akasha 2, a state-of-the-art multimodal architecture that integrates Hamiltonian State Space Duality (H-SSD) with Visual-Language Joint Embedding Predictive Architecture (VL-JEPA). The system leverages the Mamba-3 Selective State Space Model (SSM) augmented by a Sparse Mixture of Hamiltonian Experts (SMoE-HE) that enforces latent physical conservation laws through symplectic integration. For visual synthesis, we introduce Hamiltonian Flow Matching (HFM) and persistent 3D Gaussian Splatting (3DGS), enabling ultra-low latency (<50ms) on mobile hardware. This work establishes a new paradigm in latent world models, achieving unprecedented spatiotemporal coherence through a holographic memory architecture. Our approach demonstrates that incorporating physics-inspired inductive biases into neural architectures yields significant improvements: state-of-the-art video prediction (FVD: 287), 4x faster visual synthesis than diffusion models, and 3-18x inference speedup over transformer baselines while maintaining energy conservation over extended horizons.

  </details>



- **VerseCrafter: Dynamic Realistic Video World Model with 4D Geometric Control**  
  Sixiao Zheng, Minghao Yin, Wenbo Hu, Xiaoyu Li, Ying Shan, Yanwei Fu  
  _2026-01-08_ · https://arxiv.org/abs/2601.05138v1  
  <details><summary>Abstract</summary>

  Video world models aim to simulate dynamic, real-world environments, yet existing methods struggle to provide unified and precise control over camera and multi-object motion, as videos inherently operate dynamics in the projected 2D image plane. To bridge this gap, we introduce VerseCrafter, a 4D-aware video world model that enables explicit and coherent control over both camera and object dynamics within a unified 4D geometric world state. Our approach is centered on a novel 4D Geometric Control representation, which encodes the world state through a static background point cloud and per-object 3D Gaussian trajectories. This representation captures not only an object's path but also its probabilistic 3D occupancy over time, offering a flexible, category-agnostic alternative to rigid bounding boxes or parametric models. These 4D controls are rendered into conditioning signals for a pretrained video diffusion model, enabling the generation of high-fidelity, view-consistent videos that precisely adhere to the specified dynamics. Unfortunately, another major challenge lies in the scarcity of large-scale training data with explicit 4D annotations. We address this by developing an automatic data engine that extracts the required 4D controls from in-the-wild videos, allowing us to train our model on a massive and diverse dataset.

  </details>



- **OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction**  
  Minseong Kweon, Jinsun Park  
  _2026-01-08_ · https://arxiv.org/abs/2601.04984v1  
  <details><summary>Abstract</summary>

  We introduce OceanSplat, a novel 3D Gaussian Splatting-based approach for accurately representing 3D geometry in underwater scenes. To overcome multi-view inconsistencies caused by underwater optical degradation, our method enforces trinocular view consistency by rendering horizontally and vertically translated camera views relative to each input view and aligning them via inverse warping. Furthermore, these translated camera views are used to derive a synthetic epipolar depth prior through triangulation, which serves as a self-supervised depth regularizer. These geometric constraints facilitate the spatial optimization of 3D Gaussians and preserve scene structure in underwater environments. We also propose a depth-aware alpha adjustment that modulates the opacity of 3D Gaussians during early training based on their $z$-component and viewing direction, deterring the formation of medium-induced primitives. With our contributions, 3D Gaussians are disentangled from the scattering medium, enabling robust representation of object geometry and significantly reducing floating artifacts in reconstructed underwater scenes. Experiments on real-world underwater and simulated scenes demonstrate that OceanSplat substantially outperforms existing methods for both scene reconstruction and restoration in scattering media.

  </details>



- **ProFuse: Efficient Cross-View Context Fusion for Open-Vocabulary 3D Gaussian Splatting**  
  Yen-Jen Chiou, Wei-Tse Cheng, Yuan-Fu Yang  
  _2026-01-08_ · https://arxiv.org/abs/2601.04754v1  
  <details><summary>Abstract</summary>

  We present ProFuse, an efficient context-aware framework for open-vocabulary 3D scene understanding with 3D Gaussian Splatting (3DGS). The pipeline enhances cross-view consistency and intra-mask cohesion within a direct registration setup, adding minimal overhead and requiring no render-supervised fine-tuning. Instead of relying on a pretrained 3DGS scene, we introduce a dense correspondence-guided pre-registration phase that initializes Gaussians with accurate geometry while jointly constructing 3D Context Proposals via cross-view clustering. Each proposal carries a global feature obtained through weighted aggregation of member embeddings, and this feature is fused onto Gaussians during direct registration to maintain per-primitive language coherence across views. With associations established in advance, semantic fusion requires no additional optimization beyond standard reconstruction, and the model retains geometric refinement without densification. ProFuse achieves strong open-vocabulary 3DGS understanding while completing semantic attachment in about five minutes per scene, which is two times faster than SOTA.

  </details>


