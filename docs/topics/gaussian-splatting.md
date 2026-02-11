# Gaussian Splatting & 3DGS

_Updated: 2026-02-11 07:17 UTC_

Total papers shown: **11**


---

- **Faster-GS: Analyzing and Improving Gaussian Splatting Optimization**  
  Florian Hahlbohm, Linus Franke, Martin Eisemann, Marcus Magnor  
  _2026-02-10_ · https://arxiv.org/abs/2602.09999v1  
  <details><summary>Abstract</summary>

  Recent advances in 3D Gaussian Splatting (3DGS) have focused on accelerating optimization while preserving reconstruction quality. However, many proposed methods entangle implementation-level improvements with fundamental algorithmic modifications or trade performance for fidelity, leading to a fragmented research landscape that complicates fair comparison. In this work, we consolidate and evaluate the most effective and broadly applicable strategies from prior 3DGS research and augment them with several novel optimizations. We further investigate underexplored aspects of the framework, including numerical stability, Gaussian truncation, and gradient approximation. The resulting system, Faster-GS, provides a rigorously optimized algorithm that we evaluate across a comprehensive suite of benchmarks. Our experiments demonstrate that Faster-GS achieves up to 5$\times$ faster training while maintaining visual quality, establishing a new cost-effective and resource efficient baseline for 3DGS optimization. Furthermore, we demonstrate that optimizations can be applied to 4D Gaussian reconstruction, leading to efficient non-rigid scene optimization.

  </details>



- **CompSplat: Compression-aware 3D Gaussian Splatting for Real-world Video**  
  Hojun Song, Heejung Choi, Aro Kim, Chae-yeong Song, Gahyeon Kim, Soo Ye Kim, Jaehyup Lee, Sang-hyo Park  
  _2026-02-10_ · https://arxiv.org/abs/2602.09816v1  
  <details><summary>Abstract</summary>

  High-quality novel view synthesis (NVS) from real-world videos is crucial for applications such as cultural heritage preservation, digital twins, and immersive media. However, real-world videos typically contain long sequences with irregular camera trajectories and unknown poses, leading to pose drift, feature misalignment, and geometric distortion during reconstruction. Moreover, lossy compression amplifies these issues by introducing inconsistencies that gradually degrade geometry and rendering quality. While recent studies have addressed either long-sequence NVS or unposed reconstruction, compression-aware approaches still focus on specific artifacts or limited scenarios, leaving diverse compression patterns in long videos insufficiently explored. In this paper, we propose CompSplat, a compression-aware training framework that explicitly models frame-wise compression characteristics to mitigate inter-frame inconsistency and accumulated geometric errors. CompSplat incorporates compression-aware frame weighting and an adaptive pruning strategy to enhance robustness and geometric consistency, particularly under heavy compression. Extensive experiments on challenging benchmarks, including Tanks and Temples, Free, and Hike, demonstrate that CompSplat achieves state-of-the-art rendering quality and pose accuracy, significantly surpassing most recent state-of-the-art NVS approaches under severe compression conditions.

  </details>



- **Toward Fine-Grained Facial Control in 3D Talking Head Generation**  
  Shaoyang Xie, Xiaofeng Cong, Baosheng Yu, Zhipeng Gui, Jie Gui, Yuan Yan Tang, James Tin-Yau Kwok  
  _2026-02-10_ · https://arxiv.org/abs/2602.09736v1  
  <details><summary>Abstract</summary>

  Audio-driven talking head generation is a core component of digital avatars, and 3D Gaussian Splatting has shown strong performance in real-time rendering of high-fidelity talking heads. However, achieving precise control over fine-grained facial movements remains a significant challenge, particularly due to lip-synchronization inaccuracies and facial jitter, both of which can contribute to the uncanny valley effect. To address these challenges, we propose Fine-Grained 3D Gaussian Splatting (FG-3DGS), a novel framework that enables temporally consistent and high-fidelity talking head generation. Our method introduces a frequency-aware disentanglement strategy to explicitly model facial regions based on their motion characteristics. Low-frequency regions, such as the cheeks, nose, and forehead, are jointly modeled using a standard MLP, while high-frequency regions, including the eyes and mouth, are captured separately using a dedicated network guided by facial area masks. The predicted motion dynamics, represented as Gaussian deltas, are applied to the static Gaussians to generate the final head frames, which are rendered via a rasterizer using frame-specific camera parameters. Additionally, a high-frequency-refined post-rendering alignment mechanism, learned from large-scale audio-video pairs by a pretrained model, is incorporated to enhance per-frame generation and achieve more accurate lip synchronization. Extensive experiments on widely used datasets for talking head generation demonstrate that our method outperforms recent state-of-the-art approaches in producing high-fidelity, lip-synced talking head videos.

  </details>



- **Stability and Concentration in Nonlinear Inverse Problems with Block-Structured Parameters: Lipschitz Geometry, Identifiability, and an Application to Gaussian Splatting**  
  Joe-Mei Feng, Hsin-Hsiung Kao  
  _2026-02-10_ · https://arxiv.org/abs/2602.09415v1  
  <details><summary>Abstract</summary>

  We develop an operator-theoretic framework for stability and statistical concentration in nonlinear inverse problems with block-structured parameters. Under a unified set of assumptions combining blockwise Lipschitz geometry, local identifiability, and sub-Gaussian noise, we establish deterministic stability inequalities, global Lipschitz bounds for least-squares misfit functionals, and nonasymptotic concentration estimates. These results yield high-probability parameter error bounds that are intrinsic to the forward operator and independent of any specific reconstruction algorithm. As a concrete instantiation, we verify that the Gaussian Splatting rendering operator satisfies the proposed assumptions and derive explicit constants governing its Lipschitz continuity and resolution-dependent observability. This leads to a fundamental stability--resolution tradeoff, showing that estimation error is inherently constrained by the ratio between image resolution and model complexity. Overall, the analysis characterizes operator-level limits for a broad class of high-dimensional nonlinear inverse problems arising in modern imaging and differentiable rendering.

  </details>



- **Grow with the Flow: 4D Reconstruction of Growing Plants with Gaussian Flow Fields**  
  Weihan Luo, Lily Goli, Sherwin Bahmani, Felix Taubner, Andrea Tagliasacchi, David B. Lindell  
  _2026-02-09_ · https://arxiv.org/abs/2602.08958v2  
  <details><summary>Abstract</summary>

  Modeling the time-varying 3D appearance of plants during their growth poses unique challenges: unlike many dynamic scenes, plants generate new geometry over time as they expand, branch, and differentiate. Recent motion modeling techniques are ill-suited to this problem setting. For example, deformation fields cannot introduce new geometry, and 4D Gaussian splatting constrains motion to a linear trajectory in space and time and cannot track the same set of Gaussians over time. Here, we introduce a 3D Gaussian flow field representation that models plant growth as a time-varying derivative over Gaussian parameters -- position, scale, orientation, color, and opacity -- enabling nonlinear and continuous-time growth dynamics. To initialize a sufficient set of Gaussian primitives, we reconstruct the mature plant and learn a process of reverse growth, effectively simulating the plant's developmental history in reverse. Our approach achieves superior image quality and geometric accuracy compared to prior methods on multi-view timelapse datasets of plant growth, providing a new approach for appearance modeling of growing 3D structures.

  </details>



- **Analysis of Converged 3D Gaussian Splatting Solutions: Density Effects and Prediction Limit**  
  Zhendong Wang, Cihan Ruan, Jingchuan Xiao, Chuqing Shi, Wei Jiang, Wei Wang, Wenjie Liu, Nam Ling  
  _2026-02-09_ · https://arxiv.org/abs/2602.08909v1  
  <details><summary>Abstract</summary>

  We investigate what structure emerges in 3D Gaussian Splatting (3DGS) solutions from standard multi-view optimization. We term these Rendering-Optimal References (RORs) and analyze their statistical properties, revealing stable patterns: mixture-structured scales and bimodal radiance across diverse scenes. To understand what determines these parameters, we apply learnability probes by training predictors to reconstruct RORs from point clouds without rendering supervision. Our analysis uncovers fundamental density-stratification. Dense regions exhibit geometry-correlated parameters amenable to render-free prediction, while sparse regions show systematic failure across architectures. We formalize this through variance decomposition, demonstrating that visibility heterogeneity creates covariance-dominated coupling between geometric and appearance parameters in sparse regions. This reveals the dual character of RORs: geometric primitives where point clouds suffice, and view synthesis primitives where multi-view constraints are essential. We provide density-aware strategies that improve training robustness and discuss architectural implications for systems that adaptively balance feed-forward prediction and rendering-based refinement.

  </details>



- **Rotated Lights for Consistent and Efficient 2D Gaussians Inverse Rendering**  
  Geng Lin, Matthias Zwicker  
  _2026-02-09_ · https://arxiv.org/abs/2602.08724v1  
  <details><summary>Abstract</summary>

  Inverse rendering aims to decompose a scene into its geometry, material properties and light conditions under a certain rendering model. It has wide applications like view synthesis, relighting, and scene editing. In recent years, inverse rendering methods have been inspired by view synthesis approaches like neural radiance fields and Gaussian splatting, which are capable of efficiently decomposing a scene into its geometry and radiance. They then further estimate the material and lighting that lead to the observed scene radiance. However, the latter step is highly ambiguous and prior works suffer from inaccurate color and baked shadows in their albedo estimation albeit their regularization. To this end, we propose RotLight, a simple capturing setup, to address the ambiguity. Compared to a usual capture, RotLight only requires the object to be rotated several times during the process. We show that as few as two rotations is effective in reducing artifacts. To further improve 2DGS-based inverse rendering, we additionally introduce a proxy mesh that not only allows accurate incident light tracing, but also enables a residual constraint and improves global illumination handling. We demonstrate with both synthetic and real world datasets that our method achieves superior albedo estimation while keeping efficient computation.

  </details>



- **FLAG-4D: Flow-Guided Local-Global Dual-Deformation Model for 4D Reconstruction**  
  Guan Yuan Tan, Ngoc Tuan Vu, Arghya Pal, Sailaja Rajanala, Raphael Phan C. -W., Mettu Srinivas, Chee-Ming Ting  
  _2026-02-09_ · https://arxiv.org/abs/2602.08558v1  
  <details><summary>Abstract</summary>

  We introduce FLAG-4D, a novel framework for generating novel views of dynamic scenes by reconstructing how 3D Gaussian primitives evolve through space and time. Existing methods typically rely on a single Multilayer Perceptron (MLP) to model temporal deformations, and they often struggle to capture complex point motions and fine-grained dynamic details consistently over time, especially from sparse input views. Our approach, FLAG-4D, overcomes this by employing a dual-deformation network that dynamically warps a canonical set of 3D Gaussians over time into new positions and anisotropic shapes. This dual-deformation network consists of an Instantaneous Deformation Network (IDN) for modeling fine-grained, local deformations and a Global Motion Network (GMN) for capturing long-range dynamics, refined through mutual learning. To ensure these deformations are both accurate and temporally smooth, FLAG-4D incorporates dense motion features from a pretrained optical flow backbone. We fuse these motion cues from adjacent timeframes and use a deformation-guided attention mechanism to align this flow information with the current state of each evolving 3D Gaussian. Extensive experiments demonstrate that FLAG-4D achieves higher-fidelity and more temporally coherent reconstructions with finer detail preservation than state-of-the-art methods.

  </details>



- **TIBR4D: Tracing-Guided Iterative Boundary Refinement for Efficient 4D Gaussian Segmentation**  
  He Wu, Xia Yan, Yanghui Xu, Liegang Xia, Jiazhou Chen  
  _2026-02-09_ · https://arxiv.org/abs/2602.08540v1  
  <details><summary>Abstract</summary>

  Object-level segmentation in dynamic 4D Gaussian scenes remains challenging due to complex motion, occlusions, and ambiguous boundaries. In this paper, we present an efficient learning-free 4D Gaussian segmentation framework that lifts video segmentation masks to 4D spaces, whose core is a two-stage iterative boundary refinement, TIBR4D. The first stage is an Iterative Gaussian Instance Tracing (IGIT) at the temporal segment level. It progressively refines Gaussian-to-instance probabilities through iterative tracing, and extracts corresponding Gaussian point clouds that better handle occlusions and preserve completeness of object structures compared to existing one-shot threshold-based methods. The second stage is a frame-wise Gaussian Rendering Range Control (RCC) via suppressing highly uncertain Gaussians near object boundaries while retaining their core contributions for more accurate boundaries. Furthermore, a temporal segmentation merging strategy is proposed for IGIT to balance identity consistency and dynamic awareness. Longer segments enforce stronger multi-frame constraints for stable identities, while shorter segments allow identity changes to be captured promptly. Experiments on HyperNeRF and Neu3D demonstrate that our method produces accurate object Gaussian point clouds with clearer boundaries and higher efficiency compared to SOTA methods.

  </details>



- **Informative Object-centric Next Best View for Object-aware 3D Gaussian Splatting in Cluttered Scenes**  
  Seunghoon Jeong, Eunho Lee, Jeongyun Kim, Ayoung Kim  
  _2026-02-09_ · https://arxiv.org/abs/2602.08266v1  
  <details><summary>Abstract</summary>

  In cluttered scenes with inevitable occlusions and incomplete observations, selecting informative viewpoints is essential for building a reliable representation. In this context, 3D Gaussian Splatting (3DGS) offers a distinct advantage, as it can explicitly guide the selection of subsequent viewpoints and then refine the representation with new observations. However, existing approaches rely solely on geometric cues, neglect manipulation-relevant semantics, and tend to prioritize exploitation over exploration. To tackle these limitations, we introduce an instance-aware Next Best View (NBV) policy that prioritizes underexplored regions by leveraging object features. Specifically, our object-aware 3DGS distills instancelevel information into one-hot object vectors, which are used to compute confidence-weighted information gain that guides the identification of regions associated with erroneous and uncertain Gaussians. Furthermore, our method can be easily adapted to an object-centric NBV, which focuses view selection on a target object, thereby improving reconstruction robustness to object placement. Experiments demonstrate that our NBV policy reduces depth error by up to 77.14% on the synthetic dataset and 34.10% on the real-world GraspNet dataset compared to baselines. Moreover, compared to targeting the entire scene, performing NBV on a specific object yields an additional reduction of 25.60% in depth error for that object. We further validate the effectiveness of our approach through real-world robotic manipulation tasks.

  </details>



- **Scalable Adaptation of 3D Geometric Foundation Models via Weak Supervision from Internet Video**  
  Zihui Gao, Ke Liu, Donny Y. Chen, Duochao Shi, Guosheng Lin, Hao Chen, Chunhua Shen  
  _2026-02-08_ · https://arxiv.org/abs/2602.07891v1  
  <details><summary>Abstract</summary>

  Geometric foundation models show promise in 3D reconstruction, yet their progress is severely constrained by the scarcity of diverse, large-scale 3D annotations. While Internet videos offer virtually unlimited raw data, utilizing them as a scaling source for geometric learning is challenging due to the absence of ground-truth geometry and the presence of observational noise. To address this, we propose SAGE, a framework for Scalable Adaptation of GEometric foundation models from raw video streams. SAGE leverages a hierarchical mining pipeline to transform videos into training trajectories and hybrid supervision: (1) Informative training trajectory selection; (2) Sparse Geometric Anchoring via SfM point clouds for global structural guidance; and (3) Dense Differentiable Consistency via 3D Gaussian rendering for multi-view constraints. To prevent catastrophic forgetting, we introduce a regularization strategy using anchor data. Extensive experiments show that SAGE significantly enhances zero-shot generalization, reducing Chamfer Distance by 20-42% on unseen benchmarks (7Scenes, TUM-RGBD, Matterport3D) compared to state-of-the-art baselines. To our knowledge, SAGE pioneers the adaptation of geometric foundation models via Internet video, establishing a scalable paradigm for general-purpose 3D learning.

  </details>


