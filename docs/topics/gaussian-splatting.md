# Gaussian Splatting & 3DGS

_Updated: 2026-03-31 07:42 UTC_

Total papers shown: **10**


---

- **GeoHCC: Local Geometry-Aware Hierarchical Context Compression for 3D Gaussian Splatting**  
  Xuan Deng, Xiandong Meng, Hengyu Man, Qiang Zhu, Tiange Zhang, Debin Zhao, Xiaopeng Fan  
  _2026-03-30_ · https://arxiv.org/abs/2603.28431v1  
  <details><summary>Abstract</summary>

  Although 3D Gaussian Splatting (3DGS) enables high-fidelity real-time rendering, its prohibitive storage overhead severely hinders practical deployment. Recent anchor-based 3DGS compression schemes reduce redundancy through context modeling, yet overlook explicit geometric dependencies, leading to structural degradation and suboptimal rate-distortion performance. In this paper, we propose GeoHCC, a geometry-aware 3DGS compression framework that incorporates inter-anchor geometric correlations into anchor pruning and entropy coding for compact representation. We first introduce Neighborhood-Aware Anchor Pruning (NAAP), which evaluates anchor importance via weighted neighborhood feature aggregation and merges redundant anchors into salient neighbors, yielding a compact yet geometry-consistent anchor set. Building upon this optimized structure, we further develop a hierarchical entropy coding scheme, in which coarse-to-fine priors are exploited through a lightweight Geometry-Guided Convolution (GG-Conv) operator to enable spatially adaptive context modeling and rate-distortion optimization. Extensive experiments demonstrate that GeoHCC effectively resolves the structure preservation bottleneck, maintaining superior geometric integrity and rendering fidelity over state-of-the-art anchor-based approaches.

  </details>



- **ObjectMorpher: 3D-Aware Image Editing via Deformable 3DGS Models**  
  Yuhuan Xie, Aoxuan Pan, Yi-Hua Huang, Chirui Chang, Peng Dai, Xin Yu, Xiaojuan Qi  
  _2026-03-30_ · https://arxiv.org/abs/2603.28152v1  
  <details><summary>Abstract</summary>

  Achieving precise, object-level control in image editing remains challenging: 2D methods lack 3D awareness and often yield ambiguous or implausible results, while existing 3D-aware approaches rely on heavy optimization or incomplete monocular reconstructions. We present ObjectMorpher, a unified, interactive framework that converts ambiguous 2D edits into geometry-grounded operations. ObjectMorpher lifts target instances with an image-to-3D generator into editable 3D Gaussian Splatting (3DGS), enabling fast, identity-preserving manipulation. Users drag control points; a graph-based non-rigid deformation with as-rigid-as-possible (ARAP) constraints ensures physically sensible shape and pose changes. A composite diffusion module harmonizes lighting, color, and boundaries for seamless reintegration. Across diverse categories, ObjectMorpher delivers fine-grained, photorealistic edits with superior controllability and efficiency, outperforming 2D drag and 3D-aware baselines on KID, LPIPS, SIFID, and user preference.

  </details>



- **SVGS: Single-View to 3D Object Editing via Gaussian Splatting**  
  Pengcheng Xue, Yan Tian, Qiutao Song, Ziyi Wang, Linyang He, Weiping Ding, Mahmoud Hassaballah, Karen Egiazarian, Wei-Fa Yang, Leszek Rutkowski  
  _2026-03-30_ · https://arxiv.org/abs/2603.28126v1  
  <details><summary>Abstract</summary>

  Text-driven 3D scene editing has attracted considerable interest due to its convenience and user-friendliness. However, methods that rely on implicit 3D representations, such as Neural Radiance Fields (NeRF), while effective in rendering complex scenes, are hindered by slow processing speeds and limited control over specific regions of the scene. Moreover, existing approaches, including Instruct-NeRF2NeRF and GaussianEditor, which utilize multi-view editing strategies, frequently produce inconsistent results across different views when executing text instructions. This inconsistency can adversely affect the overall performance of the model, complicating the task of balancing the consistency of editing results with editing efficiency. To address these challenges, we propose a novel method termed Single-View to 3D Object Editing via Gaussian Splatting (SVGS), which is a single-view text-driven editing technique based on 3D Gaussian Splatting (3DGS). Specifically, in response to text instructions, we introduce a single-view editing strategy grounded in multi-view diffusion models, which reconstructs 3D scenes by leveraging only those views that yield consistent editing results. Additionally, we employ sparse 3D Gaussian Splatting as the 3D representation, which significantly enhances editing efficiency. We conducted a comparative analysis of SVGS against existing baseline methods across various scene settings, and the results indicate that SVGS outperforms its counterparts in both editing capability and processing speed, representing a significant advancement in 3D editing technology. For further details, please visit our project page at: https://amateurc.github.io/svgs.github.io.

  </details>



- **\textit{4DSurf}: High-Fidelity Dynamic Scene Surface Reconstruction**  
  Renjie Wu, Hongdong Li, Jose M. Alvarez, Miaomiao Liu  
  _2026-03-30_ · https://arxiv.org/abs/2603.28064v1  
  <details><summary>Abstract</summary>

  This paper addresses the problem of dynamic scene surface reconstruction using Gaussian Splatting (GS), aiming to recover temporally consistent geometry. While existing GS-based dynamic surface reconstruction methods can yield superior reconstruction, they are typically limited to either a single object or objects with only small deformations, struggling to maintain temporally consistent surface reconstruction of large deformations over time. We propose ``\textit{4DSurf}'', a novel and unified framework for generic dynamic surface reconstruction that does not require specifying the number or types of objects in the scene, can handle large surface deformations and temporal inconsistency in reconstruction. The key innovation of our framework is the introduction of Gaussian deformations induced Signed Distance Function Flow Regularization that constrains the motion of Gaussians to align with the evolving surface. To handle large deformations, we introduce an Overlapping Segment Partitioning strategy that divides the sequence into overlapping segments with small deformations and incrementally passes geometric information across segments through the shared overlapping timestep. Experiments on two challenging dynamic scene datasets, Hi4D and CMU Panoptic, demonstrate that our method outperforms state-of-the-art surface reconstruction methods by 49\% and 19\% in Chamfer distance, respectively, and achieves superior temporal consistency under sparse-view settings.

  </details>



- **Physically Inspired Gaussian Splatting for HDR Novel View Synthesis**  
  Huimin Zeng, Yue Bai, Hailing Wang, Yun Fu  
  _2026-03-30_ · https://arxiv.org/abs/2603.28020v1  
  <details><summary>Abstract</summary>

  High dynamic range novel view synthesis (HDR-NVS) reconstructs scenes with dynamic details by fusing multi-exposure low dynamic range (LDR) views, yet it struggles to capture ambient illumination-dependent appearance. Implicitly supervising HDR content by constraining tone-mapped results fails in correcting abnormal HDR values, and results in limited gradients for Gaussians in under/over-exposed regions. To this end, we introduce PhysHDR-GS, a physically inspired HDR-NVS framework that models scene appearance via intrinsic reflectance and adjustable ambient illumination. PhysHDR-GS employs a complementary image-exposure (IE) branch and Gaussian-illumination (GI) branch to faithfully reproduce standard camera observations and capture illumination-dependent appearance changes, respectively. During training, the proposed cross-branch HDR consistency loss provides explicit supervision for HDR content, while an illumination-guided gradient scaling strategy mitigates exposure-biased gradient starvation and reduces under-densified representations. Experimental results across realistic and synthetic datasets demonstrate our superiority in reconstructing HDR details (e.g., a PSNR gain of 2.04 dB over HDR-GS), while maintaining real-time rendering speed (up to 76 FPS). Code and models are available at https://huimin-zeng.github.io/PhysHDR-GS/.

  </details>



- **DipGuava: Disentangling Personalized Gaussian Features for 3D Head Avatars from Monocular Video**  
  Jeonghaeng Lee, Seok Keun Choi, Zhixuan Li, Weisi Lin, Sanghoon Lee  
  _2026-03-30_ · https://arxiv.org/abs/2603.28003v1  
  <details><summary>Abstract</summary>

  While recent 3D head avatar creation methods attempt to animate facial dynamics, they often fail to capture personalized details, limiting realism and expressiveness. To fill this gap, we present DipGuava (Disentangled and Personalized Gaussian UV Avatar), a novel 3D Gaussian head avatar creation method that successfully generates avatars with personalized attributes from monocular video. DipGuava is the first method to explicitly disentangle facial appearance into two complementary components, trained in a structured two-stage pipeline that significantly reduces learning ambiguity and enhances reconstruction fidelity. In the first stage, we learn a stable geometry-driven base appearance that captures global facial structure and coarse expression-dependent variations. In the second stage, the personalized residual details not captured in the first stage are predicted, including high-frequency components and nonlinearly varying features such as wrinkles and subtle skin deformations. These components are fused via dynamic appearance fusion that integrates residual details after deformation, ensuring spatial and semantic alignment. This disentangled design enables DipGuava to generate photorealistic, identity-preserving avatars, consistently outperforming prior methods in both visual quality and quantitativeperformance, as demonstrated in extensive experiments.

  </details>



- **GS3LAM: Gaussian Semantic Splatting SLAM**  
  Linfei Li, Lin Zhang, Zhong Wang, Ying Shen  
  _2026-03-29_ · https://arxiv.org/abs/2603.27781v1  
  <details><summary>Abstract</summary>

  Recently, the multi-modal fusion of RGB, depth, and semantics has shown great potential in dense Simultaneous Localization and Mapping (SLAM). However, a prerequisite for generating consistent semantic maps is the availability of dense, efficient, and scalable scene representations. Existing semantic SLAM systems based on explicit representations are often limited by resolution and an inability to predict unknown areas. Conversely, implicit representations typically rely on time-consuming ray tracing, failing to meet real-time requirements. Fortunately, 3D Gaussian Splatting (3DGS) has emerged as a promising representation that combines the efficiency of point-based methods with the continuity of geometric structures. To this end, we propose GS3LAM, a Gaussian Semantic Splatting SLAM framework that processes multimodal data to render consistent, dense semantic maps in real-time. GS3LAM models the scene as a Semantic Gaussian Field (SG-Field) and jointly optimizes camera poses and the field via multimodal error constraints. Furthermore, a Depth-adaptive Scale Regularization (DSR) scheme is introduced to resolve misalignments between scale-invariant Gaussians and geometric surfaces. To mitigate catastrophic forgetting, we propose a Random Sampling-based Keyframe Mapping (RSKM) strategy, which demonstrates superior performance over common local covisibility optimization methods. Extensive experiments on benchmark datasets show that GS3LAM achieves increased tracking robustness, superior rendering quality, and enhanced semantic precision compared to state-of-the-art methods. Source code is available at https://github.com/lif314/GS3LAM.

  </details>



- **SGS-Intrinsic: Semantic-Invariant Gaussian Splatting for Sparse-View Indoor Inverse Rendering**  
  Jiahao Niu, Rongjia Zheng, Wenju Xu, WeiShi Zheng, Qing Zhang  
  _2026-03-29_ · https://arxiv.org/abs/2603.27516v1  
  <details><summary>Abstract</summary>

  We present SGS-Intrinsic, an indoor inverse rendering framework that works well for sparse-view images. Unlike existing 3D Gaussian Splatting (3DGS) based methods that focus on object-centric reconstruction and fail to work under sparse view settings, our method allows to achieve high-quality geometry reconstruction and accurate disentanglement of material and illumination. The core idea is to construct a dense and geometry-consistent Gaussian semantic field guided by semantic and geometric priors, providing a reliable foundation for subsequent inverse rendering. Building upon this, we perform material-illumination disentanglement by combining a hybrid illumination model and material prior to effectively capture illumination-material interactions. To mitigate the impact of cast shadows and enhance the robustness of material recovery, we introduce illumination-invariant material constraint together with a deshadowing model. Extensive experiments on benchmark datasets show that our method consistently improves both reconstruction fidelity and inverse rendering quality over existing 3DGS-based inverse rendering approaches. Our code is available at https://github.com/GrumpySloths/SGS_Intrinsic.github.io.

  </details>



- **From None to All: Self-Supervised 3D Reconstruction via Novel View Synthesis**  
  Ranran Huang, Weixun Luo, Ye Mao, Krystian Mikolajczyk  
  _2026-03-29_ · https://arxiv.org/abs/2603.27455v1  
  <details><summary>Abstract</summary>

  In this paper, we introduce NAS3R, a self-supervised feed-forward framework that jointly learns explicit 3D geometry and camera parameters with no ground-truth annotations and no pretrained priors. During training, NAS3R reconstructs 3D Gaussians from uncalibrated and unposed context views and renders target views using its self-predicted camera parameters, enabling self-supervised training from 2D photometric supervision. To ensure stable convergence, NAS3R integrates reconstruction and camera prediction within a shared transformer backbone regulated by masked attention, and adopts a depth-based Gaussian formulation that facilitates well-conditioned optimization. The framework is compatible with state-of-the-art supervised 3D reconstruction architectures and can incorporate pretrained priors or intrinsic information when available. Extensive experiments show that NAS3R achieves superior results to other self-supervised methods, establishing a scalable and geometry-aware paradigm for 3D reconstruction from unconstrained data. Code and models are publicly available at https://ranrhuang.github.io/nas3r/.

  </details>



- **NimbusGS: Unified 3D Scene Reconstruction under Hybrid Weather**  
  Yanying Li, Jinyang Li, Shengfeng He, Yangyang Xu, Junyu Dong, Yong Du  
  _2026-03-28_ · https://arxiv.org/abs/2603.27228v1  
  <details><summary>Abstract</summary>

  We present NimbusGS, a unified framework for reconstructing high-quality 3D scenes from degraded multi-view inputs captured under diverse and mixed adverse weather conditions. Unlike existing methods that target specific weather types, NimbusGS addresses the broader challenge of generalization by modeling the dual nature of weather: a continuous, view-consistent medium that attenuates light, and dynamic, view-dependent particles that cause scattering and occlusion. To capture this structure, we decompose degradations into a global transmission field and per-view particulate residuals. The transmission field represents static atmospheric effects shared across views, while the residuals model transient disturbances unique to each input. To enable stable geometry learning under severe visibility degradation, we introduce a geometry-guided gradient scaling mechanism that mitigates gradient imbalance during the self-supervised optimization of 3D Gaussian representations. This physically grounded formulation allows NimbusGS to disentangle complex degradations while preserving scene structure, yielding superior geometry reconstruction and outperforming task-specific methods across diverse and challenging weather conditions. Code is available at https://github.com/lyy-ovo/NimbusGS.

  </details>


