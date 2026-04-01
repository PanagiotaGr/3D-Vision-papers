# NeRF & Neural Radiance Fields

_Updated: 2026-04-01 07:51 UTC_

Total papers shown: **9**


---

- **GRVS: a Generalizable and Recurrent Approach to Monocular Dynamic View Synthesis**  
  Thomas Tanay, Mohammed Brahimi, Michal Nazarczuk, Qingwen Zhang, Sibi Catley-Chandar, Arthur Moreau, Zhensong Zhang, Eduardo Pérez-Pellitero  
  _2026-03-31_ · https://arxiv.org/abs/2603.29734v1  
  <details><summary>Abstract</summary>

  Synthesizing novel views from monocular videos of dynamic scenes remains a challenging problem. Scene-specific methods that optimize 4D representations with explicit motion priors often break down in highly dynamic regions where multi-view information is hard to exploit. Diffusion-based approaches that integrate camera control into large pre-trained models can produce visually plausible videos but frequently suffer from geometric inconsistencies across both static and dynamic areas. Both families of methods also require substantial computational resources. Building on the success of generalizable models for static novel view synthesis, we adapt the framework to dynamic inputs and propose a new model with two key components: (1) a recurrent loop that enables unbounded and asynchronous mapping between input and target videos and (2) an efficient use of plane sweeps over dynamic inputs to disentangle camera and scene motion, and achieve fine-grained, six-degrees-of-freedom camera controls. We train and evaluate our model on the UCSD dataset and on Kubric-4D-dyn, a new monocular dynamic dataset featuring longer, higher resolution sequences with more complex scene dynamics than existing alternatives. Our model outperforms four Gaussian Splatting-based scene-specific approaches, as well as two diffusion-based approaches in reconstructing fine-grained geometric details across both static and dynamic regions.

  </details>



- **AA-Splat: Anti-Aliased Feed-forward Gaussian Splatting**  
  Taewoo Suh, Sungpyo Kim, Jongmin Park, Munchurl Kim  
  _2026-03-31_ · https://arxiv.org/abs/2603.29394v1  
  <details><summary>Abstract</summary>

  Feed-forward 3D Gaussian Splatting (FF-3DGS) emerges as a fast and robust solution for sparse-view 3D reconstruction and novel view synthesis (NVS). However, existing FF-3DGS methods are built on incorrect screen-space dilation filters, causing severe rendering artifacts when rendering at out-of-distribution sampling rates. We firstly propose an FF-3DGS model, called AA-Splat, to enable robust anti-aliased rendering at any resolution. AA-Splat utilizes an opacity-balanced band-limiting (OBBL) design, which combines two components: a 3D band-limiting post-filter integrates multi-view maximal frequency bounds into the feed-forward reconstruction pipeline, effectively band-limiting the resulting 3D scene representations and eliminating degenerate Gaussians; an Opacity Balancing (OB) to seamlessly integrate all pixel-aligned Gaussian primitives into the rendering process, compensating for the increased overlap between expanded Gaussian primitives. AA-Splat demonstrates drastic improvements with average 5.4$\sim$7.5dB PSNR gains on NVS performance over a state-of-the-art (SOTA) baseline, DepthSplat, at all resolutions, between $4\times$ and $1/4\times$. Code will be made available.

  </details>



- **Hierarchical Visual Relocalization with Nearest View Synthesis from Feature Gaussian Splatting**  
  Huaqi Tao, Bingxi Liu, Guangcheng Chen, Fulin Tang, Li He, Hong Zhang  
  _2026-03-31_ · https://arxiv.org/abs/2603.29185v1  
  <details><summary>Abstract</summary>

  Visual relocalization is a fundamental task in the field of 3D computer vision, estimating a camera's pose when it revisits a previously known scene. While point-based hierarchical relocalization methods have shown strong scalability and efficiency, they are often limited by sparse image observations and weak feature matching. In this work, we propose SplatHLoc, a novel hierarchical visual relocalization framework that uses Feature Gaussian Splatting as the scene representation. To address the sparsity of database images, we propose an adaptive viewpoint retrieval method that synthesizes virtual candidates with viewpoints more closely aligned with the query, thereby improving the accuracy of initial pose estimation. For feature matching, we observe that Gaussian-rendered features and those extracted directly from images exhibit different strengths across the two-stage matching process: the former performs better in the coarse stage, while the latter proves more effective in the fine stage. Therefore, we introduce a hybrid feature matching strategy, enabling more accurate and efficient pose estimation. Extensive experiments on both indoor and outdoor datasets show that SplatHLoc enhances the robustness of visual relocalization, setting a new state-of-the-art.

  </details>



- **UltraG-Ray: Physics-Based Gaussian Ray Casting for Novel Ultrasound View Synthesis**  
  Felix Duelmer, Jakob Klaushofer, Magdalena Wysocki, Nassir Navab, Mohammad Farid Azampour  
  _2026-03-30_ · https://arxiv.org/abs/2603.29022v1  
  <details><summary>Abstract</summary>

  Novel view synthesis (NVS) in ultrasound has gained attention as a technique for generating anatomically plausible views beyond the acquired frames, offering new capabilities for training clinicians or data augmentation. However, current methods struggle with complex tissue and view-dependent acoustic effects. Physics-based NVS aims to address these limitations by including the ultrasound image formation process into the simulation. Recent approaches combine a learnable implicit scene representation with an ultrasound-specific rendering module, yet a substantial gap between simulation and reality remains. In this work, we introduce UltraG-Ray, a novel ultrasound scene representation based on a learnable 3D Gaussian field, coupled to an efficient physics-based module for B-mode synthesis. We explicitly encode ultrasound-specific parameters, such as attenuation and reflection, into a Gaussian-based spatial representation and realize image synthesis within a novel ray casting scheme. In contrast to previous methods, this approach naturally captures view-dependent attenuation effects, thereby enabling the generation of physically informed B-mode images with increased realism. We compare our method to state-of-the-art and observe consistent gains in image quality metrics (up to 15% increase on MS-SSIM), demonstrating clear improvement in terms of realism of the synthesized ultrasound images.

  </details>



- **SVGS: Single-View to 3D Object Editing via Gaussian Splatting**  
  Pengcheng Xue, Yan Tian, Qiutao Song, Ziyi Wang, Linyang He, Weiping Ding, Mahmoud Hassaballah, Karen Egiazarian, Wei-Fa Yang, Leszek Rutkowski  
  _2026-03-30_ · https://arxiv.org/abs/2603.28126v1  
  <details><summary>Abstract</summary>

  Text-driven 3D scene editing has attracted considerable interest due to its convenience and user-friendliness. However, methods that rely on implicit 3D representations, such as Neural Radiance Fields (NeRF), while effective in rendering complex scenes, are hindered by slow processing speeds and limited control over specific regions of the scene. Moreover, existing approaches, including Instruct-NeRF2NeRF and GaussianEditor, which utilize multi-view editing strategies, frequently produce inconsistent results across different views when executing text instructions. This inconsistency can adversely affect the overall performance of the model, complicating the task of balancing the consistency of editing results with editing efficiency. To address these challenges, we propose a novel method termed Single-View to 3D Object Editing via Gaussian Splatting (SVGS), which is a single-view text-driven editing technique based on 3D Gaussian Splatting (3DGS). Specifically, in response to text instructions, we introduce a single-view editing strategy grounded in multi-view diffusion models, which reconstructs 3D scenes by leveraging only those views that yield consistent editing results. Additionally, we employ sparse 3D Gaussian Splatting as the 3D representation, which significantly enhances editing efficiency. We conducted a comparative analysis of SVGS against existing baseline methods across various scene settings, and the results indicate that SVGS outperforms its counterparts in both editing capability and processing speed, representing a significant advancement in 3D editing technology. For further details, please visit our project page at: https://amateurc.github.io/svgs.github.io.

  </details>



- **To View Transform or Not to View Transform: NeRF-based Pre-training Perspective**  
  Hyeonjun Jeong, Juyeb Shin, Dongsuk Kum  
  _2026-03-30_ · https://arxiv.org/abs/2603.28090v1  
  <details><summary>Abstract</summary>

  Neural radiance fields (NeRFs) have emerged as a prominent pre-training paradigm for vision-centric autonomous driving, which enhances 3D geometry and appearance understanding in a fully self-supervised manner. To apply NeRF-based pretraining to 3D perception models, recent approaches have simply applied NeRFs to volumetric features obtained from view transformation. However, coupling NeRFs with view transformation inherits conflicting priors; view transformation imposes discrete and rigid representations, whereas radiance fields assume continuous and adaptive functions. When these opposing assumptions are forced into a single pipeline, the misalignment surfaces as blurry and ambiguous 3D representations that ultimately limit 3D scene understanding. Moreover, the NeRF network for pre-training is discarded during downstream tasks, resulting in inefficient utilization of enhanced 3D representations through NeRF. In this paper, we propose a novel NeRF-Resembled Point-based 3D detector that can learn continuous 3D representation and thus avoid the misaligned priors from view transformation. NeRP3D preserves the pre-trained NeRF network regardless of the tasks, inheriting the principle of continuous 3D representation learning and leading to greater potentials for both scene reconstruction and detection tasks. Experiments on nuScenes dataset demonstrate that our proposed approach significantly improves previous state-of-the-art methods, outperforming not only pretext scene reconstruction tasks but also downstream detection tasks.

  </details>



- **Physically Inspired Gaussian Splatting for HDR Novel View Synthesis**  
  Huimin Zeng, Yue Bai, Hailing Wang, Yun Fu  
  _2026-03-30_ · https://arxiv.org/abs/2603.28020v1  
  <details><summary>Abstract</summary>

  High dynamic range novel view synthesis (HDR-NVS) reconstructs scenes with dynamic details by fusing multi-exposure low dynamic range (LDR) views, yet it struggles to capture ambient illumination-dependent appearance. Implicitly supervising HDR content by constraining tone-mapped results fails in correcting abnormal HDR values, and results in limited gradients for Gaussians in under/over-exposed regions. To this end, we introduce PhysHDR-GS, a physically inspired HDR-NVS framework that models scene appearance via intrinsic reflectance and adjustable ambient illumination. PhysHDR-GS employs a complementary image-exposure (IE) branch and Gaussian-illumination (GI) branch to faithfully reproduce standard camera observations and capture illumination-dependent appearance changes, respectively. During training, the proposed cross-branch HDR consistency loss provides explicit supervision for HDR content, while an illumination-guided gradient scaling strategy mitigates exposure-biased gradient starvation and reduces under-densified representations. Experimental results across realistic and synthetic datasets demonstrate our superiority in reconstructing HDR details (e.g., a PSNR gain of 2.04 dB over HDR-GS), while maintaining real-time rendering speed (up to 76 FPS). Code and models are available at https://huimin-zeng.github.io/PhysHDR-GS/.

  </details>



- **RehearsalNeRF: Decoupling Intrinsic Neural Fields of Dynamic Illuminations for Scene Editing**  
  Changyeon Won, Hyunjun Jung, Jungu Cho, Seonmi Park, Chi-Hoon Lee, Hae-Gon Jeon  
  _2026-03-30_ · https://arxiv.org/abs/2603.27948v1  
  <details><summary>Abstract</summary>

  Although there has been significant progress in neural radiance fields, an issue on dynamic illumination changes still remains unsolved. Different from relevant works that parameterize time-variant/-invariant components in scenes, subjects' radiance is highly entangled with their own emitted radiance and lighting colors in spatio-temporal domain. In this paper, we present a new effective method to learn disentangled neural fields under the severe illumination changes, named RehearsalNeRF. Our key idea is to leverage scenes captured under stable lighting like rehearsal stages, easily taken before dynamic illumination occurs, to enforce geometric consistency between the different lighting conditions. In particular, RehearsalNeRF employs a learnable vector for lighting effects which represents illumination colors in a temporal dimension and is used to disentangle projected light colors from scene radiance. Furthermore, our RehearsalNeRF is also able to reconstruct the neural fields of dynamic objects by simply adopting off-the-shelf interactive masks. To decouple the dynamic objects, we propose a new regularization leveraging optical flow, which provides coarse supervision for the color disentanglement. We demonstrate the effectiveness of RehearsalNeRF by showing robust performances on novel view synthesis and scene editing under dynamic illumination conditions. Our source code and video datasets will be publicly available.

  </details>



- **GS3LAM: Gaussian Semantic Splatting SLAM**  
  Linfei Li, Lin Zhang, Zhong Wang, Ying Shen  
  _2026-03-29_ · https://arxiv.org/abs/2603.27781v1  
  <details><summary>Abstract</summary>

  Recently, the multi-modal fusion of RGB, depth, and semantics has shown great potential in dense Simultaneous Localization and Mapping (SLAM). However, a prerequisite for generating consistent semantic maps is the availability of dense, efficient, and scalable scene representations. Existing semantic SLAM systems based on explicit representations are often limited by resolution and an inability to predict unknown areas. Conversely, implicit representations typically rely on time-consuming ray tracing, failing to meet real-time requirements. Fortunately, 3D Gaussian Splatting (3DGS) has emerged as a promising representation that combines the efficiency of point-based methods with the continuity of geometric structures. To this end, we propose GS3LAM, a Gaussian Semantic Splatting SLAM framework that processes multimodal data to render consistent, dense semantic maps in real-time. GS3LAM models the scene as a Semantic Gaussian Field (SG-Field) and jointly optimizes camera poses and the field via multimodal error constraints. Furthermore, a Depth-adaptive Scale Regularization (DSR) scheme is introduced to resolve misalignments between scale-invariant Gaussians and geometric surfaces. To mitigate catastrophic forgetting, we propose a Random Sampling-based Keyframe Mapping (RSKM) strategy, which demonstrates superior performance over common local covisibility optimization methods. Extensive experiments on benchmark datasets show that GS3LAM achieves increased tracking robustness, superior rendering quality, and enhanced semantic precision compared to state-of-the-art methods. Source code is available at https://github.com/lif314/GS3LAM.

  </details>


