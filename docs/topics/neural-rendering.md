# Neural Rendering & View Synthesis

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



- **MotionScale: Reconstructing Appearance, Geometry, and Motion of Dynamic Scenes with Scalable 4D Gaussian Splatting**  
  Haoran Zhou, Gim Hee Lee  
  _2026-03-31_ · https://arxiv.org/abs/2603.29296v1  
  <details><summary>Abstract</summary>

  Realistic reconstruction of dynamic 4D scenes from monocular videos is essential for understanding the physical world. Despite recent progress in neural rendering, existing methods often struggle to recover accurate 3D geometry and temporally consistent motion in complex environments. To address these challenges, we propose MotionScale, a 4D Gaussian Splatting framework that scales efficiently to large scenes and extended sequences while maintaining high-fidelity structural and motion coherence. At the core of our approach is a scalable motion field parameterized by cluster-centric basis transformations that adaptively expand to capture diverse and evolving motion patterns. To ensure robust reconstruction over long durations, we introduce a progressive optimization strategy comprising two decoupled propagation stages: 1) A background extension stage that adapts to newly visible regions, refines camera poses, and explicitly models transient shadows; 2) A foreground propagation stage that enforces motion consistency through a specialized three-stage refinement process. Extensive experiments on challenging real-world benchmarks demonstrate that MotionScale significantly outperforms state-of-the-art methods in both reconstruction quality and temporal stability. Project page: https://hrzhou2.github.io/motion-scale-web/.

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



- **GenFusion: Feed-forward Human Performance Capture via Progressive Canonical Space Updates**  
  Youngjoong Kwon, Yao He, Heejung Choi, Chen Geng, Zhengmao Liu, Jiajun Wu, Ehsan Adeli  
  _2026-03-30_ · https://arxiv.org/abs/2603.28997v1  
  <details><summary>Abstract</summary>

  We present a feed-forward human performance capture method that renders novel views of a performer from a monocular RGB stream. A key challenge in this setting is the lack of sufficient observations, especially for unseen regions. Assuming the subject moves continuously over time, we take advantage of the fact that more body parts become observable by maintaining a canonical space that is progressively updated with each incoming frame. This canonical space accumulates appearance information over time and serves as a context bank when direct observations are missing in the current live frame. To effectively utilize this context while respecting the deformation of the live state, we formulate the rendering process as probabilistic regression. This resolves conflicts between past and current observations, producing sharper reconstructions than deterministic regression approaches. Furthermore, it enables plausible synthesis even in regions with no prior observations. Experiments on in-domain (4D-Dress) and out-of-distribution (MVHumanNet) datasets demonstrate the effectiveness of our approach.

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



- **Poppy: Polarization-based Plug-and-Play Guidance for Enhancing Monocular Normal Estimation**  
  Irene Kim, Sai Tanmay Reddy Chakkera, Alexandros Graikos, Dimitris Samaras, Akshat Dave  
  _2026-03-29_ · https://arxiv.org/abs/2603.27891v1  
  <details><summary>Abstract</summary>

  Monocular surface normal estimators trained on large-scale RGB-normal data often perform poorly in the edge cases of reflective, textureless, and dark surfaces. Polarization encodes surface orientation independently of texture and albedo, offering a physics-based complement for these cases. Existing polarization methods, however, require multi-view capture or specialized training data, limiting generalization. We introduce Poppy, a training-free framework that refines normals from any frozen RGB backbone using single-shot polarization measurements at test time. Keeping backbone weights frozen, Poppy optimizes per-pixel offsets to the input RGB and output normal along with a learned reflectance decomposition. A differentiable rendering layer converts the refined normals into polarization predictions and penalizes mismatches with the observed signal. Across seven benchmarks and three backbone architectures (diffusion, flow, and feed-forward), Poppy reduces mean angular error by 23-26% on synthetic data and 6-16% on real data. These results show that guiding learned RGB-based normal estimators with polarization cues at test time refines normals on challenging surfaces without retraining.

  </details>


