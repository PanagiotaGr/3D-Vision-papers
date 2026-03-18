# Gaussian Splatting & 3DGS

_Updated: 2026-03-18 07:16 UTC_

Total papers shown: **9**


---

- **M^3: Dense Matching Meets Multi-View Foundation Models for Monocular Gaussian Splatting SLAM**  
  Kerui Ren, Guanghao Li, Changjian Jiang, Yingxiang Xu, Tao Lu, Linning Xu, Junting Dong, Jiangmiao Pang, Mulin Yu, Bo Dai  
  _2026-03-17_ · https://arxiv.org/abs/2603.16844v1  
  <details><summary>Abstract</summary>

  Streaming reconstruction from uncalibrated monocular video remains challenging, as it requires both high-precision pose estimation and computationally efficient online refinement in dynamic environments. While coupling 3D foundation models with SLAM frameworks is a promising paradigm, a critical bottleneck persists: most multi-view foundation models estimate poses in a feed-forward manner, yielding pixel-level correspondences that lack the requisite precision for rigorous geometric optimization. To address this, we present M^3, which augments the Multi-view foundation model with a dedicated Matching head to facilitate fine-grained dense correspondences and integrates it into a robust Monocular Gaussian Splatting SLAM. M^3 further enhances tracking stability by incorporating dynamic area suppression and cross-inference intrinsic alignment. Extensive experiments on diverse indoor and outdoor benchmarks demonstrate state-of-the-art accuracy in both pose estimation and scene reconstruction. Notably, M^3 reduces ATE RMSE by 64.3% compared to VGGT-SLAM 2.0 and outperforms ARTDECO by 2.11 dB in PSNR on the ScanNet++ dataset.

  </details>



- **Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty**  
  Mangyu Kong, Jaewon Lee, Seongwon Lee, Euntai Kim  
  _2026-03-17_ · https://arxiv.org/abs/2603.16538v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has recently emerged as a powerful scene representation and is increasingly used for visual localization and pose refinement. However, despite its high-quality differentiable rendering, the robustness of 3DGS-based pose refinement remains highly sensitive to both the initial camera pose and the reconstructed geometry. In this work, we take a closer look at these limitations and identify two major sources of uncertainty: (i) pose prior uncertainty, which often arises from regression or retrieval models that output a single deterministic estimate, and (ii) geometric uncertainty, caused by imperfections in the 3DGS reconstruction that propagate errors into PnP solvers. Such uncertainties can distort reprojection geometry and destabilize optimization, even when the rendered appearance still looks plausible. To address these uncertainties, we introduce a relocalization framework that combines Monte Carlo pose sampling with Fisher Information-based PnP optimization. Our method explicitly accounts for both pose and geometric uncertainty and requires no retraining or additional supervision. Across diverse indoor and outdoor benchmarks, our approach consistently improves localization accuracy and significantly increases stability under pose and depth noise.

  </details>



- **ProgressiveAvatars: Progressive Animatable 3D Gaussian Avatars**  
  Kaiwen Song, Jinkai Cui, Juyong Zhang  
  _2026-03-17_ · https://arxiv.org/abs/2603.16447v1  
  <details><summary>Abstract</summary>

  In practical real-time XR and telepresence applications, network and computing resources fluctuate frequently. Therefore, a progressive 3D representation is needed. To this end, we propose ProgressiveAvatars, a progressive avatar representation built on a hierarchy of 3D Gaussians grown by adaptive implicit subdivision on a template mesh. 3D Gaussians are defined in face-local coordinates to remain animatable under varying expressions and head motion across multiple detail levels. The hierarchy expands when screen-space signals indicate a lack of detail, allocating resources to important areas. Leveraging importance ranking, ProgressiveAvatars supports incremental loading and rendering, adding new Gaussians as they arrive while preserving previous content, thus achieving smooth quality improvements across varying bandwidths. ProgressiveAvatars enables progressive delivery and progressive rendering under fluctuating network bandwidth and varying compute and memory resources.

  </details>



- **Leveling3D: Leveling Up 3D Reconstruction with Feed-Forward 3D Gaussian Splatting and Geometry-Aware Generation**  
  Yiming Huang, Baixiang Huang, Beilei Cui, Chi Kit Ng, Long Bai, Hongliang Ren  
  _2026-03-17_ · https://arxiv.org/abs/2603.16211v1  
  <details><summary>Abstract</summary>

  Feed-forward 3D reconstruction has revolutionized 3D vision, providing a powerful baseline for downstream tasks such as novel-view synthesis with 3D Gaussian Splatting. Previous works explore fixing the corrupted rendering results with a diffusion model. However, they lack geometric concern and fail at filling the missing area on the extrapolated view. In this work, we introduce Leveling3D, a novel pipeline that integrates feed-forward 3D reconstruction with geometrical-consistent generation to enable holistic simultaneous reconstruction and generation. We propose a geometry-aware leveling adapter, a lightweight technique that aligns internal knowledge in the diffusion model with the geometry prior from the feed-forward model. The leveling adapter enables generation on the artifact area of the extrapolated novel views caused by underconstrained regions of the 3D representation. Specifically, to learn a more diverse distributed generation, we introduce the palette filtering strategy for training, and a test-time masking refinement to prevent messy boundaries along the fixing regions. More importantly, the enhanced extrapolated novel views from Leveling3D could be used as the inputs for feed-forward 3DGS, leveling up the 3D reconstruction. We achieve SOTA performance on public datasets, including tasks such as novel-view synthesis and depth estimation.

  </details>



- **NanoGS: Training-Free Gaussian Splat Simplification**  
  Butian Xiong, Rong Liu, Tiantian Zhou, Meida Chen, Zhiwen Fan, Andrew Feng  
  _2026-03-17_ · https://arxiv.org/abs/2603.16103v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splat (3DGS) enables high-fidelity, real-time novel view synthesis by representing scenes with large sets of anisotropic primitives, but often requires millions of Splats, incurring significant storage and transmission costs. Most existing compression methods rely on GPU-intensive post-training optimization with calibrated images, limiting practical deployment. We introduce NanoGS, a training-free and lightweight framework for Gaussian Splat simplification. Instead of relying on image-based rendering supervision, NanoGS formulates simplification as local pairwise merging over a sparse spatial graph. The method approximates a pair of Gaussians with a single primitive using mass preserved moment matching and evaluates merge quality through a principled merge cost between the original mixture and its approximation. By restricting merge candidates to local neighborhoods and selecting compatible pairs efficiently, NanoGS produces compact Gaussian representations while preserving scene structure and appearance. NanoGS operates directly on existing Gaussian Splat models, runs efficiently on CPU, and preserves the standard 3DGS parameterization, enabling seamless integration with existing rendering pipelines. Experiments demonstrate that NanoGS substantially reduces primitive count while maintaining high rendering fidelity, providing an efficient and practical solution for Gaussian Splat simplification. Our project website is available at https://saliteta.github.io/NanoGS/.

  </details>



- **IRIS: Intersection-aware Ray-based Implicit Editable Scenes**  
  Grzegorz Wilczyński, Mikołaj Zieliński, Krzysztof Byrski, Joanna Waczyńska, Dominik Belter, Przemysław Spurek  
  _2026-03-16_ · https://arxiv.org/abs/2603.15368v1  
  <details><summary>Abstract</summary>

  Neural Radiance Fields achieve high-fidelity scene representation but suffer from costly training and rendering, while 3D Gaussian splatting offers real-time performance with strong empirical results. Recently, solutions that harness the best of both worlds by using Gaussians as proxies to guide neural field evaluations, still suffer from significant computational inefficiencies. They typically rely on stochastic volumetric sampling to aggregate features, which severely limits rendering performance. To address this issue, a novel framework named IRIS (Intersection-aware Ray-based Implicit Editable Scenes) is introduced as a method designed for efficient and interactive scene editing. To overcome the limitations of standard ray marching, an analytical sampling strategy is employed that precisely identifies interaction points between rays and scene primitives, effectively eliminating empty space processing. Furthermore, to address the computational bottleneck of spatial neighbor lookups, a continuous feature aggregation mechanism is introduced that operates directly along the ray. By interpolating latent attributes from sorted intersections, costly 3D searches are bypassed, ensuring geometric consistency, enabling high-fidelity, real-time rendering, and flexible shape editing. Code can be found at https://github.com/gwilczynski95/iris.

  </details>



- **GeoNVS: Geometry Grounded Video Diffusion for Novel View Synthesis**  
  Minjun Kang, Inkyu Shin, Taeyeop Lee, Myungchul Kim, In So Kweon, Kuk-Jin Yoon  
  _2026-03-16_ · https://arxiv.org/abs/2603.14965v1  
  <details><summary>Abstract</summary>

  Novel view synthesis requires strong 3D geometric consistency and the ability to generate visually coherent images across diverse viewpoints. While recent camera-controlled video diffusion models show promising results, they often suffer from geometric distortions and limited camera controllability. To overcome these challenges, we introduce GeoNVS, a geometry-grounded novel-view synthesizer that enhances both geometric fidelity and camera controllability through explicit 3D geometric guidance. Our key innovation is the Gaussian Splat Feature Adapter (GS-Adapter), which lifts input-view diffusion features into 3D Gaussian representations, renders geometry-constrained novel-view features, and adaptively fuses them with diffusion features to correct geometrically inconsistent representations. Unlike prior methods that inject geometry at the input level, GS-Adapter operates in feature space, avoiding view-dependent color noise that degrades structural consistency. Its plug-and-play design enables zero-shot compatibility with diverse feed-forward geometry models without additional training, and can be adapted to other video diffusion backbones. Experiments across 9 scenes and 18 settings demonstrate state-of-the-art performance, achieving 11.3% and 14.9% improvements over SEVA and CameraCtrl, with up to 2x reduction in translation error and 7x in Chamfer Distance.

  </details>



- **Zero-Shot Reconstruction of Animatable 3D Avatars with Cloth Dynamics from a Single Image**  
  Joohyun Kwon, Geonhee Sim, Gyeongsik Moon  
  _2026-03-16_ · https://arxiv.org/abs/2603.14772v1  
  <details><summary>Abstract</summary>

  Existing single-image 3D human avatar methods primarily rely on rigid joint transformations, limiting their ability to model realistic cloth dynamics. We present DynaAvatar, a zero-shot framework that reconstructs animatable 3D human avatars with motion-dependent cloth dynamics from a single image. Trained on large-scale multi-person motion datasets, DynaAvatar employs a Transformer-based feed-forward architecture that directly predicts dynamic 3D Gaussian deformations without subject-specific optimization. To overcome the scarcity of dynamic captures, we introduce a static-to-dynamic knowledge transfer strategy: a Transformer pretrained on large-scale static captures provides strong geometric and appearance priors, which are efficiently adapted to motion-dependent deformations through lightweight LoRA fine-tuning on dynamic captures. We further propose the DynaFlow loss, an optical flow-guided objective that provides reliable motion-direction geometric cues for cloth dynamics in rendered space. Finally, we reannotate the missing or noisy SMPL-X fittings in existing dynamic capture datasets, as most public dynamic capture datasets contain incomplete or unreliable fittings that are unsuitable for training high-quality 3D avatar reconstruction models. Experiments demonstrate that DynaAvatar produces visually rich and generalizable animations, outperforming prior methods.

  </details>



- **LiDAR-EVS: Enhance Extrapolated View Synthesis for 3D Gaussian Splatting with Pseudo-LiDAR Supervision**  
  Yiming Huang, Xin Kang, Sipeng Zhang, Hongliang Ren, Weihua Zhang, Junjie Lai  
  _2026-03-16_ · https://arxiv.org/abs/2603.14763v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has emerged as a powerful technique for real-time LiDAR and camera synthesis in autonomous driving simulation. However, simulating LiDAR with 3DGS remains challenging for extrapolated views beyond the training trajectory, as existing methods are typically trained on single-traversal sensor scans, suffer from severe overfitting and poor generalization to novel ego-vehicle paths. To enable reliable simulation of LiDAR along unseen driving trajectories without external multi-pass data, we present LiDAR-EVS, a lightweight framework for robust extrapolated-view LiDAR simulation in autonomous driving. Designed to be plug-and-play, LiDAR-EVS readily extends to diverse LiDAR sensors and neural rendering baselines with minimal modification. Our framework comprises two key components: (1) pseudo extrapolated-view point cloud supervision with multi-frame LiDAR fusion, view transformation, occlusion curling, and intensity adjustment; (2) spatially-constrained dropout regularization that promotes robustness to diverse trajectory variations encountered in real-world driving. Extensive experiments demonstrate that LiDAR-EVS achieves SOTA performance on extrapolated-view LiDAR synthesis across three datasets, making it a promising tool for data-driven simulation, closed-loop evaluation, and synthetic data generation in autonomous driving systems.

  </details>


