# NeRF & Neural Radiance Fields

_Updated: 2026-03-18 07:16 UTC_

Total papers shown: **8**


---

- **DriveFix: Spatio-Temporally Coherent Driving Scene Restoration**  
  Heyu Si, Brandon James Denis, Muyang Sun, Dragos Datcu, Yaoru Li, Xin Jin, Ruiju Fu, Yuliia Tatarinova, Federico Landi, Jie Song, et al.  
  _2026-03-17_ · https://arxiv.org/abs/2603.16306v1  
  <details><summary>Abstract</summary>

  Recent advancements in 4D scene reconstruction, particularly those leveraging diffusion priors, have shown promise for novel view synthesis in autonomous driving. However, these methods often process frames independently or in a view-by-view manner, leading to a critical lack of spatio-temporal synergy. This results in spatial misalignment across cameras and temporal drift in sequences. We propose DriveFix, a novel multi-view restoration framework that ensures spatio-temporal coherence for driving scenes. Our approach employs an interleaved diffusion transformer architecture with specialized blocks to explicitly model both temporal dependencies and cross-camera spatial consistency. By conditioning the generation on historical context and integrating geometry-aware training losses, DriveFix enforces that the restored views adhere to a unified 3D geometry. This enables the consistent propagation of high-fidelity textures and significantly reduces artifacts. Extensive evaluations on the Waymo, nuScenes, and PandaSet datasets demonstrate that DriveFix achieves state-of-the-art performance in both reconstruction and novel view synthesis, marking a substantial step toward robust 4D world modeling for real-world deployment.

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



- **Feed-forward Gaussian Registration for Head Avatar Creation and Editing**  
  Malte Prinzler, Paulo Gotardo, Siyu Tang, Timo Bolkart  
  _2026-03-16_ · https://arxiv.org/abs/2603.15811v1  
  <details><summary>Abstract</summary>

  We present MATCH (Multi-view Avatars from Topologically Corresponding Heads), a multi-view Gaussian registration method for high-quality head avatar creation and editing. State-of-the-art multi-view head avatar methods require time-consuming head tracking followed by expensive avatar optimization, often resulting in a total creation time of more than one day. MATCH, in contrast, directly predicts Gaussian splat textures in correspondence from calibrated multi-view images in just 0.5 seconds per frame, without requiring data preprocessing. The learned intra-subject correspondence across frames enables fast creation of personalized head avatars, while correspondence across subjects supports applications such as expression transfer, optimization-free tracking, semantic editing, and identity interpolation. We establish these correspondences end-to-end using a transformer-based model that predicts Gaussian splat textures in the fixed UV layout of a template mesh. To achieve this, we introduce a novel registration-guided attention block, where each UV-map token attends exclusively to image tokens depicting its corresponding mesh region. This design improves efficiency and performance compared to dense cross-view attention. MATCH outperforms existing methods in novel-view synthesis, geometry registration, and head avatar generation, while making avatar creation 10 times faster than the closest competing baseline. The code and model weights are available on the project website.

  </details>



- **Real-Time Human Frontal View Synthesis from a Single Image**  
  Fangyu Lin, Yingdong Hu, Lunjie Zhu, Zhening Liu, Yushi Huang, Zehong Lin, Jun Zhang  
  _2026-03-16_ · https://arxiv.org/abs/2603.15433v1  
  <details><summary>Abstract</summary>

  Photorealistic human novel view synthesis from a single image is crucial for democratizing immersive 3D telepresence, eliminating the need for complex multi-camera setups. However, current rendering-centric methods prioritize visual fidelity over explicit geometric understanding and struggle with intricate regions like faces and hands, leading to temporal instability. Meanwhile, human-centric frameworks suffer from memory bottlenecks since they typically rely on an auxiliary model to provide informative structural priors for geometric modeling, which limits real-time performance. To address these challenges, we propose PrismMirror, a geometry-guided framework for instant frontal view synthesis from a single image. By avoiding external geometric modeling and focusing on frontal view synthesis, our model optimizes visual integrity for telepresence. Specifically, PrismMirror introduces a novel cascade learning strategy that enables coarse-to-fine geometric feature learning. It first directly learns coarse geometric features, such as SMPL-X meshes and point clouds, and then refines textures through rendering supervision. To achieve real-time efficiency, we distill this unified framework into a lightweight linear attention model. Notably, PrismMirror is the first monocular human frontal view synthesis model that achieves real-time inference at 24 FPS, significantly outperforming previous methods in both visual authenticity and structural accuracy.

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



- **LiDAR-EVS: Enhance Extrapolated View Synthesis for 3D Gaussian Splatting with Pseudo-LiDAR Supervision**  
  Yiming Huang, Xin Kang, Sipeng Zhang, Hongliang Ren, Weihua Zhang, Junjie Lai  
  _2026-03-16_ · https://arxiv.org/abs/2603.14763v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has emerged as a powerful technique for real-time LiDAR and camera synthesis in autonomous driving simulation. However, simulating LiDAR with 3DGS remains challenging for extrapolated views beyond the training trajectory, as existing methods are typically trained on single-traversal sensor scans, suffer from severe overfitting and poor generalization to novel ego-vehicle paths. To enable reliable simulation of LiDAR along unseen driving trajectories without external multi-pass data, we present LiDAR-EVS, a lightweight framework for robust extrapolated-view LiDAR simulation in autonomous driving. Designed to be plug-and-play, LiDAR-EVS readily extends to diverse LiDAR sensors and neural rendering baselines with minimal modification. Our framework comprises two key components: (1) pseudo extrapolated-view point cloud supervision with multi-frame LiDAR fusion, view transformation, occlusion curling, and intensity adjustment; (2) spatially-constrained dropout regularization that promotes robustness to diverse trajectory variations encountered in real-world driving. Extensive experiments demonstrate that LiDAR-EVS achieves SOTA performance on extrapolated-view LiDAR synthesis across three datasets, making it a promising tool for data-driven simulation, closed-loop evaluation, and synthetic data generation in autonomous driving systems.

  </details>


