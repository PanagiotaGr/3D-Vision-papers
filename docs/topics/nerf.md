# NeRF & Neural Radiance Fields

_Updated: 2026-03-17 07:17 UTC_

Total papers shown: **8**


---

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



- **E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction**  
  Yunsoo Kim, Changki Sung, Dasol Hong, Hyun Myung  
  _2026-03-16_ · https://arxiv.org/abs/2603.14684v1  
  <details><summary>Abstract</summary>

  The emergence of neural radiance fields (NeRF) and 3D Gaussian splatting (3DGS) has advanced novel view synthesis (NVS). These methods, however, require high-quality RGB inputs and accurate corresponding poses, limiting robustness under real-world conditions such as fast camera motion or adverse lighting. Event cameras, which capture brightness changes at each pixel with high temporal resolution and wide dynamic range, enable precise sensing of dynamic scenes and offer a promising solution. However, existing event-based NVS methods either assume known poses or rely on depth estimation models that are bounded by their initial observations, failing to generalize as the camera traverses previously unseen regions. We present E2EGS, a pose-free framework operating solely on event streams. Our key insight is that edge information provides rich structural cues essential for accurate trajectory estimation and high-quality NVS. To extract edges from noisy event streams, we exploit the distinct spatio-temporal characteristics of edges and non-edge regions. The event camera's movement induces consistent events along edges, while non-edge regions produce sparse noise. We leverage this through a patch-based temporal coherence analysis that measures local variance to extract edges while robustly suppressing noise. The extracted edges guide structure-aware Gaussian initialization and enable edge-weighted losses throughout initialization, tracking, and bundle adjustment. Extensive experiments on both synthetic and real datasets demonstrate that E2EGS achieves superior reconstruction quality and trajectory accuracy, establishing a fully pose-free paradigm for event-based 3D reconstruction.

  </details>



- **OAHuman: Occlusion-Aware 3D Human Reconstruction from Monocular Images**  
  Yuanwang Yang, Hongliang Liu, Muxin Zhang, Nan Ma, Jingyu Yang, Yu-Kun Lai, Kun Li  
  _2026-03-15_ · https://arxiv.org/abs/2603.14249v1  
  <details><summary>Abstract</summary>

  Monocular 3D human reconstruction in real-world scenarios remains highly challenging due to frequent occlusions from surrounding objects, people, or image truncation. Such occlusions lead to missing geometry and unreliable appearance cues, severely degrading the completeness and realism of reconstructed human models. Although recent neural implicit methods achieve impressive results on clean inputs, they struggle under occlusion due to entangled modeling of shape and texture. In this paper, we propose OAHuman, an occlusion-aware framework that explicitly decouples geometry reconstruction and texture synthesis for robust 3D human modeling from a single RGB image. The core innovation lies in the decoupling-perception paradigm, which addresses the fundamental issue of geometry-texture cross-contamination in occluded regions. Our framework ensures that geometry reconstruction is perceptually reinforced even in occluded areas, isolating it from texture interference. In parallel, texture synthesis is learned exclusively from visible regions, preventing texture errors from being transferred to the occluded areas. This decoupling approach enables OAHuman to achieve robust and high-fidelity reconstruction under occlusion, which has been a long-standing challenge in the field. Extensive experiments on occlusion-rich benchmarks demonstrate that OAHuman achieves superior performance in terms of structural completeness, surface detail, and texture realism, significantly improving monocular 3D human reconstruction under occlusion conditions.

  </details>



- **CamLit: Unified Video Diffusion with Explicit Camera and Lighting Control**  
  Zhiyi Kuang, Chengan He, Egor Zakharov, Yuxuan Xue, Shunsuke Saito, Olivier Maury, Timur Bagautdinov, Youyi Zheng, Giljoo Nam  
  _2026-03-15_ · https://arxiv.org/abs/2603.14241v1  
  <details><summary>Abstract</summary>

  We present CamLit, the first unified video diffusion model that jointly performs novel view synthesis (NVS) and relighting from a single input image. Given one reference image, a user-defined camera trajectory, and an environment map, CamLit synthesizes a video of the scene from new viewpoints under the specified illumination. Within a single generative process, our model produces temporally coherent and spatially aligned outputs, including relit novel-view frames and corresponding albedo frames, enabling high-quality control of both camera pose and lighting. Qualitative and quantitative experiments demonstrate that CamLit achieves high-fidelity outputs on par with state-of-the-art methods in both novel view synthesis and relighting, without sacrificing visual quality in either task. We show that a single generative model can effectively integrate camera and lighting control, simplifying the video generation pipeline while maintaining competitive performance and consistent realism.

  </details>



- **Geo-ID: Test-Time Geometric Consensus for Cross-View Consistent Intrinsics**  
  Alara Dirik, Stefanos Zafeiriou  
  _2026-03-14_ · https://arxiv.org/abs/2603.13859v1  
  <details><summary>Abstract</summary>

  Intrinsic image decomposition aims to estimate physically based rendering (PBR) parameters such as albedo, roughness, and metallicity from images. While recent methods achieve strong single-view predictions, applying them independently to multiple views of the same scene often yields inconsistent estimates, limiting their use in downstream applications such as editable neural scenes and 3D reconstruction. Video-based models can improve cross-frame consistency but require dense, ordered sequences and substantial compute, limiting their applicability to sparse, unordered image collections. We propose Geo-ID, a novel test-time framework that repurposes pretrained single-view intrinsic predictors to produce cross-view consistent decompositions by coupling independent per-view predictions through sparse geometric correspondences that form uncertainty-aware consensus targets. Geo-ID is model-agnostic, requires no retraining or inverse rendering, and applies directly to off-the-shelf intrinsic predictors. Experiments on synthetic benchmarks and real-world scenes demonstrate substantial improvements in cross-view intrinsic consistency as the number of views increases, while maintaining comparable single-view decomposition performance. We further show that the resulting consistent intrinsics enable coherent appearance editing and relighting in downstream neural scene representations.

  </details>


