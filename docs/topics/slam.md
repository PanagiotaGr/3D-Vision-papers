# SLAM & Localization

_Updated: 2026-04-10 07:57 UTC_

Total papers shown: **7**


---

- **Novel View Synthesis as Video Completion**  
  Qi Wu, Khiem Vuong, Minsik Jeon, Srinivasa Narasimhan, Deva Ramanan  
  _2026-04-09_ · https://arxiv.org/abs/2604.08500v1  
  <details><summary>Abstract</summary>

  We tackle the problem of sparse novel view synthesis (NVS) using video diffusion models; given $K$ ($\approx 5$) multi-view images of a scene and their camera poses, we predict the view from a target camera pose. Many prior approaches leverage generative image priors encoded via diffusion models. However, models trained on single images lack multi-view knowledge. We instead argue that video models already contain implicit multi-view knowledge and so should be easier to adapt for NVS. Our key insight is to formulate sparse NVS as a low frame-rate video completion task. However, one challenge is that sparse NVS is defined over an unordered set of inputs, often too sparse to admit a meaningful order, so the models should be $\textit{invariant}$ to permutations of that input set. To this end, we present FrameCrafter, which adapts video models (naturally trained with coherent frame orderings) to permutation-invariant NVS through several architectural modifications, including per-frame latent encodings and removal of temporal positional embeddings. Our results suggest that video models can be easily trained to "forget" about time with minimal supervision, producing competitive performance on sparse-view NVS benchmarks. Project page: https://frame-crafter.github.io/

  </details>



- **State and Trajectory Estimation of Tensegrity Robots via Factor Graphs and Chebyshev Polynomials**  
  Edgar Granados, Patrick Meng, Charles Tang, Shrimed Sangani, William R. Johnson, Rebecca Kramer-Bottiglio, Kostas Bekris  
  _2026-04-09_ · https://arxiv.org/abs/2604.08185v1  
  <details><summary>Abstract</summary>

  Tensegrity robots offer compliance and adaptability, but their nonlinear, and underconstrained dynamics make state estimation challenging. Reliable continuous-time estimation of all rigid links is crucial for closed-loop control, system identification, and machine learning; however, conventional methods often fall short. This paper proposes a two-stage approach for robust state or trajectory estimation (i.e., filtering or smoothing) of a cable-driven tensegrity robot. For online state estimation, this work introduces a factor-graph-based method, which fuses measurements from an RGB-D camera with on-board cable length sensors. To the best of the authors' knowledge, this is the first application of factor graphs in this domain. Factor graphs are a natural choice, as they exploit the robot's structural properties and provide effective sensor fusion solutions capable of handling nonlinearities in practice. Both the Mahalanobis distance-based clustering algorithm, used to handle noise, and the Chebyshev polynomial method, used to estimate the most probable velocities and intermediate states, are shown to perform well on simulated and real-world data, compared to an ICP-based algorithm. Results show that the approach provides high fidelity, continuous-time state and trajectory estimates for complex tensegrity robot motions.

  </details>



- **MotionScape: A Large-Scale Real-World Highly Dynamic UAV Video Dataset for World Models**  
  Zile Guo, Zhan Chen, Enze Zhu, Kan Wei, Yongkang Zou, Xiaoxuan Liu, Lei Wang  
  _2026-04-09_ · https://arxiv.org/abs/2604.07991v1  
  <details><summary>Abstract</summary>

  Recent advances in world models have demonstrated strong capabilities in simulating physical reality, making them an increasingly important foundation for embodied intelligence. For UAV agents in particular, accurate prediction of complex 3D dynamics is essential for autonomous navigation and robust decision-making in unconstrained environments. However, under the highly dynamic camera trajectories typical of UAV views, existing world models often struggle to maintain spatiotemporal physical consistency. A key reason lies in the distribution bias of current training data: most existing datasets exhibit restricted 2.5D motion patterns, such as ground-constrained autonomous driving scenes or relatively smooth human-centric egocentric videos, and therefore lack realistic high-dynamic 6-DoF UAV motion priors. To address this gap, we present MotionScape, a large-scale real-world UAV-view video dataset with highly dynamic motion for world modeling. MotionScape contains over 30 hours of 4K UAV-view videos, totaling more than 4.5M frames. This novel dataset features semantically and geometrically aligned training samples, where diverse real-world UAV videos are tightly coupled with accurate 6-DoF camera trajectories and fine-grained natural language descriptions. To build the dataset, we develop an automated multi-stage processing pipeline that integrates CLIP-based relevance filtering, temporal segmentation, robust visual SLAM for trajectory recovery, and large-language-model-driven semantic annotation. Extensive experiments show that incorporating such semantically and geometrically aligned annotations effectively improves the ability of existing world models to simulate complex 3D dynamics and handle large viewpoint shifts, thereby benefiting decision-making and planning for UAV agents in complex environments. The dataset is publicly available at https://github.com/Thelegendzz/MotionScape

  </details>



- **RoSHI: A Versatile Robot-oriented Suit for Human Data In-the-Wild**  
  Wenjing Margaret Mao, Jefferson Ng, Luyang Hu, Daniel Gehrig, Antonio Loquercio  
  _2026-04-08_ · https://arxiv.org/abs/2604.07331v1  
  <details><summary>Abstract</summary>

  Scaling up robot learning will likely require human data containing rich and long-horizon interactions in the wild. Existing approaches for collecting such data trade off portability, robustness to occlusion, and global consistency. We introduce RoSHI, a hybrid wearable that fuses low-cost sparse IMUs with the Project Aria glasses to estimate the full 3D pose and body shape of the wearer in a metric global coordinate frame from egocentric perception. This system is motivated by the complementarity of the two sensors: IMUs provide robustness to occlusions and high-speed motions, while egocentric SLAM anchors long-horizon motion and stabilizes upper body pose. We collect a dataset of agile activities to evaluate RoSHI. On this dataset, we generally outperform other egocentric baselines and perform comparably to a state-of-the-art exocentric baseline (SAM3D). Finally, we demonstrate that the motion data recorded from our system are suitable for real-world humanoid policy learning. For videos, data and more, visit the project webpage: https://roshi-mocap.github.io/

  </details>



- **VGGT-SLAM++**  
  Avilasha Mandal, Rajesh Kumar, Sudarshan Sunil Harithas, Chetan Arora  
  _2026-04-08_ · https://arxiv.org/abs/2604.06830v1  
  <details><summary>Abstract</summary>

  We introduce VGGT-SLAM++, a complete visual SLAM system that leverages the geometry-rich outputs of the Visual Geometry Grounded Transformer (VGGT). The system comprises a visual odometry (front-end) fusing the VGGT feed-forward transformer and a Sim(3) solution, a Digital Elevation Map (DEM)-based graph construction module, and a back-end that jointly enable accurate large-scale mapping with bounded memory. While prior transformer-based SLAM pipelines such as VGGT-SLAM rely primarily on sparse loop closures or global Sim(3) manifold constraints - allowing short-horizon pose drift - VGGT-SLAM++ restores high-cadence local bundle adjustment (LBA) through a spatially corrective back-end. For each VGGT submap, we construct a dense planar-canonical DEM, partition it into patches, and compute their DINOv2 embeddings to integrate the submap into a covisibility graph. Spatial neighbors are retrieved using a Visual Place Recognition (VPR) module within the covisibility window, triggering frequent local optimization that stabilizes trajectories. Across standard SLAM benchmarks, VGGT-SLAM++ achieves state-of-the-art accuracy, substantially reducing short-term drift, accelerating graph convergence, and maintaining global consistency with compact DEM tiles and sublinear retrieval.

  </details>



- **LiveStre4m: Feed-Forward Live Streaming of Novel Views from Unposed Multi-View Video**  
  Pedro Quesado, Erkut Akdag, Yasaman Kashefbahrami, Willem Menu, Egor Bondarev  
  _2026-04-08_ · https://arxiv.org/abs/2604.06740v1  
  <details><summary>Abstract</summary>

  Live-streaming Novel View Synthesis (NVS) from unposed multi-view video remains an open challenge in a wide range of applications. Existing methods for dynamic scene representation typically require ground-truth camera parameters and involve lengthy optimizations ($\approx 2.67$s), which makes them unsuitable for live streaming scenarios. To address this issue, we propose a novel viewpoint video live-streaming method (LiveStre4m), a feed-forward model for real-time NVS from unposed sparse multi-view inputs. LiveStre4m introduces a multi-view vision transformer for keyframe 3D scene reconstruction coupled with a diffusion-transformer interpolation module that ensures temporal consistency and stable streaming. In addition, a Camera Pose Predictor module is proposed to efficiently estimate both poses and intrinsics directly from RGB images, removing the reliance on known camera calibration information. Our approach enables temporally consistent novel-view video streaming in real-time using as few as two synchronized unposed input streams. LiveStre4m attains an average reconstruction time of $ 0.07$s per-frame at $ 1024 \times 768$ resolution, outperforming the optimization-based dynamic scene representation methods by orders of magnitude in runtime. These results demonstrate that LiveStre4m makes real-time NVS streaming feasible in practical settings, marking a substantial step toward deployable live novel-view synthesis systems. Code available at: https://github.com/pedro-quesado/LiveStre4m

  </details>



- **Exploring 6D Object Pose Estimation with Deformation**  
  Zhiqiang Liu, Rui Song, Duanmu Chuangqi, Jiaojiao Li, David Ferstl, Yinlin Hu  
  _2026-04-08_ · https://arxiv.org/abs/2604.06720v1  
  <details><summary>Abstract</summary>

  We present DeSOPE, a large-scale dataset for 6DoF deformed objects. Most 6D object pose methods assume rigid or articulated objects, an assumption that fails in practice as objects deviate from their canonical shapes due to wear, impact, or deformation. To model this, we introduce the DeSOPE dataset, which features high-fidelity 3D scans of 26 common object categories, each captured in one canonical state and three deformed configurations, with accurate 3D registration to the canonical mesh. Additionally, it features an RGB-D dataset with 133K frames across diverse scenarios and 665K pose annotations produced via a semi-automatic pipeline. We begin by annotating 2D masks for each instance, then compute initial poses using an object pose method, refine them through an object-level SLAM system, and finally perform manual verification to produce the final annotations. We evaluate several object pose methods and find that performance drops sharply with increasing deformation, suggesting that robust handling of such deformations is critical for practical applications. The project page and dataset are available at https://desope-6d.github.io/}{https://desope-6d.github.io/.

  </details>


