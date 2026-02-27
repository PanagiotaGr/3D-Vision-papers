# SLAM & Localization

_Updated: 2026-02-27 07:10 UTC_

Total papers shown: **15**


---

- **VGG-T$^3$: Offline Feed-Forward 3D Reconstruction at Scale**  
  Sven Elflein, Ruilong Li, Sérgio Agostinho, Zan Gojcic, Laura Leal-Taixé, Qunjie Zhou, Aljosa Osep  
  _2026-02-26_ · https://arxiv.org/abs/2602.23361v1  
  <details><summary>Abstract</summary>

  We present a scalable 3D reconstruction model that addresses a critical limitation in offline feed-forward methods: their computational and memory requirements grow quadratically w.r.t. the number of input images. Our approach is built on the key insight that this bottleneck stems from the varying-length Key-Value (KV) space representation of scene geometry, which we distill into a fixed-size Multi-Layer Perceptron (MLP) via test-time training. VGG-T$^3$ (Visual Geometry Grounded Test Time Training) scales linearly w.r.t. the number of input views, similar to online models, and reconstructs a $1k$ image collection in just $54$ seconds, achieving a $11.6\times$ speed-up over baselines that rely on softmax attention. Since our method retains global scene aggregation capability, our point map reconstruction error outperforming other linear-time methods by large margins. Finally, we demonstrate visual localization capabilities of our model by querying the scene representation with unseen images.

  </details>



- **UniScale: Unified Scale-Aware 3D Reconstruction for Multi-View Understanding via Prior Injection for Robotic Perception**  
  Mohammad Mahdavian, Gordon Tan, Binbin Xu, Yuan Ren, Dongfeng Bai, Bingbing Liu  
  _2026-02-26_ · https://arxiv.org/abs/2602.23224v1  
  <details><summary>Abstract</summary>

  We present UniScale, a unified, scale-aware multi-view 3D reconstruction framework for robotic applications that flexibly integrates geometric priors through a modular, semantically informed design. In vision-based robotic navigation, the accurate extraction of environmental structure from raw image sequences is critical for downstream tasks. UniScale addresses this challenge with a single feed-forward network that jointly estimates camera intrinsics and extrinsics, scale-invariant depth and point maps, and the metric scale of a scene from multi-view images, while optionally incorporating auxiliary geometric priors when available. By combining global contextual reasoning with camera-aware feature representations, UniScale is able to recover the metric-scale of the scene. In robotic settings where camera intrinsics are known, they can be easily incorporated to improve performance, with additional gains obtained when camera poses are also available. This co-design enables robust, metric-aware 3D reconstruction within a single unified model. Importantly, UniScale does not require training from scratch, and leverages world priors exhibited in pre-existing models without geometric encoding strategies, making it particularly suitable for resource-constrained robotic teams. We evaluate UniScale on multiple benchmarks, demonstrating strong generalization and consistent performance across diverse environments. We will release our implementation upon acceptance.

  </details>



- **Motion-aware Event Suppression for Event Cameras**  
  Roberto Pellerito, Nico Messikommer, Giovanni Cioffi, Marco Cannici, Davide Scaramuzza  
  _2026-02-26_ · https://arxiv.org/abs/2602.23204v1  
  <details><summary>Abstract</summary>

  In this work, we introduce the first framework for Motion-aware Event Suppression, which learns to filter events triggered by IMOs and ego-motion in real time. Our model jointly segments IMOs in the current event stream while predicting their future motion, enabling anticipatory suppression of dynamic events before they occur. Our lightweight architecture achieves 173 Hz inference on consumer-grade GPUs with less than 1 GB of memory usage, outperforming previous state-of-the-art methods on the challenging EVIMO benchmark by 67\% in segmentation accuracy while operating at a 53\% higher inference rate. Moreover, we demonstrate significant benefits for downstream applications: our method accelerates Vision Transformer inference by 83\% via token pruning and improves event-based visual odometry accuracy, reducing Absolute Trajectory Error (ATE) by 13\%.

  </details>



- **FLIGHT: Fibonacci Lattice-based Inference for Geometric Heading in real-Time**  
  David Dirnfeld, Fabien Delattre, Pedro Miraldo, Erik Learned-Miller  
  _2026-02-26_ · https://arxiv.org/abs/2602.23115v1  
  <details><summary>Abstract</summary>

  Estimating camera motion from monocular video is a fundamental problem in computer vision, central to tasks such as SLAM, visual odometry, and structure-from-motion. Existing methods that recover the camera's heading under known rotation, whether from an IMU or an optimization algorithm, tend to perform well in low-noise, low-outlier conditions, but often decrease in accuracy or become computationally expensive as noise and outlier levels increase. To address these limitations, we propose a novel generalization of the Hough transform on the unit sphere (S(2)) to estimate the camera's heading. First, the method extracts correspondences between two frames and generates a great circle of directions compatible with each pair of correspondences. Then, by discretizing the unit sphere using a Fibonacci lattice as bin centers, each great circle casts votes for a range of directions, ensuring that features unaffected by noise or dynamic objects vote consistently for the correct motion direction. Experimental results on three datasets demonstrate that the proposed method is on the Pareto frontier of accuracy versus efficiency. Additionally, experiments on SLAM show that the proposed method reduces RMSE by correcting the heading during camera pose initialization.

  </details>



- **Sapling-NeRF: Geo-Localised Sapling Reconstruction in Forests for Ecological Monitoring**  
  Miguel Ángel Muñoz-Bañón, Nived Chebrolu, Sruthi M. Krishna Moorthy, Yifu Tao, Fernando Torres, Roberto Salguero-Gómez, Maurice Fallon  
  _2026-02-26_ · https://arxiv.org/abs/2602.22731v1  
  <details><summary>Abstract</summary>

  Saplings are key indicators of forest regeneration and overall forest health. However, their fine-scale architectural traits are difficult to capture with existing 3D sensing methods, which make quantitative evaluation difficult. Terrestrial Laser Scanners (TLS), Mobile Laser Scanners (MLS), or traditional photogrammetry approaches poorly reconstruct thin branches, dense foliage, and lack the scale consistency needed for long-term monitoring. Implicit 3D reconstruction methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) are promising alternatives, but cannot recover the true scale of a scene and lack any means to be accurately geo-localised. In this paper, we present a pipeline which fuses NeRF, LiDAR SLAM, and GNSS to enable repeatable, geo-localised ecological monitoring of saplings. Our system proposes a three-level representation: (i) coarse Earth-frame localisation using GNSS, (ii) LiDAR-based SLAM for centimetre-accurate localisation and reconstruction, and (iii) NeRF-derived object-centric dense reconstruction of individual saplings. This approach enables repeatable quantitative evaluation and long-term monitoring of sapling traits. Our experiments in forest plots in Wytham Woods (Oxford, UK) and Evo (Finland) show that stem height, branching patterns, and leaf-to-wood ratios can be captured with increased accuracy as compared to TLS. We demonstrate that accurate stem skeletons and leaf distributions can be measured for saplings with heights between 0.5m and 2m in situ, giving ecologists access to richer structural and quantitative data for analysing forest dynamics.

  </details>



- **GIFSplat: Generative Prior-Guided Iterative Feed-Forward 3D Gaussian Splatting from Sparse Views**  
  Tianyu Chen, Wei Xiang, Kang Han, Yu Lu, Di Wu, Gaowen Liu, Ramana Rao Kompella  
  _2026-02-26_ · https://arxiv.org/abs/2602.22571v1  
  <details><summary>Abstract</summary>

  Feed-forward 3D reconstruction offers substantial runtime advantages over per-scene optimization, which remains slow at inference and often fragile under sparse views. However, existing feed-forward methods still have potential for further performance gains, especially for out-of-domain data, and struggle to retain second-level inference time once a generative prior is introduced. These limitations stem from the one-shot prediction paradigm in existing feed-forward pipeline: models are strictly bounded by capacity, lack inference-time refinement, and are ill-suited for continuously injecting generative priors. We introduce GIFSplat, a purely feed-forward iterative refinement framework for 3D Gaussian Splatting from sparse unposed views. A small number of forward-only residual updates progressively refine current 3D scene using rendering evidence, achieve favorable balance between efficiency and quality. Furthermore, we distill a frozen diffusion prior into Gaussian-level cues from enhanced novel renderings without gradient backpropagation or ever-increasing view-set expansion, thereby enabling per-scene adaptation with generative prior while preserving feed-forward efficiency. Across DL3DV, RealEstate10K, and DTU, GIFSplat consistently outperforms state-of-the-art feed-forward baselines, improving PSNR by up to +2.1 dB, and it maintains second-scale inference time without requiring camera poses or any test-time gradient optimization.

  </details>



- **Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos**  
  Matthew Strong, Wei-Jer Chang, Quentin Herau, Jiezhi Yang, Yihan Hu, Chensheng Peng, Wei Zhan  
  _2026-02-25_ · https://arxiv.org/abs/2602.22091v1  
  <details><summary>Abstract</summary>

  Ego-centric driving videos available online provide an abundant source of visual data for autonomous driving, yet their lack of annotations makes it difficult to learn representations that capture both semantic structure and 3D geometry. Recent advances in large feedforward spatial models demonstrate that point maps and ego-motion can be inferred in a single forward pass, suggesting a promising direction for scalable driving perception. We therefore propose a label-free, teacher-guided framework for learning autonomous driving representations directly from unposed videos. Unlike prior self-supervised approaches that focus primarily on frame-to-frame consistency, we posit that safe and reactive driving depends critically on temporal context. To this end, we leverage a feedforward architecture equipped with a lightweight autoregressive module, trained using multi-modal supervisory signals that guide the model to jointly predict current and future point maps, camera poses, semantic segmentation, and motion masks. Multi-modal teachers provide sequence-level pseudo-supervision, enabling LFG to learn a unified pseudo-4D representation from raw YouTube videos without poses, labels, or LiDAR. The resulting encoder not only transfers effectively to downstream autonomous driving planning on the NAVSIM benchmark, surpassing multi-camera and LiDAR baselines with only a single monocular camera, but also yields strong performance when evaluated on a range of semantic, geometric, and qualitative motion prediction tasks. These geometry and motion-aware features position LFG as a compelling video-centric foundation model for autonomous driving.

  </details>



- **Parallel Continuous-Time Relative Localization with Augmented Clamped Non-Uniform B-Splines**  
  Jiadong Lu, Zhehan Li, Tao Han, Miao Xu, Chao Xu, Yanjun Cao  
  _2026-02-25_ · https://arxiv.org/abs/2602.22006v1  
  <details><summary>Abstract</summary>

  Accurate relative localization is critical for multi-robot cooperation. In robot swarms, measurements from different robots arrive asynchronously and with clock time-offsets. Although Continuous-Time (CT) formulations have proved effective for handling asynchronous measurements in single-robot SLAM and calibration, extending CT methods to multi-robot settings faces great challenges to achieve high-accuracy, low-latency, and high-frequency performance. Especially, existing CT methods suffer from the inherent query-time delay of unclamped B-splines and high computational cost. This paper proposes CT-RIO, a novel Continuous-Time Relative-Inertial Odometry framework. We employ Clamped Non-Uniform B-splines (C-NUBS) to represent robot states for the first time, eliminating the query-time delay. We further augment C-NUBS with closed-form extension and shrinkage operations that preserve the spline shape, making it suitable for online estimation and enabling flexible knot management. This flexibility leads to the concept of knot-keyknot strategy, which supports spline extension at high-frequency while retaining sparse keyknots for adaptive relative-motion modeling. We then formulate a sliding-window relative localization problem that operates purely on relative kinematics and inter-robot constraints. To meet the demanding computation required at swarm scale, we decompose the tightly-coupled optimization into robot-wise sub-problems and solve them in parallel using incremental asynchronous block coordinate descent. Extensive experiments show that CT-RIO converges from time-offsets as large as 263 ms to sub-millisecond within 3 s, and achieves RMSEs of 0.046 m and 1.8 °. It consistently outperforms state-of-the-art methods, with improvements of up to 60% under high-speed motion.

  </details>



- **Dream-SLAM: Dreaming the Unseen for Active SLAM in Dynamic Environments**  
  Xiangqi Meng, Pengxu Hou, Zhenjun Zhao, Javier Civera, Daniel Cremers, Hesheng Wang, Haoang Li  
  _2026-02-25_ · https://arxiv.org/abs/2602.21967v1  
  <details><summary>Abstract</summary>

  In addition to the core tasks of simultaneous localization and mapping (SLAM), active SLAM additionally in- volves generating robot actions that enable effective and efficient exploration of unknown environments. However, existing active SLAM pipelines are limited by three main factors. First, they inherit the restrictions of the underlying SLAM modules that they may be using. Second, their motion planning strategies are typically shortsighted and lack long-term vision. Third, most approaches struggle to handle dynamic scenes. To address these limitations, we propose a novel monocular active SLAM method, Dream-SLAM, which is based on dreaming cross-spatio-temporal images and semantically plausible structures of partially observed dynamic environments. The generated cross-spatio-temporal im- ages are fused with real observations to mitigate noise and data incompleteness, leading to more accurate camera pose estimation and a more coherent 3D scene representation. Furthermore, we integrate dreamed and observed scene structures to enable long- horizon planning, producing farsighted trajectories that promote efficient and thorough exploration. Extensive experiments on both public and self-collected datasets demonstrate that Dream-SLAM outperforms state-of-the-art methods in localization accuracy, mapping quality, and exploration efficiency. Source code will be publicly available upon paper acceptance.

  </details>



- **Global-Aware Edge Prioritization for Pose Graph Initialization**  
  Tong Wei, Giorgos Tolias, Jiri Matas, Daniel Barath  
  _2026-02-25_ · https://arxiv.org/abs/2602.21963v1  
  <details><summary>Abstract</summary>

  The pose graph is a core component of Structure-from-Motion (SfM), where images act as nodes and edges encode relative poses. Since geometric verification is expensive, SfM pipelines restrict the pose graph to a sparse set of candidate edges, making initialization critical. Existing methods rely on image retrieval to connect each image to its $k$ nearest neighbors, treating pairs independently and ignoring global consistency. We address this limitation through the concept of edge prioritization, ranking candidate edges by their utility for SfM. Our approach has three components: (1) a GNN trained with SfM-derived supervision to predict globally consistent edge reliability; (2) multi-minimal-spanning-tree-based pose graph construction guided by these ranks; and (3) connectivity-aware score modulation that reinforces weak regions and reduces graph diameter. This globally informed initialization yields more reliable and compact pose graphs, improving reconstruction accuracy in sparse and high-speed settings and outperforming SOTA retrieval methods on ambiguous scenes. The ode and trained models are available at https://github.com/weitong8591/global_edge_prior.

  </details>



- **Geometry-as-context: Modulating Explicit 3D in Scene-consistent Video Generation to Geometry Context**  
  JiaKui Hu, Jialun Liu, Liying Yang, Xinliang Zhang, Kaiwen Li, Shuang Zeng, Yuanwei Li, Haibin Huang, Chi Zhang, Yanye Lu  
  _2026-02-25_ · https://arxiv.org/abs/2602.21929v1  
  <details><summary>Abstract</summary>

  Scene-consistent video generation aims to create videos that explore 3D scenes based on a camera trajectory. Previous methods rely on video generation models with external memory for consistency, or iterative 3D reconstruction and inpainting, which accumulate errors during inference due to incorrect intermediary outputs, non-differentiable processes, and separate models. To overcome these limitations, we introduce ``geometry-as-context". It iteratively completes the following steps using an autoregressive camera-controlled video generation model: (1) estimates the geometry of the current view necessary for 3D reconstruction, and (2) simulates and restores novel view images rendered by the 3D scene. Under this multi-task framework, we develop the camera gated attention module to enhance the model's capability to effectively leverage camera poses. During the training phase, text contexts are utilized to ascertain whether geometric or RGB images should be generated. To ensure that the model can generate RGB-only outputs during inference, the geometry context is randomly dropped from the interleaved text-image-geometry training sequence. The method has been tested on scene video generation with one-direction and forth-and-back trajectories. The results show its superiority over previous approaches in maintaining scene consistency and camera control.

  </details>



- **GeoMotion: Rethinking Motion Segmentation via Latent 4D Geometry**  
  Xiankang He, Peile Lin, Ying Cui, Dongyan Guo, Chunhua Shen, Xiaoqin Zhang  
  _2026-02-25_ · https://arxiv.org/abs/2602.21810v1  
  <details><summary>Abstract</summary>

  Motion segmentation in dynamic scenes is highly challenging, as conventional methods heavily rely on estimating camera poses and point correspondences from inherently noisy motion cues. Existing statistical inference or iterative optimization techniques that struggle to mitigate the cumulative errors in multi-stage pipelines often lead to limited performance or high computational cost. In contrast, we propose a fully learning-based approach that directly infers moving objects from latent feature representations via attention mechanisms, thus enabling end-to-end feed-forward motion segmentation. Our key insight is to bypass explicit correspondence estimation and instead let the model learn to implicitly disentangle object and camera motion. Supported by recent advances in 4D scene geometry reconstruction (e.g., $π^3$), the proposed method leverages reliable camera poses and rich spatial-temporal priors, which ensure stable training and robust inference for the model. Extensive experiments demonstrate that by eliminating complex pre-processing and iterative refinement, our approach achieves state-of-the-art motion segmentation performance with high efficiency. The code is available at:https://github.com/zjutcvg/GeoMotion.

  </details>



- **LiREC-Net: A Target-Free and Learning-Based Network for LiDAR, RGB, and Event Calibration**  
  Aditya Ranjan Dash, Ramy Battrawy, René Schuster, Didier Stricker  
  _2026-02-25_ · https://arxiv.org/abs/2602.21754v1  
  <details><summary>Abstract</summary>

  Advanced autonomous systems rely on multi-sensor fusion for safer and more robust perception. To enable effective fusion, calibrating directly from natural driving scenes (i.e., target-free) with high accuracy is crucial for precise multi-sensor alignment. Existing learning-based calibration methods are typically designed for only a single pair of sensor modalities (i.e., a bi-modal setup). Unlike these methods, we propose LiREC-Net, a target-free, learning-based calibration network that jointly calibrates multiple sensor modality pairs, including LiDAR, RGB, and event data, within a unified framework. To reduce redundant computation and improve efficiency, we introduce a shared LiDAR representation that leverages features from both its 3D nature and projected depth map, ensuring better consistency across modalities. Trained and evaluated on established datasets, such as KITTI and DSEC, our LiREC-Net achieves competitive performance to bi-modal models and sets a new strong baseline for the tri-modal use case.

  </details>



- **DAGS-SLAM: Dynamic-Aware 3DGS SLAM via Spatiotemporal Motion Probability and Uncertainty-Aware Scheduling**  
  Li Zhang, Yu-An Liu, Xijia Jiang, Conghao Huang, Danyang Li, Yanyong Zhang  
  _2026-02-25_ · https://arxiv.org/abs/2602.21644v1  
  <details><summary>Abstract</summary>

  Mobile robots and IoT devices demand real-time localization and dense reconstruction under tight compute and energy budgets. While 3D Gaussian Splatting (3DGS) enables efficient dense SLAM, dynamic objects and occlusions still degrade tracking and mapping. Existing dynamic 3DGS-SLAM often relies on heavy optical flow and per-frame segmentation, which is costly for mobile deployment and brittle under challenging illumination. We present DAGS-SLAM, a dynamic-aware 3DGS-SLAM system that maintains a spatiotemporal motion probability (MP) state per Gaussian and triggers semantics on demand via an uncertainty-aware scheduler. DAGS-SLAM fuses lightweight YOLO instance priors with geometric cues to estimate and temporally update MP, propagates MP to the front-end for dynamic-aware correspondence selection, and suppresses dynamic artifacts in the back-end via MP-guided optimization. Experiments on public dynamic RGB-D benchmarks show improved reconstruction and robust tracking while sustaining real-time throughput on a commodity GPU, demonstrating a practical speed-accuracy tradeoff with reduced semantic invocations toward mobile deployment.

  </details>



- **Automatic Map Density Selection for Locally-Performant Visual Place Recognition**  
  Somayeh Hussaini, Tobias Fischer, Michael Milford  
  _2026-02-25_ · https://arxiv.org/abs/2602.21473v1  
  <details><summary>Abstract</summary>

  A key challenge in translating Visual Place Recognition (VPR) from the lab to long-term deployment is ensuring a priori that a system can meet user-specified performance requirements across different parts of an environment, rather than just on average globally. A critical mechanism for controlling local VPR performance is the density of the reference mapping database, yet this factor is largely neglected in existing work, where benchmark datasets with fixed, engineering-driven (sensors, storage, GPS frequency) sampling densities are typically used. In this paper, we propose a dynamic VPR mapping approach that uses pairs of reference traverses from the target environment to automatically select an appropriate map density to satisfy two user-defined requirements: (1) a target Local Recall@1 level, and (2) the proportion of the operational environment over which this requirement must be met or exceeded, which we term the Recall Achievement Rate (RAR). Our approach is based on the hypothesis that match patterns between multiple reference traverses, evaluated across different map densities, can be modelled to predict the density required to meet these performance targets on unseen deployment data. Through extensive experiments across multiple VPR methods and the Nordland and Oxford RobotCar benchmarks, we show that our system consistently achieves or exceeds the specified local recall level over at least the user-specified proportion of the environment. Comparisons with alternative baselines demonstrate that our approach reliably selects the correct operating point in map density, avoiding unnecessary over-densification. Finally, ablation studies and analysis evaluate sensitivity to reference map choice and local space definitions, and reveal that conventional global Recall@1 is a poor predictor of the often more operationally meaningful RAR metric.

  </details>


