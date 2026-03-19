# SLAM & Localization

_Updated: 2026-03-19 07:12 UTC_

Total papers shown: **12**


---

- **Loc3R-VLM: Language-based Localization and 3D Reasoning with Vision-Language Models**  
  Kevin Qu, Haozhe Qi, Mihai Dusmanu, Mahdi Rad, Rui Wang, Marc Pollefeys  
  _2026-03-18_ · https://arxiv.org/abs/2603.18002v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have made impressive progress in connecting vision and language, but they still struggle with spatial understanding and viewpoint-aware reasoning. Recent efforts aim to augment the input representations with geometric cues rather than explicitly teaching models to reason in 3D space. We introduce Loc3R-VLM, a framework that equips 2D Vision-Language Models with advanced 3D understanding capabilities from monocular video input. Inspired by human spatial cognition, Loc3R-VLM relies on two joint objectives: global layout reconstruction to build a holistic representation of the scene structure, and explicit situation modeling to anchor egocentric perspective. These objectives provide direct spatial supervision that grounds both perception and language in a 3D context. To ensure geometric consistency and metric-scale alignment, we leverage lightweight camera pose priors extracted from a pre-trained 3D foundation model. Loc3R-VLM achieves state-of-the-art performance in language-based localization and outperforms existing 2D- and video-based approaches on situated and general 3D question-answering benchmarks, demonstrating that our spatial supervision framework enables strong 3D understanding. Project page: https://kevinqu7.github.io/loc3r-vlm

  </details>



- **PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery**  
  Yijing Guo, Mengjun Chao, Luo Wang, Tianyang Zhao, Haizhao Dai, Yingliang Zhang, Jingyi Yu, Yujiao Shi  
  _2026-03-18_ · https://arxiv.org/abs/2603.17571v1  
  <details><summary>Abstract</summary>

  Panoramic imagery offers a full 360° field of view and is increasingly common in consumer devices. However, it introduces non-pinhole distortions that challenge joint pose estimation and 3D reconstruction. Existing feed-forward models, built for perspective cameras, generalize poorly to this setting. We propose PanoVGGT, a permutation-equivariant Transformer framework that jointly predicts camera poses, depth maps, and 3D point clouds from one or multiple panoramas in a single forward pass. The model incorporates spherical-aware positional embeddings and a panorama-specific three-axis SO(3) rotation augmentation, enabling effective geometric reasoning in the spherical domain. To resolve inherent global-frame ambiguity, we further introduce a stochastic anchoring strategy during training. In addition, we contribute PanoCity, a large-scale outdoor panoramic dataset with dense depth and 6-DoF pose annotations. Extensive experiments on PanoCity and standard benchmarks demonstrate that PanoVGGT achieves competitive accuracy, strong robustness, and improved cross-domain generalization. Code and dataset will be released.

  </details>



- **Physics-informed Deep Mixture-of-Koopmans Vehicle Dynamics Model with Dual-branch Encoder for Distributed Electric-drive Trucks**  
  Jinyu Miao, Pu Zhang, Rujun Yan, Yifei He, Bowei Zhang, Zheng Fu, Ke Wang, Qi Song, Kun Jiang, Mengmeng Yang, et al.  
  _2026-03-18_ · https://arxiv.org/abs/2603.17416v1  
  <details><summary>Abstract</summary>

  Advanced autonomous driving systems require accurate vehicle dynamics modeling. However, identifying a precise dynamics model remains challenging due to strong nonlinearities and the coupled longitudinal and lateral dynamic characteristics. Previous research has employed physics-based analytical models or neural networks to construct vehicle dynamics representations. Nevertheless, these approaches often struggle to simultaneously achieve satisfactory performance in terms of system identification efficiency, modeling accuracy, and compatibility with linear control strategies. In this paper, we propose a fully data-driven dynamics modeling method tailored for complex distributed electric-drive trucks (DETs), leveraging Koopman operator theory to represent highly nonlinear dynamics in a lifted linear embedding space. To achieve high-precision modeling, we first propose a novel dual-branch encoder which encodes dynamic states and provides a powerful basis for the proposed Koopman-based methods entitled KODE. A physics-informed supervision mechanism, grounded in the geometric consistency of temporal vehicle motion, is incorporated into the training process to facilitate effective learning of both the encoder and the Koopman operator. Furthermore, to accommodate the diverse driving patterns of DETs, we extend the vanilla Koopman operator to a mixture-of-Koopman operator framework, enhancing modeling capability. Simulations conducted in a high-fidelity TruckSim environment and real-world experiments demonstrate that the proposed approach achieves state-of-the-art performance in long-term dynamics state estimation.

  </details>



- **OnlineHMR: Video-based Online World-Grounded Human Mesh Recovery**  
  Yiwen Zhao, Ce Zheng, Yufu Wang, Hsueh-Han Daniel Yang, Liting Wen, Laszlo A. Jeni  
  _2026-03-18_ · https://arxiv.org/abs/2603.17355v1  
  <details><summary>Abstract</summary>

  Human mesh recovery (HMR) models 3D human body from monocular videos, with recent works extending it to world-coordinate human trajectory and motion reconstruction. However, most existing methods remain offline, relying on future frames or global optimization, which limits their applicability in interactive feedback and perception-action loop scenarios such as AR/VR and telepresence. To address this, we propose OnlineHMR, a fully online framework that jointly satisfies four essential criteria of online processing, including system-level causality, faithfulness, temporal consistency, and efficiency. Built upon a two-branch architecture, OnlineHMR enables streaming inference via a causal key-value cache design and a curated sliding-window learning strategy. Meanwhile, a human-centric incremental SLAM provides online world-grounded alignment under physically plausible trajectory correction. Experimental results show that our method achieves performance comparable to existing chunk-based approaches on the standard EMDB benchmark and highly dynamic custom videos, while uniquely supporting online processing. Page and code are available at https://tsukasane.github.io/Video-OnlineHMR/.

  </details>



- **Full Stack Navigation, Mapping, and Planning for the Lunar Autonomy Challenge**  
  Adam Dai, Asta Wu, Keidai Iiyama, Guillem Casadesus Vila, Kaila Coimbra, Thomas Deng, Grace Gao  
  _2026-03-18_ · https://arxiv.org/abs/2603.17232v1  
  <details><summary>Abstract</summary>

  We present a modular, full-stack autonomy system for lunar surface navigation and mapping developed for the Lunar Autonomy Challenge. Operating in a GNSS-denied, visually challenging environment, our pipeline integrates semantic segmentation, stereo visual odometry, pose graph SLAM with loop closures, and layered planning and control. We leverage lightweight learning-based perception models for real-time segmentation and feature tracking and use a factor-graph backend to maintain globally consistent localization. High-level waypoint planning is designed to promote mapping coverage while encouraging frequent loop closures, and local motion planning uses arc sampling with geometric obstacle checks for efficient, reactive control. We evaluate our approach in the competition's high-fidelity lunar simulator, demonstrating centimeter-level localization accuracy, high-fidelity map generation, and strong repeatability across random seeds and rock distributions. Our solution achieved first place in the final competition evaluation.

  </details>



- **Visual SLAM with DEM Anchoring for Lunar Surface Navigation**  
  Adam Dai, Guillem Casadesus Vila, Grace Gao  
  _2026-03-18_ · https://arxiv.org/abs/2603.17229v1  
  <details><summary>Abstract</summary>

  Future lunar missions will require autonomous rovers capable of traversing tens of kilometers across challenging terrain while maintaining accurate localization and producing globally consistent maps. However, the absence of global positioning systems, extreme illumination, and low-texture regolith make long-range navigation on the Moon particularly difficult, as visual-inertial odometry pipelines accumulate drift over extended traverses. To address this challenge, we present a stereo visual simultaneous localization and mapping (SLAM) system that integrates learned feature detection and matching with global constraints from digital elevation models (DEMs). Our front-end employs learning-based feature extraction and matching to achieve robustness to illumination extremes and repetitive terrain, while the back-end incorporates DEM-derived height and surface-normal factors into a pose graph, providing absolute surface constraints that mitigate long-term drift. We validate our approach using both simulated lunar traverse data generated in Unreal Engine and real Moon/Mars analog data collected from Mt. Etna. Results demonstrate that DEM anchoring consistently reduces absolute trajectory error compared to baseline SLAM methods, lowering drift in long-range navigation even in repetitive or visually aliased terrain.

  </details>



- **FastLoop: Parallel Loop Closing with GPU-Acceleration in Visual SLAM**  
  Soudabeh Mohammadhashemi, Shishir Gopinath, Kimia Khabiri, Parsa Hosseininejad, Karthik Dantu, Steven Y. Ko  
  _2026-03-17_ · https://arxiv.org/abs/2603.17201v1  
  <details><summary>Abstract</summary>

  Visual SLAM systems combine visual tracking with global loop closure to maintain a consistent map and accurate localization. Loop closure is a computationally expensive process as we need to search across the whole map for matches. This paper presents FastLoop, a GPU-accelerated loop closing module to alleviate this computational complexity. We identify key performance bottlenecks in the loop closing pipeline of visual SLAM and address them through parallel optimizations on the GPU. Specifically, we use task-level and data-level parallelism and integrate a GPU-accelerated pose graph optimization. Our implementation is built on top of ORB-SLAM3 and leverages CUDA for GPU programming. Experimental results show that FastLoop achieves an average speedup of 1.4x and 1.3x on the EuRoC dataset and 3.0x and 2.4x on the TUM-VI dataset for the loop closing module on desktop and embedded platforms, respectively, while maintaining the accuracy of the original system.

  </details>



- **SLAM Adversarial Lab: An Extensible Framework for Visual SLAM Robustness Evaluation under Adverse Conditions**  
  Mohamed Hefny, Karthik Dantu, Steven Y. Ko  
  _2026-03-17_ · https://arxiv.org/abs/2603.17165v1  
  <details><summary>Abstract</summary>

  We present SAL (SLAM Adversarial Lab), a modular framework for evaluating visual SLAM systems under adversarial conditions such as fog and rain. SAL represents each adversarial condition as a perturbation that transforms an existing dataset into an adversarial dataset. When transforming a dataset, SAL supports severity levels using easily-interpretable real-world units such as meters for fog visibility. SAL's extensible architecture decouples datasets, perturbations, and SLAM algorithms through common interfaces, so users can add new components without rewriting integration code. Moreover, SAL includes a search procedure that finds the severity level of a perturbation at which a SLAM system fails. To showcase the capabilities of SAL, our evaluation integrates seven SLAM algorithms and evaluates them across three datasets under weather, camera, and video transport perturbations.

  </details>



- **WorldCam: Interactive Autoregressive 3D Gaming Worlds with Camera Pose as a Unifying Geometric Representation**  
  Jisu Nam, Yicong Hong, Chun-Hao Paul Huang, Feng Liu, JoungBin Lee, Jiyoung Kim, Siyoon Jin, Yunsung Lee, Jaeyoon Jung, Suhwan Choi, et al.  
  _2026-03-17_ · https://arxiv.org/abs/2603.16871v1  
  <details><summary>Abstract</summary>

  Recent advances in video diffusion transformers have enabled interactive gaming world models that allow users to explore generated environments over extended horizons. However, existing approaches struggle with precise action control and long-horizon 3D consistency. Most prior works treat user actions as abstract conditioning signals, overlooking the fundamental geometric coupling between actions and the 3D world, whereby actions induce relative camera motions that accumulate into a global camera pose within a 3D world. In this paper, we establish camera pose as a unifying geometric representation to jointly ground immediate action control and long-term 3D consistency. First, we define a physics-based continuous action space and represent user inputs in the Lie algebra to derive precise 6-DoF camera poses, which are injected into the generative model via a camera embedder to ensure accurate action alignment. Second, we use global camera poses as spatial indices to retrieve relevant past observations, enabling geometrically consistent revisiting of locations during long-horizon navigation. To support this research, we introduce a large-scale dataset comprising 3,000 minutes of authentic human gameplay annotated with camera trajectories and textual descriptions. Extensive experiments show that our approach substantially outperforms state-of-the-art interactive gaming world models in action controllability, long-horizon visual quality, and 3D spatial consistency.

  </details>



- **M^3: Dense Matching Meets Multi-View Foundation Models for Monocular Gaussian Splatting SLAM**  
  Kerui Ren, Guanghao Li, Changjian Jiang, Yingxiang Xu, Tao Lu, Linning Xu, Junting Dong, Jiangmiao Pang, Mulin Yu, Bo Dai  
  _2026-03-17_ · https://arxiv.org/abs/2603.16844v1  
  <details><summary>Abstract</summary>

  Streaming reconstruction from uncalibrated monocular video remains challenging, as it requires both high-precision pose estimation and computationally efficient online refinement in dynamic environments. While coupling 3D foundation models with SLAM frameworks is a promising paradigm, a critical bottleneck persists: most multi-view foundation models estimate poses in a feed-forward manner, yielding pixel-level correspondences that lack the requisite precision for rigorous geometric optimization. To address this, we present M^3, which augments the Multi-view foundation model with a dedicated Matching head to facilitate fine-grained dense correspondences and integrates it into a robust Monocular Gaussian Splatting SLAM. M^3 further enhances tracking stability by incorporating dynamic area suppression and cross-inference intrinsic alignment. Extensive experiments on diverse indoor and outdoor benchmarks demonstrate state-of-the-art accuracy in both pose estimation and scene reconstruction. Notably, M^3 reduces ATE RMSE by 64.3% compared to VGGT-SLAM 2.0 and outperforms ARTDECO by 2.11 dB in PSNR on the ScanNet++ dataset.

  </details>



- **Reconciling distributed compliance with high-performance control in continuum soft robotics**  
  Vito Daniele Perfetta, Daniel Feliu Talegon, Ebrahim Shahabi, Cosimo Della Santina  
  _2026-03-17_ · https://arxiv.org/abs/2603.16630v1  
  <details><summary>Abstract</summary>

  High-performance closed-loop control of truly soft continuum manipulators has remained elusive. Experimental demonstrations have largely relied on sufficiently stiff, piecewise architectures in which each actuated segment behaves as a distributed yet effectively rigid element, while deformation modes beyond simple bending are suppressed. This strategy simplifies modeling and control, but sidesteps the intrinsic complexity of a fully compliant body and makes the system behave as a serial kinematic chain, much like a conventional articulated robot. An implicit conclusion has consequently emerged within the community: distributed softness and dynamic precision are incompatible. Here we show this trade-off is not fundamental. We present a highly compliant, fully continuum robotic arm - without hardware discretization or stiffness-based mode suppression - that achieves fast, precise task-space convergence under dynamic conditions. The platform integrates direct-drive actuation, a tendon routing scheme enabling coupled bending and twisting, and a structured nonlinear control architecture grounded in reduced-order strain modeling of underactuated systems. Modeling, actuation, and control are co-designed to preserve essential mechanical complexity while enabling high-bandwidth loop closure. Experiments demonstrate accurate, repeatable execution of dynamic Cartesian tasks, including fast positioning and interaction. The proposed system achieves the fastest reported task-execution speed among soft robots. At millimetric precision, execution speed increases nearly fourfold compared with prior approaches, while operating on a fully compliant continuum body. These results show that distributed compliance and high-performance dynamic control can coexist, opening a path toward truly soft manipulators approaching the operational capabilities of rigid robots without sacrificing morphological richness.

  </details>



- **Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty**  
  Mangyu Kong, Jaewon Lee, Seongwon Lee, Euntai Kim  
  _2026-03-17_ · https://arxiv.org/abs/2603.16538v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has recently emerged as a powerful scene representation and is increasingly used for visual localization and pose refinement. However, despite its high-quality differentiable rendering, the robustness of 3DGS-based pose refinement remains highly sensitive to both the initial camera pose and the reconstructed geometry. In this work, we take a closer look at these limitations and identify two major sources of uncertainty: (i) pose prior uncertainty, which often arises from regression or retrieval models that output a single deterministic estimate, and (ii) geometric uncertainty, caused by imperfections in the 3DGS reconstruction that propagate errors into PnP solvers. Such uncertainties can distort reprojection geometry and destabilize optimization, even when the rendered appearance still looks plausible. To address these uncertainties, we introduce a relocalization framework that combines Monte Carlo pose sampling with Fisher Information-based PnP optimization. Our method explicitly accounts for both pose and geometric uncertainty and requires no retraining or additional supervision. Across diverse indoor and outdoor benchmarks, our approach consistently improves localization accuracy and significantly increases stability under pose and depth noise.

  </details>


