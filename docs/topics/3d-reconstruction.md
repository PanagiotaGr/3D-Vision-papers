# 3D Reconstruction

_Updated: 2026-02-24 07:14 UTC_

Total papers shown: **15**


---

- **tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction**  
  Chen Wang, Hao Tan, Wang Yifan, Zhiqin Chen, Yuheng Liu, Kalyan Sunkavalli, Sai Bi, Lingjie Liu, Yiwei Hu  
  _2026-02-23_ · https://arxiv.org/abs/2602.20160v1  
  <details><summary>Abstract</summary>

  We propose tttLRM, a novel large 3D reconstruction model that leverages a Test-Time Training (TTT) layer to enable long-context, autoregressive 3D reconstruction with linear computational complexity, further scaling the model's capability. Our framework efficiently compresses multiple image observations into the fast weights of the TTT layer, forming an implicit 3D representation in the latent space that can be decoded into various explicit formats, such as Gaussian Splats (GS) for downstream applications. The online learning variant of our model supports progressive 3D reconstruction and refinement from streaming observations. We demonstrate that pretraining on novel view synthesis tasks effectively transfers to explicit 3D modeling, resulting in improved reconstruction quality and faster convergence. Extensive experiments show that our method achieves superior performance in feedforward 3D Gaussian reconstruction compared to state-of-the-art approaches on both objects and scenes.

  </details>



- **RADE-Net: Robust Attention Network for Radar-Only Object Detection in Adverse Weather**  
  Christof Leitgeb, Thomas Puchleitner, Max Peter Ronecker, Daniel Watzenig  
  _2026-02-23_ · https://arxiv.org/abs/2602.19994v1  
  <details><summary>Abstract</summary>

  Automotive perception systems are obligated to meet high requirements. While optical sensors such as Camera and Lidar struggle in adverse weather conditions, Radar provides a more robust perception performance, effectively penetrating fog, rain, and snow. Since full Radar tensors have large data sizes and very few datasets provide them, most Radar-based approaches work with sparse point clouds or 2D projections, which can result in information loss. Additionally, deep learning methods show potential to extract richer and more dense features from low level Radar data and therefore significantly increase the perception performance. Therefore, we propose a 3D projection method for fast-Fourier-transformed 4D Range-Azimuth-Doppler-Elevation (RADE) tensors. Our method preserves rich Doppler and Elevation features while reducing the required data size for a single frame by 91.9% compared to a full tensor, thus achieving higher training and inference speed as well as lower model complexity. We introduce RADE-Net, a lightweight model tailored to 3D projections of the RADE tensor. The backbone enables exploitation of low-level and high-level cues of Radar tensors with spatial and channel-attention. The decoupled detection heads predict object center-points directly in the Range-Azimuth domain and regress rotated 3D bounding boxes from rich feature maps in the cartesian scene. We evaluate the model on scenes with multiple different road users and under various weather conditions on the large-scale K-Radar dataset and achieve a 16.7% improvement compared to their baseline, as well as 6.5% improvement over current Radar-only models. Additionally, we outperform several Lidar approaches in scenarios with adverse weather conditions. The code is available under https://github.com/chr-is-tof/RADE-Net.

  </details>



- **Monocular Mesh Recovery and Body Measurement of Female Saanen Goats**  
  Bo Jin, Shichao Zhao, Jin Lyu, Bin Zhang, Tao Yu, Liang An, Yebin Liu, Meili Wang  
  _2026-02-23_ · https://arxiv.org/abs/2602.19896v1  
  <details><summary>Abstract</summary>

  The lactation performance of Saanen dairy goats, renowned for their high milk yield, is intrinsically linked to their body size, making accurate 3D body measurement essential for assessing milk production potential, yet existing reconstruction methods lack goat-specific authentic 3D data. To address this limitation, we establish the FemaleSaanenGoat dataset containing synchronized eight-view RGBD videos of 55 female Saanen goats (6-18 months). Using multi-view DynamicFusion, we fuse noisy, non-rigid point cloud sequences into high-fidelity 3D scans, overcoming challenges from irregular surfaces and rapid movement. Based on these scans, we develop SaanenGoat, a parametric 3D shape model specifically designed for female Saanen goats. This model features a refined template with 41 skeletal joints and enhanced udder representation, registered with our scan data. A comprehensive shape space constructed from 48 goats enables precise representation of diverse individual variations. With the help of SaanenGoat model, we get high-precision 3D reconstruction from single-view RGBD input, and achieve automated measurement of six critical body dimensions: body length, height, chest width, chest girth, hip width, and hip height. Experimental results demonstrate the superior accuracy of our method in both 3D reconstruction and body measurement, presenting a novel paradigm for large-scale 3D vision applications in precision livestock farming.

  </details>



- **One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image**  
  Pengfei Wang, Liyi Chen, Zhiyuan Ma, Yanjun Guo, Guowen Zhang, Lei Zhang  
  _2026-02-23_ · https://arxiv.org/abs/2602.19766v1  
  <details><summary>Abstract</summary>

  Generating explorable 3D scenes from a single image is a highly challenging problem in 3D vision. Existing methods struggle to support free exploration, often producing severe geometric distortions and noisy artifacts when the viewpoint moves far from the original perspective. We introduce \textbf{One2Scene}, an effective framework that decomposes this ill-posed problem into three tractable sub-tasks to enable immersive explorable scene generation. We first use a panorama generator to produce anchor views from a single input image as initialization. Then, we lift these 2D anchors into an explicit 3D geometric scaffold via a generalizable, feed-forward Gaussian Splatting network. Instead of treating the panorama as a single image for reconstruction, we project it into multiple sparse anchor views and reformulate the reconstruction task as multi-view stereo matching, which allows us to leverage robust geometric priors learned from large-scale multi-view datasets. A bidirectional feature fusion module is used to enforce cross-view consistency, yielding an efficient and geometrically reliable scaffold. Finally, the scaffold serves as a strong prior for a novel view generator to produce photorealistic and geometrically accurate views at arbitrary cameras. By explicitly conditioning on a 3D-consistent scaffold to perform reconstruction, One2Scene works stably under large camera motions, supporting immersive scene exploration. Extensive experiments show that One2Scene substantially outperforms state-of-the-art methods in panorama depth estimation, feed-forward 360° reconstruction, and explorable 3D scene generation. Code and models will be released.

  </details>



- **Training Deep Stereo Matching Networks on Tree Branch Imagery: A Benchmark Study for Real-Time UAV Forestry Applications**  
  Yida Lin, Bing Xue, Mengjie Zhang, Sam Schofield, Richard Green  
  _2026-02-23_ · https://arxiv.org/abs/2602.19763v1  
  <details><summary>Abstract</summary>

  Autonomous drone-based tree pruning needs accurate, real-time depth estimation from stereo cameras. Depth is computed from disparity maps using $Z = f B/d$, so even small disparity errors cause noticeable depth mistakes at working distances. Building on our earlier work that identified DEFOM-Stereo as the best reference disparity generator for vegetation scenes, we present the first study to train and test ten deep stereo matching networks on real tree branch images. We use the Canterbury Tree Branches dataset -- 5,313 stereo pairs from a ZED Mini camera at 1080P and 720P -- with DEFOM-generated disparity maps as training targets. The ten methods cover step-by-step refinement, 3D convolution, edge-aware attention, and lightweight designs. Using perceptual metrics (SSIM, LPIPS, ViTScore) and structural metrics (SIFT/ORB feature matching), we find that BANet-3D produces the best overall quality (SSIM = 0.883, LPIPS = 0.157), while RAFT-Stereo scores highest on scene-level understanding (ViTScore = 0.799). Testing on an NVIDIA Jetson Orin Super (16 GB, independently powered) mounted on our drone shows that AnyNet reaches 6.99 FPS at 1080P -- the only near-real-time option -- while BANet-2D gives the best quality-speed balance at 1.21 FPS. We also compare 720P and 1080P processing times to guide resolution choices for forestry drone systems.

  </details>



- **VGGT-MPR: VGGT-Enhanced Multimodal Place Recognition in Autonomous Driving Environments**  
  Jingyi Xu, Zhangshuo Qi, Zhongmiao Yan, Xuyu Gao, Qianyun Jiao, Songpengcheng Xia, Xieyuanli Chen, Ling Pei  
  _2026-02-23_ · https://arxiv.org/abs/2602.19735v1  
  <details><summary>Abstract</summary>

  In autonomous driving, robust place recognition is critical for global localization and loop closure detection. While inter-modality fusion of camera and LiDAR data in multimodal place recognition (MPR) has shown promise in overcoming the limitations of unimodal counterparts, existing MPR methods basically attend to hand-crafted fusion strategies and heavily parameterized backbones that require costly retraining. To address this, we propose VGGT-MPR, a multimodal place recognition framework that adopts the Visual Geometry Grounded Transformer (VGGT) as a unified geometric engine for both global retrieval and re-ranking. In the global retrieval stage, VGGT extracts geometrically-rich visual embeddings through prior depth-aware and point map supervision, and densifies sparse LiDAR point clouds with predicted depth maps to improve structural representation. This enhances the discriminative ability of fused multimodal features and produces global descriptors for fast retrieval. Beyond global retrieval, we design a training-free re-ranking mechanism that exploits VGGT's cross-view keypoint-tracking capability. By combining mask-guided keypoint extraction with confidence-aware correspondence scoring, our proposed re-ranking mechanism effectively refines retrieval results without additional parameter optimization. Extensive experiments on large-scale autonomous driving benchmarks and our self-collected data demonstrate that VGGT-MPR achieves state-of-the-art performance, exhibiting strong robustness to severe environmental changes, viewpoint shifts, and occlusions. Our code and data will be made publicly available.

  </details>



- **BayesFusion-SDF: Probabilistic Signed Distance Fusion with View Planning on CPU**  
  Soumya Mazumdar, Vineet Kumar Rakesh, Tapas Samanta  
  _2026-02-23_ · https://arxiv.org/abs/2602.19697v1  
  <details><summary>Abstract</summary>

  Key part of robotics, augmented reality, and digital inspection is dense 3D reconstruction from depth observations. Traditional volumetric fusion techniques, including truncated signed distance functions (TSDF), enable efficient and deterministic geometry reconstruction; however, they depend on heuristic weighting and fail to transparently convey uncertainty in a systematic way. Recent neural implicit methods, on the other hand, get very high fidelity but usually need a lot of GPU power for optimization and aren't very easy to understand for making decisions later on. This work presents BayesFusion-SDF, a CPU-centric probabilistic signed distance fusion framework that conceptualizes geometry as a sparse Gaussian random field with a defined posterior distribution over voxel distances. First, a rough TSDF reconstruction is used to create an adaptive narrow-band domain. Then, depth observations are combined using a heteroscedastic Bayesian formulation that is solved using sparse linear algebra and preconditioned conjugate gradients. Randomized diagonal estimators are a quick way to get an idea of posterior uncertainty. This makes it possible to extract surfaces and plan the next best view while taking into account uncertainty. Tests on a controlled ablation scene and a CO3D object sequence show that the new method is more accurate geometrically than TSDF baselines and gives useful estimates of uncertainty for active sensing. The proposed formulation provides a clear and easy-to-use alternative to GPU-heavy neural reconstruction methods while still being able to be understood in a probabilistic way and acting in a predictable way. GitHub: https://mazumdarsoumya.github.io/BayesFusionSDF

  </details>



- **TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures**  
  Hyeongjin Nam, Daniel Sungho Jung, Kyoung Mu Lee  
  _2026-02-23_ · https://arxiv.org/abs/2602.19679v1  
  <details><summary>Abstract</summary>

  Joint reconstruction of 3D human and object from a single image is an active research area, with pivotal applications in robotics and digital content creation. Despite recent advances, existing approaches suffer from two fundamental limitations. First, their reconstructions rely heavily on physical contact information, which inherently cannot capture non-contact human-object interactions, such as gazing at or pointing toward an object. Second, the reconstruction process is primarily driven by local geometric proximity, neglecting the human and object appearances that provide global context crucial for understanding holistic interactions. To address these issues, we introduce TeHOR, a framework built upon two core designs. First, beyond contact information, our framework leverages text descriptions of human-object interactions to enforce semantic alignment between the 3D reconstruction and its textual cues, enabling reasoning over a wider spectrum of interactions, including non-contact cases. Second, we incorporate appearance cues of the 3D human and object into the alignment process to capture holistic contextual information, thereby ensuring visually plausible reconstructions. As a result, our framework produces accurate and semantically coherent reconstructions, achieving state-of-the-art performance.

  </details>



- **PoseCraft: Tokenized 3D Body Landmark and Camera Conditioning for Photorealistic Human Image Synthesis**  
  Zhilin Guo, Jing Yang, Kyle Fogarty, Jingyi Wan, Boqiao Zhang, Tianhao Wu, Weihao Xia, Chenliang Zhou, Sakar Khattar, Fangcheng Zhong, et al.  
  _2026-02-22_ · https://arxiv.org/abs/2602.19350v1  
  <details><summary>Abstract</summary>

  Digitizing humans and synthesizing photorealistic avatars with explicit 3D pose and camera controls are central to VR, telepresence, and entertainment. Existing skinning-based workflows require laborious manual rigging or template-based fittings, while neural volumetric methods rely on canonical templates and re-optimization for each unseen pose. We present PoseCraft, a diffusion framework built around tokenized 3D interface: instead of relying only on rasterized geometry as 2D control images, we encode sparse 3D landmarks and camera extrinsics as discrete conditioning tokens and inject them into diffusion via cross-attention. Our approach preserves 3D semantics by avoiding 2D re-projection ambiguity under large pose and viewpoint changes, and produces photorealistic imagery that faithfully captures identity and appearance. To train and evaluate at scale, we also implement GenHumanRF, a data generation workflow that renders diverse supervision from volumetric reconstructions. Our experiments show that PoseCraft achieves significant perceptual quality improvement over diffusion-centric methods, and attains better or comparable metrics to latest volumetric rendering SOTA while better preserving fabric and hair details.

  </details>



- **DefenseSplat: Enhancing the Robustness of 3D Gaussian Splatting via Frequency-Aware Filtering**  
  Yiran Qiao, Yiren Lu, Yunlai Zhou, Rui Yang, Linlin Hou, Yu Yin, Jing Ma  
  _2026-02-22_ · https://arxiv.org/abs/2602.19323v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has emerged as a powerful paradigm for real-time and high-fidelity 3D reconstruction from posed images. However, recent studies reveal its vulnerability to adversarial corruptions in input views, where imperceptible yet consistent perturbations can drastically degrade rendering quality, increase training and rendering time, and inflate memory usage, even leading to server denial-of-service. In our work, to mitigate this issue, we begin by analyzing the distinct behaviors of adversarial perturbations in the low- and high-frequency components of input images using wavelet transforms. Based on this observation, we design a simple yet effective frequency-aware defense strategy that reconstructs training views by filtering high-frequency noise while preserving low-frequency content. This approach effectively suppresses adversarial artifacts while maintaining the authenticity of the original scene. Notably, it does not significantly impair training on clean data, achieving a desirable trade-off between robustness and performance on clean inputs. Through extensive experiments under a wide range of attack intensities on multiple benchmarks, we demonstrate that our method substantially enhances the robustness of 3DGS without access to clean ground-truth supervision. By highlighting and addressing the overlooked vulnerabilities of 3D Gaussian Splatting, our work paves the way for more robust and secure 3D reconstructions.

  </details>



- **GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning**  
  Zehao Deng, An Liu, Yan Wang  
  _2026-02-22_ · https://arxiv.org/abs/2602.19206v1  
  <details><summary>Abstract</summary>

  Zero-shot 3D Anomaly Detection is an emerging task that aims to detect anomalies in a target dataset without any target training data, which is particularly important in scenarios constrained by sample scarcity and data privacy concerns. While current methods adapt CLIP by projecting 3D point clouds into 2D representations, they face challenges. The projection inherently loses some geometric details, and the reliance on a single 2D modality provides an incomplete visual understanding, limiting their ability to detect diverse anomaly types. To address these limitations, we propose the Geometry-Aware Prompt and Synergistic View Representation Learning (GS-CLIP) framework, which enables the model to identify geometric anomalies through a two-stage learning process. In stage 1, we dynamically generate text prompts embedded with 3D geometric priors. These prompts contain global shape context and local defect information distilled by our Geometric Defect Distillation Module (GDDM). In stage 2, we introduce Synergistic View Representation Learning architecture that processes rendered and depth images in parallel. A Synergistic Refinement Module (SRM) subsequently fuses the features of both streams, capitalizing on their complementary strengths. Comprehensive experimental results on four large-scale public datasets show that GS-CLIP achieves superior performance in detection. Code can be available at https://github.com/zhushengxinyue/GS-CLIP.

  </details>



- **Direction-aware 3D Large Multimodal Models**  
  Quan Liu, Weihao Xuan, Junjue Wang, Naoto Yokoya, Ling Shao, Shijian Lu  
  _2026-02-22_ · https://arxiv.org/abs/2602.19063v1  
  <details><summary>Abstract</summary>

  3D large multimodal models (3D LMMs) rely heavily on ego poses for enabling directional question-answering and spatial reasoning. However, most existing point cloud benchmarks contain rich directional queries but lack the corresponding ego poses, making them inherently ill-posed in 3D large multimodal modelling. In this work, we redefine a new and rigorous paradigm that enables direction-aware 3D LMMs by identifying and supplementing ego poses into point cloud benchmarks and transforming the corresponding point cloud data according to the identified ego poses. We enable direction-aware 3D LMMs with two novel designs. The first is PoseRecover, a fully automatic pose recovery pipeline that matches questions with ego poses from RGB-D video extrinsics via object-frustum intersection and visibility check with Z-buffers. The second is PoseAlign that transforms the point cloud data to be aligned with the identified ego poses instead of either injecting ego poses into textual prompts or introducing pose-encoded features in the projection layers. Extensive experiments show that our designs yield consistent improvements across multiple 3D LMM backbones such as LL3DA, LL3DA-SONATA, Chat-Scene, and 3D-LLAVA, improving ScanRefer mIoU by 30.0% and Scan2Cap LLM-as-judge accuracy by 11.7%. In addition, our approach is simple, generic, and training-efficient, requiring only instruction tuning while establishing a strong baseline for direction-aware 3D-LMMs.

  </details>



- **OpenVO: Open-World Visual Odometry with Temporal Dynamics Awareness**  
  Phuc D. A. Nguyen, Anh N. Nhu, Ming C. Lin  
  _2026-02-22_ · https://arxiv.org/abs/2602.19035v1  
  <details><summary>Abstract</summary>

  We introduce OpenVO, a novel framework for Open-world Visual Odometry (VO) with temporal awareness under limited input conditions. OpenVO effectively estimates real-world-scale ego-motion from monocular dashcam footage with varying observation rates and uncalibrated cameras, enabling robust trajectory dataset construction from rare driving events recorded in dashcam. Existing VO methods are trained on fixed observation frequency (e.g., 10Hz or 12Hz), completely overlooking temporal dynamics information. Many prior methods also require calibrated cameras with known intrinsic parameters. Consequently, their performance degrades when (1) deployed under unseen observation frequencies or (2) applied to uncalibrated cameras. These significantly limit their generalizability to many downstream tasks, such as extracting trajectories from dashcam footage. To address these challenges, OpenVO (1) explicitly encodes temporal dynamics information within a two-frame pose regression framework and (2) leverages 3D geometric priors derived from foundation models. We validate our method on three major autonomous-driving benchmarks - KITTI, nuScenes, and Argoverse 2 - achieving more than 20 performance improvement over state-of-the-art approaches. Under varying observation rate settings, our method is significantly more robust, achieving 46%-92% lower errors across all metrics. These results demonstrate the versatility of OpenVO for real-world 3D reconstruction and diverse downstream applications.

  </details>



- **Marginalized Bundle Adjustment: Multi-View Camera Pose from Monocular Depth Estimates**  
  Shengjie Zhu, Ahmed Abdelkader, Mark J. Matthews, Xiaoming Liu, Wen-Sheng Chu  
  _2026-02-21_ · https://arxiv.org/abs/2602.18906v1  
  <details><summary>Abstract</summary>

  Structure-from-Motion (SfM) is a fundamental 3D vision task for recovering camera parameters and scene geometry from multi-view images. While recent deep learning advances enable accurate Monocular Depth Estimation (MDE) from single images without depending on camera motion, integrating MDE into SfM remains a challenge. Unlike conventional triangulated sparse point clouds, MDE produces dense depth maps with significantly higher error variance. Inspired by modern RANSAC estimators, we propose Marginalized Bundle Adjustment (MBA) to mitigate MDE error variance leveraging its density. With MBA, we show that MDE depth maps are sufficiently accurate to yield SoTA or competitive results in SfM and camera relocalization tasks. Through extensive evaluations, we demonstrate consistently robust performance across varying scales, ranging from few-frame setups to large multi-view systems with thousands of images. Our method highlights the significant potential of MDE in multi-view 3D vision.

  </details>



- **Enhancing 3D LiDAR Segmentation by Shaping Dense and Accurate 2D Semantic Predictions**  
  Xiaoyu Dong, Tiankui Xian, Wanshui Gan, Naoto Yokoya  
  _2026-02-21_ · https://arxiv.org/abs/2602.18869v1  
  <details><summary>Abstract</summary>

  Semantic segmentation of 3D LiDAR point clouds is important in urban remote sensing for understanding real-world street environments. This task, by projecting LiDAR point clouds and 3D semantic labels as sparse maps, can be reformulated as a 2D problem. However, the intrinsic sparsity of the projected LiDAR and label maps can result in sparse and inaccurate intermediate 2D semantic predictions, which in return limits the final 3D accuracy. To address this issue, we enhance this task by shaping dense and accurate 2D predictions. Specifically, we develop a multi-modal segmentation model, MM2D3D. By leveraging camera images as auxiliary data, we introduce cross-modal guided filtering to overcome label map sparsity by constraining intermediate 2D semantic predictions with dense semantic relations derived from the camera images; and we introduce dynamic cross pseudo supervision to overcome LiDAR map sparsity by encouraging the 2D predictions to emulate the dense distribution of the semantic predictions from the camera images. Experiments show that our techniques enable our model to achieve intermediate 2D semantic predictions with dense distribution and higher accuracy, which effectively enhances the final 3D accuracy. Comparisons with previous methods demonstrate our superior performance in both 2D and 3D spaces.

  </details>


