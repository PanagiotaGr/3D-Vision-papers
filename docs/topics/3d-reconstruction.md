# 3D Reconstruction

_Updated: 2026-02-25 07:15 UTC_

Total papers shown: **23**


---

- **Region of Interest Segmentation and Morphological Analysis for Membranes in Cryo-Electron Tomography**  
  Xingyi Cheng, Julien Maufront, Aurélie Di Cicco, Daniël M. Pelt, Manuela Dezi, Daniel Lévy  
  _2026-02-24_ · https://arxiv.org/abs/2602.21195v1  
  <details><summary>Abstract</summary>

  Cryo-electron tomography (cryo-ET) enables high resolution, three-dimensional reconstruction of biological structures, including membranes and membrane proteins. Identification of regions of interest (ROIs) is central to scientific imaging, as it enables isolation and quantitative analysis of specific structural features within complex datasets. In practice, however, ROIs are typically derived indirectly through full structure segmentation followed by post hoc analysis. This limitation is especially apparent for continuous and geometrically complex structures such as membranes, which are segmented as single entities. Here, we developed TomoROIS-SurfORA, a two step framework for direct, shape-agnostic ROI segmentation and morphological surface analysis. TomoROIS performs deep learning-based ROI segmentation and can be trained from scratch using small annotated datasets, enabling practical application across diverse imaging data. SurfORA processes segmented structures as point clouds and surface meshes to extract quantitative morphological features, including inter-membrane distances, curvature, and surface roughness. It supports both closed and open surfaces, with specific considerations for open surfaces, which are common in cryo-ET due to the missing wedge effect. We demonstrate both tools using in vitro reconstituted membrane systems containing deformable vesicles with complex geometries, enabling automatic quantitative analysis of membrane contact sites and remodeling events such as invagination. While demonstrated here on cryo-ET membrane data, the combined approach is applicable to ROI detection and surface analysis in broader scientific imaging contexts.

  </details>



- **BrepGaussian: CAD reconstruction from Multi-View Images with Gaussian Splatting**  
  Jiaxing Yu, Dongyang Ren, Hangyu Xu, Zhouyuxiao Yang, Yuanqi Li, Jie Guo, Zhengkang Zhou, Yanwen Guo  
  _2026-02-24_ · https://arxiv.org/abs/2602.21105v1  
  <details><summary>Abstract</summary>

  The boundary representation (B-rep) models a 3D solid as its explicit boundaries: trimmed corners, edges, and faces. Recovering B-rep representation from unstructured data is a challenging and valuable task of computer vision and graphics. Recent advances in deep learning have greatly improved the recovery of 3D shape geometry, but still depend on dense and clean point clouds and struggle to generalize to novel shapes. We propose B-rep Gaussian Splatting (BrepGaussian), a novel framework that learns 3D parametric representations from 2D images. We employ a Gaussian Splatting renderer with learnable features, followed by a specific fitting strategy. To disentangle geometry reconstruction and feature learning, we introduce a two-stage learning framework that first captures geometry and edges and then refines patch features to achieve clean geometry and coherent instance representations. Extensive experiments demonstrate the superior performance of our approach to state-of-the-art methods. We will release our code and datasets upon acceptance.

  </details>



- **Event-Aided Sharp Radiance Field Reconstruction for Fast-Flying Drones**  
  Rong Zou, Marco Cannici, Davide Scaramuzza  
  _2026-02-24_ · https://arxiv.org/abs/2602.21101v1  
  <details><summary>Abstract</summary>

  Fast-flying aerial robots promise rapid inspection under limited battery constraints, with direct applications in infrastructure inspection, terrain exploration, and search and rescue. However, high speeds lead to severe motion blur in images and induce significant drift and noise in pose estimates, making dense 3D reconstruction with Neural Radiance Fields (NeRFs) particularly challenging due to their high sensitivity to such degradations. In this work, we present a unified framework that leverages asynchronous event streams alongside motion-blurred frames to reconstruct high-fidelity radiance fields from agile drone flights. By embedding event-image fusion into NeRF optimization and jointly refining event-based visual-inertial odometry priors using both event and frame modalities, our method recovers sharp radiance fields and accurate camera trajectories without ground-truth supervision. We validate our approach on both synthetic data and real-world sequences captured by a fast-flying drone. Despite highly dynamic drone flights, where RGB frames are severely degraded by motion blur and pose priors become unreliable, our method reconstructs high-fidelity radiance fields and preserves fine scene details, delivering a performance gain of over 50% on real-world data compared to state-of-the-art methods.

  </details>



- **UFO: Unifying Feed-Forward and Optimization-based Methods for Large Driving Scene Modeling**  
  Kaiyuan Tan, Yingying Shen, Mingfei Tu, Haohui Zhu, Bing Wang, Guang Chen, Hangjun Ye, Haiyang Sun  
  _2026-02-24_ · https://arxiv.org/abs/2602.20943v1  
  <details><summary>Abstract</summary>

  Dynamic driving scene reconstruction is critical for autonomous driving simulation and closed-loop learning. While recent feed-forward methods have shown promise for 3D reconstruction, they struggle with long-range driving sequences due to quadratic complexity in sequence length and challenges in modeling dynamic objects over extended durations. We propose UFO, a novel recurrent paradigm that combines the benefits of optimization-based and feed-forward methods for efficient long-range 4D reconstruction. Our approach maintains a 4D scene representation that is iteratively refined as new observations arrive, using a visibility-based filtering mechanism to select informative scene tokens and enable efficient processing of long sequences. For dynamic objects, we introduce an object pose-guided modeling approach that supports accurate long-range motion capture. Experiments on the Waymo Open Dataset demonstrate that our method significantly outperforms both per-scene optimization and existing feed-forward methods across various sequence lengths. Notably, our approach can reconstruct 16-second driving logs within 0.5 second while maintaining superior visual quality and geometric accuracy.

  </details>



- **RU4D-SLAM: Reweighting Uncertainty in Gaussian Splatting SLAM for 4D Scene Reconstruction**  
  Yangfan Zhao, Hanwei Zhang, Ke Huang, Qiufeng Wang, Zhenzhou Shao, Dengyu Wu  
  _2026-02-24_ · https://arxiv.org/abs/2602.20807v1  
  <details><summary>Abstract</summary>

  Combining 3D Gaussian splatting with Simultaneous Localization and Mapping (SLAM) has gained popularity as it enables continuous 3D environment reconstruction during motion. However, existing methods struggle in dynamic environments, particularly moving objects complicate 3D reconstruction and, in turn, hinder reliable tracking. The emergence of 4D reconstruction, especially 4D Gaussian splatting, offers a promising direction for addressing these challenges, yet its potential for 4D-aware SLAM remains largely underexplored. Along this direction, we propose a robust and efficient framework, namely Reweighting Uncertainty in Gaussian Splatting SLAM (RU4D-SLAM) for 4D scene reconstruction, that introduces temporal factors into spatial 3D representation while incorporating uncertainty-aware perception of scene changes, blurred image synthesis, and dynamic scene reconstruction. We enhance dynamic scene representation by integrating motion blur rendering, and improve uncertainty-aware tracking by extending per-pixel uncertainty modeling, which is originally designed for static scenarios, to handle blurred images. Furthermore, we propose a semantic-guided reweighting mechanism for per-pixel uncertainty estimation in dynamic scenes, and introduce a learnable opacity weight to support adaptive 4D mapping. Extensive experiments on standard benchmarks demonstrate that our method substantially outperforms state-of-the-art approaches in both trajectory accuracy and 4D scene reconstruction, particularly in dynamic environments with moving objects and low-quality inputs. Code available: https://ru4d-slam.github.io

  </details>



- **Monocular Endoscopic Tissue 3D Reconstruction with Multi-Level Geometry Regularization**  
  Yangsen Chen, Hao Wang  
  _2026-02-24_ · https://arxiv.org/abs/2602.20718v1  
  <details><summary>Abstract</summary>

  Reconstructing deformable endoscopic tissues is crucial for achieving robot-assisted surgery. However, 3D Gaussian Splatting-based approaches encounter challenges in achieving consistent tissue surface reconstruction, while existing NeRF-based methods lack real-time rendering capabilities. In pursuit of both smooth deformable surfaces and real-time rendering, we introduce a novel approach based on 3D Gaussian Splatting. Specifically, we introduce surface-aware reconstruction, initially employing a Sign Distance Field-based method to construct a mesh, subsequently utilizing this mesh to constrain the Gaussian Splatting reconstruction process. Furthermore, to ensure the generation of physically plausible deformations, we incorporate local rigidity and global non-rigidity restrictions to guide Gaussian deformation, tailored for the highly deformable nature of soft endoscopic tissue. Based on 3D Gaussian Splatting, our proposed method delivers a fast rendering process and smooth surface appearances. Quantitative and qualitative analysis against alternative methodologies shows that our approach achieves solid reconstruction quality in both textures and geometries.

  </details>



- **SD4R: Sparse-to-Dense Learning for 3D Object Detection with 4D Radar**  
  Xiaokai Bai, Jiahao Cheng, Songkai Wang, Yixuan Luo, Lianqing Zheng, Xiaohan Zhang, Si-Yuan Cao, Hui-Liang Shen  
  _2026-02-24_ · https://arxiv.org/abs/2602.20653v1  
  <details><summary>Abstract</summary>

  4D radar measurements offer an affordable and weather-robust solution for 3D perception. However, the inherent sparsity and noise of radar point clouds present significant challenges for accurate 3D object detection, underscoring the need for effective and robust point clouds densification. Despite recent progress, existing densification methods often fail to address the extreme sparsity of 4D radar point clouds and exhibit limited robustness when processing scenes with a small number of points. In this paper, we propose SD4R, a novel framework that transforms sparse radar point clouds into dense representations. SD4R begins by utilizing a foreground point generator (FPG) to mitigate noise propagation and produce densified point clouds. Subsequently, a logit-query encoder (LQE) enhances conventional pillarization, resulting in robust feature representations. Through these innovations, our SD4R demonstrates strong capability in both noise reduction and foreground point densification. Extensive experiments conducted on the publicly available View-of-Delft dataset demonstrate that SD4R achieves state-of-the-art performance. Source code is available at https://github.com/lancelot0805/SD4R.

  </details>



- **From Pairs to Sequences: Track-Aware Policy Gradients for Keypoint Detection**  
  Yepeng Liu, Hao Li, Liwen Yang, Fangzhen Li, Xudi Ge, Yuliang Gu, kuang Gao, Bing Wang, Guang Chen, Hangjun Ye, et al.  
  _2026-02-24_ · https://arxiv.org/abs/2602.20630v1  
  <details><summary>Abstract</summary>

  Keypoint-based matching is a fundamental component of modern 3D vision systems, such as Structure-from-Motion (SfM) and SLAM. Most existing learning-based methods are trained on image pairs, a paradigm that fails to explicitly optimize for the long-term trackability of keypoints across sequences under challenging viewpoint and illumination changes. In this paper, we reframe keypoint detection as a sequential decision-making problem. We introduce TraqPoint, a novel, end-to-end Reinforcement Learning (RL) framework designed to optimize the \textbf{Tra}ck-\textbf{q}uality (Traq) of keypoints directly on image sequences. Our core innovation is a track-aware reward mechanism that jointly encourages the consistency and distinctiveness of keypoints across multiple views, guided by a policy gradient method. Extensive evaluations on sparse matching benchmarks, including relative pose estimation and 3D reconstruction, demonstrate that TraqPoint significantly outperforms some state-of-the-art (SOTA) keypoint detection and description methods.

  </details>



- **Progressive Per-Branch Depth Optimization for DEFOM-Stereo and SAM3 Joint Analysis in UAV Forestry Applications**  
  Yida Lin, Bing Xue, Mengjie Zhang, Sam Schofield, Richard Green  
  _2026-02-24_ · https://arxiv.org/abs/2602.20539v1  
  <details><summary>Abstract</summary>

  Accurate per-branch 3D reconstruction is a prerequisite for autonomous UAV-based tree pruning; however, dense disparity maps from modern stereo matchers often remain too noisy for individual branch analysis in complex forest canopies. This paper introduces a progressive pipeline integrating DEFOM-Stereo foundation-model disparity estimation, SAM3 instance segmentation, and multi-stage depth optimization to deliver robust per-branch point clouds. Starting from a naive baseline, we systematically identify and resolve three error families through successive refinements. Mask boundary contamination is first addressed through morphological erosion and subsequently refined via a skeleton-preserving variant to safeguard thin-branch topology. Segmentation inaccuracy is then mitigated using LAB-space Mahalanobis color validation coupled with cross-branch overlap arbitration. Finally, depth noise - the most persistent error source - is initially reduced by outlier removal and median filtering, before being superseded by a robust five-stage scheme comprising MAD global detection, spatial density consensus, local MAD filtering, RGB-guided filtering, and adaptive bilateral filtering. Evaluated on 1920x1080 stereo imagery of Radiata pine (Pinus radiata) acquired with a ZED Mini camera (63 mm baseline) from a UAV in Canterbury, New Zealand, the proposed pipeline reduces the average per-branch depth standard deviation by 82% while retaining edge fidelity. The result is geometrically coherent 3D point clouds suitable for autonomous pruning tool positioning. All code and processed data are publicly released to facilitate further UAV forestry research.

  </details>



- **SceMoS: Scene-Aware 3D Human Motion Synthesis by Planning with Geometry-Grounded Tokens**  
  Anindita Ghosh, Vladislav Golyanik, Taku Komura, Philipp Slusallek, Christian Theobalt, Rishabh Dabral  
  _2026-02-24_ · https://arxiv.org/abs/2602.20476v1  
  <details><summary>Abstract</summary>

  Synthesizing text-driven 3D human motion within realistic scenes requires learning both semantic intent ("walk to the couch") and physical feasibility (e.g., avoiding collisions). Current methods use generative frameworks that simultaneously learn high-level planning and low-level contact reasoning, and rely on computationally expensive 3D scene data such as point clouds or voxel occupancy grids. We propose SceMoS, a scene-aware motion synthesis framework that shows that structured 2D scene representations can serve as a powerful alternative to full 3D supervision in physically grounded motion synthesis. SceMoS disentangles global planning from local execution using lightweight 2D cues and relying on (1) a text-conditioned autoregressive global motion planner that operates on a bird's-eye-view (BEV) image rendered from an elevated corner of the scene, encoded with DINOv2 features, as the scene representation, and (2) a geometry-grounded motion tokenizer trained via a conditional VQ-VAE, that uses 2D local scene heightmap, thus embedding surface physics directly into a discrete vocabulary. This 2D factorization reaches an efficiency-fidelity trade-off: BEV semantics capture spatial layout and affordance for global reasoning, while local heightmaps enforce fine-grained physical adherence without full 3D volumetric reasoning. SceMoS achieves state-of-the-art motion realism and contact accuracy on the TRUMANS benchmark, reducing the number of trainable parameters for scene encoding by over 50%, showing that 2D scene cues can effectively ground 3D human-scene interaction.

  </details>



- **CLIPoint3D: Language-Grounded Few-Shot Unsupervised 3D Point Cloud Domain Adaptation**  
  Mainak Singha, Sarthak Mehrotra, Paolo Casari, Subhasis Chaudhuri, Elisa Ricci, Biplab Banerjee  
  _2026-02-23_ · https://arxiv.org/abs/2602.20409v1  
  <details><summary>Abstract</summary>

  Recent vision-language models (VLMs) such as CLIP demonstrate impressive cross-modal reasoning, extending beyond images to 3D perception. Yet, these models remain fragile under domain shifts, especially when adapting from synthetic to real-world point clouds. Conventional 3D domain adaptation approaches rely on heavy trainable encoders, yielding strong accuracy but at the cost of efficiency. We introduce CLIPoint3D, the first framework for few-shot unsupervised 3D point cloud domain adaptation built upon CLIP. Our approach projects 3D samples into multiple depth maps and exploits the frozen CLIP backbone, refined through a knowledge-driven prompt tuning scheme that integrates high-level language priors with geometric cues from a lightweight 3D encoder. To adapt task-specific features effectively, we apply parameter-efficient fine-tuning to CLIP's encoders and design an entropy-guided view sampling strategy for selecting confident projections. Furthermore, an optimal transport-based alignment loss and an uncertainty-aware prototype alignment loss collaboratively bridge source-target distribution gaps while maintaining class separability. Extensive experiments on PointDA-10 and GraspNetPC-10 benchmarks show that CLIPoint3D achieves consistent 3-16% accuracy gains over both CLIP-based and conventional encoder-based baselines. Codes are available at https://github.com/SarthakM320/CLIPoint3D.

  </details>



- **Large-scale Photorealistic Outdoor 3D Scene Reconstruction from UAV Imagery Using Gaussian Splatting Techniques**  
  Christos Maikos, Georgios Angelidis, Georgios Th. Papadopoulos  
  _2026-02-23_ · https://arxiv.org/abs/2602.20342v1  
  <details><summary>Abstract</summary>

  In this study, we present an end-to-end pipeline capable of converting drone-captured video streams into high-fidelity 3D reconstructions with minimal latency. Unmanned aerial vehicles (UAVs) are extensively used in aerial real-time perception applications. Moreover, recent advances in 3D Gaussian Splatting (3DGS) have demonstrated significant potential for real-time neural rendering. However, their integration into end-to-end UAV-based reconstruction and visualization systems remains underexplored. Our goal is to propose an efficient architecture that combines live video acquisition via RTMP streaming, synchronized sensor fusion, camera pose estimation, and 3DGS optimization, achieving continuous model updates and low-latency deployment within interactive visualization environments that supports immersive augmented and virtual reality (AR/VR) applications. Experimental results demonstrate that the proposed method achieves competitive visual fidelity, while delivering significantly higher rendering performance and substantially reduced end-to-end latency, compared to NeRF-based approaches. Reconstruction quality remains within 4-7\% of high-fidelity offline references, confirming the suitability of the proposed system for real-time, scalable augmented perception from aerial platforms.

  </details>



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


