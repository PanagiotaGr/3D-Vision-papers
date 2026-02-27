# Gaussian Splatting & 3DGS

_Updated: 2026-02-27 07:10 UTC_

Total papers shown: **12**


---

- **Latent Gaussian Splatting for 4D Panoptic Occupancy Tracking**  
  Maximilian Luz, Rohit Mohan, Thomas Nürnberg, Yakov Miron, Daniele Cattaneo, Abhinav Valada  
  _2026-02-26_ · https://arxiv.org/abs/2602.23172v1  
  <details><summary>Abstract</summary>

  Capturing 4D spatiotemporal surroundings is crucial for the safe and reliable operation of robots in dynamic environments. However, most existing methods address only one side of the problem: they either provide coarse geometric tracking via bounding boxes, or detailed 3D structures like voxel-based occupancy that lack explicit temporal association. In this work, we present Latent Gaussian Splatting for 4D Panoptic Occupancy Tracking (LaGS) that advances spatiotemporal scene understanding in a holistic direction. Our approach incorporates camera-based end-to-end tracking with mask-based multi-view panoptic occupancy prediction, and addresses the key challenge of efficiently aggregating multi-view information into 3D voxel grids via a novel latent Gaussian splatting approach. Specifically, we first fuse observations into 3D Gaussians that serve as a sparse point-centric latent representation of the 3D scene, and then splat the aggregated features onto a 3D voxel grid that is decoded by a mask-based segmentation head. We evaluate LaGS on the Occ3D nuScenes and Waymo datasets, achieving state-of-the-art performance for 4D panoptic occupancy tracking. We make our code available at https://lags.cs.uni-freiburg.de/.

  </details>



- **PackUV: Packed Gaussian UV Maps for 4D Volumetric Video**  
  Aashish Rai, Angela Xing, Anushka Agarwal, Xiaoyan Cong, Zekun Li, Tao Lu, Aayush Prakash, Srinath Sridhar  
  _2026-02-26_ · https://arxiv.org/abs/2602.23040v1  
  <details><summary>Abstract</summary>

  Volumetric videos offer immersive 4D experiences, but remain difficult to reconstruct, store, and stream at scale. Existing Gaussian Splatting based methods achieve high-quality reconstruction but break down on long sequences, temporal inconsistency, and fail under large motions and disocclusions. Moreover, their outputs are typically incompatible with conventional video coding pipelines, preventing practical applications. We introduce PackUV, a novel 4D Gaussian representation that maps all Gaussian attributes into a sequence of structured, multi-scale UV atlas, enabling compact, image-native storage. To fit this representation from multi-view videos, we propose PackUV-GS, a temporally consistent fitting method that directly optimizes Gaussian parameters in the UV domain. A flow-guided Gaussian labeling and video keyframing module identifies dynamic Gaussians, stabilizes static regions, and preserves temporal coherence even under large motions and disocclusions. The resulting UV atlas format is the first unified volumetric video representation compatible with standard video codecs (e.g., FFV1) without losing quality, enabling efficient streaming within existing multimedia infrastructure. To evaluate long-duration volumetric capture, we present PackUV-2B, the largest multi-view video dataset to date, featuring more than 50 synchronized cameras, substantial motion, and frequent disocclusions across 100 sequences and 2B (billion) frames. Extensive experiments demonstrate that our method surpasses existing baselines in rendering fidelity while scaling to sequences up to 30 minutes with consistent quality.

  </details>



- **GSTurb: Gaussian Splatting for Atmospheric Turbulence Mitigation**  
  Hanliang Du, Zhangji Lu, Zewei Cai, Qijian Tang, Qifeng Yu, Xiaoli Liu  
  _2026-02-26_ · https://arxiv.org/abs/2602.22800v1  
  <details><summary>Abstract</summary>

  Atmospheric turbulence causes significant image degradation due to pixel displacement (tilt) and blur, particularly in long-range imaging applications. In this paper, we propose a novel framework for atmospheric turbulence mitigation, GSTurb, which integrates optical flow-guided tilt correction and Gaussian splatting for modeling non-isoplanatic blur. The framework employs Gaussian parameters to represent tilt and blur, and optimizes them across multiple frames to enhance restoration. Experimental results on the ATSyn-static dataset demonstrate the effectiveness of our method, achieving a peak PSNR of 27.67 dB and SSIM of 0.8735. Compared to the state-of-the-art method, GSTurb improves PSNR by 1.3 dB (a 4.5% increase) and SSIM by 0.048 (a 5.8% increase). Additionally, on real datasets, including the TSRWGAN Real-World and CLEAR datasets, GSTurb outperforms existing methods, showing significant improvements in both qualitative and quantitative performance. These results highlight that combining optical flow-guided tilt correction with Gaussian splatting effectively enhances image restoration under both synthetic and real-world turbulence conditions. The code for this method will be available at https://github.com/DuhlLiamz/3DGS_turbulence/tree/main.

  </details>



- **Sapling-NeRF: Geo-Localised Sapling Reconstruction in Forests for Ecological Monitoring**  
  Miguel Ángel Muñoz-Bañón, Nived Chebrolu, Sruthi M. Krishna Moorthy, Yifu Tao, Fernando Torres, Roberto Salguero-Gómez, Maurice Fallon  
  _2026-02-26_ · https://arxiv.org/abs/2602.22731v1  
  <details><summary>Abstract</summary>

  Saplings are key indicators of forest regeneration and overall forest health. However, their fine-scale architectural traits are difficult to capture with existing 3D sensing methods, which make quantitative evaluation difficult. Terrestrial Laser Scanners (TLS), Mobile Laser Scanners (MLS), or traditional photogrammetry approaches poorly reconstruct thin branches, dense foliage, and lack the scale consistency needed for long-term monitoring. Implicit 3D reconstruction methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) are promising alternatives, but cannot recover the true scale of a scene and lack any means to be accurately geo-localised. In this paper, we present a pipeline which fuses NeRF, LiDAR SLAM, and GNSS to enable repeatable, geo-localised ecological monitoring of saplings. Our system proposes a three-level representation: (i) coarse Earth-frame localisation using GNSS, (ii) LiDAR-based SLAM for centimetre-accurate localisation and reconstruction, and (iii) NeRF-derived object-centric dense reconstruction of individual saplings. This approach enables repeatable quantitative evaluation and long-term monitoring of sapling traits. Our experiments in forest plots in Wytham Woods (Oxford, UK) and Evo (Finland) show that stem height, branching patterns, and leaf-to-wood ratios can be captured with increased accuracy as compared to TLS. We demonstrate that accurate stem skeletons and leaf distributions can be measured for saplings with heights between 0.5m and 2m in situ, giving ecologists access to richer structural and quantitative data for analysing forest dynamics.

  </details>



- **ArtPro: Self-Supervised Articulated Object Reconstruction with Adaptive Integration of Mobility Proposals**  
  Xuelu Li, Zhaonan Wang, Xiaogang Wang, Lei Wu, Manyi Li, Changhe Tu  
  _2026-02-26_ · https://arxiv.org/abs/2602.22666v1  
  <details><summary>Abstract</summary>

  Reconstructing articulated objects into high-fidelity digital twins is crucial for applications such as robotic manipulation and interactive simulation. Recent self-supervised methods using differentiable rendering frameworks like 3D Gaussian Splatting remain highly sensitive to the initial part segmentation. Their reliance on heuristic clustering or pre-trained models often causes optimization to converge to local minima, especially for complex multi-part objects. To address these limitations, we propose ArtPro, a novel self-supervised framework that introduces adaptive integration of mobility proposals. Our approach begins with an over-segmentation initialization guided by geometry features and motion priors, generating part proposals with plausible motion hypotheses. During optimization, we dynamically merge these proposals by analyzing motion consistency among spatial neighbors, while a collision-aware motion pruning mechanism prevents erroneous kinematic estimation. Extensive experiments on both synthetic and real-world objects demonstrate that ArtPro achieves robust reconstruction of complex multi-part objects, significantly outperforming existing methods in accuracy and stability.

  </details>



- **BetterScene: 3D Scene Synthesis with Representation-Aligned Generative Model**  
  Yuci Han, Charles Toth, John E. Anderson, William J. Shuart, Alper Yilmaz  
  _2026-02-26_ · https://arxiv.org/abs/2602.22596v1  
  <details><summary>Abstract</summary>

  We present BetterScene, an approach to enhance novel view synthesis (NVS) quality for diverse real-world scenes using extremely sparse, unconstrained photos. BetterScene leverages the production-ready Stable Video Diffusion (SVD) model pretrained on billions of frames as a strong backbone, aiming to mitigate artifacts and recover view-consistent details at inference time. Conventional methods have developed similar diffusion-based solutions to address these challenges of novel view synthesis. Despite significant improvements, these methods typically rely on off-the-shelf pretrained diffusion priors and fine-tune only the UNet module while keeping other components frozen, which still leads to inconsistent details and artifacts even when incorporating geometry-aware regularizations like depth or semantic conditions. To address this, we investigate the latent space of the diffusion model and introduce two components: (1) temporal equivariance regularization and (2) vision foundation model-aligned representation, both applied to the variational autoencoder (VAE) module within the SVD pipeline. BetterScene integrates a feed-forward 3D Gaussian Splatting (3DGS) model to render features as inputs for the SVD enhancer and generate continuous, artifact-free, consistent novel views. We evaluate on the challenging DL3DV-10K dataset and demonstrate superior performance compared to state-of-the-art methods.

  </details>



- **GIFSplat: Generative Prior-Guided Iterative Feed-Forward 3D Gaussian Splatting from Sparse Views**  
  Tianyu Chen, Wei Xiang, Kang Han, Yu Lu, Di Wu, Gaowen Liu, Ramana Rao Kompella  
  _2026-02-26_ · https://arxiv.org/abs/2602.22571v1  
  <details><summary>Abstract</summary>

  Feed-forward 3D reconstruction offers substantial runtime advantages over per-scene optimization, which remains slow at inference and often fragile under sparse views. However, existing feed-forward methods still have potential for further performance gains, especially for out-of-domain data, and struggle to retain second-level inference time once a generative prior is introduced. These limitations stem from the one-shot prediction paradigm in existing feed-forward pipeline: models are strictly bounded by capacity, lack inference-time refinement, and are ill-suited for continuously injecting generative priors. We introduce GIFSplat, a purely feed-forward iterative refinement framework for 3D Gaussian Splatting from sparse unposed views. A small number of forward-only residual updates progressively refine current 3D scene using rendering evidence, achieve favorable balance between efficiency and quality. Furthermore, we distill a frozen diffusion prior into Gaussian-level cues from enhanced novel renderings without gradient backpropagation or ever-increasing view-set expansion, thereby enabling per-scene adaptation with generative prior while preserving feed-forward efficiency. Across DL3DV, RealEstate10K, and DTU, GIFSplat consistently outperforms state-of-the-art feed-forward baselines, improving PSNR by up to +2.1 dB, and it maintains second-scale inference time without requiring camera poses or any test-time gradient optimization.

  </details>



- **SwiftNDC: Fast Neural Depth Correction for High-Fidelity 3D Reconstruction**  
  Kang Han, Wei Xiang, Lu Yu, Mathew Wyatt, Gaowen Liu, Ramana Rao Kompella  
  _2026-02-26_ · https://arxiv.org/abs/2602.22565v1  
  <details><summary>Abstract</summary>

  Depth-guided 3D reconstruction has gained popularity as a fast alternative to optimization-heavy approaches, yet existing methods still suffer from scale drift, multi-view inconsistencies, and the need for substantial refinement to achieve high-fidelity geometry. Here, we propose SwiftNDC, a fast and general framework built around a Neural Depth Correction field that produces cross-view consistent depth maps. From these refined depths, we generate a dense point cloud through back-projection and robust reprojection-error filtering, obtaining a clean and uniformly distributed geometric initialization for downstream reconstruction. This reliable dense geometry substantially accelerates 3D Gaussian Splatting (3DGS) for mesh reconstruction, enabling high-quality surfaces with significantly fewer optimization iterations. For novel-view synthesis, SwiftNDC can also improve 3DGS rendering quality, highlighting the benefits of strong geometric initialization. We conduct a comprehensive study across five datasets, including two for mesh reconstruction, as well as three for novel-view synthesis. SwiftNDC consistently reduces running time for accurate mesh reconstruction and boosts rendering fidelity for view synthesis, demonstrating the effectiveness of combining neural depth refinement with robust geometric initialization for high-fidelity and efficient 3D reconstruction.

  </details>



- **AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction**  
  Hanyang Liu, Rongjun Qin  
  _2026-02-25_ · https://arxiv.org/abs/2602.22376v1  
  <details><summary>Abstract</summary>

  Recent advances in 4D scene reconstruction have significantly improved dynamic modeling across various domains. However, existing approaches remain limited under aerial conditions with single-view capture, wide spatial range, and dynamic objects of limited spatial footprint and large motion disparity. These challenges cause severe depth ambiguity and unstable motion estimation, making monocular aerial reconstruction inherently ill-posed. To this end, we present AeroDGS, a physics-guided 4D Gaussian splatting framework for monocular UAV videos. AeroDGS introduces a Monocular Geometry Lifting module that reconstructs reliable static and dynamic geometry from a single aerial sequence, providing a robust basis for dynamic estimation. To further resolve monocular ambiguity, we propose a Physics-Guided Optimization module that incorporates differentiable ground-support, upright-stability, and trajectory-smoothness priors, transforming ambiguous image cues into physically consistent motion. The framework jointly refines static backgrounds and dynamic entities with stable geometry and coherent temporal evolution. We additionally build a real-world UAV dataset that spans various altitudes and motion conditions to evaluate dynamic aerial reconstruction. Experiments on synthetic and real UAV scenes demonstrate that AeroDGS outperforms state-of-the-art methods, achieving superior reconstruction fidelity in dynamic aerial environments.

  </details>



- **Space-Time Forecasting of Dynamic Scenes with Motion-aware Gaussian Grouping**  
  Junmyeong Lee, Hoseung Choi, Minsu Cho  
  _2026-02-25_ · https://arxiv.org/abs/2602.21668v1  
  <details><summary>Abstract</summary>

  Forecasting dynamic scenes remains a fundamental challenge in computer vision, as limited observations make it difficult to capture coherent object-level motion and long-term temporal evolution. We present Motion Group-aware Gaussian Forecasting (MoGaF), a framework for long-term scene extrapolation built upon the 4D Gaussian Splatting representation. MoGaF introduces motion-aware Gaussian grouping and group-wise optimization to enforce physically consistent motion across both rigid and non-rigid regions, yielding spatially coherent dynamic representations. Leveraging this structured space-time representation, a lightweight forecasting module predicts future motion, enabling realistic and temporally stable scene evolution. Experiments on synthetic and real-world datasets demonstrate that MoGaF consistently outperforms existing baselines in rendering quality, motion plausibility, and long-term forecasting stability. Our project page is available at https://slime0519.github.io/mogaf

  </details>



- **Generalizing Visual Geometry Priors to Sparse Gaussian Occupancy Prediction**  
  Changqing Zhou, Yueru Luo, Changhao Chen  
  _2026-02-25_ · https://arxiv.org/abs/2602.21552v1  
  <details><summary>Abstract</summary>

  Accurate 3D scene understanding is essential for embodied intelligence, with occupancy prediction emerging as a key task for reasoning about both objects and free space. Existing approaches largely rely on depth priors (e.g., DepthAnything) but make only limited use of 3D cues, restricting performance and generalization. Recently, visual geometry models such as VGGT have shown strong capability in providing rich 3D priors, but similar to monocular depth foundation models, they still operate at the level of visible surfaces rather than volumetric interiors, motivating us to explore how to more effectively leverage these increasingly powerful geometry priors for 3D occupancy prediction. We present GPOcc, a framework that leverages generalizable visual geometry priors (GPs) for monocular occupancy prediction. Our method extends surface points inward along camera rays to generate volumetric samples, which are represented as Gaussian primitives for probabilistic occupancy inference. To handle streaming input, we further design a training-free incremental update strategy that fuses per-frame Gaussians into a unified global representation. Experiments on Occ-ScanNet and EmbodiedOcc-ScanNet demonstrate significant gains: GPOcc improves mIoU by +9.99 in the monocular setting and +11.79 in the streaming setting over prior state of the art. Under the same depth prior, it achieves +6.73 mIoU while running 2.65$\times$ faster. These results highlight that GPOcc leverages geometry priors more effectively and efficiently. Code will be released at https://github.com/JuIvyy/GPOcc.

  </details>



- **BrepGaussian: CAD reconstruction from Multi-View Images with Gaussian Splatting**  
  Jiaxing Yu, Dongyang Ren, Hangyu Xu, Zhouyuxiao Yang, Yuanqi Li, Jie Guo, Zhengkang Zhou, Yanwen Guo  
  _2026-02-24_ · https://arxiv.org/abs/2602.21105v1  
  <details><summary>Abstract</summary>

  The boundary representation (B-rep) models a 3D solid as its explicit boundaries: trimmed corners, edges, and faces. Recovering B-rep representation from unstructured data is a challenging and valuable task of computer vision and graphics. Recent advances in deep learning have greatly improved the recovery of 3D shape geometry, but still depend on dense and clean point clouds and struggle to generalize to novel shapes. We propose B-rep Gaussian Splatting (BrepGaussian), a novel framework that learns 3D parametric representations from 2D images. We employ a Gaussian Splatting renderer with learnable features, followed by a specific fitting strategy. To disentangle geometry reconstruction and feature learning, we introduce a two-stage learning framework that first captures geometry and edges and then refines patch features to achieve clean geometry and coherent instance representations. Extensive experiments demonstrate the superior performance of our approach to state-of-the-art methods. We will release our code and datasets upon acceptance.

  </details>


