# 3D Reconstruction

_Updated: 2026-04-02 07:39 UTC_

Total papers shown: **21**


---

- **Neural Harmonic Textures for High-Quality Primitive Based Neural Reconstruction**  
  Jorge Condor, Nicolas Moenne-Loccoz, Merlin Nimier-David, Piotr Didyk, Zan Gojcic, Qi Wu  
  _2026-04-01_ · https://arxiv.org/abs/2604.01204v1  
  <details><summary>Abstract</summary>

  Primitive-based methods such as 3D Gaussian Splatting have recently become the state-of-the-art for novel-view synthesis and related reconstruction tasks. Compared to neural fields, these representations are more flexible, adaptive, and scale better to large scenes. However, the limited expressivity of individual primitives makes modeling high-frequency detail challenging. We introduce Neural Harmonic Textures, a neural representation approach that anchors latent feature vectors on a virtual scaffold surrounding each primitive. These features are interpolated within the primitive at ray intersection points. Inspired by Fourier analysis, we apply periodic activations to the interpolated features, turning alpha blending into a weighted sum of harmonic components. The resulting signal is then decoded in a single deferred pass using a small neural network, significantly reducing computational cost. Neural Harmonic Textures yield state-of-the-art results in real-time novel view synthesis while bridging the gap between primitive- and neural-field-based reconstruction. Our method integrates seamlessly into existing primitive-based pipelines such as 3DGUT, Triangle Splatting, and 2DGS. We further demonstrate its generality with applications to 2D image fitting and semantic reconstruction.

  </details>



- **Open-Set Supervised 3D Anomaly Detection: An Industrial Dataset and a Generalisable Framework for Unknown Defects**  
  Hanzhe Liang, Luocheng Zhang, Junyang Xia, HanLiang Zhou, Bingyang Guo, Yingxi Xie, Can Gao, Ruiyun Yu, Jinbao Wang, Pan Li  
  _2026-04-01_ · https://arxiv.org/abs/2604.01171v1  
  <details><summary>Abstract</summary>

  Although self-supervised 3D anomaly detection assumes that acquiring high-precision point clouds is computationally expensive, in real manufacturing scenarios it is often feasible to collect a limited number of anomalous samples. Therefore, we study open-set supervised 3D anomaly detection, where the model is trained with only normal samples and a small number of known anomalous samples, aiming to identify unknown anomalies at test time. We present Open-Industry, a high-quality industrial dataset containing 15 categories, each with five real anomaly types collected from production lines. We first adapt general open-set anomaly detection methods to accommodate 3D point cloud inputs better. Building upon this, we propose Open3D-AD, a point-cloud-oriented approach that leverages normal samples, simulated anomalies, and partially observed real anomalies to model the probability density distributions of normal and anomalous data. Then, we introduce a simple Correspondence Distributions Subsampling to reduce the overlap between normal and non-normal distributions, enabling stronger dual distributions modeling. Based on these contributions, we establish a comprehensive benchmark and evaluate the proposed method extensively on Open-Industry as well as established datasets including Real3D-AD and Anomaly-ShapeNet. Benchmark results and ablation studies demonstrate the effectiveness of Open3D-AD and further reveal the potential of open-set supervised 3D anomaly detection.

  </details>



- **ReinDriveGen: Reinforcement Post-Training for Out-of-Distribution Driving Scene Generation**  
  Hao Zhang, Lue Fan, Weikang Bian, Zehuan Wu, Lewei Lu, Zhaoxiang Zhang, Hongsheng Li  
  _2026-04-01_ · https://arxiv.org/abs/2604.01129v1  
  <details><summary>Abstract</summary>

  We present ReinDriveGen, a framework that enables full controllability over dynamic driving scenes, allowing users to freely edit actor trajectories to simulate safety-critical corner cases such as front-vehicle collisions, drifting cars, vehicles spinning out of control, pedestrians jaywalking, and cyclists cutting across lanes. Our approach constructs a dynamic 3D point cloud scene from multi-frame LiDAR data, introduces a vehicle completion module to reconstruct full 360° geometry from partial observations, and renders the edited scene into 2D condition images that guide a video diffusion model to synthesize realistic driving videos. Since such edited scenarios inevitably fall outside the training distribution, we further propose an RL-based post-training strategy with a pairwise preference model and a pairwise reward mechanism, enabling robust quality improvement under out-of-distribution conditions without ground-truth supervision. Extensive experiments demonstrate that ReinDriveGen outperforms existing approaches on edited driving scenarios and achieves state-of-the-art results on novel ego viewpoint synthesis.

  </details>



- **Lightweight Prompt-Guided CLIP Adaptation for Monocular Depth Estimation**  
  Reyhaneh Ahani Manghotay, Jie Liang  
  _2026-04-01_ · https://arxiv.org/abs/2604.01118v1  
  <details><summary>Abstract</summary>

  Leveraging the rich semantic features of vision-language models (VLMs) like CLIP for monocular depth estimation tasks is a promising direction, yet often requires extensive fine-tuning or lacks geometric precision. We present a parameter-efficient framework, named MoA-DepthCLIP, that adapts pretrained CLIP representations for monocular depth estimation with minimal supervision. Our method integrates a lightweight Mixture-of-Adapters (MoA) module into the pretrained Vision Transformer (ViT-B/32) backbone combined with selective fine-tuning of the final layers. This design enables spatially-aware adaptation, guided by a global semantic context vector and a hybrid prediction architecture that synergizes depth bin classification with direct regression. To enhance structural accuracy, we employ a composite loss function that enforces geometric constraints. On the NYU Depth V2 benchmark, MoA-DepthCLIP achieves competitive results, significantly outperforming the DepthCLIP baseline by improving the $δ_1$ accuracy from 0.390 to 0.745 and reducing the RMSE from 1.176 to 0.520. These results are achieved while requiring substantially few trainable parameters, demonstrating that lightweight, prompt-guided MoA is a highly effective strategy for transferring VLM knowledge to fine-grained monocular depth estimation tasks.

  </details>



- **Sub-metre Lunar DEM Generation and Validation from Chandrayaan-2 OHRC Multi-View Imagery Using Open-Source Photogrammetry**  
  Aaranay Aadi, Jai Singla, Nitant Dube, Oleg Alexandrov  
  _2026-04-01_ · https://arxiv.org/abs/2604.01032v1  
  <details><summary>Abstract</summary>

  High-resolution digital elevation models (DEMs) of the lunar surface are essential for surface mobility planning, landing site characterization, and planetary science. The Orbiter High Resolution Camera (OHRC) on board Chandrayaan-2 has the best ground sampling capabilities of any lunar orbital imaging currently in use by acquiring panchromatic imagery at a resolution of roughly 20-30 cm per pixel. This work presents, for the first time, the generation of sub-metre DEMs from OHRC multi-view imagery using an exclusively open-source pipeline. Candidate stereo pairs are identified from non-paired OHRC archives through geometric analysis of image metadata, employing baseline-to-height (B/H) ratio computation and convergence angle estimation. Dense stereo correspondence and ray triangulation are then applied to generate point clouds, which are gridded into DEMs at effective spatial resolutions between approximately 24 and 54 cm across five geographically distributed lunar sites. Absolute elevation consistency is established through Iterative Closest Point (ICP) alignment against Lunar Reconnaissance Orbiter Narrow Angle Camera (NAC) Digital Terrain Models, followed by constant-bias offset correction. Validation against NAC reference terrain yields a vertical RMSE of 5.85 m (at native OHRC resolution), and a horizontal accuracy of less than 30 cm assessed by planimetric feature matching.

  </details>



- **EgoSim: Egocentric World Simulator for Embodied Interaction Generation**  
  Jinkun Hao, Mingda Jia, Ruiyan Wang, Xihui Liu, Ran Yi, Lizhuang Ma, Jiangmiao Pang, Xudong Xu  
  _2026-04-01_ · https://arxiv.org/abs/2604.01001v1  
  <details><summary>Abstract</summary>

  We introduce EgoSim, a closed-loop egocentric world simulator that generates spatially consistent interaction videos and persistently updates the underlying 3D scene state for continuous simulation. Existing egocentric simulators either lack explicit 3D grounding, causing structural drift under viewpoint changes, or treat the scene as static, failing to update world states across multi-stage interactions. EgoSim addresses both limitations by modeling 3D scenes as updatable world states. We generate embodiment interactions via a Geometry-action-aware Observation Simulation model, with spatial consistency from an Interaction-aware State Updating module. To overcome the critical data bottleneck posed by the difficulty in acquiring densely aligned scene-interaction training pairs, we design a scalable pipeline that extracts static point clouds, camera trajectories, and embodiment actions from in-the-wild large-scale monocular egocentric videos. We further introduce EgoCap, a capture system that enables low-cost real-world data collection with uncalibrated smartphones. Extensive experiments demonstrate that EgoSim significantly outperforms existing methods in terms of visual quality, spatial consistency, and generalization to complex scenes and in-the-wild dexterous interactions, while supporting cross-embodiment transfer to robotic manipulation. Codes and datasets will be open soon. The project page is at egosimulator.github.io.

  </details>



- **Shape Representation using Gaussian Process mixture models**  
  Panagiotis Sapoutzoglou, George Terzakis, Georgios Floros, Maria Pateraki  
  _2026-04-01_ · https://arxiv.org/abs/2604.00862v1  
  <details><summary>Abstract</summary>

  Traditional explicit 3D representations, such as point clouds and meshes, demand significant storage to capture fine geometric details and require complex indexing systems for surface lookups, making functional representations an efficient, compact, and continuous alternative. In this work, we propose a novel, object-specific functional shape representation that models surface geometry with Gaussian Process (GP) mixture models. Rather than relying on computationally heavy neural architectures, our method is lightweight, leveraging GPs to learn continuous directional distance fields from sparsely sampled point clouds. We capture complex topologies by anchoring local GP priors at strategic reference points, which can be flexibly extracted using any structural decomposition method (e.g. skeletonization, distance-based clustering). Extensive evaluations on the ShapeNetCore and IndustryShapes datasets demonstrate that our method can efficiently and accurately represent complex geometries.

  </details>



- **Sparkle: A Robust and Versatile Representation for Point Cloud based Human Motion Capture**  
  Yiming Ren, Yujing Sun, Aoru Xue, Kwok-Yan Lam, Yuexin Ma  
  _2026-04-01_ · https://arxiv.org/abs/2604.00857v1  
  <details><summary>Abstract</summary>

  Point cloud-based motion capture leverages rich spatial geometry and privacy-preserving sensing, but learning robust representations from noisy, unstructured point clouds remains challenging. Existing approaches face a struggle trade-off between point-based methods (geometrically detailed but noisy) and skeleton-based ones (robust but oversimplified). We address the fundamental challenge: how to construct an effective representation for human motion capture that can balance expressiveness and robustness. In this paper, we propose Sparkle, a structured representation unifying skeletal joints and surface anchors with explicit kinematic-geometric factorization. Our framework, SparkleMotion, learns this representation through hierarchical modules embedding geometric continuity and kinematic constraints. By explicitly disentangling internal kinematic structure from external surface geometry, SparkleMotion achieves state-of-the-art performance not only in accuracy but crucially in robustness and generalization under severe domain shifts, noise, and occlusion. Extensive experiments demonstrate our superiority across diverse sensor types and challenging real-world scenarios.

  </details>



- **MoonAnything: A Vision Benchmark with Large-Scale Lunar Supervised Data**  
  Clémentine Grethen, Yuang Shi, Simone Gasparini, Géraldine Morin  
  _2026-04-01_ · https://arxiv.org/abs/2604.00682v1  
  <details><summary>Abstract</summary>

  Accurate perception of lunar surfaces is critical for modern lunar exploration missions. However, developing robust learning-based perception systems is hindered by the lack of datasets that provide both geometric and photometric supervision. Existing lunar datasets typically lack either geometric ground truth, photometric realism, illumination diversity, or large-scale coverage. In this paper, we introduce MoonAnything, a unified benchmark built on real lunar topography with physically-based rendering, providing the first comprehensive geometric and photometric supervision under diverse illumination with large scale. The benchmark comprises two complementary sub-datasets : i) LunarGeo provides stereo images with corresponding dense depth maps and camera calibration enabling 3D reconstruction and pose estimation; ii) LunarPhoto provides photorealistic images using a spatially-varying BRDF model, along with multi-illumination renderings under real solar configurations, enabling reflectance estimation and illumination-robust perception. Together, these datasets offer over 130K samples with comprehensive supervision. Beyond lunar applications, MoonAnything offers a unique setting and challenging testbed for algorithms under low-textured, high-contrast conditions and applies to other airless celestial bodies and could generalize beyond. We establish baselines using state-of-the-art methods and release the complete dataset along with generation tools to support community extension: https://github.com/clementinegrethen/MoonAnything.

  </details>



- **Reliev3R: Relieving Feed-forward Reconstruction from Multi-View Geometric Annotations**  
  Youyu Chen, Junjun Jiang, Yueru Luo, Kui Jiang, Xianming Liu, Xu Yan, Dave Zhenyu Chen  
  _2026-04-01_ · https://arxiv.org/abs/2604.00548v1  
  <details><summary>Abstract</summary>

  With recent advances, Feed-forward Reconstruction Models (FFRMs) have demonstrated great potential in reconstruction quality and adaptiveness to multiple downstream tasks. However, the excessive reliance on multi-view geometric annotations, e.g. 3D point maps and camera poses, makes the fully-supervised training scheme of FFRMs difficult to scale up. In this paper, we propose Reliev3R, a weakly-supervised paradigm for training FFRMs from scratch without cost-prohibitive multi-view geometric annotations. Relieving the reliance on geometric sensory data and compute-exhaustive structure-from-motion preprocessing, our method draws 3D knowledge directly from monocular relative depths and image sparse correspondences given by zero-shot predictions of pretrained models. At the core of Reliev3R, we design an ambiguity-aware relative depth loss and a trigonometry-based reprojection loss to facilitate supervision for multi-view geometric consistency. Training from scratch with the less data, Reliev3R catches up with its fully-supervised sibling models, taking a step towards low-cost 3D reconstruction supervisions and scalable FFRMs.

  </details>



- **Think, Act, Build: An Agentic Framework with Vision Language Models for Zero-Shot 3D Visual Grounding**  
  Haibo Wang, Zihao Lin, Zhiyang Xu, Lifu Huang  
  _2026-04-01_ · https://arxiv.org/abs/2604.00528v1  
  <details><summary>Abstract</summary>

  3D Visual Grounding (3D-VG) aims to localize objects in 3D scenes via natural language descriptions. While recent advancements leveraging Vision-Language Models (VLMs) have explored zero-shot possibilities, they typically suffer from a static workflow relying on preprocessed 3D point clouds, essentially degrading grounding into proposal matching. To bypass this reliance, our core motivation is to decouple the task: leveraging 2D VLMs to resolve complex spatial semantics, while relying on deterministic multi-view geometry to instantiate the 3D structure. Driven by this insight, we propose "Think, Act, Build (TAB)", a dynamic agentic framework that reformulates 3D-VG tasks as a generative 2D-to-3D reconstruction paradigm operating directly on raw RGB-D streams. Specifically, guided by a specialized 3D-VG skill, our VLM agent dynamically invokes visual tools to track and reconstruct the target across 2D frames. Crucially, to overcome the multi-view coverage deficit caused by strict VLM semantic tracking, we introduce the Semantic-Anchored Geometric Expansion, a mechanism that first anchors the target in a reference video clip and then leverages multi-view geometry to propagate its spatial location across unobserved frames. This enables the agent to "Build" the target's 3D representation by aggregating these multi-view features via camera parameters, directly mapping 2D visual cues to 3D coordinates. Furthermore, to ensure rigorous assessment, we identify flaws such as reference ambiguity and category errors in existing benchmarks and manually refine the incorrect queries. Extensive experiments on ScanRefer and Nr3D demonstrate that our framework, relying entirely on open-source models, significantly outperforms previous zero-shot methods and even surpasses fully supervised baselines.

  </details>



- **Neural Reconstruction of LiDAR Point Clouds under Jamming Attacks via Full-Waveform Representation and Simultaneous Laser Sensing**  
  Ryo Yoshida, Takami Sato, Wenlun Zhang, Yuki Hayakawa, Shota Nagai, Takahiro Kado, Taro Beppu, Ibuki Fujioka, Yunshan Zhong, Kentaro Yoshioka  
  _2026-04-01_ · https://arxiv.org/abs/2604.00371v1  
  <details><summary>Abstract</summary>

  LiDAR sensors are critical for autonomous driving perception, yet remain vulnerable to spoofing attacks. Jamming attacks inject high-frequency laser pulses that completely blind LiDAR sensors by overwhelming authentic returns with malicious signals. We discover that while point clouds become randomized, the underlying full-waveform data retains distinguishable signatures between attack and legitimate signals. In this work, we propose PULSAR-Net, capable of reconstructing authentic point clouds under jamming attacks by leveraging previously underutilized intermediate full-waveform representations and simultaneous laser sensing in modern LiDAR systems. PULSAR-Net adopts a novel U-Net architecture with axial spatial attention mechanisms specifically designed to identify attack-induced signals from authentic object returns in the full-waveform representation. To address the lack of full-waveform representations in existing LiDAR datasets under jamming attacks, we introduce a physics-aware dataset generation pipeline that synthesizes realistic full-waveform representations under jamming attacks. Despite being trained exclusively on synthetic data, PULSAR-Net achieves reconstruction rates of 92% and 73% for vehicles obscured by jamming attacks in real-world static and driving scenarios, respectively.

  </details>



- **Dual Contouring of Signed Distance Data**  
  Xiana Carrera, Ningna Wang, Christopher Batty, Oded Stein, Silvia Sellán  
  _2026-03-31_ · https://arxiv.org/abs/2604.00157v1  
  <details><summary>Abstract</summary>

  We propose an algorithm to reconstruct explicit polygonal meshes from discretely sampled Signed Distance Function (SDF) data, which is especially effective at recovering sharp features. Building on the traditional Dual Contouring of Hermite Data method, we design and solve a quadratic optimization problem to decide the optimal placement of the mesh's vertices within each cell of a regular grid. Critically, this optimization relies solely on discretely sampled SDF data, without requiring arbitrary access to the function, gradient information, or training on large-scale datasets. Our method sets a new state of the art in surface reconstruction from SDFs at medium and high resolutions, and opens the door for applications in 3D modeling and design.

  </details>



- **OmniRoam: World Wandering via Long-Horizon Panoramic Video Generation**  
  Yuheng Liu, Xin Lin, Xinke Li, Baihan Yang, Chen Wang, Kalyan Sunkavalli, Yannick Hold-Geoffroy, Hao Tan, Kai Zhang, Xiaohui Xie, et al.  
  _2026-03-31_ · https://arxiv.org/abs/2603.30045v1  
  <details><summary>Abstract</summary>

  Modeling scenes using video generation models has garnered growing research interest in recent years. However, most existing approaches rely on perspective video models that synthesize only limited observations of a scene, leading to issues of completeness and global consistency. We propose OmniRoam, a controllable panoramic video generation framework that exploits the rich per-frame scene coverage and inherent long-term spatial and temporal consistency of panoramic representation, enabling long-horizon scene wandering. Our framework begins with a preview stage, where a trajectory-controlled video generation model creates a quick overview of the scene from a given input image or video. Then, in the refine stage, this video is temporally extended and spatially upsampled to produce long-range, high-resolution videos, thus enabling high-fidelity world wandering. To train our model, we introduce two panoramic video datasets that incorporate both synthetic and real-world captured videos. Experiments show that our framework consistently outperforms state-of-the-art methods in terms of visual quality, controllability, and long-term scene consistency, both qualitatively and quantitatively. We further showcase several extensions of this framework, including real-time video generation and 3D reconstruction. Code is available at https://github.com/yuhengliu02/OmniRoam.

  </details>



- **AA-Splat: Anti-Aliased Feed-forward Gaussian Splatting**  
  Taewoo Suh, Sungpyo Kim, Jongmin Park, Munchurl Kim  
  _2026-03-31_ · https://arxiv.org/abs/2603.29394v1  
  <details><summary>Abstract</summary>

  Feed-forward 3D Gaussian Splatting (FF-3DGS) emerges as a fast and robust solution for sparse-view 3D reconstruction and novel view synthesis (NVS). However, existing FF-3DGS methods are built on incorrect screen-space dilation filters, causing severe rendering artifacts when rendering at out-of-distribution sampling rates. We firstly propose an FF-3DGS model, called AA-Splat, to enable robust anti-aliased rendering at any resolution. AA-Splat utilizes an opacity-balanced band-limiting (OBBL) design, which combines two components: a 3D band-limiting post-filter integrates multi-view maximal frequency bounds into the feed-forward reconstruction pipeline, effectively band-limiting the resulting 3D scene representations and eliminating degenerate Gaussians; an Opacity Balancing (OB) to seamlessly integrate all pixel-aligned Gaussian primitives into the rendering process, compensating for the increased overlap between expanded Gaussian primitives. AA-Splat demonstrates drastic improvements with average 5.4$\sim$7.5dB PSNR gains on NVS performance over a state-of-the-art (SOTA) baseline, DepthSplat, at all resolutions, between $4\times$ and $1/4\times$. Code will be made available.

  </details>



- **Extend3D: Town-Scale 3D Generation**  
  Seungwoo Yoon, Jinmo Kim, Jaesik Park  
  _2026-03-31_ · https://arxiv.org/abs/2603.29387v1  
  <details><summary>Abstract</summary>

  In this paper, we propose Extend3D, a training-free pipeline for 3D scene generation from a single image, built upon an object-centric 3D generative model. To overcome the limitations of fixed-size latent spaces in object-centric models for representing wide scenes, we extend the latent space in the $x$ and $y$ directions. Then, by dividing the extended latent space into overlapping patches, we apply the object-centric 3D generative model to each patch and couple them at each time step. Since patch-wise 3D generation with image conditioning requires strict spatial alignment between image and latent patches, we initialize the scene using a point cloud prior from a monocular depth estimator and iteratively refine occluded regions through SDEdit. We discovered that treating the incompleteness of 3D structure as noise during 3D refinement enables 3D completion via a concept, which we term under-noising. Furthermore, to address the sub-optimality of object-centric models for sub-scene generation, we optimize the extended latent during denoising, ensuring that the denoising trajectories remain consistent with the sub-scene dynamics. To this end, we introduce 3D-aware optimization objectives for improved geometric structure and texture fidelity. We demonstrate that our method yields better results than prior methods, as evidenced by human preference and quantitative experiments.

  </details>



- **StereoVGGT: A Training-Free Visual Geometry Transformer for Stereo Vision**  
  Ziyang Chen, Yansong Qu, You Shen, Xuan Cheng, Liujuan Cao  
  _2026-03-31_ · https://arxiv.org/abs/2603.29368v1  
  <details><summary>Abstract</summary>

  Driven by the advancement of 3D devices, stereo vision tasks including stereo matching and stereo conversion have emerged as a critical research frontier. Contemporary stereo vision backbones typically rely on either monocular depth estimation (MDE) models or visual foundation models (VFMs). Crucially, these models are predominantly pretrained without explicit supervision of camera poses. Given that such geometric knowledge is indispensable for stereo vision, the absence of explicit spatial constraints constitutes a significant performance bottleneck for existing architectures. Recognizing that the Visual Geometry Grounded Transformer (VGGT) operates as a foundation model pretrained on extensive 3D priors, including camera poses, we investigate its potential as a robust backbone for stereo vision tasks. Nevertheless, empirical results indicate that its direct application to stereo vision yields suboptimal performance. We observe that VGGT suffers from a more significant degradation of geometric details during feature extraction. Such characteristics conflict with the requirements of binocular stereo vision, thereby constraining its efficacy for relative tasks. To bridge this gap, we propose StereoVGGT, a feature backbone specifically tailored for stereo vision. By leveraging the frozen VGGT and introducing a training-free feature adjustment pipeline, we mitigate geometric degradation and harness the latent camera calibration knowledge embedded within the model. StereoVGGT-based stereo matching network achieved the $1^{st}$ rank among all published methods on the KITTI benchmark, validating that StereoVGGT serves as a highly effective backbone for stereo vision.

  </details>



- **Stepper: Stepwise Immersive Scene Generation with Multiview Panoramas**  
  Felix Wimbauer, Fabian Manhardt, Michael Oechsle, Nikolai Kalischek, Christian Rupprecht, Daniel Cremers, Federico Tombari  
  _2026-03-30_ · https://arxiv.org/abs/2603.28980v1  
  <details><summary>Abstract</summary>

  The synthesis of immersive 3D scenes from text is rapidly maturing, driven by novel video generative models and feed-forward 3D reconstruction, with vast potential in AR/VR and world modeling. While panoramic images have proven effective for scene initialization, existing approaches suffer from a trade-off between visual fidelity and explorability: autoregressive expansion suffers from context drift, while panoramic video generation is limited to low resolution. We present Stepper, a unified framework for text-driven immersive 3D scene synthesis that circumvents these limitations via stepwise panoramic scene expansion. Stepper leverages a novel multi-view 360° diffusion model that enables consistent, high-resolution expansion, coupled with a geometry reconstruction pipeline that enforces geometric coherence. Trained on a new large-scale, multi-view panorama dataset, Stepper achieves state-of-the-art fidelity and structural consistency, outperforming prior approaches, thereby setting a new standard for immersive scene generation.

  </details>



- **Fisheye3R: Adapting Unified 3D Feed-Forward Foundation Models to Fisheye Lenses**  
  Ruxiao Duan, Erin Hong, Dongxu Zhao, Eric Turner, Alex Wong, Yunwen Zhou  
  _2026-03-30_ · https://arxiv.org/abs/2603.28896v1  
  <details><summary>Abstract</summary>

  Feed-forward foundation models for multi-view 3-dimensional (3D) reconstruction have been trained on large-scale datasets of perspective images; when tested on wide field-of-view images, e.g., from a fisheye camera, their performance degrades. Their error arises from changes in spatial positions of pixels due to a non-linear projection model that maps 3D points onto the 2D image plane. While one may surmise that training on fisheye images would resolve this problem, there are far fewer fisheye images with ground truth than perspective images, which limit generalization. To enable inference on imagery exhibiting high radial distortion, we propose Fisheye3R, a novel adaptation framework that extends these multi-view 3D reconstruction foundation models to natively accommodate fisheye inputs without performance regression on perspective images. To address the scarcity of fisheye images and ground truth, we introduce flexible learning schemes that support self-supervised adaptation using only unlabeled perspective images and supervised adaptation without any fisheye training data. Extensive experiments across three foundation models, including VGGT, $π^3$, and MapAnything, demonstrate that our approach consistently improves camera pose, depth, point map, and field-of-view estimation on fisheye images.

  </details>



- **Industrial3D: A Terrestrial LiDAR Point Cloud Dataset and CrossParadigm Benchmark for Industrial Infrastructure**  
  Chao Yin, Hongzhe Yue, Qing Han, Difeng Hu, Zhenyu Liang, Fangzhou Lin, Bing Sun, Boyu Wang, Mingkai Li, Wei Yao, et al.  
  _2026-03-30_ · https://arxiv.org/abs/2603.28660v1  
  <details><summary>Abstract</summary>

  Automated semantic understanding of dense point clouds is a prerequisite for Scan-to-BIM pipelines, digital twin construction, and as-built verification--core tasks in the digital transformation of the construction industry. Yet for industrial mechanical, electrical, and plumbing (MEP) facilities, this challenge remains largely unsolved: TLS acquisitions of water treatment plants, chiller halls, and pumping stations exhibit extreme geometric ambiguity, severe occlusion, and extreme class imbalance that architectural benchmarks (e.g., S3DIS or ScanNet) cannot adequately represent. We present Industrial3D, a terrestrial LiDAR dataset comprising 612 million expertly labelled points at 6 mm resolution from 13 water treatment facilities. At 6.6x the scale of the closest comparable MEP dataset, Industrial3D provides the largest and most demanding testbed for industrial 3D scene understanding to date. We further establish the first industrial cross-paradigm benchmark, evaluating nine representative methods across fully supervised, weakly supervised, unsupervised, and foundation model settings under a unified benchmark protocol. The best supervised method achieves 55.74% mIoU, whereas zero-shot Point-SAM reaches only 15.79%--a 39.95 percentage-point gap that quantifies the unresolved domain-transfer challenge for industrial TLS data. Systematic analysis reveals that this gap originates from a dual crisis: statistical rarity (215:1 imbalance, 3.5x more severe than S3DIS) and geometric ambiguity (tail-class points share cylindrical primitives with head-class pipes) that frequency-based re-weighting alone cannot resolve. Industrial3D, along with benchmark code and pre-trained models, will be publicly available at https://github.com/pointcloudyc/Industrial3D.

  </details>



- **Seen2Scene: Completing Realistic 3D Scenes with Visibility-Guided Flow**  
  Quan Meng, Yujin Chen, Lei Li, Matthias Nießner, Angela Dai  
  _2026-03-30_ · https://arxiv.org/abs/2603.28548v1  
  <details><summary>Abstract</summary>

  We present Seen2Scene, the first flow matching-based approach that trains directly on incomplete, real-world 3D scans for scene completion and generation. Unlike prior methods that rely on complete and hence synthetic 3D data, our approach introduces visibility-guided flow matching, which explicitly masks out unknown regions in real scans, enabling effective learning from real-world, partial observations. We represent 3D scenes using truncated signed distance field (TSDF) volumes encoded in sparse grids and employ a sparse transformer to efficiently model complex scene structures while masking unknown regions. We employ 3D layout boxes as an input conditioning signal, and our approach is flexibly adapted to various other inputs such as text or partial scans. By learning directly from real-world, incomplete 3D scans, Seen2Scene enables realistic 3D scene completion for complex, cluttered real environments. Experiments demonstrate that our model produces coherent, complete, and realistic 3D scenes, outperforming baselines in completion accuracy and generation quality.

  </details>


