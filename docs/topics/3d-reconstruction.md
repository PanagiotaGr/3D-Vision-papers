# 3D Reconstruction

_Updated: 2026-03-26 07:34 UTC_

Total papers shown: **21**


---

- **EndoVGGT: GNN-Enhanced Depth Estimation for Surgical 3D Reconstruction**  
  Falong Fan, Yi Xie, Arnis Lektauers, Bo Liu, Jerzy Rozenblit  
  _2026-03-25_ · https://arxiv.org/abs/2603.24577v1  
  <details><summary>Abstract</summary>

  Accurate 3D reconstruction of deformable soft tissues is essential for surgical robotic perception. However, low-texture surfaces, specular highlights, and instrument occlusions often fragment geometric continuity, posing a challenge for existing fixed-topology approaches. To address this, we propose EndoVGGT, a geometry-centric framework equipped with a Deformation-aware Graph Attention (DeGAT) module. Rather than using static spatial neighborhoods, DeGAT dynamically constructs feature-space semantic graphs to capture long-range correlations among coherent tissue regions. This enables robust propagation of structural cues across occlusions, enforcing global consistency and improving non-rigid deformation recovery. Extensive experiments on SCARED show that our method significantly improves fidelity, increasing PSNR by 24.6% and SSIM by 9.1% over prior state-of-the-art. Crucially, EndoVGGT exhibits strong zero-shot cross-dataset generalization to the unseen SCARED and EndoNeRF domains, confirming that DeGAT learns domain-agnostic geometric priors. These results highlight the efficacy of dynamic feature-space modeling for consistent surgical 3D reconstruction.

  </details>



- **HGGT: Robust and Flexible 3D Hand Mesh Reconstruction from Uncalibrated Images**  
  Yumeng Liu, Xiao-Xiao Long, Marc Habermann, Xuanze Yang, Cheng Lin, Yuan Liu, Yuexin Ma, Wenping Wang, Ligang Liu  
  _2026-03-25_ · https://arxiv.org/abs/2603.23997v1  
  <details><summary>Abstract</summary>

  Recovering high-fidelity 3D hand geometry from images is a critical task in computer vision, holding significant value for domains such as robotics, animation and VR/AR. Crucially, scalable applications demand both accuracy and deployment flexibility, requiring the ability to leverage massive amounts of unstructured image data from the internet or enable deployment on consumer-grade RGB cameras without complex calibration. However, current methods face a dilemma. While single-view approaches are easy to deploy, they suffer from depth ambiguity and occlusion. Conversely, multi-view systems resolve these uncertainties but typically demand fixed, calibrated setups, limiting their real-world utility. To bridge this gap, we draw inspiration from 3D foundation models that learn explicit geometry directly from visual data. By reformulating hand reconstruction from arbitrary views as a visual-geometry grounded task, we propose a feed-forward architecture that, for the first time in literature, jointly infers 3D hand meshes and camera poses from uncalibrated views. Extensive evaluations show that our approach outperforms state-of-the-art benchmarks and demonstrates strong generalization to uncalibrated, in-the-wild scenarios. Here is the link of our project page: https://lym29.github.io/HGGT/.

  </details>



- **SLAT-Phys: Fast Material Property Field Prediction from Structured 3D Latents**  
  Rocktim Jyoti Das, Dinesh Manocha  
  _2026-03-25_ · https://arxiv.org/abs/2603.23973v1  
  <details><summary>Abstract</summary>

  Estimating the material property field of 3D assets is critical for physics-based simulation, robotics, and digital twin generation. Existing vision-based approaches are either too expensive and slow or rely on 3D information. We present SLAT-Phys, an end-to-end method that predicts spatially varying material property fields of 3D assets directly from a single RGB image without explicit 3D reconstruction. Our approach leverages spatially organised latent features from a pretrained 3D asset generation model that encodes rich geometry and semantic prior, and trains a lightweight neural decoder to estimate Young's modulus, density, and Poisson's ratio. The coarse volumetric layout and semantic cues of the latent representation about object geometry and appearance enable accurate material estimation. Our experiments demonstrate that our method provides competitive accuracy in predicting continuous material parameters when compared against prior approaches, while significantly reducing computation time. In particular, SLAT-Phys requires only 9.9 seconds per object on an NVIDIA RTXA5000 GPU and avoids reconstruction and voxelization preprocessing. This results in 120x speedup compared to prior methods and enables faster material property estimation from a single image.

  </details>



- **PointRFT: Explicit Reinforcement Fine-tuning for Point Cloud Few-shot Learning**  
  Yankai Wang, Yiding Sun, Qirui Wang, Pengbo Li, Chaoyi Lu, Dongxu Zhang  
  _2026-03-25_ · https://arxiv.org/abs/2603.23957v1  
  <details><summary>Abstract</summary>

  Understanding spatial dynamics and semantics in point cloud is fundamental for comprehensive 3D comprehension. While reinforcement learning algorithms such as Group Relative Policy Optimization (GRPO) have recently achieved remarkable breakthroughs in large language models by incentivizing reasoning capabilities through strategic reward design, their potential remains largely unexplored in the 3D perception domain. This naturally raises a pivotal question: Can RL-based methods effectively empower 3D point cloud fine-tuning? In this paper, we propose PointRFT, the first reinforcement fine-tuning paradigm tailored specifically for point cloud representation learning. We select three prevalent 3D foundation models and devise specialized accuracy reward and dispersion reward functions to stabilize training and mitigate distribution shifts. Through comprehensive few-shot classification experiments comparing distinct training paradigms, we demonstrate that PointRFT consistently outperforms vanilla supervised fine-tuning (SFT) across diverse benchmarks. Furthermore, when organically integrated into a hybrid Pretraining-SFT-RFT paradigm, the representational capacity of point cloud foundation models is substantially unleashed, achieving state-of-the-art performance particularly under data-scarce scenarios.

  </details>



- **An Adapter-free Fine-tuning Approach for Tuning 3D Foundation Models**  
  Sneha Paul, Zachary Patterson, Nizar Bouguila  
  _2026-03-24_ · https://arxiv.org/abs/2603.23730v1  
  <details><summary>Abstract</summary>

  Point cloud foundation models demonstrate strong generalization, yet adapting them to downstream tasks remains challenging in low-data regimes. Full fine-tuning often leads to overfitting and significant drift from pre-trained representations, while existing parameter-efficient fine-tuning (PEFT) methods mitigate this issue by introducing additional trainable components at the cost of increased inference-time latency. We propose Momentum-Consistency Fine-Tuning (MCFT), an adapter-free approach that bridges the gap between full and parameter-efficient fine-tuning. MCFT selectively fine-tunes a portion of the pre-trained encoder while enforcing a momentum-based consistency constraint to preserve task-agnostic representations. Unlike PEFT methods, MCFT introduces no additional representation learning parameters beyond a standard task head, maintaining the original model's parameter count and inference efficiency. We further extend MCFT with two variants: a semi-supervised framework that leverages abundant unlabeled data to enhance few-shot performance, and a pruning-based variant that improves computational efficiency through structured layer removal. Extensive experiments on object recognition and part segmentation benchmarks demonstrate that MCFT consistently outperforms prior methods, achieving a 3.30% gain in 5-shot settings and up to a 6.13% improvement with semi-supervised learning, while remaining well-suited for resource-constrained deployment.

  </details>



- **AdvSplat: Adversarial Attacks on Feed-Forward Gaussian Splatting Models**  
  Yiran Qiao, Yiren Lu, Yunlai Zhou, Rui Yang, Linlin Hou, Yu Yin, Jing Ma  
  _2026-03-24_ · https://arxiv.org/abs/2603.23686v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) is increasingly recognized as a powerful paradigm for real-time, high-fidelity 3D reconstruction. However, its per-scene optimization pipeline limits scalability and generalization, and prevents efficient inference. Recently emerged feed-forward 3DGS models address these limitations by enabling fast reconstruction from a few input views after large-scale pretraining, without scene-specific optimization. Despite their advantages and strong potential for commercial deployment, the use of neural networks as the backbone also amplifies the risk of adversarial manipulation. In this paper, we introduce AdvSplat, the first systematic study of adversarial attacks on feed-forward 3DGS. We first employ white-box attacks to reveal fundamental vulnerabilities of this model family. We then develop two improved, practically relevant, query-efficient black-box algorithms that optimize pixel-space perturbations via a frequency-domain parameterization: one based on gradient estimation and the other gradient-free, without requiring any access to model internals. Extensive experiments across multiple datasets demonstrate that AdvSplat can significantly disrupt reconstruction results by injecting imperceptible perturbations into the input images. Our findings surface an overlooked yet urgent problem in this domain, and we hope to draw the community's attention to this emerging security and robustness challenge.

  </details>



- **I3DM: Implicit 3D-aware Memory Retrieval and Injection for Consistent Video Scene Generation**  
  Jia Li, Han Yan, Yihang Chen, Siqi Li, Xibin Song, Yifu Wang, Jianfei Cai, Tien-Tsin Wong, Pan Ji  
  _2026-03-24_ · https://arxiv.org/abs/2603.23413v1  
  <details><summary>Abstract</summary>

  Despite remarkable progress in video generation, maintaining long-term scene consistency upon revisiting previously explored areas remains challenging. Existing solutions rely either on explicitly constructing 3D geometry, which suffers from error accumulation and scale ambiguity, or on naive camera Field-of-View (FoV) retrieval, which typically fails under complex occlusions. To overcome these limitations, we propose I3DM, a novel implicit 3D-aware memory mechanism for consistent video scene generation that bypasses explicit 3D reconstruction. At the core of our approach is a 3D-aware memory retrieval strategy, which leverages the intermediate features of a pre-trained Feed-Forward Novel View Synthesis (FF-NVS) model to score view relevance, enabling robust retrieval even in highly occluded scenarios. Furthermore, to fully utilize the retrieved historical frames, we introduce a 3D-aligned memory injection module. This module implicitly warps historical content to the target view and adaptively conditions the generation on reliable warping regions, leading to improved revisit consistency and accurate camera control. Extensive experiments demonstrate that our method outperforms state-of-the-art approaches, achieving superior revisit consistency, generation fidelity, and camera control precision.

  </details>



- **Contrastive Metric Learning for Point Cloud Segmentation in Highly Granular Detectors**  
  Max Marriott-Clarke, Lazar Novakovic, Elizabeth Ratzer, Robert J. Bainbridge, Loukas Gouskos, Benedikt Maier  
  _2026-03-24_ · https://arxiv.org/abs/2603.23356v1  
  <details><summary>Abstract</summary>

  We propose a novel clustering approach for point-cloud segmentation based on supervised contrastive metric learning (CML). Rather than predicting cluster assignments or object-centric variables, the method learns a latent representation in which points belonging to the same object are embedded nearby while unrelated points are separated. Clusters are then reconstructed using a density-based readout in the learned metric space, decoupling representation learning from cluster formation and enabling flexible inference. The approach is evaluated on simulated data from a highly granular calorimeter, where the task is to separate highly overlapping particle showers represented as sets of calorimeter hits. A direct comparison with object condensation (OC) is performed using identical graph neural network backbones and equal latent dimensionality, isolating the effect of the learning objective. The CML method produces a more stable and separable embedding geometry for both electromagnetic and hadronic particle showers, leading to improved local neighbourhood consistency, a more reliable separation of overlapping showers, and better generalization when extrapolating to unseen multiplicities and energies. This translates directly into higher reconstruction efficiency and purity, particularly in high-multiplicity regimes, as well as improved energy resolution. In mixed-particle environments, CML maintains strong performance, suggesting robust learning of the shower topology, while OC exhibits significant degradation. These results demonstrate that similarity-based representation learning combined with density-based aggregation is a promising alternative to object-centric approaches for point cloud segmentation in highly granular detectors.

  </details>



- **Pose-Free Omnidirectional Gaussian Splatting for 360-Degree Videos with Consistent Depth Priors**  
  Chuanqing Zhuang, Xin Lu, Zehui Deng, Zhengda Lu, Yiqun Wang, Junqi Diao, Jun Xiao  
  _2026-03-24_ · https://arxiv.org/abs/2603.23324v1  
  <details><summary>Abstract</summary>

  Omnidirectional 3D Gaussian Splatting with panoramas is a key technique for 3D scene representation, and existing methods typically rely on slow SfM to provide camera poses and sparse points priors. In this work, we propose a pose-free omnidirectional 3DGS method, named PFGS360, that reconstructs 3D Gaussians from unposed omnidirectional videos. To achieve accurate camera pose estimation, we first construct a spherical consistency-aware pose estimation module, which recovers poses by establishing consistent 2D-3D correspondences between the reconstructed Gaussians and the unposed images using Gaussians' internal depth priors. Besides, to enhance the fidelity of novel view synthesis, we introduce a depth-inlier-aware densification module to extract depth inliers and Gaussian outliers with consistent monocular depth priors, enabling efficient Gaussian densification and achieving photorealistic novel view synthesis. The experiments show significant outperformance over existing pose-free and pose-aware 3DGS methods on both real-world and synthetic 360-degree videos. Code is available at https://github.com/zcq15/PFGS360.

  </details>



- **CCF: Complementary Collaborative Fusion for Domain Generalized Multi-Modal 3D Object Detection**  
  Yuchen Wu, Kun Wang, Yining Pan, Na Zhao  
  _2026-03-24_ · https://arxiv.org/abs/2603.23276v1  
  <details><summary>Abstract</summary>

  Multi-modal fusion has emerged as a promising paradigm for accurate 3D object detection. However, performance degrades substantially when deployed in target domains different from training. In this work, focusing on dual-branch proposal-level detectors, we identify two factors that limit robust cross-domain generalization: 1) in challenging domains such as rain or nighttime, one modality may undergo severe degradation; 2) the LiDAR branch often dominates the detection process, leading to systematic underutilization of visual cues and vulnerability when point clouds are compromised. To address these challenges, we propose three components. First, Query-Decoupled Loss provides independent supervision for 2D-only, 3D-only, and fused queries, rebalancing gradient flow across modalities. Second, LiDAR-Guided Depth Prior augments 2D queries with instance-aware geometric priors through probabilistic fusion of image-predicted and LiDAR-derived depth distributions, improving their spatial initialization. Third, Complementary Cross-Modal Masking applies complementary spatial masks to the image and point cloud, encouraging queries from both modalities to compete within the fused decoder and thereby promoting adaptive fusion. Extensive experiments demonstrate substantial gains over state-of-the-art baselines while preserving source-domain performance. Code and models are publicly available at https://github.com/IMPL-Lab/CCF.

  </details>



- **GO-Renderer: Generative Object Rendering with 3D-aware Controllable Video Diffusion Models**  
  Zekai Gu, Shuoxuan Feng, Yansong Wang, Hanzhuo Huang, Zhongshuo Du, Chengfeng Zhao, Chengwei Ren, Peng Wang, Yuan Liu  
  _2026-03-24_ · https://arxiv.org/abs/2603.23246v1  
  <details><summary>Abstract</summary>

  Reconstructing a renderable 3D model from images is a useful but challenging task. Recent feedforward 3D reconstruction methods have demonstrated remarkable success in efficiently recovering geometry, but still cannot accurately model the complex appearances of these 3D reconstructed models. Recent diffusion-based generative models can synthesize realistic images or videos of an object using reference images without explicitly modeling its appearance, which provides a promising direction for object rendering, but lacks accurate control over the viewpoints. In this paper, we propose GO-Renderer, a unified framework integrating the reconstructed 3D proxies to guide the video generative models to achieve high-quality object rendering on arbitrary viewpoints under arbitrary lighting conditions. Our method not only enjoys the accurate viewpoint control using the reconstructed 3D proxy but also enables high-quality rendering in different lighting environments using diffusion generative models without explicitly modeling complex materials and lighting. Extensive experiments demonstrate that GO-Renderer achieves state-of-the-art performance across the object rendering tasks, including synthesizing images on new viewpoints, rendering the objects in a novel lighting environment, and inserting an object into an existing video.

  </details>



- **GTLR-GS: Geometry-Texture Aware LiDAR-Regularized 3D Gaussian Splatting for Realistic Scene Reconstruction**  
  Yan Fang, Jianfei Ge, Jiangjian Xiao  
  _2026-03-24_ · https://arxiv.org/abs/2603.23192v1  
  <details><summary>Abstract</summary>

  Recent advances in 3D Gaussian Splatting (3DGS) have enabled real-time, photorealistic scene reconstruction. However, conventional 3DGS frameworks typically rely on sparse point clouds derived from Structure-from-Motion (SfM), which inherently suffer from scale ambiguity, limited geometric consistency, and strong view dependency due to the lack of geometric priors. In this work, a LiDAR-centric 3D Gaussian Splatting framework is proposed that explicitly incorporates metric geometric priors into the entire Gaussian optimization process. Instead of treating LiDAR data as a passive initialization source, 3DGS optimization is reformulated as a geometry-conditioned allocation and refinement problem under a fixed representational budget. Specifically, this work introduces (i) a geometry-texture-aware allocation strategy that selectively assigns Gaussian primitives to regions with high structural or appearance complexity, (ii) a curvature-adaptive refinement mechanism that dynamically guides Gaussian splitting toward geometrically complex areas during training, and (iii) a confidence-aware metric depth regularization that anchors the reconstructed geometry to absolute scale using LiDAR measurements while maintaining optimization stability. Extensive experiments on the ScanNet++ dataset and a custom real-world dataset validate the proposed approach. The results demonstrate state-of-the-art performance in metric-scale reconstruction with high geometric fidelity.

  </details>



- **Generative Event Pretraining with Foundation Model Alignment**  
  Jianwen Cao, Jiaxu Xing, Nico Messikommer, Davide Scaramuzza  
  _2026-03-24_ · https://arxiv.org/abs/2603.23032v1  
  <details><summary>Abstract</summary>

  Event cameras provide robust visual signals under fast motion and challenging illumination conditions thanks to their microsecond latency and high dynamic range. However, their unique sensing characteristics and limited labeled data make it challenging to train event-based visual foundation models (VFMs), which are crucial for learning visual features transferable across tasks. To tackle this problem, we propose GEP (Generative Event Pretraining), a two-stage framework that transfers semantic knowledge learned from internet-scale image datasets to event data while learning event-specific temporal dynamics. First, an event encoder is aligned to a frozen VFM through a joint regression-contrastive objective, grounding event features in image semantics. Second, a transformer backbone is autoregressively pretrained on mixed event-image sequences to capture the temporal structure unique to events. Our approach outperforms state-of-the-art event pretraining methods on a diverse range of downstream tasks, including object recognition, segmentation, and depth estimation. Together, VFM-guided alignment and generative sequence modeling yield a semantically rich, temporally aware event model that generalizes robustly across domains.

  </details>



- **UniQueR: Unified Query-based Feedforward 3D Reconstruction**  
  Chensheng Peng, Quentin Herau, Jiezhi Yang, Yichen Xie, Yihan Hu, Wenzhao Zheng, Matthew Strong, Masayoshi Tomizuka, Wei Zhan  
  _2026-03-24_ · https://arxiv.org/abs/2603.22851v1  
  <details><summary>Abstract</summary>

  We present UniQueR, a unified query-based feedforward framework for efficient and accurate 3D reconstruction from unposed images. Existing feedforward models such as DUSt3R, VGGT, and AnySplat typically predict per-pixel point maps or pixel-aligned Gaussians, which remain fundamentally 2.5D and limited to visible surfaces. In contrast, UniQueR formulates reconstruction as a sparse 3D query inference problem. Our model learns a compact set of 3D anchor points that act as explicit geometric queries, enabling the network to infer scene structure, including geometry in occluded regions--in a single forward pass. Each query encodes spatial and appearance priors directly in global 3D space (instead of per-frame camera space) and spawns a set of 3D Gaussians for differentiable rendering. By leveraging unified query interactions across multi-view features and a decoupled cross-attention design, UniQueR achieves strong geometric expressiveness while substantially reducing memory and computational cost. Experiments on Mip-NeRF 360 and VR-NeRF demonstrate that UniQueR surpasses state-of-the-art feedforward methods in both rendering quality and geometric accuracy, using an order of magnitude fewer primitives than dense alternatives.

  </details>



- **Multimodal Industrial Anomaly Detection via Geometric Prior**  
  Min Li, Jinghui He, Gang Li, Jiachen Li, Jin Wan, Delong Han  
  _2026-03-24_ · https://arxiv.org/abs/2603.22757v1  
  <details><summary>Abstract</summary>

  The purpose of multimodal industrial anomaly detection is to detect complex geometric shape defects such as subtle surface deformations and irregular contours that are difficult to detect in 2D-based methods. However, current multimodal industrial anomaly detection lacks the effective use of crucial geometric information like surface normal vectors and 3D shape topology, resulting in low detection accuracy. In this paper, we propose a novel Geometric Prior-based Anomaly Detection network (GPAD). Firstly, we propose a point cloud expert model to perform fine-grained geometric feature extraction, employing differential normal vector computation to enhance the geometric details of the extracted features and generate geometric prior. Secondly, we propose a two-stage fusion strategy to efficiently leverage the complementarity of multimodal data as well as the geometric prior inherent in 3D points. We further propose attention fusion and anomaly regions segmentation based on geometric prior, which enhance the model's ability to perceive geometric defects. Extensive experiments show that our multimodal industrial anomaly detection model outperforms the State-of-the-art (SOTA) methods in detection accuracy on both MVTec-3D AD and Eyecandies datasets.

  </details>



- **CAM3R: Camera-Agnostic Model for 3D Reconstruction**  
  Namitha Guruprasad, Abhay Yadav, Cheng Peng, Rama Chellappa  
  _2026-03-23_ · https://arxiv.org/abs/2603.22631v1  
  <details><summary>Abstract</summary>

  Recovering dense 3D geometry from unposed images remains a foundational challenge in computer vision. Current state-of-the-art models are predominantly trained on perspective datasets, which implicitly constrains them to a standard pinhole camera geometry. As a result, these models suffer from significant geometric degradation when applied to wide-angle imagery captured via non-rectilinear optics, such as fisheye or panoramic sensors. To address this, we present CAM3R, a Camera-Agnostic, feed-forward Model for 3D Reconstruction capable of processing images from wide-angle camera models without prior calibration. Our framework consists of a two-view network which is bifurcated into a Ray Module (RM) to estimate per-pixel ray directions and a Cross-view Module (CVM) to infer radial distance with confidence maps, pointmaps, and relative poses. To unify these pairwise predictions into a consistent 3D scene, we introduce a Ray-Aware Global Alignment framework for pose refinement and scale optimization while strictly preserving the predicted local geometry. Extensive experiments on various camera model datasets, including panorama, fisheye and pinhole imagery, demonstrate that CAM3R establishes a new state-of-the-art in pose estimation and reconstruction.

  </details>



- **FullCircle: Effortless 3D Reconstruction from Casual 360$^\circ$ Captures**  
  Yalda Foroutan, Ipek Oztas, Daniel Rebain, Aysegul Dundar, Kwang Moo Yi, Lily Goli, Andrea Tagliasacchi  
  _2026-03-23_ · https://arxiv.org/abs/2603.22572v1  
  <details><summary>Abstract</summary>

  Radiance fields have emerged as powerful tools for 3D scene reconstruction. However, casual capture remains challenging due to the narrow field of view of perspective cameras, which limits viewpoint coverage and feature correspondences necessary for reliable camera calibration and reconstruction. While commercially available 360$^\circ$ cameras offer significantly broader coverage than perspective cameras for the same capture effort, existing 360$^\circ$ reconstruction methods require special capture protocols and pre-processing steps that undermine the promise of radiance fields: effortless workflows to capture and reconstruct 3D scenes. We propose a practical pipeline for reconstructing 3D scenes directly from raw 360$^\circ$ camera captures. We require no special capture protocols or pre-processing, and exhibit robustness to a prevalent source of reconstruction errors: the human operator that is visible in all 360$^\circ$ imagery. To facilitate evaluation, we introduce a multi-tiered dataset of scenes captured as raw dual-fisheye images, establishing a benchmark for robust casual 360$^\circ$ reconstruction. Our method significantly outperforms not only vanilla 3DGS for 360$^\circ$ cameras but also robust perspective baselines when perspective cameras are simulated from the same capture, demonstrating the advantages of 360$^\circ$ capture for casual reconstruction. Additional results are available at: https://theialab.github.io/fullcircle

  </details>



- **Static Scene Reconstruction from Dynamic Egocentric Videos**  
  Qifei Cui, Patrick Chen  
  _2026-03-23_ · https://arxiv.org/abs/2603.22450v1  
  <details><summary>Abstract</summary>

  Egocentric videos present unique challenges for 3D reconstruction due to rapid camera motion and frequent dynamic interactions. State-of-the-art static reconstruction systems, such as MapAnything, often degrade in these settings, suffering from catastrophic trajectory drift and "ghost" geometry caused by moving hands. We bridge this gap by proposing a robust pipeline that adapts static reconstruction backbones to long-form egocentric video. Our approach introduces a mask-aware reconstruction mechanism that explicitly suppresses dynamic foreground in the attention layers, preventing hand artifacts from contaminating the static map. Furthermore, we employ a chunked reconstruction strategy with pose-graph stitching to ensure global consistency and eliminate long-term drift. Experiments on HD-EPIC and indoor drone datasets demonstrate that our pipeline significantly improves absolute trajectory error and yields visually clean static geometry compared to naive baselines, effectively extending the capability of foundation models to dynamic first-person scenes.

  </details>



- **Spatially-Aware Evaluation Framework for Aerial LiDAR Point Cloud Semantic Segmentation: Distance-Based Metrics on Challenging Regions**  
  Alex Salvatierra, José Antonio Sanz, Christian Gutiérrez, Mikel Galar  
  _2026-03-23_ · https://arxiv.org/abs/2603.22420v1  
  <details><summary>Abstract</summary>

  Semantic segmentation metrics for 3D point clouds, such as mean Intersection over Union (mIoU) and Overall Accuracy (OA), present two key limitations in the context of aerial LiDAR data. First, they treat all misclassifications equally regardless of their spatial context, overlooking cases where the geometric severity of errors directly impacts the quality of derived geospatial products such as Digital Terrain Models. Second, they are often dominated by the large proportion of easily classified points, which can mask meaningful differences between models and under-represent performance in challenging regions. To address these limitations, we propose a novel evaluation framework for comparing semantic segmentation models through two complementary approaches. First, we introduce distance-based metrics that account for the spatial deviation between each misclassified point and the nearest ground-truth point of the predicted class, capturing the geometric severity of errors. Second, we propose a focused evaluation on a common subset of hard points, defined as the points misclassified by at least one of the evaluated models, thereby reducing the bias introduced by easily classified points and better revealing differences in model performance in challenging regions. We validate our framework by comparing three state-of-the-art deep learning models on three aerial LiDAR datasets. Results demonstrate that the proposed metrics provide complementary information to traditional measures, revealing spatial error patterns that are critical for Earth Observation applications but invisible to conventional evaluation approaches. The proposed framework enables more informed model selection for scenarios where spatial consistency is critical.

  </details>



- **GenOpticalFlow: A Generative Approach to Unsupervised Optical Flow Learning**  
  Yixuan Luo, Feng Qiao, Zhexiao Xiong, Yanjing Li, Nathan Jacobs  
  _2026-03-23_ · https://arxiv.org/abs/2603.22270v1  
  <details><summary>Abstract</summary>

  Optical flow estimation is a fundamental problem in computer vision, yet the reliance on expensive ground-truth annotations limits the scalability of supervised approaches. Although unsupervised and semi-supervised methods alleviate this issue, they often suffer from unreliable supervision signals based on brightness constancy and smoothness assumptions, leading to inaccurate motion estimation in complex real-world scenarios. To overcome these limitations, we introduce \textbf{\modelname}, a novel framework that synthesizes large-scale, perfectly aligned frame--flow data pairs for supervised optical flow training without human annotations. Specifically, our method leverages a pre-trained depth estimation network to generate pseudo optical flows, which serve as conditioning inputs for a next-frame generation model trained to produce high-fidelity, pixel-aligned subsequent frames. This process enables the creation of abundant, high-quality synthetic data with precise motion correspondence. Furthermore, we propose an \textit{inconsistent pixel filtering} strategy that identifies and removes unreliable pixels in generated frames, effectively enhancing fine-tuning performance on real-world datasets. Extensive experiments on KITTI2012, KITTI2015, and Sintel demonstrate that \textbf{\modelname} achieves competitive or superior results compared to existing unsupervised and semi-supervised approaches, highlighting its potential as a scalable and annotation-free solution for optical flow learning. We will release our code upon acceptance.

  </details>



- **Riverine Land Cover Mapping through Semantic Segmentation of Multispectral Point Clouds**  
  Sopitta Thurachen, Josef Taher, Matti Lehtomäki, Leena Matikainen, Linnea Blåfield, Mikel Calle Navarro, Antero Kukko, Tomi Westerlund, Harri Kaartinen  
  _2026-03-23_ · https://arxiv.org/abs/2603.22230v1  
  <details><summary>Abstract</summary>

  Accurate land cover mapping in riverine environments is essential for effective river management, ecological understanding, and geomorphic change monitoring. This study explores the use of Point Transformer v2 (PTv2), an advanced deep neural network architecture designed for point cloud data, for land cover mapping through semantic segmentation of multispectral LiDAR data in real-world riverine environments. We utilize the geometric and spectral information from the 3-channel LiDAR point cloud to map land cover classes, including sand, gravel, low vegetation, high vegetation, forest floor, and water. The PTv2 model was trained and evaluated on point cloud data from the Oulanka river in northern Finland using both geometry and spectral features. To improve the model's generalization in new riverine environments, we additionally investigate multi-dataset training that adds sparsely annotated data from an additional river dataset. Results demonstrated that using the full-feature configuration resulted in performance with a mean Intersection over Union (mIoU) of 0.950, significantly outperforming the geometry baseline. Other ablation studies revealed that intensity and reflectance features were the key for accurate land cover mapping. The multi-dataset training experiment showed improved generalization performance, suggesting potential for developing more robust models despite limited high-quality annotated data. Our work demonstrates the potential of applying transformer-based architectures to multispectral point clouds in riverine environments. The approach offers new capabilities for monitoring sediment transport and other river management applications.

  </details>


