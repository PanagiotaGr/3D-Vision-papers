# 3D Reconstruction

_Updated: 2026-03-25 07:16 UTC_

Total papers shown: **23**


---

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



- **Omni-WorldBench: Towards a Comprehensive Interaction-Centric Evaluation for World Models**  
  Meiqi Wu, Zhixin Cai, Fufangchen Zhao, Xiaokun Feng, Rujing Dang, Bingze Song, Ruitian Tian, Jiashu Zhu, Jiachen Lei, Hao Dou, et al.  
  _2026-03-23_ · https://arxiv.org/abs/2603.22212v1  
  <details><summary>Abstract</summary>

  Video--based world models have emerged along two dominant paradigms: video generation and 3D reconstruction. However, existing evaluation benchmarks either focus narrowly on visual fidelity and text--video alignment for generative models, or rely on static 3D reconstruction metrics that fundamentally neglect temporal dynamics. We argue that the future of world modeling lies in 4D generation, which jointly models spatial structure and temporal evolution. In this paradigm, the core capability is interactive response: the ability to faithfully reflect how interaction actions drive state transitions across space and time. Yet no existing benchmark systematically evaluates this critical dimension. To address this gap, we propose Omni--WorldBench, a comprehensive benchmark specifically designed to evaluate the interactive response capabilities of world models in 4D settings. Omni--WorldBench comprises two key components: Omni--WorldSuite, a systematic prompt suite spanning diverse interaction levels and scene types; and Omni--Metrics, an agent-based evaluation framework that quantifies world modeling capabilities by measuring the causal impact of interaction actions on both final outcomes and intermediate state evolution trajectories. We conduct extensive evaluations of 18 representative world models across multiple paradigms. Our analysis reveals critical limitations of current world models in interactive response, providing actionable insights for future research. Omni-WorldBench will be publicly released to foster progress in interactive 4D world modeling.

  </details>



- **Adapting Point Cloud Analysis via Multimodal Bayesian Distribution Learning**  
  Xingyu Zhu, Liang Yi, Shuo Wang, Wenbo Zhu, Yonglinag Wu, Beier Zhu, Hanwang Zhang  
  _2026-03-23_ · https://arxiv.org/abs/2603.22070v1  
  <details><summary>Abstract</summary>

  Multimodal 3D vision-language models show strong generalization across diverse 3D tasks, but their performance still degrades notably under domain shifts. This has motivated recent studies on test-time adaptation (TTA), which enables models to adapt online using test-time data. Among existing TTA methods, cache-based mechanisms are widely adopted for leveraging previously observed samples in online prediction refinement. However, they store only limited historical information, leading to progressive information loss as the test stream evolves. In addition, their prediction logits are fused heuristically, making adaptation unstable. To address these limitations, we propose BayesMM, a Multimodal Bayesian Distribution Learning framework for test-time point cloud analysis. BayesMM models textual priors and streaming visual features of each class as Gaussian distributions: textual parameters are derived from semantic prompts, while visual parameters are updated online with arriving samples. The two modalities are fused via Bayesian model averaging, which automatically adjusts their contributions based on posterior evidence, yielding a unified prediction that adapts continually to evolving test-time data without training. Extensive experiments on multiple point cloud benchmarks demonstrate that BayesMM maintains robustness under distributional shifts, yielding over 4% average improvement.

  </details>



- **GTSR: Subsurface Scattering Awared 3D Gaussians for Translucent Surface Reconstruction**  
  Youwen Yuan, Xi Zhao  
  _2026-03-23_ · https://arxiv.org/abs/2603.22036v1  
  <details><summary>Abstract</summary>

  Reconstructing translucent objects from multi-view images is a difficult problem. Previously, researchers have used differentiable path tracing and the neural implicit field, which require relatively large computational costs. Recently, many works have achieved good reconstruction results for opaque objects based on a 3DGS pipeline with much higher efficiency. However, such methods have difficulty dealing with translucent objects, because they do not consider the optical properties of translucent objects. In this paper, we propose a novel 3DGS-based pipeline (GTSR) to reconstruct the surface geometry of translucent objects. GTSR combines two sets of Gaussians, surface and interior Gaussians, which are used to model the surface and scattering color when lights pass translucent objects. To render the appearance of translucent objects, we introduce a method that uses the Fresnel term to blend two sets of Gaussians. Furthermore, to improve the reconstructed details of non-contour areas, we introduce the Disney BSDF model with deferred rendering to enhance constraints of the normal and depth. Experimental results demonstrate that our method outperforms baseline reconstruction methods on the NeuralTO Syn dataset while showing great real-time rendering performance. We also extend the dataset with new translucent objects of varying material properties and demonstrate our method can adapt to different translucent materials.

  </details>



- **SARe: Structure-Aware Large-Scale 3D Fragment Reassembly**  
  Hanze Jia, Chunshi Wang, Yuxiao Yang, Zhonghua Jiang, Yawei Luo, Shuainan Ye, Tan Tang  
  _2026-03-23_ · https://arxiv.org/abs/2603.21611v1  
  <details><summary>Abstract</summary>

  3D fragment reassembly aims to recover the rigid poses of unordered fragment point clouds or meshes in a common object coordinate system to reconstruct the complete shape. The problem becomes particularly challenging as the number of fragments grows, since the target shape is unknown and fragments provide weak semantic cues. Existing end-to-end approaches are prone to cascading failures due to unreliable contact reasoning, most notably inaccurate fragment adjacencies. To address this, we propose Structure-Aware Reassembly (SARe), a generative framework with SARe-Gen for Euclidean-space assembly generation and SARe-Refine for inference-time refinement, with explicit contact modeling. SARe-Gen jointly predicts fracture-surface token probabilities and an inter-fragment contact graph to localize contact regions and infer candidate adjacencies. It adopts a query-point-based conditioning scheme and extracts aligned local geometric tokens at query locations from a frozen geometry encoder, yielding queryable structural representations without additional structural pretraining. We further introduce an inference-time refinement stage, SARe-Refine. By verifying candidate contact edges with geometric-consistency checks, it selects reliable substructures and resamples the remaining uncertain regions while keeping verified parts fixed, leading to more stable and consistent assemblies in the many-fragment regime. We evaluate SARe across three settings, including synthetic fractures, simulated fractures from scanned real objects, and real physically fractured scans. The results demonstrate state-of-the-art performance, with more graceful degradation and higher success rates as the fragment count increases in challenging large-scale reassembly.

  </details>



- **HACMatch Semi-Supervised Rotation Regression with Hardness-Aware Curriculum Pseudo Labeling**  
  Mei Li, Huayi Zhou, Suizhi Huang, Yuxiang Lu, Yue Ding, Hongtao Lu  
  _2026-03-23_ · https://arxiv.org/abs/2603.21583v1  
  <details><summary>Abstract</summary>

  Regressing 3D rotations of objects from 2D images is a crucial yet challenging task, with broad applications in autonomous driving, virtual reality, and robotic control. Existing rotation regression models often rely on large amounts of labeled data for training or require additional information beyond 2D images, such as point clouds or CAD models. Therefore, exploring semi-supervised rotation regression using only a limited number of labeled 2D images is highly valuable. While recent work FisherMatch introduces semi-supervised learning to rotation regression, it suffers from rigid entropy-based pseudo-label filtering that fails to effectively distinguish between reliable and unreliable unlabeled samples. To address this limitation, we propose a hardness-aware curriculum learning framework that dynamically selects pseudo-labeled samples based on their difficulty, progressing from easy to complex examples. We introduce both multi-stage and adaptive curriculum strategies to replace fixed-threshold filtering with more flexible, hardness-aware mechanisms. Additionally, we present a novel structured data augmentation strategy specifically tailored for rotation estimation, which assembles composite images from augmented patches to introduce feature diversity while preserving critical geometric integrity. Comprehensive experiments on PASCAL3D+ and ObjectNet3D demonstrate that our method outperforms existing supervised and semi-supervised baselines, particularly in low-data regimes, validating the effectiveness of our curriculum learning framework and structured augmentation approach.

  </details>



- **Back to Point: Exploring Point-Language Models for Zero-Shot 3D Anomaly Detection**  
  Kaiqiang Li, Gang Li, Mingle Zhou, Min Li, Delong Han, Jin Wan  
  _2026-03-23_ · https://arxiv.org/abs/2603.21511v1  
  <details><summary>Abstract</summary>

  Zero-shot (ZS) 3D anomaly detection is crucial for reliable industrial inspection, as it enables detecting and localizing defects without requiring any target-category training data. Existing approaches render 3D point clouds into 2D images and leverage pre-trained Vision-Language Models (VLMs) for anomaly detection. However, such strategies inevitably discard geometric details and exhibit limited sensitivity to local anomalies. In this paper, we revisit intrinsic 3D representations and explore the potential of pre-trained Point-Language Models (PLMs) for ZS 3D anomaly detection. We propose BTP (Back To Point), a novel framework that effectively aligns 3D point cloud and textual embeddings. Specifically, BTP aligns multi-granularity patch features with textual representations for localized anomaly detection, while incorporating geometric descriptors to enhance sensitivity to structural anomalies. Furthermore, we introduce a joint representation learning strategy that leverages auxiliary point cloud data to improve robustness and enrich anomaly semantics. Extensive experiments on Real3D-AD and Anomaly-ShapeNet demonstrate that BTP achieves superior performance in ZS 3D anomaly detection. Code will be available at \href{https://github.com/wistful-8029/BTP-3DAD}{https://github.com/wistful-8029/BTP-3DAD}.

  </details>



- **PAS3R: Pose-Adaptive Streaming 3D Reconstruction for Long Video Sequences**  
  Lanbo Xu, Liang Guo, Caigui Jiang, Cheng Wang  
  _2026-03-22_ · https://arxiv.org/abs/2603.21436v1  
  <details><summary>Abstract</summary>

  Online monocular 3D reconstruction enables dense scene recovery from streaming video but remains fundamentally limited by the stability-adaptation dilemma: the reconstruction model must rapidly incorporate novel viewpoints while preserving previously accumulated scene structure. Existing streaming approaches rely on uniform or attention-based update mechanisms that often fail to account for abrupt viewpoint transitions, leading to trajectory drift and geometric inconsistencies over long sequences. We introduce PAS3R, a pose-adaptive streaming reconstruction framework that dynamically modulates state updates according to camera motion and scene structure. Our key insight is that frames contributing significant geometric novelty should exert stronger influence on the reconstruction state, while frames with minor viewpoint variation should prioritize preserving historical context. PAS3R operationalizes this principle through a motion-aware update mechanism that jointly leverages inter-frame pose variation and image frequency cues to estimate frame importance. To further stabilize long-horizon reconstruction, we introduce trajectory-consistent training objectives that incorporate relative pose constraints and acceleration regularization. A lightweight online stabilization module further suppresses high-frequency trajectory jitter and geometric artifacts without increasing memory consumption. Extensive experiments across multiple benchmarks demonstrate that PAS3R significantly improves trajectory accuracy, depth estimation, and point cloud reconstruction quality in long video sequences while maintaining competitive performance on shorter sequences.

  </details>



- **FluidGaussian: Propagating Simulation-Based Uncertainty Toward Functionally-Intelligent 3D Reconstruction**  
  Yuqiu Liu, Jialin Song, Marissa Ramirez de Chanlatte, Rochishnu Chowdhury, Rushil Paresh Desai, Wuyang Chen, Daniel Martin, Michael Mahoney  
  _2026-03-22_ · https://arxiv.org/abs/2603.21356v1  
  <details><summary>Abstract</summary>

  Real objects that inhabit the physical world follow physical laws and thus behave plausibly during interaction with other physical objects. However, current methods that perform 3D reconstructions of real-world scenes from multi-view 2D images optimize primarily for visual fidelity, i.e., they train with photometric losses and reason about uncertainty in the image or representation space. This appearance-centric view overlooks body contacts and couplings, conflates function-critical regions (e.g., aerodynamic or hydrodynamic surfaces) with ornamentation, and reconstructs structures suboptimally, even when physical regularizers are added. All these can lead to unphysical and implausible interactions. To address this, we consider the question: How can 3D reconstruction become aware of real-world interactions and underlying object functionality, beyond visual cues? To answer this question, we propose FluidGaussian, a plug-and-play method that tightly couples geometry reconstruction with ubiquitous fluid-structure interactions to assess surface quality at high granularity. We define a simulation-based uncertainty metric induced by fluid simulations and integrate it with active learning to prioritize views that improve both visual and physical fidelity. In an empirical evaluation on NeRF Synthetic (Blender), Mip-NeRF 360, and DrivAerNet++, our FluidGaussian method yields up to +8.6% visual PSNR (Peak Signal-to-Noise Ratio) and -62.3% velocity divergence during fluid simulations. Our code is available at https://github.com/delta-lab-ai/FluidGaussian.

  </details>


