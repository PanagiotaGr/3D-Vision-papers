# 3D Reconstruction

_Updated: 2026-04-11 07:15 UTC_

Total papers shown: **19**


---

- **ETCH-X: Robustify Expressive Body Fitting to Clothed Humans with Composable Datasets**  
  Xiaoben Li, Jingyi Wu, Zeyu Cai, Yu Siyuan, Boqian Li, Yuliang Xiu  
  _2026-04-09_ · https://arxiv.org/abs/2604.08548v1  
  <details><summary>Abstract</summary>

  Human body fitting, which aligns parametric body models such as SMPL to raw 3D point clouds of clothed humans, serves as a crucial first step for downstream tasks like animation and texturing. An effective fitting method should be both locally expressive-capturing fine details such as hands and facial features-and globally robust to handle real-world challenges, including clothing dynamics, pose variations, and noisy or partial inputs. Existing approaches typically excel in only one aspect, lacking an all-in-one solution.We upgrade ETCH to ETCH-X, which leverages a tightness-aware fitting paradigm to filter out clothing dynamics ("undress"), extends expressiveness with SMPL-X, and replaces explicit sparse markers (which are highly sensitive to partial data) with implicit dense correspondences ("dense fit") for more robust and fine-grained body fitting. Our disentangled "undress" and "dense fit" modular stages enable separate and scalable training on composable data sources, including diverse simulated garments (CLOTH3D), large-scale full-body motions (AMASS), and fine-grained hand gestures (InterHand2.6M), improving outfit generalization and pose robustness of both bodies and hands. Our approach achieves robust and expressive fitting across diverse clothing, poses, and levels of input completeness, delivering a substantial performance improvement over ETCH on both: 1) seen data, such as 4D-Dress (MPJPE-All, 33.0% ) and CAPE (V2V-Hands, 35.8% ), and 2) unseen data, such as BEDLAM2.0 (MPJPE-All, 80.8% ; V2V-All, 80.5% ). Code and models will be released at https://xiaobenli00.github.io/ETCH-X/.

  </details>



- **Scal3R: Scalable Test-Time Training for Large-Scale 3D Reconstruction**  
  Tao Xie, Peishan Yang, Yudong Jin, Yingfeng Cai, Wei Yin, Weiqiang Ren, Qian Zhang, Wei Hua, Sida Peng, Xiaoyang Guo, et al.  
  _2026-04-09_ · https://arxiv.org/abs/2604.08542v1  
  <details><summary>Abstract</summary>

  This paper addresses the task of large-scale 3D scene reconstruction from long video sequences. Recent feed-forward reconstruction models have shown promising results by directly regressing 3D geometry from RGB images without explicit 3D priors or geometric constraints. However, these methods often struggle to maintain reconstruction accuracy and consistency over long sequences due to limited memory capacity and the inability to effectively capture global contextual cues. In contrast, humans can naturally exploit the global understanding of the scene to inform local perception. Motivated by this, we propose a novel neural global context representation that efficiently compresses and retains long-range scene information, enabling the model to leverage extensive contextual cues for enhanced reconstruction accuracy and consistency. The context representation is realized through a set of lightweight neural sub-networks that are rapidly adapted during test time via self-supervised objectives, which substantially increases memory capacity without incurring significant computational overhead. The experiments on multiple large-scale benchmarks, including the KITTI Odometry~\cite{Geiger2012CVPR} and Oxford Spires~\cite{tao2025spires} datasets, demonstrate the effectiveness of our approach in handling ultra-large scenes, achieving leading pose accuracy and state-of-the-art 3D reconstruction accuracy while maintaining efficiency. Code is available at https://zju3dv.github.io/scal3r.

  </details>



- **Self-Improving 4D Perception via Self-Distillation**  
  Nan Huang, Pengcheng Yu, Weijia Zeng, James M. Rehg, Angjoo Kanazawa, Haiwen Feng, Qianqian Wang  
  _2026-04-09_ · https://arxiv.org/abs/2604.08532v1  
  <details><summary>Abstract</summary>

  Large-scale multi-view reconstruction models have made remarkable progress, but most existing approaches still rely on fully supervised training with ground-truth 3D/4D annotations. Such annotations are expensive and particularly scarce for dynamic scenes, limiting scalability. We propose SelfEvo, a self-improving framework that continually improves pretrained multi-view reconstruction models using unlabeled videos. SelfEvo introduces a self-distillation scheme using spatiotemporal context asymmetry, enabling self-improvement for learning-based 4D perception without external annotations. We systematically study design choices that make self-improvement effective, including loss signals, forms of asymmetry, and other training strategies. Across eight benchmarks spanning diverse datasets and domains, SelfEvo consistently improves pretrained baselines and generalizes across base models (e.g. VGGT and $π^3$), with significant gains on dynamic scenes. Overall, SelfEvo achieves up to 36.5% relative improvement in video depth estimation and 20.1% in camera estimation, without using any labeled data. Project Page: https://self-evo.github.io/.

  </details>



- **SurfelSplat: Learning Efficient and Generalizable Gaussian Surfel Representations for Sparse-View Surface Reconstruction**  
  Chensheng Dai, Shengjun Zhang, Min Chen, Yueqi Duan  
  _2026-04-09_ · https://arxiv.org/abs/2604.08370v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has demonstrated impressive performance in 3D scene reconstruction. Beyond novel view synthesis, it shows great potential for multi-view surface reconstruction. Existing methods employ optimization-based reconstruction pipelines that achieve precise and complete surface extractions. However, these approaches typically require dense input views and high time consumption for per-scene optimization. To address these limitations, we propose SurfelSplat, a feed-forward framework that generates efficient and generalizable pixel-aligned Gaussian surfel representations from sparse-view images. We observe that conventional feed-forward structures struggle to recover accurate geometric attributes of Gaussian surfels because the spatial frequency of pixel-aligned primitives exceeds Nyquist sampling rates. Therefore, we propose a cross-view feature aggregation module based on the Nyquist sampling theorem. Specifically, we first adapt the geometric forms of Gaussian surfels with spatial sampling rate-guided low-pass filters. We then project the filtered surfels across all input views to obtain cross-view feature correlations. By processing these correlations through a specially designed feature fusion network, we can finally regress Gaussian surfels with precise geometry. Extensive experiments on DTU reconstruction benchmarks demonstrate that our model achieves comparable results with state-of-the-art methods, and predict Gaussian surfels within 1 second, offering a 100x speedup without costly per-scene training.

  </details>



- **Revisiting Radar Perception With Spectral Point Clouds**  
  Hamza Alsharif, Jing Gu, Pavol Jancura, Satish Ravindran, Gijs Dubbelman  
  _2026-04-09_ · https://arxiv.org/abs/2604.08282v1  
  <details><summary>Abstract</summary>

  Radar perception models are trained with different inputs, from range-Doppler spectra to sparse point clouds. Dense spectra are assumed to outperform sparse point clouds, yet they can vary considerably across sensors and configurations, which hinders transfer. In this paper, we provide alternatives for incorporating spectral information into radar point clouds and show that, point clouds need not underperform compared to spectra. We introduce the spectral point cloud paradigm, where point clouds are treated as sparse, compressed representations of the radar spectra, and argue that, when enriched with spectral information, they serve as strong candidates for a unified input representation that is more robust against sensor-specific differences. We develop an experimental framework that compares spectral point cloud (PC) models at varying densities against a dense range-Doppler (RD) benchmark, and report the density levels where the PC configurations meet the performance of the RD benchmark. Furthermore, we experiment with two basic spectral enrichment approaches, that inject additional target-relevant information into the point clouds. Contrary to the common belief that the dense RD approach is superior, we show that point clouds can do just as well, and can surpass the RD benchmark when enrichment is applied. Spectral point clouds can therefore serve as strong candidates for unified radar perception, paving the way for future radar foundation models.

  </details>



- **Brain3D: EEG-to-3D Decoding of Visual Representations via Multimodal Reasoning**  
  Emanuele Balloni, Emanuele Frontoni, Chiara Matti, Marina Paolanti, Roberto Pierdicca, Emiliano Santarnecchi  
  _2026-04-09_ · https://arxiv.org/abs/2604.08068v1  
  <details><summary>Abstract</summary>

  Decoding visual information from electroencephalography (EEG) has recently achieved promising results, primarily focusing on reconstructing two-dimensional (2D) images from brain activity. However, the reconstruction of three-dimensional (3D) representations remains largely unexplored. This limits the geometric understanding and reduces the applicability of neural decoding in different contexts. To address this gap, we propose Brain3D, a multimodal architecture for EEG-to-3D reconstruction based on EEG-to-image decoding. It progressively transforms neural representations into the 3D domain using geometry-aware generative reasoning. Our pipeline first produces visually grounded images from EEG signals, then employs a multimodal large language model to extract structured 3D-aware descriptions, which guide a diffusion-based generation stage whose outputs are finally converted into coherent 3D meshes via a single-image-to-3D model. By decomposing the problem into structured stages, the proposed approach avoids direct EEG-to-3D mappings and enables scalable brain-driven 3D generation. We conduct a comprehensive evaluation comparing the reconstructed 3D outputs against the original visual stimuli, assessing both semantic alignment and geometric fidelity. Experimental results demonstrate strong performance of the proposed architecture, achieving up to 85.4% 10-way Top-1 EEG decoding accuracy and 0.648 CLIPScore, supporting the feasibility of multimodal EEG-driven 3D reconstruction.

  </details>



- **SceneScribe-1M: A Large-Scale Video Dataset with Comprehensive Geometric and Semantic Annotations**  
  Yunnan Wang, Kecheng Zheng, Jianyuan Wang, Minghao Chen, David Novotny, Christian Rupprecht, Yinghao Xu, Xing Zhu, Wenjun Zeng, Xin Jin, et al.  
  _2026-04-09_ · https://arxiv.org/abs/2604.07990v1  
  <details><summary>Abstract</summary>

  The convergence of 3D geometric perception and video synthesis has created an unprecedented demand for large-scale video data that is rich in both semantic and spatio-temporal information. While existing datasets have advanced either 3D understanding or video generation, a significant gap remains in providing a unified resource that supports both domains at scale. To bridge this chasm, we introduce SceneScribe-1M, a new large-scale, multi-modal video dataset. It comprises one million in-the-wild videos, each meticulously annotated with detailed textual descriptions, precise camera parameters, dense depth maps, and consistent 3D point tracks. We demonstrate the versatility and value of SceneScribe-1M by establishing benchmarks across a wide array of downstream tasks, including monocular depth estimation, scene reconstruction, and dynamic point tracking, as well as generative tasks such as text-to-video synthesis, with or without camera control. By open-sourcing SceneScribe-1M, we aim to provide a comprehensive benchmark and a catalyst for research, fostering the development of models that can both perceive the dynamic 3D world and generate controllable, realistic video content.

  </details>



- **Object-Centric Stereo Ranging for Autonomous Driving: From Dense Disparity to Census-Based Template Matching**  
  Qihao Huang  
  _2026-04-09_ · https://arxiv.org/abs/2604.07980v1  
  <details><summary>Abstract</summary>

  Accurate depth estimation is critical for autonomous driving perception systems, particularly for long range vehicle detection on highways. Traditional dense stereo matching methods such as Block Matching (BM) and Semi Global Matching (SGM) produce per pixel disparity maps but suffer from high computational cost, sensitivity to radiometric differences between stereo cameras, and poor accuracy at long range where disparity values are small. In this report, we present a comprehensive stereo ranging system that integrates three complementary depth estimation approaches: dense BM/SGM disparity, object centric Census based template matching, and monocular geometric priors, within a unified detection ranging tracking pipeline. Our key contribution is a novel object centric Census based template matching algorithm that performs GPU accelerated sparse stereo matching directly within detected bounding boxes, employing a far close divide and conquer strategy, forward backward verification, occlusion aware sampling, and robust multi block aggregation. We further describe an online calibration refinement framework that combines auto rectification offset search, radar stereo voting based disparity correction, and object level radar stereo association for continuous extrinsic drift compensation. The complete system achieves real time performance through asynchronous GPU pipeline design and delivers robust ranging across diverse driving conditions including nighttime, rain, and varying illumination.

  </details>



- **Sampling-Aware 3D Spatial Analysis in Multiplexed Imaging**  
  Ido Harlev, Tamar Oukhanov, Raz Ben-Uri, Leeat Keren, Shai Bagon  
  _2026-04-09_ · https://arxiv.org/abs/2604.07890v1  
  <details><summary>Abstract</summary>

  Highly multiplexed microscopy enables rich spatial characterization of tissues at single-cell resolution, yet most analyses rely on two-dimensional sections despite inherently three-dimensional tissue organization. Acquiring dense volumetric data in spatial proteomics remains costly and technically challenging, leaving practitioners to choose between 2D sections or 3D serial sections under limited imaging budgets. In this work, we study how sampling geometry impacts the stability of commonly used spatial statistics, and we introduce a geometry-aware reconstruction module that enables sparse yet consistent 3D analysis from serial sections. Using controlled simulations, we show that planar sampling reliably recovers global cell-type abundance but exhibits high variance for local statistics such as cell clustering and cell-cell interactions, particularly for rare or spatially localized populations. We observe consistent behavior in real multiplexed datasets, where interaction metrics and neighborhood relationships fluctuate substantially across individual sections. To support sparse 3D analysis in practice, we present a reconstruction approach that links cell projections across adjacent sections using phenotype and proximity constraints and recovers single-cell 3D centroids using cell-type-specific shape priors. We further analyze the trade-off between section spacing, coverage, and redundancy, identifying acquisition regimes that maximize reconstruction utility under fixed imaging budgets. We validate the reconstruction module on a public imaging mass cytometry dataset with dense axial sampling and demonstrate its downstream utility on an in-house CODEX dataset by enabling structure-level 3D analyses that are unreliable in 2D. Together, our results provide diagnostic tools and practical guidance for deciding when 2D sampling suffices and when sparse 3D reconstruction is warranted.

  </details>



- **Adaptive Depth-converted-Scale Convolution for Self-supervised Monocular Depth Estimation**  
  Yanbo Gao, Huibin Bai, Huasong Zhou, Xingyu Gao, Shuai Li, Xun Cai, Hui Yuan, Wei Hua, Tian Xie  
  _2026-04-09_ · https://arxiv.org/abs/2604.07665v1  
  <details><summary>Abstract</summary>

  Self-supervised monocular depth estimation (MDE) has received increasing interests in the last few years. The objects in the scene, including the object size and relationship among different objects, are the main clues to extract the scene structure. However, previous works lack the explicit handling of the changing sizes of the object due to the change of its depth. Especially in a monocular video, the size of the same object is continuously changed, resulting in size and depth ambiguity. To address this problem, we propose a Depth-converted-Scale Convolution (DcSConv) enhanced monocular depth estimation framework, by incorporating the prior relationship between the object depth and object scale to extract features from appropriate scales of the convolution receptive field. The proposed DcSConv focuses on the adaptive scale of the convolution filter instead of the local deformation of its shape. It establishes that the scale of the convolution filter matters no less (or even more in the evaluated task) than its local deformation. Moreover, a Depth-converted-Scale aware Fusion (DcS-F) is developed to adaptively fuse the DcSConv features and the conventional convolution features. Our DcSConv enhanced monocular depth estimation framework can be applied on top of existing CNN based methods as a plug-and-play module to enhance the conventional convolution block. Extensive experiments with different baselines have been conducted on the KITTI benchmark and our method achieves the best results with an improvement up to 11.6% in terms of SqRel reduction. Ablation study also validates the effectiveness of each proposed module.

  </details>



- **Monocular Depth Estimation From the Perspective of Feature Restoration: A Diffusion Enhanced Depth Restoration Approach**  
  Huibin Bai, Shuai Li, Hanxiao Zhai, Yanbo Gao, Chong Lv, Yibo Wang, Haipeng Ping, Wei Hua, Xingyu Gao  
  _2026-04-09_ · https://arxiv.org/abs/2604.07664v1  
  <details><summary>Abstract</summary>

  Monocular Depth Estimation (MDE) is a fundamental computer vision task with important applications in 3D vision. The current mainstream MDE methods employ an encoder-decoder architecture with multi-level/scale feature processing. However, the limitations of the current architecture and the effects of different-level features on the prediction accuracy are not evaluated. In this paper, we first investigate the above problem and show that there is still substantial potential in the current framework if encoder features can be improved. Therefore, we propose to formulate the depth estimation problem from the feature restoration perspective, by treating pretrained encoder features as degraded features of an assumed ground truth feature that yields the ground truth depth map. Then an Invertible Transform-enhanced Indirect Diffusion (InvT-IndDiffusion) module is developed for feature restoration. Due to the absence of direct supervision on feature, only indirect supervision from the final sparse depth map is used. During the iterative procedure of diffusion, this results in feature deviations among steps. The proposed InvT-IndDiffusion solves this problem by using an invertible transform-based decoder under the bi-Lipschitz condition. Finally, a plug-and-play Auxiliary Viewpoint-based Low-level Feature Enhancement module (AV-LFE) is developed to enhance local details with auxiliary viewpoint when available. Experiments demonstrate that the proposed method achieves better performance than the state-of-the-art methods on various datasets. Specifically on the KITTI benchmark, compared with the baseline, the performance is improved by 4.09% and 37.77% under different training settings in terms of RMSE. Code is available at https://github.com/whitehb1/IID-RDepth.

  </details>



- **Fast Spatial Memory with Elastic Test-Time Training**  
  Ziqiao Ma, Xueyang Yu, Haoyu Zhen, Yuncong Yang, Joyce Chai, Chuang Gan  
  _2026-04-08_ · https://arxiv.org/abs/2604.07350v1  
  <details><summary>Abstract</summary>

  Large Chunk Test-Time Training (LaCT) has shown strong performance on long-context 3D reconstruction, but its fully plastic inference-time updates remain vulnerable to catastrophic forgetting and overfitting. As a result, LaCT is typically instantiated with a single large chunk spanning the full input sequence, falling short of the broader goal of handling arbitrarily long sequences in a single pass. We propose Elastic Test-Time Training inspired by elastic weight consolidation, that stabilizes LaCT fast-weight updates with a Fisher-weighted elastic prior around a maintained anchor state. The anchor evolves as an exponential moving average of past fast weights to balance stability and plasticity. Based on this updated architecture, we introduce Fast Spatial Memory (FSM), an efficient and scalable model for 4D reconstruction that learns spatiotemporal representations from long observation sequences and renders novel view-time combinations. We pre-trained FSM on large-scale curated 3D/4D data to capture the dynamics and semantics of complex spatial environments. Extensive experiments show that FSM supports fast adaptation over long sequences and delivers high-quality 3D/4D reconstruction with smaller chunks and mitigating the camera-interpolation shortcut. Overall, we hope to advance LaCT beyond the bounded single-chunk setting toward robust multi-chunk adaptation, a necessary step for generalization to genuinely longer sequences, while substantially alleviating the activation-memory bottleneck.

  </details>



- **From Blobs to Spokes: High-Fidelity Surface Reconstruction via Oriented Gaussians**  
  Diego Gomez, Antoine Guédon, Nissim Maruani, Bingchen Gong, Maks Ovsjanikov  
  _2026-04-08_ · https://arxiv.org/abs/2604.07337v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has revolutionized fast novel view synthesis, yet its opacity-based formulation makes surface extraction fundamentally difficult. Unlike implicit methods built on Signed Distance Fields or occupancy, 3DGS lacks a global geometric field, forcing existing approaches to resort to heuristics such as TSDF fusion of blended depth maps. Inspired by the Objects as Volumes framework, we derive a principled occupancy field for Gaussian Splatting and show how it can be used to extract highly accurate watertight meshes of complex scenes. Our key contribution is to introduce a learnable oriented normal at each Gaussian element and to define an adapted attenuation formulation, which leads to closed-form expressions for both the normal and occupancy fields at arbitrary locations in space. We further introduce a novel consistency loss and a dedicated densification strategy to enforce Gaussians to wrap the entire surface by closing geometric holes, ensuring a complete shell of oriented primitives. We modify the differentiable rasterizer to output depth as an isosurface of our continuous model, and introduce Primal Adaptive Meshing for Region-of-Interest meshing at arbitrary resolution. We additionally expose fundamental biases in standard surface evaluation protocols and propose two more rigorous alternatives. Overall, our method Gaussian Wrapping sets a new state-of-the-art on DTU and Tanks and Temples, producing complete, watertight meshes at a fraction of the size of concurrent work-recovering thin structures such as the notoriously elusive bicycle spokes.

  </details>



- **Are Face Embeddings Compatible Across Deep Neural Network Models?**  
  Fizza Rubab, Yiying Tong, Arun Ross  
  _2026-04-08_ · https://arxiv.org/abs/2604.07282v1  
  <details><summary>Abstract</summary>

  Automated face recognition has made rapid strides over the past decade due to the unprecedented rise of deep neural network (DNN) models that can be trained for domain-specific tasks. At the same time, foundation models that are pretrained on broad vision or vision-language tasks have shown impressive generalization across diverse domains, including biometrics. This raises an important question: Do different DNN models--both domain-specific and foundation models--encode facial identity in similar ways, despite being trained on different datasets, loss functions, and architectures? In this regard, we directly analyze the geometric structure of embedding spaces imputed by different DNN models. Treating embeddings of face images as point clouds, we study whether simple affine transformations can align face representations of one model with another. Our findings reveal surprising cross-model compatibility: low-capacity linear mappings substantially improve cross-model face recognition over unaligned baselines for both face identification and verification tasks. Alignment patterns generalize across datasets and vary systematically across model families, indicating representational convergence in facial identity encoding. These findings have implications for model interoperability, ensemble design, and biometric template security.

  </details>



- **Mem3R: Streaming 3D Reconstruction with Hybrid Memory via Test-Time Training**  
  Changkun Liu, Jiezhi Yang, Zeman Li, Yuan Deng, Jiancong Guo, Luca Ballan  
  _2026-04-08_ · https://arxiv.org/abs/2604.07279v1  
  <details><summary>Abstract</summary>

  Streaming 3D perception is well suited to robotics and augmented reality, where long visual streams must be processed efficiently and consistently. Recent recurrent models offer a promising solution by maintaining fixed-size states and enabling linear-time inference, but they often suffer from drift accumulation and temporal forgetting over long sequences due to the limited capacity of compressed latent memories. We propose Mem3R, a streaming 3D reconstruction model with a hybrid memory design that decouples camera tracking from geometric mapping to improve temporal consistency over long sequences. For camera tracking, Mem3R employs an implicit fast-weight memory implemented as a lightweight Multi-Layer Perceptron updated via Test-Time Training. For geometric mapping, Mem3R maintains an explicit token-based fixed-size state. Compared with CUT3R, this design not only significantly improves long-sequence performance but also reduces the model size from 793M to 644M parameters. Mem3R supports existing improved plug-and-play state update strategies developed for CUT3R. Specifically, integrating it with TTT3R decreases Absolute Trajectory Error by up to 39% over the base implementation on 500 to 1000 frame sequences. The resulting improvements also extend to other downstream tasks, including video depth estimation and 3D reconstruction, while preserving constant GPU memory usage and comparable inference throughput. Project page: https://lck666666.github.io/Mem3R/

  </details>



- **Geo-EVS: Geometry-Conditioned Extrapolative View Synthesis for Autonomous Driving**  
  Yatong Lan, Rongkui Tang, Lei He  
  _2026-04-08_ · https://arxiv.org/abs/2604.07250v1  
  <details><summary>Abstract</summary>

  Extrapolative novel view synthesis can reduce camera-rig dependency in autonomous driving by generating standardized virtual views from heterogeneous sensors. Existing methods degrade outside recorded trajectories because extrapolated poses provide weak geometric support and no dense target-view supervision. The key is to explicitly expose the model to out-of-trajectory condition defects during training. We propose Geo-EVS, a geometry-conditioned framework under sparse supervision. Geo-EVS has two components. Geometry-Aware Reprojection (GAR) uses fine-tuned VGGT to reconstruct colored point clouds and reproject them to observed and virtual target poses, producing geometric condition maps. This design unifies the reprojection path between training and inference. Artifact-Guided Latent Diffusion (AGLD) injects reprojection-derived artifact masks during training so the model learns to recover structure under missing support. For evaluation, we use a LiDAR-Projected Sparse-Reference (LPSR) protocol when dense extrapolated-view ground truth is unavailable. On Waymo, Geo-EVS improves sparse-view synthesis quality and geometric accuracy, especially in high-angle and low-coverage settings. It also improves downstream 3D detection.

  </details>



- **AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors**  
  Xiaoxue Zhang, Xiaoxu Zheng, Yixuan Yin, Tiao Zhao, Kaihua Tang, Michael Bi Mi, Zhan Xu, Dave Zhenyu Chen  
  _2026-04-08_ · https://arxiv.org/abs/2604.07053v2  
  <details><summary>Abstract</summary>

  Recent feed-forward Gaussian reconstruction models adopt a pixel-aligned formulation that maps each 2D pixel to a 3D Gaussian, entangling Gaussian representations tightly with the input images. In this paper, we propose AnchorSplat, a novel feed-forward 3DGS framework for scene-level reconstruction that represents the scene directly in 3D space. AnchorSplat introduces an anchor-aligned Gaussian representation guided by 3D geometric priors (e.g., sparse point clouds, voxels, or RGB-D point clouds), enabling a more geometry-aware renderable 3D Gaussians that is independent of image resolution and number of views. This design substantially reduces the number of required Gaussians, improving computational efficiency while enhancing reconstruction fidelity. Beyond the anchor-aligned design, we utilize a Gaussian Refiner to adjust the intermediate Gaussiansy via merely a few forward passes. Experiments on the ScanNet++ v2 NVS benchmark demonstrate the SOTA performance, outperforming previous methods with more view-consistent and substantially fewer Gaussian primitives.

  </details>



- **Synthetic Dataset Generation for Partially Observed Indoor Objects**  
  Jelle Vermandere, Maarten Bassier, Maarten Vergauwen  
  _2026-04-08_ · https://arxiv.org/abs/2604.07010v1  
  <details><summary>Abstract</summary>

  Learning-based methods for 3D scene reconstruction and object completion require large datasets containing partial scans paired with complete ground-truth geometry. However, acquiring such datasets using real-world scanning systems is costly and time-consuming, particularly when accurate ground truth for occluded regions is required. In this work, we present a virtual scanning framework implemented in Unity for generating realistic synthetic 3D scan datasets. The proposed system simulates the behaviour of real-world scanners using configurable parameters such as scan resolution, measurement range, and distance-dependent noise. Instead of directly sampling mesh surfaces, the framework performs ray-based scanning from virtual viewpoints, enabling realistic modelling of sensor visibility and occlusion effects. In addition, panoramic images captured at the scanner location are used to assign colours to the resulting point clouds. To support scalable dataset creation, the scanner is integrated with a procedural indoor scene generation pipeline that automatically produces diverse room layouts and furniture arrangements. Using this system, we introduce the \textit{V-Scan} dataset, which contains synthetic indoor scans together with object-level partial point clouds, voxel-based occlusion grids, and complete ground-truth geometry. The resulting dataset provides valuable supervision for training and evaluating learning-based methods for scene reconstruction and object completion.

  </details>



- **FORGE:Fine-grained Multimodal Evaluation for Manufacturing Scenarios**  
  Xiangru Jian, Hao Xu, Wei Pang, Xinjian Zhao, Chengyu Tao, Qixin Zhang, Xikun Zhang, Chao Zhang, Guanzhi Deng, Alex Xue, et al.  
  _2026-04-08_ · https://arxiv.org/abs/2604.07413v1  
  <details><summary>Abstract</summary>

  The manufacturing sector is increasingly adopting Multimodal Large Language Models (MLLMs) to transition from simple perception to autonomous execution, yet current evaluations fail to reflect the rigorous demands of real-world manufacturing environments. Progress is hindered by data scarcity and a lack of fine-grained domain semantics in existing datasets. To bridge this gap, we introduce FORGE. Wefirst construct a high-quality multimodal dataset that combines real-world 2D images and 3D point clouds, annotated with fine-grained domain semantics (e.g., exact model numbers). We then evaluate 18 state-of-the-art MLLMs across three manufacturing tasks, namely workpiece verification, structural surface inspection, and assembly verification, revealing significant performance gaps. Counter to conventional understanding, the bottleneck analysis shows that visual grounding is not the primary limiting factor. Instead, insufficient domain-specific knowledge is the key bottleneck, setting a clear direction for future research. Beyond evaluation, we show that our structured annotations can serve as an actionable training resource: supervised fine-tuning of a compact 3B-parameter model on our data yields up to 90.8% relative improvement in accuracy on held-out manufacturing scenarios, providing preliminary evidence for a practical pathway toward domain-adapted manufacturing MLLMs. The code and datasets are available at https://ai4manufacturing.github.io/forge-web.

  </details>


