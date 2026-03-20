# 3D Reconstruction

_Updated: 2026-03-20 07:11 UTC_

Total papers shown: **25**


---

- **MonoArt: Progressive Structural Reasoning for Monocular Articulated 3D Reconstruction**  
  Haitian Li, Haozhe Xie, Junxiang Xu, Beichen Wen, Fangzhou Hong, Ziwei Liu  
  _2026-03-19_ · https://arxiv.org/abs/2603.19231v1  
  <details><summary>Abstract</summary>

  Reconstructing articulated 3D objects from a single image requires jointly inferring object geometry, part structure, and motion parameters from limited visual evidence. A key difficulty lies in the entanglement between motion cues and object structure, which makes direct articulation regression unstable. Existing methods address this challenge through multi-view supervision, retrieval-based assembly, or auxiliary video generation, often sacrificing scalability or efficiency. We present MonoArt, a unified framework grounded in progressive structural reasoning. Rather than predicting articulation directly from image features, MonoArt progressively transforms visual observations into canonical geometry, structured part representations, and motion-aware embeddings within a single architecture. This structured reasoning process enables stable and interpretable articulation inference without external motion templates or multi-stage pipelines. Extensive experiments on PartNet-Mobility demonstrate that OM achieves state-of-the-art performance in both reconstruction accuracy and inference speed. The framework further generalizes to robotic manipulation and articulated scene reconstruction.

  </details>



- **Reconstruction Matters: Learning Geometry-Aligned BEV Representation through 3D Gaussian Splatting**  
  Yiren Lu, Xin Ye, Burhaneddin Yaman, Jingru Luo, Zhexiao Xiong, Liu Ren, Yu Yin  
  _2026-03-19_ · https://arxiv.org/abs/2603.19193v1  
  <details><summary>Abstract</summary>

  Bird's-Eye-View (BEV) perception serves as a cornerstone for autonomous driving, offering a unified spatial representation that fuses surrounding-view images to enable reasoning for various downstream tasks, such as semantic segmentation, 3D object detection, and motion prediction. However, most existing BEV perception frameworks adopt an end-to-end training paradigm, where image features are directly transformed into the BEV space and optimized solely through downstream task supervision. This formulation treats the entire perception process as a black box, often lacking explicit 3D geometric understanding and interpretability, leading to suboptimal performance. In this paper, we claim that an explicit 3D representation matters for accurate BEV perception, and we propose Splat2BEV, a Gaussian Splatting-assisted framework for BEV tasks. Splat2BEV aims to learn BEV feature representations that are both semantically rich and geometrically precise. We first pre-train a Gaussian generator that explicitly reconstructs 3D scenes from multi-view inputs, enabling the generation of geometry-aligned feature representations. These representations are then projected into the BEV space to serve as inputs for downstream tasks. Extensive experiments on nuScenes and argoverse dataset demonstrate that Splat2BEV achieves state-of-the-art performance and validate the effectiveness of incorporating explicit 3D reconstruction into BEV perception.

  </details>



- **Generalized Hand-Object Pose Estimation with Occlusion Awareness**  
  Hui Yang, Wei Sun, Jian Liu, Jian Xiao Tao Xie, Hossein Rahmani, Ajmal Saeed mian, Nicu Sebe, Gim Hee Lee  
  _2026-03-19_ · https://arxiv.org/abs/2603.19013v1  
  <details><summary>Abstract</summary>

  Generalized 3D hand-object pose estimation from a single RGB image remains challenging due to the large variations in object appearances and interaction patterns, especially under heavy occlusion. We propose GenHOI, a framework for generalized hand-object pose estimation with occlusion awareness. GenHOI integrates hierarchical semantic knowledge with hand priors to enhance model generalization under challenging occlusion conditions. Specifically, we introduce a hierarchical semantic prompt that encodes object states, hand configurations, and interaction patterns via textual descriptions. This enables the model to learn abstract high-level representations of hand-object interactions for generalization to unseen objects and novel interactions while compensating for missing or ambiguous visual cues. To enable robust occlusion reasoning, we adopt a multi-modal masked modeling strategy over RGB images, predicted point clouds, and textual descriptions. Moreover, we leverage hand priors as stable spatial references to extract implicit interaction constraints. This allows reliable pose inference even under significant variations in object shapes and interaction patterns. Extensive experiments on the challenging DexYCB and HO3Dv2 benchmarks demonstrate that our method achieves state-of-the-art performance in hand-object pose estimation.

  </details>



- **VGGT-360: Geometry-Consistent Zero-Shot Panoramic Depth Estimation**  
  Jiayi Yuan, Haobo Jiang, De Wen Soh, Na Zhao  
  _2026-03-19_ · https://arxiv.org/abs/2603.18943v1  
  <details><summary>Abstract</summary>

  This paper presents VGGT-360, a novel training-free framework for zero-shot, geometry-consistent panoramic depth estimation. Unlike prior view-independent training-free approaches, VGGT-360 reformulates the task as panoramic reprojection over multi-view reconstructed 3D models by leveraging the intrinsic 3D consistency of VGGT-like foundation models, thereby unifying fragmented per-view reasoning into a coherent panoramic understanding. To achieve robust and accurate estimation, VGGT-360 integrates three plug-and-play modules that form a unified panorama-to-3D-to-depth framework: (i) Uncertainty-guided adaptive projection slices panoramas into perspective views to bridge the domain gap between panoramic inputs and VGGT's perspective prior. It estimates gradient-based uncertainty to allocate denser views to geometry-poor regions, yielding geometry-informative inputs for VGGT. (ii) Structure-saliency enhanced attention strengthens VGGT's robustness during 3D reconstruction by injecting structure-aware confidence into its attention layers, guiding focus toward geometrically reliable regions and enhancing cross-view coherence. (iii) Correlation-weighted 3D model correction refines the reconstructed 3D model by reweighting overlapping points using attention-inferred correlation scores, providing a consistent geometric basis for accurate panoramic reprojection. Extensive experiments show that VGGT-360 outperforms both trained and training-free state-of-the-art methods across multiple resolutions and diverse indoor and outdoor datasets.

  </details>



- **GHOST: Fast Category-agnostic Hand-Object Interaction Reconstruction from RGB Videos using Gaussian Splatting**  
  Ahmed Tawfik Aboukhadra, Marcel Rogge, Nadia Robertini, Abdalla Arafa, Jameel Malik, Ahmed Elhayek, Didier Stricker  
  _2026-03-19_ · https://arxiv.org/abs/2603.18912v1  
  <details><summary>Abstract</summary>

  Understanding realistic hand-object interactions from monocular RGB videos is essential for AR/VR, robotics, and embodied AI. Existing methods rely on category-specific templates or heavy computation, yet still produce physically inconsistent hand-object alignment in 3D. We introduce GHOST (Gaussian Hand-Object Splatting), a fast, category-agnostic framework for reconstructing dynamic hand-object interactions using 2D Gaussian Splatting. GHOST represents both hands and objects as dense, view-consistent Gaussian discs and introduces three key innovations: (1) a geometric-prior retrieval and consistency loss that completes occluded object regions, (2) a grasp-aware alignment that refines hand translations and object scale to ensure realistic contact, and (3) a hand-aware background loss that prevents penalizing hand-occluded object regions. GHOST achieves complete, physically consistent, and animatable reconstructions from a single RGB video while running an order of magnitude faster than prior category-agnostic methods. Extensive experiments on ARCTIC, HO3D, and in-the-wild datasets demonstrate state-of-the-art accuracy in 3D reconstruction and 2D rendering quality, establishing GHOST as an efficient and robust solution for realistic hand-object interaction modeling. Code is available at https://github.com/ATAboukhadra/GHOST.

  </details>



- **Points-to-3D: Structure-Aware 3D Generation with Point Cloud Priors**  
  Jiatong Xia, Zicheng Duan, Anton van den Hengel, Lingqiao Liu  
  _2026-03-19_ · https://arxiv.org/abs/2603.18782v1  
  <details><summary>Abstract</summary>

  Recent progress in 3D generation has been driven largely by models conditioned on images or text, while readily available 3D priors are still underused. In many real-world scenarios, the visible-region point cloud are easy to obtain from active sensors such as LiDAR or from feed-forward predictors like VGGT, offering explicit geometric constraints that current methods fail to exploit. In this work, we introduce Points-to-3D, a diffusion-based framework that leverages point cloud priors for geometry-controllable 3D asset and scene generation. Built on a latent 3D diffusion model TRELLIS, Points-to-3D first replaces pure-noise sparse structure latent initialization with a point cloud priors tailored input formulation.A structure inpainting network, trained within the TRELLIS framework on task-specific data designed to learn global structural inpainting, is then used for inference with a staged sampling strategy (structural inpainting followed by boundary refinement), completing the global geometry while preserving the visible regions of the input priors.In practice, Points-to-3D can take either accurate point-cloud priors or VGGT-estimated point clouds from single images as input. Experiments on both objects and scene scenarios consistently demonstrate superior performance over state-of-the-art baselines in terms of rendering quality and geometric fidelity, highlighting the effectiveness of explicitly embedding point-cloud priors for achieving more accurate and structurally controllable 3D generation.

  </details>



- **SEAR: Simple and Efficient Adaptation of Visual Geometric Transformers for RGB+Thermal 3D Reconstruction**  
  Vsevolod Skorokhodov, Chenghao Xu, Shuo Sun, Olga Fink, Malcolm Mielle  
  _2026-03-19_ · https://arxiv.org/abs/2603.18774v1  
  <details><summary>Abstract</summary>

  Foundational feed-forward visual geometry models enable accurate and efficient camera pose estimation and scene reconstruction by learning strong scene priors from massive RGB datasets. However, their effectiveness drops when applied to mixed sensing modalities, such as RGB-thermal (RGB-T) images. We observe that while a visual geometry grounded transformer pretrained on RGB data generalizes well to thermal-only reconstruction, it struggles to align RGB and thermal modalities when processed jointly. To address this, we propose SEAR, a simple yet efficient fine-tuning strategy that adapts a pretrained geometry transformer to multimodal RGB-T inputs. Despite being trained on a relatively small RGB-T dataset, our approach significantly outperforms state-of-the-art methods for 3D reconstruction and camera pose estimation, achieving significant improvements over all metrics (e.g., over 29\% in AUC@30) and delivering higher detail and consistency between modalities with negligible overhead in inference time compared to the original pretrained model. Notably, SEAR enables reliable multimodal pose estimation and reconstruction even under challenging conditions, such as low lighting and dense smoke. We validate our architecture through extensive ablation studies, demonstrating how the model aligns both modalities. Additionally, we introduce a new dataset featuring RGB and thermal sequences captured at different times, viewpoints, and illumination conditions, providing a robust benchmark for future work in multimodal 3D scene reconstruction. Code and models are publicly available at https://www.github.com/Schindler-EPFL-Lab/SEAR.

  </details>



- **SwiftGS: Episodic Priors for Immediate Satellite Surface Recovery**  
  Rong Fu, Jiekai Wu, Haiyun Wei, Xiaowen Ma, Shiyin Lin, Kangan Qian, Chuang Liu, Jianyuan Ni, Simon James Fong  
  _2026-03-19_ · https://arxiv.org/abs/2603.18634v1  
  <details><summary>Abstract</summary>

  Rapid, large-scale 3D reconstruction from multi-date satellite imagery is vital for environmental monitoring, urban planning, and disaster response, yet remains difficult due to illumination changes, sensor heterogeneity, and the cost of per-scene optimization. We introduce SwiftGS, a meta-learned system that reconstructs 3D surfaces in a single forward pass by predicting geometry-radiation-decoupled Gaussian primitives together with a lightweight SDF, replacing expensive per-scene fitting with episodic training that captures transferable priors. The model couples a differentiable physics graph for projection, illumination, and sensor response with spatial gating that blends sparse Gaussian detail and global SDF structure, and incorporates semantic-geometric fusion, conditional lightweight task heads, and multi-view supervision from a frozen geometric teacher under an uncertainty-aware multi-task loss. At inference, SwiftGS operates zero-shot with optional compact calibration and achieves accurate DSM reconstruction and view-consistent rendering at significantly reduced computational cost, with ablations highlighting the benefits of the hybrid representation, physics-aware rendering, and episodic meta-training.

  </details>



- **NymeriaPlus: Enriching Nymeria Dataset with Additional Annotations and Data**  
  Daniel DeTone, Federica Bogo, Eric-Tuan Le, Duncan Frost, Julian Straub, Yawar Siddiqui, Yuting Ye, Jakob Engel, Richard Newcombe, Lingni Ma  
  _2026-03-19_ · https://arxiv.org/abs/2603.18496v1  
  <details><summary>Abstract</summary>

  The Nymeria Dataset, released in 2024, is a large-scale collection of in-the-wild human activities captured with multiple egocentric wearable devices that are spatially localized and temporally synchronized. It provides body-motion ground truth recorded with a motion-capture suit, device trajectories, semi-dense 3D point clouds, and in-context narrations. In this paper, we upgrade Nymeria and introduce NymeriaPlus. NymeriaPlus features: (1) improved human motion in Momentum Human Rig (MHR) and SMPL formats; (2) dense 3D and 2D bounding box annotations for indoor objects and structural elements; (3) instance-level 3D object reconstructions; and (4) additional modalities e.g., basemap recordings, audio, and wristband videos. By consolidating these complementary modalities and annotations into a single, coherent benchmark, NymeriaPlus strengthens Nymeria into a more powerful in-the-wild egocentric dataset. We expect NymeriaPlus to bridge a key gap in existing egocentric resources and to support a broader range of research, including unique explorations of multimodal learning for embodied AI.

  </details>



- **FILT3R: Latent State Adaptive Kalman Filter for Streaming 3D Reconstruction**  
  Seonghyun Jin, Jong Chul Ye  
  _2026-03-19_ · https://arxiv.org/abs/2603.18493v1  
  <details><summary>Abstract</summary>

  Streaming 3D reconstruction maintains a persistent latent state that is updated online from incoming frames, enabling constant-memory inference. A key failure mode is the state update rule: aggressive overwrites forget useful history, while conservative updates fail to track new evidence, and both behaviors become unstable beyond the training horizon. To address this challenge, we propose FILT3R, a training-free latent filtering layer that casts recurrent state updates as stochastic state estimation in token space. FILT3R maintains a per-token variance and computes a Kalman-style gain that adaptively balances memory retention against new observations. Process noise -- governing how much the latent state is expected to change between frames -- is estimated online from EMA-normalized temporal drift of candidate tokens. Using extensive experiments, we demonstrate that FILT3R yields an interpretable, plug-in update rule that generalizes common overwrite and gating policies as special cases. Specifically, we show that gains shrink in stable regimes as uncertainty contracts with accumulated evidence, and rise when genuine scene change increases process uncertainty, improving long-horizon stability for depth, pose, and 3D reconstruction, compared to the existing methods. Code will be released at https://github.com/jinotter3/FILT3R.

  </details>



- **Pixel-Accurate Epipolar Guided Matching**  
  Oleksii Nasypanyi, Francois Rameau  
  _2026-03-19_ · https://arxiv.org/abs/2603.18401v1  
  <details><summary>Abstract</summary>

  Keypoint matching can be slow and unreliable in challenging conditions such as repetitive textures or wide-baseline views. In such cases, known geometric relations (e.g., the fundamental matrix) can be used to restrict potential correspondences to a narrow epipolar envelope, thereby reducing the search space and improving robustness. These epipolar-guided matching approaches have proved effective in tasks such as SfM; however, most rely on coarse spatial binning, which introduces approximation errors, requires costly post-processing, and may miss valid correspondences. We address these limitations with an exact formulation that performs candidate selection directly in angular space. In our approach, each keypoint is assigned a tolerance circle which, when viewed from the epipole, defines an angular interval. Matching then becomes a 1D angular interval query, solved efficiently in logarithmic time with a segment tree. This guarantees pixel-level tolerance, supports per-keypoint control, and removes unnecessary descriptor comparisons. Extensive evaluation on ETH3D demonstrates noticeable speedups over existing approaches while recovering exact correspondence sets.

  </details>



- **Fast and Generalizable NeRF Architecture Selection for Satellite Scene Reconstruction**  
  Devjyoti Chakraborty, Zaki Sukma, Rakandhiya D. Rachmanto, Kriti Ghosh, In Kee Kim, Suchendra M. Bhandarkar, Lakshmish Ramaswamy, Nancy K. O'Hare, Deepak Mishra  
  _2026-03-18_ · https://arxiv.org/abs/2603.18306v1  
  <details><summary>Abstract</summary>

  Neural Radiance Fields (NeRF) have emerged as a powerful approach for photorealistic 3D reconstruction from multi-view images. However, deploying NeRF for satellite imagery remains challenging. Each scene requires individual training, and optimizing architectures via Neural Architecture Search (NAS) demands hours to days of GPU time. While existing approaches focus on architectural improvements, our SHAP analysis reveals that multi-view consistency, rather than model architecture, determines reconstruction quality. Based on this insight, we develop PreSCAN, a predictive framework that estimates NeRF quality prior to training using lightweight geometric and photometric descriptors. PreSCAN selects suitable architectures in < 30 seconds with < 1 dB prediction error, achieving 1000$\times$ speedup over NAS. We further demonstrate PreSCAN's deployment utility on edge platforms (Jetson Orin), where combining its predictions with offline cost profiling reduces inference power by 26% and latency by 43% with minimal quality loss. Experiments on DFC2019 datasets confirm that PreSCAN generalizes across diverse satellite scenes without retraining.

  </details>



- **Semantic Segmentation and Depth Estimation for Real-Time Lunar Surface Mapping Using 3D Gaussian Splatting**  
  Guillem Casadesus Vila, Adam Dai, Grace Gao  
  _2026-03-18_ · https://arxiv.org/abs/2603.18218v1  
  <details><summary>Abstract</summary>

  Navigation and mapping on the lunar surface require robust perception under challenging conditions, including poorly textured environments, high-contrast lighting, and limited computational resources. This paper presents a real-time mapping framework that integrates dense perception models with a 3D Gaussian Splatting (3DGS) representation. We first benchmark several models on synthetic datasets generated with the LuPNT simulator, selecting a stereo dense depth estimation model based on Gated Recurrent Units for its balance of speed and accuracy in depth estimation, and a convolutional neural network for its superior performance in detecting semantic segments. Using ground truth poses to decouple the local scene understanding from the global state estimation, our pipeline reconstructs a 120-meter traverse with a geometric height accuracy of approximately 3 cm, outperforming a traditional point cloud baseline without LiDAR. The resulting 3DGS map enables novel view synthesis and serves as a foundation for a full SLAM system, where its capacity for joint map and pose optimization would offer significant advantages. Our results demonstrate that combining semantic segmentation and dense depth estimation with learned map representations is an effective approach for creating detailed, large-scale maps to support future lunar surface missions.

  </details>



- **GMT: Goal-Conditioned Multimodal Transformer for 6-DOF Object Trajectory Synthesis in 3D Scenes**  
  Huajian Zeng, Abhishek Saroha, Daniel Cremers, Xi Wang  
  _2026-03-18_ · https://arxiv.org/abs/2603.17993v1  
  <details><summary>Abstract</summary>

  Synthesizing controllable 6-DOF object manipulation trajectories in 3D environments is essential for enabling robots to interact with complex scenes, yet remains challenging due to the need for accurate spatial reasoning, physical feasibility, and multimodal scene understanding. Existing approaches often rely on 2D or partial 3D representations, limiting their ability to capture full scene geometry and constraining trajectory precision. We present GMT, a multimodal transformer framework that generates realistic and goal-directed object trajectories by jointly leveraging 3D bounding box geometry, point cloud context, semantic object categories, and target end poses. The model represents trajectories as continuous 6-DOF pose sequences and employs a tailored conditioning strategy that fuses geometric, semantic, contextual, and goaloriented information. Extensive experiments on synthetic and real-world benchmarks demonstrate that GMT outperforms state-of-the-art human motion and human-object interaction baselines, such as CHOIS and GIMO, achieving substantial gains in spatial accuracy and orientation control. Our method establishes a new benchmark for learningbased manipulation planning and shows strong generalization to diverse objects and cluttered 3D environments. Project page: https://huajian- zeng.github. io/projects/gmt/.

  </details>



- **Feeling the Space: Egomotion-Aware Video Representation for Efficient and Accurate 3D Scene Understanding**  
  Shuyao Shi, Kang G. Shin  
  _2026-03-18_ · https://arxiv.org/abs/2603.17980v1  
  <details><summary>Abstract</summary>

  Recent Multimodal Large Language Models (MLLMs) have shown high potential for spatial reasoning within 3D scenes. However, they typically rely on computationally expensive 3D representations like point clouds or reconstructed Bird's-Eye View (BEV) maps, or lack physical grounding to resolve ambiguities in scale and size. This paper significantly enhances MLLMs with egomotion modality data, captured by Inertial Measurement Units (IMUs) concurrently with the video. In particular, we propose a novel framework, called Motion-MLLM, introducing two key components: (1) a cascaded motion-visual keyframe filtering module that leverages both IMU data and visual features to efficiently select a sparse yet representative set of keyframes, and (2) an asymmetric cross-modal fusion module where motion tokens serve as intermediaries that channel egomotion cues and cross-frame visual context into the visual representation. By grounding visual content in physical egomotion trajectories, Motion-MLLM can reason about absolute scale and spatial relationships across the scene. Our extensive evaluation shows that Motion-MLLM makes significant improvements in various tasks related to 3D scene understanding and spatial reasoning. Compared to state-of-the-art (SOTA) methods based on video frames and explicit 3D data, Motion-MLLM exhibits similar or even higher accuracy with significantly less overhead (i.e., 1.40$\times$ and 1.63$\times$ higher cost-effectiveness, respectively).

  </details>



- **SegFly: A 2D-3D-2D Paradigm for Aerial RGB-Thermal Semantic Segmentation at Scale**  
  Markus Gross, Sai Bharadhwaj Matha, Rui Song, Viswanathan Muthuveerappan, Conrad Christoph, Julius Huber, Daniel Cremers  
  _2026-03-18_ · https://arxiv.org/abs/2603.17920v1  
  <details><summary>Abstract</summary>

  Semantic segmentation for uncrewed aerial vehicles (UAVs) is fundamental for aerial scene understanding, yet existing RGB and RGB-T datasets remain limited in scale, diversity, and annotation efficiency due to the high cost of manual labeling and the difficulties of accurate RGB-T alignment on off-the-shelf UAVs. To address these challenges, we propose a scalable geometry-driven 2D-3D-2D paradigm that leverages multi-view redundancy in high-overlap aerial imagery to automatically propagate labels from a small subset of manually annotated RGB images to both RGB and thermal modalities within a unified framework. By lifting less than 3% of RGB images into a semantic 3D point cloud and reprojecting it into all views, our approach enables dense pseudo ground-truth generation across large image collections, automatically producing 97% of RGB labels and 100% of thermal labels while achieving 91% and 88% annotation accuracy without any 2D manual refinement. We further extend this 2D-3D-2D paradigm to cross-modal image registration, using 3D geometry as an intermediate alignment space to obtain fully automatic, strong pixel-level RGB-T alignment with 87% registration accuracy and no hardware-level synchronization. Applying our framework to existing geo-referenced aerial imagery, we construct SegFly, a large-scale benchmark with over 20,000 high-resolution RGB images and more than 15,000 geometrically aligned RGB-T pairs spanning diverse urban, industrial, and rural environments across multiple altitudes and seasons. On SegFly, we establish the Firefly baseline for RGB and thermal semantic segmentation and show that both conventional architectures and vision foundation models benefit substantially from SegFly supervision, highlighting the potential of geometry-driven 2D-3D-2D pipelines for scalable multi-modal scene understanding. Data and Code available at https://github.com/markus-42/SegFly.

  </details>



- **PC-CrossDiff: Point-Cluster Dual-Level Cross-Modal Differential Attention for Unified 3D Referring and Segmentation**  
  Wenbin Tan, Jiawen Lin, Fangyong Wang, Yuan Xie, Yong Xie, Yachao Zhang, Yanyun Qu  
  _2026-03-18_ · https://arxiv.org/abs/2603.17753v1  
  <details><summary>Abstract</summary>

  3D Visual Grounding (3DVG) aims to localize the referent of natural language referring expressions through two core tasks: Referring Expression Comprehension (3DREC) and Segmentation (3DRES). While existing methods achieve high accuracy in simple, single-object scenes, they suffer from severe performance degradation in complex, multi-object scenes that are common in real-world settings, hindering practical deployment. Existing methods face two key challenges in complex, multi-object scenes: inadequate parsing of implicit localization cues critical for disambiguating visually similar objects, and ineffective suppression of dynamic spatial interference from co-occurring objects, resulting in degraded grounding accuracy. To address these challenges, we propose PC-CrossDiff, a unified dual-task framework with a dual-level cross-modal differential attention architecture for 3DREC and 3DRES. Specifically, the framework introduces: (i) Point-Level Differential Attention (PLDA) modules that apply bidirectional differential attention between text and point clouds, adaptively extracting implicit localization cues via learnable weights to improve discriminative representation; (ii) Cluster-Level Differential Attention (CLDA) modules that establish a hierarchical attention mechanism to adaptively enhance localization-relevant spatial relationships while suppressing ambiguous or irrelevant spatial relations through a localization-aware differential attention block. Our method achieves state-of-the-art performance on the ScanRefer, NR3D, and SR3D benchmarks. Notably, on the Implicit subsets of ScanRefer, it improves the Overall@0.50 score by +10.16% for the 3DREC task, highlighting its strong ability to parse implicit spatial cues.

  </details>



- **TAPESTRY: From Geometry to Appearance via Consistent Turntable Videos**  
  Yan Zeng, Haoran Jiang, Kaixin Yao, Qixuan Zhang, Longwen Zhang, Lan Xu, Jingyi Yu  
  _2026-03-18_ · https://arxiv.org/abs/2603.17735v1  
  <details><summary>Abstract</summary>

  Automatically generating photorealistic and self-consistent appearances for untextured 3D models is a critical challenge in digital content creation. The advancement of large-scale video generation models offers a natural approach: directly synthesizing 360-degree turntable videos (TTVs), which can serve not only as high-quality dynamic previews but also as an intermediate representation to drive texture synthesis and neural rendering. However, existing general-purpose video diffusion models struggle to maintain strict geometric consistency and appearance stability across the full range of views, making their outputs ill-suited for high-quality 3D reconstruction. To this end, we introduce TAPESTRY, a framework for generating high-fidelity TTVs conditioned on explicit 3D geometry. We reframe the 3D appearance generation task as a geometry-conditioned video diffusion problem: given a 3D mesh, we first render and encode multi-modal geometric features to constrain the video generation process with pixel-level precision, thereby enabling the creation of high-quality and consistent TTVs. Building upon this, we also design a method for downstream reconstruction tasks from the TTV input, featuring a multi-stage pipeline with 3D-Aware Inpainting. By rotating the model and performing a context-aware secondary generation, this pipeline effectively completes self-occluded regions to achieve full surface coverage. The videos generated by TAPESTRY are not only high-quality dynamic previews but also serve as a reliable, 3D-aware intermediate representation that can be seamlessly back-projected into UV textures or used to supervise neural rendering methods like 3DGS. This enables the automated creation of production-ready, complete 3D assets from untextured meshes. Experimental results demonstrate that our method outperforms existing approaches in both video consistency and final reconstruction quality.

  </details>



- **PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery**  
  Yijing Guo, Mengjun Chao, Luo Wang, Tianyang Zhao, Haizhao Dai, Yingliang Zhang, Jingyi Yu, Yujiao Shi  
  _2026-03-18_ · https://arxiv.org/abs/2603.17571v1  
  <details><summary>Abstract</summary>

  Panoramic imagery offers a full 360° field of view and is increasingly common in consumer devices. However, it introduces non-pinhole distortions that challenge joint pose estimation and 3D reconstruction. Existing feed-forward models, built for perspective cameras, generalize poorly to this setting. We propose PanoVGGT, a permutation-equivariant Transformer framework that jointly predicts camera poses, depth maps, and 3D point clouds from one or multiple panoramas in a single forward pass. The model incorporates spherical-aware positional embeddings and a panorama-specific three-axis SO(3) rotation augmentation, enabling effective geometric reasoning in the spherical domain. To resolve inherent global-frame ambiguity, we further introduce a stochastic anchoring strategy during training. In addition, we contribute PanoCity, a large-scale outdoor panoramic dataset with dense depth and 6-DoF pose annotations. Extensive experiments on PanoCity and standard benchmarks demonstrate that PanoVGGT achieves competitive accuracy, strong robustness, and improved cross-domain generalization. Code and dataset will be released.

  </details>



- **Learning Coordinate-based Convolutional Kernels for Continuous SE(3) Equivariant and Efficient Point Cloud Analysis**  
  Jaein Kim, Hee Bin Yoo, Dong-Sig Han, Byoung-Tak Zhang  
  _2026-03-18_ · https://arxiv.org/abs/2603.17538v1  
  <details><summary>Abstract</summary>

  A symmetry on rigid motion is one of the salient factors in efficient learning of 3D point cloud problems. Group convolution has been a representative method to extract equivariant features, but its realizations have struggled to retain both rigorous symmetry and scalability simultaneously. We advocate utilizing the intertwiner framework to resolve this trade-off, but previous works on it, which did not achieve complete SE(3) symmetry or scalability to large-scale problems, necessitate a more advanced kernel architecture. We present Equivariant Coordinate-based Kernel Convolution, or ECKConv. It acquires SE(3) equivariance from the kernel domain defined in a double coset space, and its explicit kernel design using coordinate-based networks enhances its learning capability and memory efficiency. The experiments on diverse point cloud tasks, e.g., classification, pose registration, part segmentation, and large-scale semantic segmentation, validate the rigid equivariance, memory scalability, and outstanding performance of ECKConv compared to state-of-the-art equivariant methods.

  </details>



- **UniSem: Generalizable Semantic 3D Reconstruction from Sparse Unposed Images**  
  Guibiao Liao, Qian Ren, Kaimin Liao, Hua Wang, Zhi Chen, Luchao Wang, Yaohua Tang  
  _2026-03-18_ · https://arxiv.org/abs/2603.17519v1  
  <details><summary>Abstract</summary>

  Semantic-aware 3D reconstruction from sparse, unposed images remains challenging for feed-forward 3D Gaussian Splatting (3DGS). Existing methods often predict an over-complete set of Gaussian primitives under sparse-view supervision, leading to unstable geometry and inferior depth quality. Meanwhile, they rely solely on 2D segmenter features for semantic lifting, which provides weak 3D-level and limited generalizable supervision, resulting in incomplete 3D semantics in novel scenes. To address these issues, we propose UniSem, a unified framework that jointly improves depth accuracy and semantic generalization via two key components. First, Error-aware Gaussian Dropout (EGD) performs error-guided capacity control by suppressing redundancy-prone Gaussians using rendering error cues, producing meaningful, geometrically stable Gaussian representations for improved depth estimation. Second, we introduce a Mix-training Curriculum (MTC) that progressively blends 2D segmenter-lifted semantics with the model's own emergent 3D semantic priors, implemented with object-level prototype alignment to enhance semantic coherence and completeness. Extensive experiments on ScanNet and Replica show that UniSem achieves superior performance in depth prediction and open-vocabulary 3D segmentation across varying numbers of input views. Notably, with 16-view inputs, UniSem reduces depth Rel by 15.2% and improves open-vocabulary segmentation mAcc by 3.7% over strong baselines.

  </details>



- **Stereo World Model: Camera-Guided Stereo Video Generation**  
  Yang-Tian Sun, Zehuan Huang, Yifan Niu, Lin Ma, Yan-Pei Cao, Yuewen Ma, Xiaojuan Qi  
  _2026-03-18_ · https://arxiv.org/abs/2603.17375v1  
  <details><summary>Abstract</summary>

  We present StereoWorld, a camera-conditioned stereo world model that jointly learns appearance and binocular geometry for end-to-end stereo video generation.Unlike monocular RGB or RGBD approaches, StereoWorld operates exclusively within the RGB modality, while simultaneously grounding geometry directly from disparity. To efficiently achieve consistent stereo generation, our approach introduces two key designs: (1) a unified camera-frame RoPE that augments latent tokens with camera-aware rotary positional encoding, enabling relative, view- and time-consistent conditioning while preserving pretrained video priors via a stable attention initialization; and (2) a stereo-aware attention decomposition that factors full 4D attention into 3D intra-view attention plus horizontal row attention, leveraging the epipolar prior to capture disparity-aligned correspondences with substantially lower compute. Across benchmarks, StereoWorld improves stereo consistency, disparity accuracy, and camera-motion fidelity over strong monocular-then-convert pipelines, achieving more than 3x faster generation with an additional 5% gain in viewpoint consistency. Beyond benchmarks, StereoWorld enables end-to-end binocular VR rendering without depth estimation or inpainting, enhances embodied policy learning through metric-scale depth grounding, and is compatible with long-video distillation for extended interactive stereo synthesis.

  </details>



- **MCoT-MVS: Multi-level Vision Selection by Multi-modal Chain-of-Thought Reasoning for Composed Image Retrieval**  
  Xuri Ge, Chunhao Wang, Xindi Wang, Zheyun Qin, Zhumin Chen, Xin Xin  
  _2026-03-18_ · https://arxiv.org/abs/2603.17360v1  
  <details><summary>Abstract</summary>

  Composed Image Retrieval (CIR) aims to retrieve target images based on a reference image and modified texts. However, existing methods often struggle to extract the correct semantic cues from the reference image that best reflect the user's intent under textual modification prompts, resulting in interference from irrelevant visual noise. In this paper, we propose a novel Multi-level Vision Selection by Multi-modal Chain-of-Thought Reasoning (MCoT-MVS) for CIR, integrating attention-aware multi-level vision features guided by reasoning cues from a multi-modal large language model (MLLM). Specifically, we leverage an MLLM to perform chain-of-thought reasoning on the multimodal composed input, generating the retained, removed, and target-inferred texts. These textual cues subsequently guide two reference visual attention selection modules to selectively extract discriminative patch-level and instance-level semantics from the reference image. Finally, to effectively fuse these multi-granular visual cues with the modified text and the imagined target description, we design a weighted hierarchical combination module to align the composed query with target images in a unified embedding space. Extensive experiments on two CIR benchmarks, namely CIRR and FashionIQ, demonstrate that our approach consistently outperforms existing methods and achieves new state-of-the-art performance. Code and trained models are publicly released.

  </details>



- **LLM-Powered Flood Depth Estimation from Social Media Imagery: A Vision-Language Model Framework with Mechanistic Interpretability for Transportation Resilience**  
  Nafis Fuad, Xiaodong Qian  
  _2026-03-17_ · https://arxiv.org/abs/2603.17108v1  
  <details><summary>Abstract</summary>

  Urban flooding poses an escalating threat to transportation network continuity, yet no operational system currently provides real-time, street-level flood depth information at the centimeter resolution required for dynamic routing, electric vehicle (EV) safety, and autonomous vehicle (AV) operations. This study presents FloodLlama, a fine-tuned open-source vision-language model (VLM) for continuous flood depth estimation from single street-level images, supported by a multimodal sensing pipeline using TikTok data. A synthetic dataset of approximately 190000 images was generated, covering seven vehicle types, four weather conditions, and 41 depth levels (0-40 cm at 1 cm resolution). Progressive curriculum training enabled coarse-to-fine learning, while LLaMA 3.2-11B Vision was fine-tuned using QLoRA. Evaluation across 34797 trials reveals a depth-dependent prompt effect: simple prompts perform better for shallow flooding, whereas chain-of-thought (CoT) reasoning improves performance at greater depths. FloodLlama achieves a mean absolute error (MAE) below 0.97 cm and Acc@5cm above 93.7% for deep flooding, exceeding 96.8% for shallow depths. A five-phase mechanistic interpretability framework identifies layer L23 as the critical depth-encoding transition and enables selective fine-tuning that reduces trainable parameters by 76-80% while maintaining accuracy. The Tier 3 configuration achieves 98.62% accuracy on real-world data and shows strong robustness under visual occlusion. A TikTok-based data pipeline, validated on 676 annotated flood frames from Detroit, demonstrates the feasibility of real-time, crowd-sourced flood sensing. The proposed framework provides a scalable, infrastructure-free solution with direct implications for EV safety, AV deployment, and resilient transportation management.

  </details>



- **MessyKitchens: Contact-rich object-level 3D scene reconstruction**  
  Junaid Ahmed Ansari, Ran Ding, Fabio Pizzati, Ivan Laptev  
  _2026-03-17_ · https://arxiv.org/abs/2603.16868v1  
  <details><summary>Abstract</summary>

  Monocular 3D scene reconstruction has recently seen significant progress. Powered by the modern neural architectures and large-scale data, recent methods achieve high performance in depth estimation from a single image. Meanwhile, reconstructing and decomposing common scenes into individual 3D objects remains a hard challenge due to the large variety of objects, frequent occlusions and complex object relations. Notably, beyond shape and pose estimation of individual objects, applications in robotics and animation require physically-plausible scene reconstruction where objects obey physical principles of non-penetration and realistic contacts. In this work we advance object-level scene reconstruction along two directions. First, we introduceMessyKitchens, a new dataset with real-world scenes featuring cluttered environments and providing high-fidelity object-level ground truth in terms of 3D object shapes, poses and accurate object contacts. Second, we build on the recent SAM 3D approach for single-object reconstruction and extend it with Multi-Object Decoder (MOD) for joint object-level scene reconstruction. To validate our contributions, we demonstrate MessyKitchens to significantly improve previous datasets in registration accuracy and inter-object penetration. We also compare our multi-object reconstruction approach on three datasets and demonstrate consistent and significant improvements of MOD over the state of the art. Our new benchmark, code and pre-trained models will become publicly available on our project website: https://messykitchens.github.io/.

  </details>


