# 3D Reconstruction

_Updated: 2026-03-19 07:12 UTC_

Total papers shown: **24**


---

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



- **WildDepth: A Multimodal Dataset for 3D Wildlife Perception and Depth Estimation**  
  Muhammad Aamir, Naoya Muramatsu, Sangyun Shin, Matthew Wijers, Jiaxing Jhong, Xinyu Hou, Amir Patel, Andrew Markham  
  _2026-03-17_ · https://arxiv.org/abs/2603.16816v1  
  <details><summary>Abstract</summary>

  Depth estimation and 3D reconstruction have been extensively studied as core topics in computer vision. Starting from rigid objects with relatively simple geometric shapes, such as vehicles, the research has expanded to address general objects, including challenging deformable objects, such as humans and animals. However, for the animal, in particular, the majority of existing models are trained based on datasets without metric scale, which can help validate image-only models. To address this limitation, we present WildDepth, a multimodal dataset and benchmark suite for depth estimation, behavior detection, and 3D reconstruction from diverse categories of animals ranging from domestic to wild environments with synchronized RGB and LiDAR. Experimental results show that the use of multi-modal data improves depth reliability by up to 10% RMSE, while RGB-LiDAR fusion enhances 3D reconstruction fidelity by 12% in Chamfer distance. By releasing WildDepth and its benchmarks, we aim to foster robust multimodal perception systems that generalize across domains.

  </details>



- **IOSVLM: A 3D Vision-Language Model for Unified Dental Diagnosis from Intraoral Scans**  
  Huimin Xiong, Zijie Meng, Tianxiang Hu, Chenyi Zhou, Yang Feng, Zuozhu Liu  
  _2026-03-17_ · https://arxiv.org/abs/2603.16781v1  
  <details><summary>Abstract</summary>

  3D intraoral scans (IOS) are increasingly adopted in routine dentistry due to abundant geometric evidence, and unified multi-disease diagnosis is desirable for clinical documentation and communication. While recent works introduce dental vision-language models (VLMs) to enable unified diagnosis and report generation on 2D images or multi-view images rendered from IOS, they do not fully leverage native 3D geometry. Such work is necessary and also challenging, due to: (i) heterogeneous scan forms and the complex IOS topology, (ii) multi-disease co-occurrence with class imbalance and fine-grained morphological ambiguity, (iii) limited paired 3D IOS-text data. Thus, we present IOSVLM, an end-to-end 3D VLM that represents scans as point clouds and follows a 3D encoder-projector-LLM design for unified diagnosis and generative visual question-answering (VQA), together with IOSVQA, a large-scale multi-source IOS diagnosis VQA dataset comprising 19,002 cases and 249,055 VQA pairs over 23 oral diseases and heterogeneous scan types. To address the distribution gap between color-free IOS data and color-dependent 3D pre-training, we propose a geometry-to-chromatic proxy that stabilizes fine-grained geometric perception and cross-modal alignment. A two-stage curriculum training strategy further enhances robustness. IOSVLM consistently outperforms strong baselines, achieving gains of at least +9.58% macro accuracy and +1.46% macro F1, indicating the effectiveness of direct 3D geometry modeling for IOS-based diagnosis.

  </details>



- **World Reconstruction From Inconsistent Views**  
  Lukas Höllein, Matthias Nießner  
  _2026-03-17_ · https://arxiv.org/abs/2603.16736v2  
  <details><summary>Abstract</summary>

  Video diffusion models generate high-quality and diverse worlds; however, individual frames often lack 3D consistency across the output sequence, which makes the reconstruction of 3D worlds difficult. To this end, we propose a new method that handles these inconsistencies by non-rigidly aligning the video frames into a globally-consistent coordinate frame that produces sharp and detailed pointcloud reconstructions. First, a geometric foundation model lifts each frame into a pixel-wise 3D pointcloud, which contains unaligned surfaces due to these inconsistencies. We then propose a tailored non-rigid iterative frame-to-model ICP to obtain an initial alignment across all frames, followed by a global optimization that further sharpens the pointcloud. Finally, we leverage this pointcloud as initialization for 3D reconstruction and propose a novel inverse deformation rendering loss to create high quality and explorable 3D environments from inconsistent views. We demonstrate that our 3D scenes achieve higher quality than baselines, effectively turning video models into 3D-consistent world generators.

  </details>



- **GAP-MLLM: Geometry-Aligned Pre-training for Activating 3D Spatial Perception in Multimodal Large Language Models**  
  Jiaxin Zhang, Junjun Jiang, Haijie Li, Youyu Chen, Kui Jiang, Dave Zhenyu Chen  
  _2026-03-17_ · https://arxiv.org/abs/2603.16461v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) demonstrate exceptional semantic reasoning but struggle with 3D spatial perception when restricted to pure RGB inputs. Despite leveraging implicit geometric priors from 3D reconstruction models, image-based methods still exhibit a notable performance gap compared to methods using explicit 3D data. We argue that this gap does not arise from insufficient geometric priors, but from a misalignment in the training paradigm: text-dominated fine-tuning fails to activate geometric representations within MLLMs. Existing approaches typically resort to naive feature concatenation and optimize directly for downstream tasks without geometry-specific supervision, leading to suboptimal structural utilization. To address this limitation, we propose GAP-MLLM, a Geometry-Aligned Pre-training paradigm that explicitly activates structural perception before downstream adaptation. Specifically, we introduce a visual-prompted joint task that compels the MLLMs to predict sparse pointmaps alongside semantic labels, thereby enforcing geometric awareness. Furthermore, we design a multi-level progressive fusion module with a token-level gating mechanism, enabling adaptive integration of geometric priors without suppressing semantic reasoning. Extensive experiments demonstrate that GAP-MLLM significantly enhances geometric feature fusion and consistently enhances performance across 3D visual grounding, 3D dense captioning, and 3D video object detection tasks.

  </details>



- **Fast-HaMeR: Boosting Hand Mesh Reconstruction using Knowledge Distillation**  
  Hunain Ahmed Jillani, Ahmed Tawfik Aboukhadra, Ahmed Elhayek, Jameel Malik, Nadia Robertini, Didier Stricker  
  _2026-03-17_ · https://arxiv.org/abs/2603.16444v1  
  <details><summary>Abstract</summary>

  Fast and accurate 3D hand reconstruction is essential for real-time applications in VR/AR, human-computer interaction, robotics, and healthcare. Most state-of-the-art methods rely on heavy models, limiting their use on resource-constrained devices like headsets, smartphones, and embedded systems. In this paper, we investigate how the use of lightweight neural networks, combined with Knowledge Distillation, can accelerate complex 3D hand reconstruction models by making them faster and lighter, while maintaining comparable reconstruction accuracy. While our approach is suited for various hand reconstruction frameworks, we focus primarily on boosting the HaMeR model, currently the leading method in terms of reconstruction accuracy. We replace its original ViT-H backbone with lighter alternatives, including MobileNet, MobileViT, ConvNeXt, and ResNet, and evaluate three knowledge distillation strategies: output-level, feature-level, and a hybrid of both. Our experiments show that using lightweight backbones that are only 35% the size of the original achieves 1.5x faster inference speed while preserving similar performance quality with only a minimal accuracy difference of 0.4mm. More specifically, we show how output-level distillation notably improves student performance, while feature-level distillation proves more effective for higher-capacity students. Overall, the findings pave the way for efficient real-world applications on low-power devices. The code and models are publicly available under https://github.com/hunainahmedj/Fast-HaMeR.

  </details>



- **$D^3$-RSMDE: 40$\times$ Faster and High-Fidelity Remote Sensing Monocular Depth Estimation**  
  Ruizhi Wang, Weihan Li, Zunlei Feng, Haofei Zhang, Mingli Song, Jiayu Wang, Jie Song, Li Sun  
  _2026-03-17_ · https://arxiv.org/abs/2603.16362v1  
  <details><summary>Abstract</summary>

  Real-time, high-fidelity monocular depth estimation from remote sensing imagery is crucial for numerous applications, yet existing methods face a stark trade-off between accuracy and efficiency. Although using Vision Transformer (ViT) backbones for dense prediction is fast, they often exhibit poor perceptual quality. Conversely, diffusion models offer high fidelity but at a prohibitive computational cost. To overcome these limitations, we propose Depth Detail Diffusion for Remote Sensing Monocular Depth Estimation ($D^3$-RSMDE), an efficient framework designed to achieve an optimal balance between speed and quality. Our framework first leverages a ViT-based module to rapidly generate a high-quality preliminary depth map construction, which serves as a structural prior, effectively replacing the time-consuming initial structure generation stage of diffusion models. Based on this prior, we propose a Progressive Linear Blending Refinement (PLBR) strategy, which uses a lightweight U-Net to refine the details in only a few iterations. The entire refinement step operates efficiently in a compact latent space supported by a Variational Autoencoder (VAE). Extensive experiments demonstrate that $D^3$-RSMDE achieves a notable 11.85% reduction in the Learned Perceptual Image Patch Similarity (LPIPS) perceptual metric over leading models like Marigold, while also achieving over a 40x speedup in inference and maintaining VRAM usage comparable to lightweight ViT models.

  </details>



- **Learning Human-Object Interaction for 3D Human Pose Estimation from LiDAR Point Clouds**  
  Daniel Sungho Jung, Dohee Cho, Kyoung Mu Lee  
  _2026-03-17_ · https://arxiv.org/abs/2603.16343v1  
  <details><summary>Abstract</summary>

  Understanding humans from LiDAR point clouds is one of the most critical tasks in autonomous driving due to its close relationships with pedestrian safety, yet it remains challenging in the presence of diverse human-object interactions and cluttered backgrounds. Nevertheless, existing methods largely overlook the potential of leveraging human-object interactions to build robust 3D human pose estimation frameworks. There are two major challenges that motivate the incorporation of human-object interaction. First, human-object interactions introduce spatial ambiguity between human and object points, which often leads to erroneous 3D human keypoint predictions in interaction regions. Second, there exists severe class imbalance in the number of points between interacting and non-interacting body parts, with the interaction-frequent regions such as hand and foot being sparsely observed in LiDAR data. To address these challenges, we propose a Human-Object Interaction Learning (HOIL) framework for robust 3D human pose estimation from LiDAR point clouds. To mitigate the spatial ambiguity issue, we present human-object interaction-aware contrastive learning (HOICL) that effectively enhances feature discrimination between human and object points, particularly in interaction regions. To alleviate the class imbalance issue, we introduce contact-aware part-guided pooling (CPPool) that adaptively reallocates representational capacity by compressing overrepresented points while preserving informative points from interacting body parts. In addition, we present an optional contact-based temporal refinement that refines erroneous per-frame keypoint estimates using contact cues over time. As a result, our HOIL effectively leverages human-object interaction to resolve spatial ambiguity and class imbalance in interaction regions. Codes will be released.

  </details>



- **Iris: Bringing Real-World Priors into Diffusion Model for Monocular Depth Estimation**  
  Xinhao Cai, Gensheng Pei, Zeren Sun, Yazhou Yao, Fumin Shen, Wenguan Wang  
  _2026-03-17_ · https://arxiv.org/abs/2603.16340v1  
  <details><summary>Abstract</summary>

  In this paper, we propose \textbf{Iris}, a deterministic framework for Monocular Depth Estimation (MDE) that integrates real-world priors into the diffusion model. Conventional feed-forward methods rely on massive training data, yet still miss details. Previous diffusion-based methods leverage rich generative priors yet struggle with synthetic-to-real domain transfer. Iris, in contrast, preserves fine details, generalizes strongly from synthetic to real scenes, and remains efficient with limited training data. To this end, we introduce a two-stage Priors-to-Geometry Deterministic (PGD) schedule: the prior stage uses Spectral-Gated Distillation (SGD) to transfer low-frequency real priors while leaving high-frequency details unconstrained, and the geometry stage applies Spectral-Gated Consistency (SGC) to enforce high-frequency fidelity while refining with synthetic ground truth. The two stages share weights and are executed with a high-to-low timestep schedule. Extensive experimental results confirm that Iris achieves significant improvements in MDE performance with strong in-the-wild generalization.

  </details>



- **PureCLIP-Depth: Prompt-Free and Decoder-Free Monocular Depth Estimation within CLIP Embedding Space**  
  Ryutaro Miya, Kazuyoshi Fushinobu, Tatsuya Kawaguchi  
  _2026-03-17_ · https://arxiv.org/abs/2603.16238v1  
  <details><summary>Abstract</summary>

  We propose PureCLIP-Depth, a completely prompt-free, decoder-free Monocular Depth Estimation (MDE) model that operates entirely within the Contrastive Language-Image Pre-training (CLIP) embedding space. Unlike recent models that rely heavily on geometric features, we explore a novel approach to MDE driven by conceptual information, performing computations directly within the conceptual CLIP space. The core of our method lies in learning a direct mapping from the RGB domain to the depth domain strictly inside this embedding space. Our approach achieves state-of-the-art performance among CLIP embedding-based models on both indoor and outdoor datasets. The code used in this research is available at: https://github.com/ryutaroLF/PureCLIP-Depth

  </details>



- **Leveling3D: Leveling Up 3D Reconstruction with Feed-Forward 3D Gaussian Splatting and Geometry-Aware Generation**  
  Yiming Huang, Baixiang Huang, Beilei Cui, Chi Kit Ng, Long Bai, Hongliang Ren  
  _2026-03-17_ · https://arxiv.org/abs/2603.16211v1  
  <details><summary>Abstract</summary>

  Feed-forward 3D reconstruction has revolutionized 3D vision, providing a powerful baseline for downstream tasks such as novel-view synthesis with 3D Gaussian Splatting. Previous works explore fixing the corrupted rendering results with a diffusion model. However, they lack geometric concern and fail at filling the missing area on the extrapolated view. In this work, we introduce Leveling3D, a novel pipeline that integrates feed-forward 3D reconstruction with geometrical-consistent generation to enable holistic simultaneous reconstruction and generation. We propose a geometry-aware leveling adapter, a lightweight technique that aligns internal knowledge in the diffusion model with the geometry prior from the feed-forward model. The leveling adapter enables generation on the artifact area of the extrapolated novel views caused by underconstrained regions of the 3D representation. Specifically, to learn a more diverse distributed generation, we introduce the palette filtering strategy for training, and a test-time masking refinement to prevent messy boundaries along the fixing regions. More importantly, the enhanced extrapolated novel views from Leveling3D could be used as the inputs for feed-forward 3DGS, leveling up the 3D reconstruction. We achieve SOTA performance on public datasets, including tasks such as novel-view synthesis and depth estimation.

  </details>



- **GATS: Gaussian Aware Temporal Scaling Transformer for Invariant 4D Spatio-Temporal Point Cloud Representation**  
  Jiayi Tian, Jiaze Wang  
  _2026-03-17_ · https://arxiv.org/abs/2603.16154v1  
  <details><summary>Abstract</summary>

  Understanding 4D point cloud videos is essential for enabling intelligent agents to perceive dynamic environments. However, temporal scale bias across varying frame rates and distributional uncertainty in irregular point clouds make it highly challenging to design a unified and robust 4D backbone. Existing CNN or Transformer based methods are constrained either by limited receptive fields or by quadratic computational complexity, while neglecting these implicit distortions. To address this problem, we propose a novel dual invariant framework, termed \textbf{Gaussian Aware Temporal Scaling (GATS)}, which explicitly resolves both distributional inconsistencies and temporal. The proposed \emph{Uncertainty Guided Gaussian Convolution (UGGC)} incorporates local Gaussian statistics and uncertainty aware gating into point convolution, thereby achieving robust neighborhood aggregation under density variation, noise, and occlusion. In parallel, the \emph{Temporal Scaling Attention (TSA)} introduces a learnable scaling factor to normalize temporal distances, ensuring frame partition invariance and consistent velocity estimation across different frame rates. These two modules are complementary: temporal scaling normalizes time intervals prior to Gaussian estimation, while Gaussian modeling enhances robustness to irregular distributions. Our experiments on mainstream benchmarks MSR-Action3D (\textbf{+6.62\%} accuracy), NTU RGBD (\textbf{+1.4\%} accuracy), and Synthia4D (\textbf{+1.8\%} mIoU) demonstrate significant performance gains, offering a more efficient and principled paradigm for invariant 4D point cloud video understanding with superior accuracy, robustness, and scalability compared to Transformer based counterparts.

  </details>



- **DualPrim: Compact 3D Reconstruction with Positive and Negative Primitives**  
  Xiaoxu Meng, Zhongmin Chen, Bo Yang, Weikai Chen, Weixiao Liu, Lin Gao  
  _2026-03-17_ · https://arxiv.org/abs/2603.16133v1  
  <details><summary>Abstract</summary>

  Neural reconstructions often trade structure for fidelity, yielding dense and unstructured meshes with irregular topology and weak part boundaries that hinder editing, animation, and downstream asset reuse. We present DualPrim, a compact and structured 3D reconstruction framework. Unlike additive-only implicit or primitive methods, DualPrim represents shapes with positive and negative superquadrics: the former builds the bases while the latter carves local volumes through a differentiable operator, enabling topology-aware modeling of holes and concavities. This additive-subtractive design increases the representational power without sacrificing compactness or differentiability. We embed DualPrim in a volumetric differentiable renderer, enabling end-to-end learning from multi-view images and seamless mesh export via closed-form boolean difference. Empirically, DualPrim delivers state-of-the-art accuracy and produces compact, structured, and interpretable outputs that better satisfy downstream needs than additive-only alternatives.

  </details>


