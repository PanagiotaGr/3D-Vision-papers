# 3D Reconstruction

_Updated: 2026-03-18 07:16 UTC_

Total papers shown: **25**


---

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
  _2026-03-17_ · https://arxiv.org/abs/2603.16736v1  
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



- **HSImul3R: Physics-in-the-Loop Reconstruction of Simulation-Ready Human-Scene Interactions**  
  Yukang Cao, Haozhe Xie, Fangzhou Hong, Long Zhuo, Zhaoxi Chen, Liang Pan, Ziwei Liu  
  _2026-03-16_ · https://arxiv.org/abs/2603.15612v1  
  <details><summary>Abstract</summary>

  We present HSImul3R, a unified framework for simulation-ready 3D reconstruction of human-scene interactions (HSI) from casual captures, including sparse-view images and monocular videos. Existing methods suffer from a perception-simulation gap: visually plausible reconstructions often violate physical constraints, leading to instability in physics engines and failure in embodied AI applications. To bridge this gap, we introduce a physically-grounded bi-directional optimization pipeline that treats the physics simulator as an active supervisor to jointly refine human dynamics and scene geometry. In the forward direction, we employ Scene-targeted Reinforcement Learning to optimize human motion under dual supervision of motion fidelity and contact stability. In the reverse direction, we propose Direct Simulation Reward Optimization, which leverages simulation feedback on gravitational stability and interaction success to refine scene geometry. We further present HSIBench, a new benchmark with diverse objects and interaction scenarios. Extensive experiments demonstrate that HSImul3R produces the first stable, simulation-ready HSI reconstructions and can be directly deployed to real-world humanoid robots.

  </details>



- **Real-Time Human Frontal View Synthesis from a Single Image**  
  Fangyu Lin, Yingdong Hu, Lunjie Zhu, Zhening Liu, Yushi Huang, Zehong Lin, Jun Zhang  
  _2026-03-16_ · https://arxiv.org/abs/2603.15433v1  
  <details><summary>Abstract</summary>

  Photorealistic human novel view synthesis from a single image is crucial for democratizing immersive 3D telepresence, eliminating the need for complex multi-camera setups. However, current rendering-centric methods prioritize visual fidelity over explicit geometric understanding and struggle with intricate regions like faces and hands, leading to temporal instability. Meanwhile, human-centric frameworks suffer from memory bottlenecks since they typically rely on an auxiliary model to provide informative structural priors for geometric modeling, which limits real-time performance. To address these challenges, we propose PrismMirror, a geometry-guided framework for instant frontal view synthesis from a single image. By avoiding external geometric modeling and focusing on frontal view synthesis, our model optimizes visual integrity for telepresence. Specifically, PrismMirror introduces a novel cascade learning strategy that enables coarse-to-fine geometric feature learning. It first directly learns coarse geometric features, such as SMPL-X meshes and point clouds, and then refines textures through rendering supervision. To achieve real-time efficiency, we distill this unified framework into a lightweight linear attention model. Notably, PrismMirror is the first monocular human frontal view synthesis model that achieves real-time inference at 24 FPS, significantly outperforming previous methods in both visual authenticity and structural accuracy.

  </details>



- **Pointing-Based Object Recognition**  
  Lukáš Hajdúch, Viktor Kocur  
  _2026-03-16_ · https://arxiv.org/abs/2603.15403v1  
  <details><summary>Abstract</summary>

  This paper presents a comprehensive pipeline for recognizing objects targeted by human pointing gestures using RGB images. As human-robot interaction moves toward more intuitive interfaces, the ability to identify targets of non-verbal communication becomes crucial. Our proposed system integrates several existing state-of-the-art methods, including object detection, body pose estimation, monocular depth estimation, and vision-language models. We evaluate the impact of 3D spatial information reconstructed from a single image and the utility of image captioning models in correcting classification errors. Experimental results on a custom dataset show that incorporating depth information significantly improves target identification, especially in complex scenes with overlapping objects. The modularity of the approach allows for deployment in environments where specialized depth sensors are unavailable.

  </details>



- **Spectral Rectification for Parameter-Efficient Adaptation of Foundation Models in Colonoscopy Depth Estimation**  
  Xiaoxian Zhang, Minghai Shi, Lei Li  
  _2026-03-16_ · https://arxiv.org/abs/2603.15374v1  
  <details><summary>Abstract</summary>

  Accurate monocular depth estimation is critical in colonoscopy for lesion localization and navigation. Foundation models trained on natural images fail to generalize directly to colonoscopy. We identify the core issue not as a semantic gap, but as a statistical shift in the frequency domain: colonoscopy images lack the strong high-frequency edge and texture gradients that these models rely on for geometric reasoning. To address this, we propose SpecDepth, a parameter-efficient adaptation framework that preserves the robust geometric representations of the pre-trained models while adapting to the colonoscopy domain. Its key innovation is an adaptive spectral rectification module, which uses a learnable wavelet decomposition to explicitly model and amplify the attenuated high-frequency components in feature maps. Different from conventional fine-tuning that risks distorting high-level semantic features, this targeted, low-level adjustment realigns the input signal with the original inductive bias of the foundational model. On the public C3VD and SimCol3D datasets, SpecDepth achieved state-of-the-art performance with an absolute relative error of 0.022 and 0.027, respectively. Our work demonstrates that directly addressing spectral mismatches is a highly effective strategy for adapting vision foundation models to specialized medical imaging tasks. The code will be released publicly after the manuscript is accepted for publication.

  </details>



- **MeMix: Writing Less, Remembering More for Streaming 3D Reconstruction**  
  Jiacheng Dong, Huan Li, Sicheng Zhou, Wenhao Hu, Weili Xu, Yan Wang  
  _2026-03-16_ · https://arxiv.org/abs/2603.15330v1  
  <details><summary>Abstract</summary>

  Reconstruction is a fundamental task in 3D vision and a fundamental capability for spatial intelligence. Particularly, streaming 3D reconstruction is central to real-time spatial perception, yet existing recurrent online models often suffer from progressive degradation on long sequences due to state drift and forgetting, motivating inference-time remedies. We present MeMix, a training-free, plug-and-play module that improves streaming reconstruction by recasting the recurrent state into a Memory Mixture. MeMix partitions the state into multiple independent memory patches and updates only the least-aligned memory patches while exactly preserving others. This selective update mitigates catastrophic forgetting while retaining $O(1)$ inference memory, and requires no fine-tuning or additional learnable parameters, making it directly applicable to existing recurrent reconstruction models. Across standard benchmarks (ScanNet, 7-Scenes, KITTI, etc.), under identical backbones and inference settings, MeMix reduces reconstruction completeness error by 15.3% on average (up to 40.0%) across 300--500 frame streams on 7-Scenes. The code is available at https://dongjiacheng06.github.io/MeMix/

  </details>



- **Reference-Free Omnidirectional Stereo Matching via Multi-View Consistency Maximization**  
  Lehuai Xu, Weiming Zhang, Yang Li, Sidan Du, Lin Wang  
  _2026-03-16_ · https://arxiv.org/abs/2603.15019v1  
  <details><summary>Abstract</summary>

  Reliable omnidirectional depth estimation from multi-fisheye stereo matching is pivotal to many applications, such as embodied robotics. Existing approaches either rely on spherical sweeping with heuristic fusion strategies to build the cost columns or perform reference-centric stereo matching based on rectified views. However, these methods fail to explicitly exploit geometric relationships between multiple views, rendering them less capable of capturing the global dependencies, visibility, or scale changes. In this paper, we shift to a new perspective and propose a novel reference-free framework, dubbed FreeOmniMVS, via multi-view consistency maximization. The highlight of FreeOmniMVS is that it can aggregate pair-wise correlations into a robust, visibility-aware, and global consensus. As such, it is tolerant to occlusions, partial overlaps, and varying baselines. Specifically, to achieve global coherence, we introduce a novel View-pair Correlation Transformer (VCT) that explicitly models pairwise correlation volumes across all camera view pairs, allowing us to drop unreliable pairs caused by occlusion or out-of-focus observations. To realize scalable and visibility-aware consensus, we propose a lightweight attention mechanism that adaptively fuses the correlation vectors, eliminating the need for a designated reference view and allowing all cameras to contribute equally to the stereo matching process. Extensive experiments on diverse benchmark datasets demonstrate the superiority of our method for globally consistent, visibility-aware, and scale-aware omnidirectional depth estimation.

  </details>



- **Thermal Image Refinement with Depth Estimation using Recurrent Networks for Monocular ORB-SLAM3**  
  Hürkan Şahin, Huy Xuan Pham, Van Huyen Dang, Alper Yegenoglu, Erdal Kayacan  
  _2026-03-16_ · https://arxiv.org/abs/2603.14998v1  
  <details><summary>Abstract</summary>

  Autonomous navigation in GPS-denied and visually degraded environments remains challenging for unmanned aerial vehicles (UAVs). To this end, we investigate the use of a monocular thermal camera as a standalone sensor on a UAV platform for real-time depth estimation and simultaneous localization and mapping (SLAM). To extract depth information from thermal images, we propose a novel pipeline employing a lightweight supervised network with recurrent blocks (RBs) integrated to capture temporal dependencies, enabling more robust predictions. The network combines lightweight convolutional backbones with a thermal refinement network (T-RefNet) to refine raw thermal inputs and enhance feature visibility. The refined thermal images and predicted depth maps are integrated into ORB-SLAM3, enabling thermal-only localization. Unlike previous methods, the network is trained on a custom non-radiometric dataset, obviating the need for high-cost radiometric thermal cameras. Experimental results on datasets and UAV flights demonstrate competitive depth accuracy and robust SLAM performance under low-light conditions. On the radiometric VIVID++ (indoor-dark) dataset, our method achieves an absolute relative error of approximately 0.06, compared to baselines exceeding 0.11. In our non-radiometric indoor set, baseline errors remain above 0.24, whereas our approach remains below 0.10. Thermal-only ORB-SLAM3 maintains a mean trajectory error under 0.4 m.

  </details>



- **GT-PCQA: Geometry-Texture Decoupled Point Cloud Quality Assessment with MLLM**  
  Guohua Zhang, Jian Jin, Meiqin Liu, Chao Yao, Weisi Lin, Yao Zhao  
  _2026-03-16_ · https://arxiv.org/abs/2603.14951v1  
  <details><summary>Abstract</summary>

  With the rapid advancement of Multi-modal Large Language Models (MLLMs), MLLM-based Image Quality Assessment (IQA) methods have shown promising generalization. However, directly extending these MLLM-based IQA methods to PCQA remains challenging. On the one hand, existing PCQA datasets are limited in scale, which hinders stable and effective instruction tuning of MLLMs. On the other hand, due to large-scale image-text pretraining, MLLMs tend to rely on texture-dominant reasoning and are insufficiently sensitive to geometric structural degradations that are critical for PCQA. To address these gaps, we propose a novel MLLM-based no-reference PCQA framework, termed GT-PCQA, which is built upon two key strategies. First, to enable stable and effective instruction tuning under scarce PCQA supervision, a 2D-3D joint training strategy is proposed. This strategy formulates PCQA as a relative quality comparison problem to unify large-scale IQA datasets with limited PCQA datasets. It incorporates a parameter-efficient Low-Rank Adaptation (LoRA) scheme to support instruction tuning. Second, a geometry-texture decoupling strategy is presented, which integrates a dual-prompt mechanism with an alternating optimization scheme to mitigate the inherent texture-dominant bias of pre-trained MLLMs, while enhancing sensitivity to geometric structural degradations. Extensive experiments demonstrate that GT-PCQA achieves competitive performance and exhibits strong generalization.

  </details>



- **ILV: Iterative Latent Volumes for Fast and Accurate Sparse-View CT Reconstruction**  
  Seungryong Lee, Woojeong Baek, Joosang Lee, Eunbyung Park  
  _2026-03-16_ · https://arxiv.org/abs/2603.14915v1  
  <details><summary>Abstract</summary>

  A long-term goal in CT imaging is to achieve fast and accurate 3D reconstruction from sparse-view projections, thereby reducing radiation exposure, lowering system cost, and enabling timely imaging in clinical workflows. Recent feed-forward approaches have shown strong potential toward this overarching goal, yet their results still suffer from artifacts and loss of fine details. In this work, we introduce Iterative Latent Volumes (ILV), a feed-forward framework that integrates data-driven priors with classical iterative reconstruction principles to overcome key limitations of prior feed-forward models in sparse-view CBCT reconstruction. At its core, ILV constructs an explicit 3D latent volume that is repeatedly updated by conditioning on multi-view X-ray features and the learned anatomical prior, enabling the recovery of fine structural details beyond the reach of prior feed-forward models. In addition, we develop and incorporate several key architectural components, including an X-ray feature volume, group cross-attention, efficient self-attention, and view-wise feature aggregation, that efficiently realize its core latent volume refinement concept. Extensive experiments on a large-scale dataset of approximately 14,000 CT volumes demonstrate that ILV significantly outperforms existing feed-forward and optimization-based methods in both reconstruction quality and speed. These results show that ILV enables fast and accurate sparse-view CBCT reconstruction suitable for clinical use. The project page is available at: https://sngryonglee.github.io/ILV/.

  </details>



- **RadarXFormer: Robust Object Detection via Cross-Dimension Fusion of 4D Radar Spectra and Images for Autonomous Driving**  
  Yue Sun, Yeqiang Qian, Zhe Wang, Tianhui Li, Chunxiang Wang, Ming Yang  
  _2026-03-16_ · https://arxiv.org/abs/2603.14822v1  
  <details><summary>Abstract</summary>

  Reliable perception is essential for autonomous driving systems to operate safely under diverse real-world traffic conditions. However, camera- and LiDAR-based perception systems suffer from performance degradation under adverse weather and lighting conditions, limiting their robustness and large-scale deployment in intelligent transportation systems. Radar-vision fusion provides a promising alternative by combining the environmental robustness and cost efficiency of millimeter-wave (mmWave) radar with the rich semantic information captured by cameras. Nevertheless, conventional 3D radar measurements lack height resolution and remain highly sparse, while emerging 4D mmWave radar introduces elevation information but also brings challenges such as signal noise and large data volume. To address these issues, this paper proposes RadarXFormer, a 3D object detection framework that enables efficient cross-modal fusion between 4D radar spectra and RGB images. Instead of relying on sparse radar point clouds, RadarXFormer directly leverages raw radar spectra and constructs an efficient 3D representation that reduces data volume while preserving complete 3D spatial information. The "X" highlights the proposed cross-dimension (3D-2D) fusion mechanism, in which multi-scale 3D spherical radar feature cubes are fused with complementary 2D image feature maps. Experiments on the K-Radar dataset demonstrate improved detection accuracy and robustness under challenging conditions while maintaining real-time inference capability.

  </details>



- **SSR: A Training-Free Approach for Streaming 3D Reconstruction**  
  Hui Deng, Yuxin Mao, Yuxin He, Yuchao Dai  
  _2026-03-16_ · https://arxiv.org/abs/2603.14765v1  
  <details><summary>Abstract</summary>

  Streaming 3D reconstruction demands long-horizon state updates under strict latency constraints, yet stateful recurrent models often suffer from geometric drift as errors accumulate over time. We revisit this problem from a Grassmannian manifold perspective: the latent persistent state can be viewed as a subspace representation, i.e., a point evolving on a Grassmannian manifold, where temporal coherence implies the state trajectory should remain on (or near) this manifold.Based on this view, we propose Self-expressive Sequence Regularization (SSR), a plug-and-play, training-free operator that enforces Grassmannian sequence regularity during inference.Given a window of historical states, SSR computes an analytical affinity matrix via the self-expressive property and uses it to regularize the current update, effectively pulling noisy predictions back toward the manifold-consistent trajectory with minimal overhead. Experiments on long-sequence benchmarks demonstrate that SSR consistently reduces drift and improves reconstruction quality across multiple streaming 3D reconstruction tasks.

  </details>



- **LiDAR-EVS: Enhance Extrapolated View Synthesis for 3D Gaussian Splatting with Pseudo-LiDAR Supervision**  
  Yiming Huang, Xin Kang, Sipeng Zhang, Hongliang Ren, Weihua Zhang, Junjie Lai  
  _2026-03-16_ · https://arxiv.org/abs/2603.14763v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has emerged as a powerful technique for real-time LiDAR and camera synthesis in autonomous driving simulation. However, simulating LiDAR with 3DGS remains challenging for extrapolated views beyond the training trajectory, as existing methods are typically trained on single-traversal sensor scans, suffer from severe overfitting and poor generalization to novel ego-vehicle paths. To enable reliable simulation of LiDAR along unseen driving trajectories without external multi-pass data, we present LiDAR-EVS, a lightweight framework for robust extrapolated-view LiDAR simulation in autonomous driving. Designed to be plug-and-play, LiDAR-EVS readily extends to diverse LiDAR sensors and neural rendering baselines with minimal modification. Our framework comprises two key components: (1) pseudo extrapolated-view point cloud supervision with multi-frame LiDAR fusion, view transformation, occlusion curling, and intensity adjustment; (2) spatially-constrained dropout regularization that promotes robustness to diverse trajectory variations encountered in real-world driving. Extensive experiments demonstrate that LiDAR-EVS achieves SOTA performance on extrapolated-view LiDAR synthesis across three datasets, making it a promising tool for data-driven simulation, closed-loop evaluation, and synthetic data generation in autonomous driving systems.

  </details>


