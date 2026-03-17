# 3D Reconstruction

_Updated: 2026-03-17 07:17 UTC_

Total papers shown: **24**


---

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



- **Fractal Autoregressive Depth Estimation with Continuous Token Diffusion**  
  Jinchang Zhang, Xinrou Kang, Guoyu Lu  
  _2026-03-16_ · https://arxiv.org/abs/2603.14702v1  
  <details><summary>Abstract</summary>

  Monocular depth estimation can benefit from autoregressive (AR) generation, but direct AR modeling is hindered by the modality gap between RGB and depth, inefficient pixel-wise generation, and instability in continuous depth prediction. We propose a Fractal Visual Autoregressive Diffusion framework that reformulates depth estimation as a coarse-to-fine, next-scale autoregressive generation process. A VCFR module fuses multi-scale image features with current depth predictions to improve cross-modal conditioning, while a conditional denoising diffusion loss models depth distributions directly in continuous space and mitigates errors caused by discrete quantization. To improve computational efficiency, we organize the scale-wise generators into a fractal recursive architecture, reusing a base visual AR unit in a self-similar hierarchy. We further introduce an uncertainty-aware robust consensus aggregation scheme for multi-sample inference to improve fusion stability and provide a practical pixel-wise reliability estimate. Experiments on standard benchmarks demonstrate strong performance and validate the effectiveness of the proposed design.

  </details>



- **E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction**  
  Yunsoo Kim, Changki Sung, Dasol Hong, Hyun Myung  
  _2026-03-16_ · https://arxiv.org/abs/2603.14684v1  
  <details><summary>Abstract</summary>

  The emergence of neural radiance fields (NeRF) and 3D Gaussian splatting (3DGS) has advanced novel view synthesis (NVS). These methods, however, require high-quality RGB inputs and accurate corresponding poses, limiting robustness under real-world conditions such as fast camera motion or adverse lighting. Event cameras, which capture brightness changes at each pixel with high temporal resolution and wide dynamic range, enable precise sensing of dynamic scenes and offer a promising solution. However, existing event-based NVS methods either assume known poses or rely on depth estimation models that are bounded by their initial observations, failing to generalize as the camera traverses previously unseen regions. We present E2EGS, a pose-free framework operating solely on event streams. Our key insight is that edge information provides rich structural cues essential for accurate trajectory estimation and high-quality NVS. To extract edges from noisy event streams, we exploit the distinct spatio-temporal characteristics of edges and non-edge regions. The event camera's movement induces consistent events along edges, while non-edge regions produce sparse noise. We leverage this through a patch-based temporal coherence analysis that measures local variance to extract edges while robustly suppressing noise. The extracted edges guide structure-aware Gaussian initialization and enable edge-weighted losses throughout initialization, tracking, and bundle adjustment. Extensive experiments on both synthetic and real datasets demonstrate that E2EGS achieves superior reconstruction quality and trajectory accuracy, establishing a fully pose-free paradigm for event-based 3D reconstruction.

  </details>



- **Expanding mmWave Datasets for Human Pose Estimation with Unlabeled Data and LiDAR Datasets**  
  Zhuoxuan Peng, Boan Zhu, Xingjian Zhang, Wenying Li, S. -H. Gary Chan  
  _2026-03-15_ · https://arxiv.org/abs/2603.14507v1  
  <details><summary>Abstract</summary>

  Current mmWave datasets for human pose estimation (HPE) are scarce and lack diversity in both point cloud (PC) attributes and human poses, severely hampering the generalization ability of their trained models. On the other hand, unlabeled mmWave HPE data and diverse LiDAR HPE datasets are readily available. We propose EMDUL, a novel approach to expand the volume and diversity of an existing mmWave dataset using unlabeled mmWave data and a LiDAR dataset. EMDUL trains a pseudo-label estimator to annotate the unlabeled mmWave data and is able to convert, or translate, a given annotated LiDAR PC to its mmWave counterpart. Expanded with both LiDAR-converted and pseudo-labeled mmWave PCs, our mmWave dataset significantly boosts the performance and generalization ability of all our HPE models, with substantial 15.1% and 18.9% error reductions for in-domain and out-of-domain settings, respectively.

  </details>



- **V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning**  
  Lorenzo Mur-Labadia, Matthew Muckley, Amir Bar, Mido Assran, Koustuv Sinha, Mike Rabbat, Yann LeCun, Nicolas Ballas, Adrien Bardes  
  _2026-03-15_ · https://arxiv.org/abs/2603.14482v1  
  <details><summary>Abstract</summary>

  We present V-JEPA 2.1, a family of self-supervised models that learn dense, high-quality visual representations for both images and videos while retaining strong global scene understanding. The approach combines four key components. First, a dense predictive loss uses a masking-based objective in which both visible and masked tokens contribute to the training signal, encouraging explicit spatial and temporal grounding. Second, deep self-supervision applies the self-supervised objective hierarchically across multiple intermediate encoder layers to improve representation quality. Third, multi-modal tokenizers enable unified training across images and videos. Finally, the model benefits from effective scaling in both model capacity and training data. Together, these design choices produce representations that are spatially structured, semantically coherent, and temporally consistent. Empirically, V-JEPA 2.1 achieves state-of-the-art performance on several challenging benchmarks, including 7.71 mAP on Ego4D for short-term object-interaction anticipation and 40.8 Recall@5 on EPIC-KITCHENS for high-level action anticipation, as well as a 20-point improvement in real-robot grasping success rate over V-JEPA-2 AC. The model also demonstrates strong performance in robotic navigation (5.687 ATE on TartanDrive), depth estimation (0.307 RMSE on NYUv2 with a linear probe), and global recognition (77.7 on Something-Something-V2). These results show that V-JEPA 2.1 significantly advances the state of the art in dense visual understanding and world modeling.

  </details>



- **OCRA: Object-Centric Learning with 3D and Tactile Priors for Human-to-Robot Action Transfer**  
  Kuanning Wang, Ke Fan, Yuqian Fu, Siyu Lin, Hu Luo, Daniel Seita, Yanwei Fu, Yu-Gang Jiang, Xiangyang Xue  
  _2026-03-15_ · https://arxiv.org/abs/2603.14401v1  
  <details><summary>Abstract</summary>

  We present OCRA, an Object-Centric framework for video-based human-to-Robot Action transfer that learns directly from human demonstration videos to enable robust manipulation. Object-centric learning emphasizes task-relevant objects and their interactions while filtering out irrelevant background, providing a natural and scalable way to teach robots. OCRA leverages multi-view RGB videos, the state-of-the-art 3D foundation model VGGT, and advanced detection and segmentation models to reconstruct object-centric 3D point clouds, capturing rich interactions between objects. To handle properties not easily perceived by vision alone, we incorporate tactile priors via a large-scale dataset of over one million tactile images. These 3D and tactile priors are fused through a multimodal module (ResFiLM) and fed into a Diffusion Policy to generate robust manipulation actions. Extensive experiments on both vision-only and visuo-tactile tasks show that OCRA significantly outperforms existing baselines and ablations, demonstrating its effectiveness for learning from human demonstration videos.

  </details>



- **Direct Object-Level Reconstruction via Probabilistic Gaussian Splatting**  
  Shuai Guo, Ao Guo, Junchao Zhao, Qi Chen, Yuxiang Qi, Zechuan Li, Dong Chen, Tianjia Shao, Mingliang Xu  
  _2026-03-15_ · https://arxiv.org/abs/2603.14316v1  
  <details><summary>Abstract</summary>

  Object-level 3D reconstruction play important roles across domains such as cultural heritage digitization, industrial manufacturing, and virtual reality. However, existing Gaussian Splatting-based approaches generally rely on full-scene reconstruction, in which substantial redundant background information is introduced, leading to increased computational and storage overhead. To address this limitation, we propose an efficient single-object 3D reconstruction method based on 2D Gaussian Splatting. By directly integrating foreground-background probability cues into Gaussian primitives and dynamically pruning low-probability Gaussians during training, the proposed method fundamentally focuses on an object of interest and improves the memory and computational efficiency. Our pipeline leverages probability masks generated by YOLO and SAM to supervise probabilistic Gaussian attributes, replacing binary masks with continuous probability values to mitigate boundary ambiguity. Additionally, we propose a dual-stage filtering strategy for training's startup to suppress background Gaussians. And, during training, rendered probability masks are conversely employed to refine supervision and enhance boundary consistency across views. Experiments conducted on the MIP-360, T&T, and NVOS datasets demonstrate that our method exhibits strong self-correction capability in the presence of mask errors and achieves reconstruction quality comparable to standard 3DGS approaches, while requiring only approximately 1/10 of their Gaussian amount. These results validate the efficiency and robustness of our method for single-object reconstruction and highlight its potential for applications requiring both high fidelity and computational efficiency.

  </details>



- **In-Field 3D Wheat Head Instance Segmentation From TLS Point Clouds Using Deep Learning Without Manual Labels**  
  Tomislav Medic, Liangliang Nan  
  _2026-03-15_ · https://arxiv.org/abs/2603.14309v1  
  <details><summary>Abstract</summary>

  3D instance segmentation for laser scanning (LiDAR) point clouds remains a challenge in many remote sensing-related domains. Successful solutions typically rely on supervised deep learning and manual annotations, and consequently focus on objects that can be well delineated through visual inspection and manual labeling of point clouds. However, for tasks with more complex and cluttered scenes, such as in-field plant phenotyping in agriculture, such approaches are often infeasible. In this study, we tackle the task of in-field wheat head instance segmentation directly from terrestrial laser scanning (TLS) point clouds. To address the problem and circumvent the need for manual annotations, we propose a novel two-stage pipeline. To obtain the initial 3D instance proposals, the first stage uses 3D-to-2D multi-view projections, the Grounded SAM pipeline for zero-shot 2D object-centric segmentation, and multi-view label fusion. The second stage uses these initial proposals as noisy pseudo-labels to train a supervised 3D panoptic-style segmentation neural network. Our results demonstrate the feasibility of the proposed approach and show performance improvementsrelative to Wheat3DGS, a recent alternative solution for in-field wheat head instance segmentation without manual 3D annotations based on multi-view RGB images and 3D Gaussian Splatting, showcasing TLS as a competitive sensing alternative. Moreover, the results show that both stages of the proposed pipeline can deliver usable 3D instance segmentation without manual annotations, indicating promising, low-effort transferability to other comparable TLS-based point cloud segmentation tasks.

  </details>



- **RegFormer++: An Efficient Large-Scale 3D LiDAR Point Registration Network with Projection-Aware 2D Transformer**  
  Jiuming Liu, Guangming Wang, Zhe Liu, Chaokang Jiang, Haoang Li, Mengmeng Liu, Tianchen Deng, Marc Pollefeys, Michael Ying Yang, Hesheng Wang  
  _2026-03-15_ · https://arxiv.org/abs/2603.14290v1  
  <details><summary>Abstract</summary>

  Although point cloud registration has achieved remarkable advances in object-level and indoor scenes, large-scale LiDAR registration methods has been rarely explored before. Challenges mainly arise from the huge point scale, complex point distribution, and numerous outliers within outdoor LiDAR scans. In addition, most existing registration works generally adopt a two-stage paradigm: They first find correspondences by extracting discriminative local descriptors and then leverage robust estimators (e.g. RANSAC) to filter outliers, which are highly dependent on well-designed descriptors and post-processing choices. To address these problems, we propose a novel end-to-end differential transformer network, termed RegFormer++, for large-scale point cloud alignment without requiring any further post-processing. Specifically, a hierarchical projection-aware 2D transformer with linear complexity is proposed to project raw LiDAR points onto a cylindrical surface and extract global point features, which can improve resilience to outliers due to long-range dependencies. Because we fill original 3D coordinates into 2D projected positions, our designed transformer can benefit from both high efficiency in 2D processing and accuracy from 3D geometric information. Furthermore, to effectively reduce wrong point matching, a Bijective Association Transformer (BAT) is designed, combining both cross attention and all-to-all point gathering. To improve training stability and robustness, a feature-transformed optimal transport module is also designed for regressing the final pose transformation. Extensive experiments on KITTI, NuScenes, and Argoverse datasets demonstrate that our model achieves state-of-the-art performance in terms of both accuracy and efficiency.

  </details>



- **Walking Further: Semantic-aware Multimodal Gait Recognition Under Long-Range Conditions**  
  Zhiyang Lu, Wen Jiang, Tianren Wu, Zhichao Wang, Changwang Zhang, Siqi Shen, Ming Cheng  
  _2026-03-15_ · https://arxiv.org/abs/2603.14189v1  
  <details><summary>Abstract</summary>

  Gait recognition is an emerging biometric technology that enables non-intrusive and hard-to-spoof human identification. However, most existing methods are confined to short-range, unimodal settings and fail to generalize to long-range and cross-distance scenarios under real-world conditions. To address this gap, we present \textbf{LRGait}, the first LiDAR-Camera multimodal benchmark designed for robust long-range gait recognition across diverse outdoor distances and environments. We further propose \textbf{EMGaitNet}, an end-to-end framework tailored for long-range multimodal gait recognition. To bridge the modality gap between RGB images and point clouds, we introduce a semantic-guided fusion pipeline. A CLIP-based Semantic Mining (SeMi) module first extracts human body-part-aware semantic cues, which are then employed to align 2D and 3D features via a Semantic-Guided Alignment (SGA) module within a unified embedding space. A Symmetric Cross-Attention Fusion (SCAF) module hierarchically integrates visual contours and 3D geometric features, and a Spatio-Temporal (ST) module captures global gait dynamics. Extensive experiments on various gait datasets validate the effectiveness of our method.

  </details>



- **CIPHER: Culvert Inspection through Pairwise Frame Selection and High-Efficiency Reconstruction**  
  Seoyoung Lee, Zhangyang Wang  
  _2026-03-14_ · https://arxiv.org/abs/2603.14150v1  
  <details><summary>Abstract</summary>

  Automated culvert inspection systems can help increase the safety and efficiency of flood management operations. As a key step to this system, we present an efficient RGB-based 3D reconstruction pipeline for culvert-like structures in visually repetitive environments. Our approach first selects informative frame pairs to maximize viewpoint diversity while ensuring valid correspondence matching using a plug-and-play module, followed by a reconstruction model that simultaneously estimates RGB appearance, geometry, and semantics in real-time. Experiments demonstrate that our method effectively generates accurate 3D reconstructions and depth maps, enhancing culvert inspection efficiency with minimal human intervention.

  </details>



- **Intrinsic Tolerance in C-Arm Imaging: How Extrinsic Re-optimization Preserves 3D Reconstruction Accuracy**  
  Lin Li, Benjamin Aubert, Paul Kemper, Aric Plumley  
  _2026-03-14_ · https://arxiv.org/abs/2603.14031v1  
  <details><summary>Abstract</summary>

  \textbf{Purpose:} C-arm fluoroscopy's 3D reconstruction relies on accurate intrinsic calibration, which is often challenging in clinical practice. This study ensures high-precision reconstruction accuracy by re-optimizing the extrinsic parameters to compensate for intrinsic calibration errors. \noindent\textbf{Methods:} We conducted both simulation and real-world experiments using five commercial C-arm systems. Intrinsic parameters were perturbed in controlled increments. Focal length was increased by 100 to 700 pixels ($\approx$20 mm to 140 mm) and principal point by 20 to 200 pixels. For each perturbation, we (1) reconstructed 3D points from known phantom geometries, (2) re-estimated extrinsic poses using standard optimization, and (3) measured reconstruction and reprojection errors relative to ground truth. \noindent\textbf{Results:} Even with focal length errors up to 500 pixels ($\approx$100 mm, assuming a nominal focal length of $\sim$1000 mm), mean 3D reconstruction error remained under 0.2 mm. Larger focal length deviations (700 pixels) elevated error to only $\approx$0.3 mm. Principal point shifts up to 200 pixels introduced negligible reconstruction error once extrinsic parameters were re-optimized, with reprojection error increases below 0.5 pixels. \noindent\textbf{Conclusion:} Moderate errors in intrinsic calibration can be effectively mitigated by extrinsic re-optimization, preserving submillimeter 3D reconstruction accuracy. This intrinsic tolerance suggests a practical pathway to relax calibration precision requirements, thereby simplifying C-arm system setup and reducing clinical workflow burden without compromising performance.

  </details>



- **Geo-ID: Test-Time Geometric Consensus for Cross-View Consistent Intrinsics**  
  Alara Dirik, Stefanos Zafeiriou  
  _2026-03-14_ · https://arxiv.org/abs/2603.13859v1  
  <details><summary>Abstract</summary>

  Intrinsic image decomposition aims to estimate physically based rendering (PBR) parameters such as albedo, roughness, and metallicity from images. While recent methods achieve strong single-view predictions, applying them independently to multiple views of the same scene often yields inconsistent estimates, limiting their use in downstream applications such as editable neural scenes and 3D reconstruction. Video-based models can improve cross-frame consistency but require dense, ordered sequences and substantial compute, limiting their applicability to sparse, unordered image collections. We propose Geo-ID, a novel test-time framework that repurposes pretrained single-view intrinsic predictors to produce cross-view consistent decompositions by coupling independent per-view predictions through sparse geometric correspondences that form uncertainty-aware consensus targets. Geo-ID is model-agnostic, requires no retraining or inverse rendering, and applies directly to off-the-shelf intrinsic predictors. Experiments on synthetic benchmarks and real-world scenes demonstrate substantial improvements in cross-view intrinsic consistency as the number of views increases, while maintaining comparable single-view decomposition performance. We further show that the resulting consistent intrinsics enable coherent appearance editing and relighting in downstream neural scene representations.

  </details>


