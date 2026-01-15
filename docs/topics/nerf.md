# NeRF

_Updated: 2026-01-15 10:46 UTC_

Total papers shown: **30**


---

- **SAM3-DMS: Decoupled Memory Selection for Multi-target Video Segmentation of SAM3**  
  Ruiqi Shen, Chang Liu, Henghui Ding  
  _2026-01-14_ · https://arxiv.org/abs/2601.09699v1  
  <details><summary>Abstract</summary>

  Segment Anything 3 (SAM3) has established a powerful foundation that robustly detects, segments, and tracks specified targets in videos. However, in its original implementation, its group-level collective memory selection is suboptimal for complex multi-object scenarios, as it employs a synchronized decision across all concurrent targets conditioned on their average performance, often overlooking individual reliability. To this end, we propose SAM3-DMS, a training-free decoupled strategy that utilizes fine-grained memory selection on individual objects. Experiments demonstrate that our approach achieves robust identity preservation and tracking stability. Notably, our advantage becomes more pronounced with increased target density, establishing a solid foundation for simultaneous multi-target video segmentation in the wild.

  </details>



- **COMPOSE: Hypergraph Cover Optimization for Multi-view 3D Human Pose Estimation**  
  Tony Danjun Wang, Tolga Birdal, Nassir Navab, Lennart Bastian  
  _2026-01-14_ · https://arxiv.org/abs/2601.09698v1  
  <details><summary>Abstract</summary>

  3D pose estimation from sparse multi-views is a critical task for numerous applications, including action recognition, sports analysis, and human-robot interaction. Optimization-based methods typically follow a two-stage pipeline, first detecting 2D keypoints in each view and then associating these detections across views to triangulate the 3D pose. Existing methods rely on mere pairwise associations to model this correspondence problem, treating global consistency between views (i.e., cycle consistency) as a soft constraint. Yet, reconciling these constraints for multiple views becomes brittle when spurious associations propagate errors. We thus propose COMPOSE, a novel framework that formulates multi-view pose correspondence matching as a hypergraph partitioning problem rather than through pairwise association. While the complexity of the resulting integer linear program grows exponentially in theory, we introduce an efficient geometric pruning strategy to substantially reduce the search space. COMPOSE achieves improvements of up to 23% in average precision over previous optimization-based methods and up to 11% over self-supervised end-to-end learned methods, offering a promising solution to a widely studied problem.

  </details>



- **Efficient Camera-Controlled Video Generation of Static Scenes via Sparse Diffusion and 3D Rendering**  
  Jieying Chen, Jeffrey Hu, Joan Lasenby, Ayush Tewari  
  _2026-01-14_ · https://arxiv.org/abs/2601.09697v1  
  <details><summary>Abstract</summary>

  Modern video generative models based on diffusion models can produce very realistic clips, but they are computationally inefficient, often requiring minutes of GPU time for just a few seconds of video. This inefficiency poses a critical barrier to deploying generative video in applications that require real-time interactions, such as embodied AI and VR/AR. This paper explores a new strategy for camera-conditioned video generation of static scenes: using diffusion-based generative models to generate a sparse set of keyframes, and then synthesizing the full video through 3D reconstruction and rendering. By lifting keyframes into a 3D representation and rendering intermediate views, our approach amortizes the generation cost across hundreds of frames while enforcing geometric consistency. We further introduce a model that predicts the optimal number of keyframes for a given camera trajectory, allowing the system to adaptively allocate computation. Our final method, SRENDER, uses very sparse keyframes for simple trajectories and denser ones for complex camera motion. This results in video generation that is more than 40 times faster than the diffusion-based baseline in generating 20 seconds of video, while maintaining high visual fidelity and temporal stability, offering a practical path toward efficient and controllable video synthesis.

  </details>



- **LLMs can Compress LLMs: Adaptive Pruning by Agents**  
  Sai Varun Kodathala, Rakesh Vunnam  
  _2026-01-14_ · https://arxiv.org/abs/2601.09694v1  
  <details><summary>Abstract</summary>

  As Large Language Models (LLMs) continue to scale, post-training pruning has emerged as a promising approach to reduce computational costs while preserving performance. Existing methods such as SparseGPT and Wanda achieve high sparsity through layer-wise weight reconstruction or activation-aware magnitude pruning, but rely on uniform or hand-crafted heuristics to determine per-layer sparsity ratios. Moreover, recent work has shown that pruned LLMs suffer from severe factual knowledge degradation, with structured pruning methods experiencing near-total collapse in factual question-answering capabilities. We introduce agent-guided pruning, where a foundation model acts as an adaptive pruning agent to intelligently select which layers to prune at each iteration while preserving critical knowledge pathways. Our method constructs layer-wise sensitivity profiles by combining Wanda-inspired weight-activation metrics with gradient importance scores, normalized as z-scores for model-agnostic comparison. These statistics are processed by an LLM agent equipped with self-reflection capabilities, enabling it to learn from previous pruning outcomes and iteratively refine its strategy. A checkpoint rollback mechanism maintains model quality by reverting when perplexity degradation exceeds a threshold. We evaluate our approach on Qwen3 models (4B and 8B parameters) at approximately 45% sparsity, demonstrating substantial improvements over structured pruning baselines: 56% relative improvement in MMLU accuracy, 19x better factual knowledge retention on FreebaseQA, and 69% lower perplexity degradation. Notably, our framework requires no retraining, operates in a model-agnostic manner, and exhibits effective self-correction with only 2-4 rollbacks across 21-40 iterations, demonstrating that foundation models can effectively guide the compression of other foundation models.

  </details>



- **STEP3-VL-10B Technical Report**  
  Ailin Huang, Chengyuan Yao, Chunrui Han, Fanqi Wan, Hangyu Guo, Haoran Lv, Hongyu Zhou, Jia Wang, Jian Zhou, Jianjian Sun, et al.  
  _2026-01-14_ · https://arxiv.org/abs/2601.09668v1  
  <details><summary>Abstract</summary>

  We present STEP3-VL-10B, a lightweight open-source foundation model designed to redefine the trade-off between compact efficiency and frontier-level multimodal intelligence. STEP3-VL-10B is realized through two strategic shifts: first, a unified, fully unfrozen pre-training strategy on 1.2T multimodal tokens that integrates a language-aligned Perception Encoder with a Qwen3-8B decoder to establish intrinsic vision-language synergy; and second, a scaled post-training pipeline featuring over 1k iterations of reinforcement learning. Crucially, we implement Parallel Coordinated Reasoning (PaCoRe) to scale test-time compute, allocating resources to scalable perceptual reasoning that explores and synthesizes diverse visual hypotheses. Consequently, despite its compact 10B footprint, STEP3-VL-10B rivals or surpasses models 10$\times$-20$\times$ larger (e.g., GLM-4.6V-106B, Qwen3-VL-235B) and top-tier proprietary flagships like Gemini 2.5 Pro and Seed-1.5-VL. Delivering best-in-class performance, it records 92.2% on MMBench and 80.11% on MMMU, while excelling in complex reasoning with 94.43% on AIME2025 and 75.95% on MathVision. We release the full model suite to provide the community with a powerful, efficient, and reproducible baseline.

  </details>



- **SCE-SLAM: Scale-Consistent Monocular SLAM via Scene Coordinate Embeddings**  
  Yuchen Wu, Jiahe Li, Xiaohan Yu, Lina Yu, Jin Zheng, Xiao Bai  
  _2026-01-14_ · https://arxiv.org/abs/2601.09665v1  
  <details><summary>Abstract</summary>

  Monocular visual SLAM enables 3D reconstruction from internet video and autonomous navigation on resource-constrained platforms, yet suffers from scale drift, i.e., the gradual divergence of estimated scale over long sequences. Existing frame-to-frame methods achieve real-time performance through local optimization but accumulate scale drift due to the lack of global constraints among independent windows. To address this, we propose SCE-SLAM, an end-to-end SLAM system that maintains scale consistency through scene coordinate embeddings, which are learned patch-level representations encoding 3D geometric relationships under a canonical scale reference. The framework consists of two key modules: geometry-guided aggregation that leverages 3D spatial proximity to propagate scale information from historical observations through geometry-modulated attention, and scene coordinate bundle adjustment that anchors current estimates to the reference scale through explicit 3D coordinate constraints decoded from the scene coordinate embeddings. Experiments on KITTI, Waymo, and vKITTI demonstrate substantial improvements: our method reduces absolute trajectory error by 8.36m on KITTI compared to the best prior approach, while maintaining 36 FPS and achieving scale consistency across large-scale scenes.

  </details>



- **Self-Supervised Animal Identification for Long Videos**  
  Xuyang Fang, Sion Hannuna, Edwin Simpson, Neill Campbell  
  _2026-01-14_ · https://arxiv.org/abs/2601.09663v1  
  <details><summary>Abstract</summary>

  Identifying individual animals in long-duration videos is essential for behavioral ecology, wildlife monitoring, and livestock management. Traditional methods require extensive manual annotation, while existing self-supervised approaches are computationally demanding and ill-suited for long sequences due to memory constraints and temporal error propagation. We introduce a highly efficient, self-supervised method that reframes animal identification as a global clustering task rather than a sequential tracking problem. Our approach assumes a known, fixed number of individuals within a single video -- a common scenario in practice -- and requires only bounding box detections and the total count. By sampling pairs of frames, using a frozen pre-trained backbone, and employing a self-bootstrapping mechanism with the Hungarian algorithm for in-batch pseudo-label assignment, our method learns discriminative features without identity labels. We adapt a Binary Cross Entropy loss from vision-language models, enabling state-of-the-art accuracy ($>$97\%) while consuming less than 1 GB of GPU memory per batch -- an order of magnitude less than standard contrastive methods. Evaluated on challenging real-world datasets (3D-POP pigeons and 8-calves feeding videos), our framework matches or surpasses supervised baselines trained on over 1,000 labeled frames, effectively removing the manual annotation bottleneck. This work enables practical, high-accuracy animal identification on consumer-grade hardware, with broad applicability in resource-constrained research settings. All code written for this paper are \href{https://huggingface.co/datasets/tonyFang04/8-calves}{here}.

  </details>



- **LiteEmbed: Adapting CLIP to Rare Classes**  
  Aishwarya Agarwal, Srikrishna Karanam, Vineet Gandhi  
  _2026-01-14_ · https://arxiv.org/abs/2601.09661v1  
  <details><summary>Abstract</summary>

  Large-scale vision-language models such as CLIP achieve strong zero-shot recognition but struggle with classes that are rarely seen during pretraining, including newly emerging entities and culturally specific categories. We introduce LiteEmbed, a lightweight framework for few-shot personalization of CLIP that enables new classes to be added without retraining its encoders. LiteEmbed performs subspace-guided optimization of text embeddings within CLIP's vocabulary, leveraging a PCA-based decomposition that disentangles coarse semantic directions from fine-grained variations. Two complementary objectives, coarse alignment and fine separation, jointly preserve global semantic consistency while enhancing discriminability among visually similar classes. Once optimized, the embeddings are plug-and-play, seamlessly substituting CLIP's original text features across classification, retrieval, segmentation, and detection tasks. Extensive experiments demonstrate substantial gains over prior methods, establishing LiteEmbed as an effective approach for adapting CLIP to underrepresented, rare, or unseen classes.

  </details>



- **Image2Garment: Simulation-ready Garment Generation from a Single Image**  
  Selim Emir Can, Jan Ackermann, Kiyohiro Nakayama, Ruofan Liu, Tong Wu, Yang Zheng, Hugo Bertiche, Menglei Chai, Thabo Beeler, Gordon Wetzstein  
  _2026-01-14_ · https://arxiv.org/abs/2601.09658v1  
  <details><summary>Abstract</summary>

  Estimating physically accurate, simulation-ready garments from a single image is challenging due to the absence of image-to-physics datasets and the ill-posed nature of this problem. Prior methods either require multi-view capture and expensive differentiable simulation or predict only garment geometry without the material properties required for realistic simulation. We propose a feed-forward framework that sidesteps these limitations by first fine-tuning a vision-language model to infer material composition and fabric attributes from real images, and then training a lightweight predictor that maps these attributes to the corresponding physical fabric parameters using a small dataset of material-physics measurements. Our approach introduces two new datasets (FTAG and T2P) and delivers simulation-ready garments from a single image without iterative optimization. Experiments show that our estimator achieves superior accuracy in material composition estimation and fabric attribute prediction, and by passing them through our physics parameter estimator, we further achieve higher-fidelity simulations compared to state-of-the-art image-to-garment methods.

  </details>



- **AquaFeat+: an Underwater Vision Learning-based Enhancement Method for Object Detection, Classification, and Tracking**  
  Emanuel da Costa Silva, Tatiana Taís Schein, José David García Ramos, Eduardo Lawson da Silva, Stephanie Loi Brião, Felipe Gomes de Oliveira, Paulo Lilles Jorge Drews-Jr  
  _2026-01-14_ · https://arxiv.org/abs/2601.09652v1  
  <details><summary>Abstract</summary>

  Underwater video analysis is particularly challenging due to factors such as low lighting, color distortion, and turbidity, which compromise visual data quality and directly impact the performance of perception modules in robotic applications. This work proposes AquaFeat+, a plug-and-play pipeline designed to enhance features specifically for automated vision tasks, rather than for human perceptual quality. The architecture includes modules for color correction, hierarchical feature enhancement, and an adaptive residual output, which are trained end-to-end and guided directly by the loss function of the final application. Trained and evaluated in the FishTrack23 dataset, AquaFeat+ achieves significant improvements in object detection, classification, and tracking metrics, validating its effectiveness for enhancing perception tasks in underwater robotic applications.

  </details>



- **Identifying Models Behind Text-to-Image Leaderboards**  
  Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaudhari, Alina Oprea, Amir Houmansadr  
  _2026-01-14_ · https://arxiv.org/abs/2601.09647v1  
  <details><summary>Abstract</summary>

  Text-to-image (T2I) models are increasingly popular, producing a large share of AI-generated images online. To compare model quality, voting-based leaderboards have become the standard, relying on anonymized model outputs for fairness. In this work, we show that such anonymity can be easily broken. We find that generations from each T2I model form distinctive clusters in the image embedding space, enabling accurate deanonymization without prompt control or training data. Using 22 models and 280 prompts (150K images), our centroid-based method achieves high accuracy and reveals systematic model-specific signatures. We further introduce a prompt-level distinguishability metric and conduct large-scale analyses showing how certain prompts can lead to near-perfect distinguishability. Our findings expose fundamental security flaws in T2I leaderboards and motivate stronger anonymization defenses.

  </details>



- **Toward Understanding Unlearning Difficulty: A Mechanistic Perspective and Circuit-Guided Difficulty Metric**  
  Jiali Cheng, Ziheng Chen, Chirag Agarwal, Hadi Amiri  
  _2026-01-14_ · https://arxiv.org/abs/2601.09624v1  
  <details><summary>Abstract</summary>

  Machine unlearning is becoming essential for building trustworthy and compliant language models. Yet unlearning success varies considerably across individual samples: some are reliably erased, while others persist despite the same procedure. We argue that this disparity is not only a data-side phenomenon, but also reflects model-internal mechanisms that encode and protect memorized information. We study this problem from a mechanistic perspective based on model circuits--structured interaction pathways that govern how predictions are formed. We propose Circuit-guided Unlearning Difficulty (CUD), a {\em pre-unlearning} metric that assigns each sample a continuous difficulty score using circuit-level signals. Extensive experiments demonstrate that CUD reliably separates intrinsically easy and hard samples, and remains stable across unlearning methods. We identify key circuit-level patterns that reveal a mechanistic signature of difficulty: easy-to-unlearn samples are associated with shorter, shallower interactions concentrated in earlier-to-intermediate parts of the original model, whereas hard samples rely on longer and deeper pathways closer to late-stage computation. Compared to existing qualitative studies, CUD takes a first step toward a principled, fine-grained, and interpretable analysis of unlearning difficulty; and motivates the development of unlearning methods grounded in model mechanisms.

  </details>



- **Sim2real Image Translation Enables Viewpoint-Robust Policies from Fixed-Camera Datasets**  
  Jeremiah Coholich, Justin Wit, Robert Azarcon, Zsolt Kira  
  _2026-01-14_ · https://arxiv.org/abs/2601.09605v1  
  <details><summary>Abstract</summary>

  Vision-based policies for robot manipulation have achieved significant recent success, but are still brittle to distribution shifts such as camera viewpoint variations. Robot demonstration data is scarce and often lacks appropriate variation in camera viewpoints. Simulation offers a way to collect robot demonstrations at scale with comprehensive coverage of different viewpoints, but presents a visual sim2real challenge. To bridge this gap, we propose MANGO -- an unpaired image translation method with a novel segmentation-conditioned InfoNCE loss, a highly-regularized discriminator design, and a modified PatchNCE loss. We find that these elements are crucial for maintaining viewpoint consistency during sim2real translation. When training MANGO, we only require a small amount of fixed-camera data from the real world, but show that our method can generate diverse unseen viewpoints by translating simulated observations. In this domain, MANGO outperforms all other image translation methods we tested. Imitation-learning policies trained on data augmented by MANGO are able to achieve success rates as high as 60\% on views that the non-augmented policy fails completely on.

  </details>



- **Iterative Differential Entropy Minimization (IDEM) method for fine rigid pairwise 3D Point Cloud Registration: A Focus on the Metric**  
  Emmanuele Barberi, Felice Sfravara, Filippo Cucinotta  
  _2026-01-14_ · https://arxiv.org/abs/2601.09601v1  
  <details><summary>Abstract</summary>

  Point cloud registration is a central theme in computer vision, with alignment algorithms continuously improving for greater robustness. Commonly used methods evaluate Euclidean distances between point clouds and minimize an objective function, such as Root Mean Square Error (RMSE). However, these approaches are most effective when the point clouds are well-prealigned and issues such as differences in density, noise, holes, and limited overlap can compromise the results. Traditional methods, such as Iterative Closest Point (ICP), require choosing one point cloud as fixed, since Euclidean distances lack commutativity. When only one point cloud has issues, adjustments can be made, but in real scenarios, both point clouds may be affected, often necessitating preprocessing. The authors introduce a novel differential entropy-based metric, designed to serve as the objective function within an optimization framework for fine rigid pairwise 3D point cloud registration, denoted as Iterative Differential Entropy Minimization (IDEM). This metric does not depend on the choice of a fixed point cloud and, during transformations, reveals a clear minimum corresponding to the best alignment. Multiple case studies are conducted, and the results are compared with those obtained using RMSE, Chamfer distance, and Hausdorff distance. The proposed metric proves effective even with density differences, noise, holes, and partial overlap, where RMSE does not always yield optimal alignment.

  </details>



- **Show, don't tell -- Providing Visual Error Feedback for Handwritten Documents**  
  Said Yasin, Torsten Zesch  
  _2026-01-14_ · https://arxiv.org/abs/2601.09586v1  
  <details><summary>Abstract</summary>

  Handwriting remains an essential skill, particularly in education. Therefore, providing visual feedback on handwritten documents is an important but understudied area. We outline the many challenges when going from an image of handwritten input to correctly placed informative error feedback. We empirically compare modular and end-to-end systems and find that both approaches currently do not achieve acceptable overall quality. We identify the major challenges and outline an agenda for future research.

  </details>



- **Multimodal Signal Processing For Thermo-Visible-Lidar Fusion In Real-time 3D Semantic Mapping**  
  Jiajun Sun, Yangyi Ou, Haoyuan Zheng, Chao yang, Yue Ma  
  _2026-01-14_ · https://arxiv.org/abs/2601.09578v1  
  <details><summary>Abstract</summary>

  In complex environments, autonomous robot navigation and environmental perception pose higher requirements for SLAM technology. This paper presents a novel method for semantically enhancing 3D point cloud maps with thermal information. By first performing pixel-level fusion of visible and infrared images, the system projects real-time LiDAR point clouds onto this fused image stream. It then segments heat source features in the thermal channel to instantly identify high temperature targets and applies this temperature information as a semantic layer on the final 3D map. This approach generates maps that not only have accurate geometry but also possess a critical semantic understanding of the environment, making it highly valuable for specific applications like rapid disaster assessment and industrial preventive maintenance.

  </details>



- **OpenVoxel: Training-Free Grouping and Captioning Voxels for Open-Vocabulary 3D Scene Understanding**  
  Sheng-Yu Huang, Jaesung Choe, Yu-Chiang Frank Wang, Cheng Sun  
  _2026-01-14_ · https://arxiv.org/abs/2601.09575v1  
  <details><summary>Abstract</summary>

  We propose OpenVoxel, a training-free algorithm for grouping and captioning sparse voxels for the open-vocabulary 3D scene understanding tasks. Given the sparse voxel rasterization (SVR) model obtained from multi-view images of a 3D scene, our OpenVoxel is able to produce meaningful groups that describe different objects in the scene. Also, by leveraging powerful Vision Language Models (VLMs) and Multi-modal Large Language Models (MLLMs), our OpenVoxel successfully build an informative scene map by captioning each group, enabling further 3D scene understanding tasks such as open-vocabulary segmentation (OVS) or referring expression segmentation (RES). Unlike previous methods, our method is training-free and does not introduce embeddings from a CLIP/BERT text encoder. Instead, we directly proceed with text-to-text search using MLLMs. Through extensive experiments, our method demonstrates superior performance compared to recent studies, particularly in complex referring expression segmentation (RES) tasks. The code will be open.

  </details>



- **Trustworthy Longitudinal Brain MRI Completion: A Deformation-Based Approach with KAN-Enhanced Diffusion Model**  
  Tianli Tao, Ziyang Wang, Delong Yang, Han Zhang, Le Zhang  
  _2026-01-14_ · https://arxiv.org/abs/2601.09572v1  
  <details><summary>Abstract</summary>

  Longitudinal brain MRI is essential for lifespan study, yet high attrition rates often lead to missing data, complicating analysis. Deep generative models have been explored, but most rely solely on image intensity, leading to two key limitations: 1) the fidelity or trustworthiness of the generated brain images are limited, making downstream studies questionable; 2) the usage flexibility is restricted due to fixed guidance rooted in the model structure, restricting full ability to versatile application scenarios. To address these challenges, we introduce DF-DiffCom, a Kolmogorov-Arnold Networks (KAN)-enhanced diffusion model that smartly leverages deformation fields for trustworthy longitudinal brain image completion. Trained on OASIS-3, DF-DiffCom outperforms state-of-the-art methods, improving PSNR by 5.6% and SSIM by 0.12. More importantly, its modality-agnostic nature allows smooth extension to varied MRI modalities, even to attribute maps such as brain tissue segmentation results.

  </details>



- **Hot-Start from Pixels: Low-Resolution Visual Tokens for Chinese Language Modeling**  
  Shuyang Xiang, Hao Guan  
  _2026-01-14_ · https://arxiv.org/abs/2601.09566v1  
  <details><summary>Abstract</summary>

  Large language models typically represent Chinese characters as discrete index-based tokens, largely ignoring their visual form. For logographic scripts, visual structure carries semantic and phonetic information, which may aid prediction. We investigate whether low-resolution visual inputs can serve as an alternative for character-level modeling. Instead of token IDs, our decoder receives grayscale images of individual characters, with resolutions as low as $8 \times 8$ pixels. Remarkably, these inputs achieve 39.2\% accuracy, comparable to the index-based baseline of 39.1\%. Such low-resource settings also exhibit a pronounced \emph{hot-start} effect: by 0.4\% of total training, accuracy reaches above 12\%, while index-based models lag at below 6\%. Overall, our results demonstrate that minimal visual structure can provide a robust and efficient signal for Chinese language modeling, offering an alternative perspective on character representation that complements traditional index-based approaches.

  </details>



- **Bipartite Mode Matching for Vision Training Set Search from a Hierarchical Data Server**  
  Yue Yao, Ruining Yang, Tom Gedeon  
  _2026-01-14_ · https://arxiv.org/abs/2601.09531v1  
  <details><summary>Abstract</summary>

  We explore a situation in which the target domain is accessible, but real-time data annotation is not feasible. Instead, we would like to construct an alternative training set from a large-scale data server so that a competitive model can be obtained. For this problem, because the target domain usually exhibits distinct modes (i.e., semantic clusters representing data distribution), if the training set does not contain these target modes, the model performance would be compromised. While prior existing works improve algorithms iteratively, our research explores the often-overlooked potential of optimizing the structure of the data server. Inspired by the hierarchical nature of web search engines, we introduce a hierarchical data server, together with a bipartite mode matching algorithm (BMM) to align source and target modes. For each target mode, we look in the server data tree for the best mode match, which might be large or small in size. Through bipartite matching, we aim for all target modes to be optimally matched with source modes in a one-on-one fashion. Compared with existing training set search algorithms, we show that the matched server modes constitute training sets that have consistently smaller domain gaps with the target domain across object re-identification (re-ID) and detection tasks. Consequently, models trained on our searched training sets have higher accuracy than those trained otherwise. BMM allows data-centric unsupervised domain adaptation (UDA) orthogonal to existing model-centric UDA methods. By combining the BMM with existing UDA methods like pseudo-labeling, further improvement is observed.

  </details>



- **Video Joint-Embedding Predictive Architectures for Facial Expression Recognition**  
  Lennart Eing, Cristina Luna-Jiménez, Silvan Mertes, Elisabeth André  
  _2026-01-14_ · https://arxiv.org/abs/2601.09524v1  
  <details><summary>Abstract</summary>

  This paper introduces a novel application of Video Joint-Embedding Predictive Architectures (V-JEPAs) for Facial Expression Recognition (FER). Departing from conventional pre-training methods for video understanding that rely on pixel-level reconstructions, V-JEPAs learn by predicting embeddings of masked regions from the embeddings of unmasked regions. This enables the trained encoder to not capture irrelevant information about a given video like the color of a region of pixels in the background. Using a pre-trained V-JEPA video encoder, we train shallow classifiers using the RAVDESS and CREMA-D datasets, achieving state-of-the-art performance on RAVDESS and outperforming all other vision-based methods on CREMA-D (+1.48 WAR). Furthermore, cross-dataset evaluations reveal strong generalization capabilities, demonstrating the potential of purely embedding-based pre-training approaches to advance FER. We release our code at https://github.com/lennarteingunia/vjepa-for-fer.

  </details>



- **Learning Whole-Body Human-Humanoid Interaction from Human-Human Demonstrations**  
  Wei-Jin Huang, Yue-Yi Zhang, Yi-Lin Wei, Zhi-Wei Xia, Juantao Tan, Yuan-Ming Li, Zhilin Zhao, Wei-Shi Zheng  
  _2026-01-14_ · https://arxiv.org/abs/2601.09518v1  
  <details><summary>Abstract</summary>

  Enabling humanoid robots to physically interact with humans is a critical frontier, but progress is hindered by the scarcity of high-quality Human-Humanoid Interaction (HHoI) data. While leveraging abundant Human-Human Interaction (HHI) data presents a scalable alternative, we first demonstrate that standard retargeting fails by breaking the essential contacts. We address this with PAIR (Physics-Aware Interaction Retargeting), a contact-centric, two-stage pipeline that preserves contact semantics across morphology differences to generate physically consistent HHoI data. This high-quality data, however, exposes a second failure: conventional imitation learning policies merely mimic trajectories and lack interactive understanding. We therefore introduce D-STAR (Decoupled Spatio-Temporal Action Reasoner), a hierarchical policy that disentangles when to act from where to act. In D-STAR, Phase Attention (when) and a Multi-Scale Spatial module (where) are fused by the diffusion head to produce synchronized whole-body behaviors beyond mimicry. By decoupling these reasoning streams, our model learns robust temporal phases without being distracted by spatial noise, leading to responsive, synchronized collaboration. We validate our framework through extensive and rigorous simulations, demonstrating significant performance gains over baseline approaches and a complete, effective pipeline for learning complex whole-body interactions from HHI data.

  </details>



- **V-DPM: 4D Video Reconstruction with Dynamic Point Maps**  
  Edgar Sucar, Eldar Insafutdinov, Zihang Lai, Andrea Vedaldi  
  _2026-01-14_ · https://arxiv.org/abs/2601.09499v1  
  <details><summary>Abstract</summary>

  Powerful 3D representations such as DUSt3R invariant point maps, which encode 3D shape and camera parameters, have significantly advanced feed forward 3D reconstruction. While point maps assume static scenes, Dynamic Point Maps (DPMs) extend this concept to dynamic 3D content by additionally representing scene motion. However, existing DPMs are limited to image pairs and, like DUSt3R, require post processing via optimization when more than two views are involved. We argue that DPMs are more useful when applied to videos and introduce V-DPM to demonstrate this. First, we show how to formulate DPMs for video input in a way that maximizes representational power, facilitates neural prediction, and enables reuse of pretrained models. Second, we implement these ideas on top of VGGT, a recent and powerful 3D reconstructor. Although VGGT was trained on static scenes, we show that a modest amount of synthetic data is sufficient to adapt it into an effective V-DPM predictor. Our approach achieves state of the art performance in 3D and 4D reconstruction for dynamic scenes. In particular, unlike recent dynamic extensions of VGGT such as P3, DPMs recover not only dynamic depth but also the full 3D motion of every point in the scene.

  </details>



- **MAD: Motion Appearance Decoupling for efficient Driving World Models**  
  Ahmad Rahimi, Valentin Gerard, Eloi Zablocki, Matthieu Cord, Alexandre Alahi  
  _2026-01-14_ · https://arxiv.org/abs/2601.09452v1  
  <details><summary>Abstract</summary>

  Recent video diffusion models generate photorealistic, temporally coherent videos, yet they fall short as reliable world models for autonomous driving, where structured motion and physically consistent interactions are essential. Adapting these generalist video models to driving domains has shown promise but typically requires massive domain-specific data and costly fine-tuning. We propose an efficient adaptation framework that converts generalist video diffusion models into controllable driving world models with minimal supervision. The key idea is to decouple motion learning from appearance synthesis. First, the model is adapted to predict structured motion in a simplified form: videos of skeletonized agents and scene elements, focusing learning on physical and social plausibility. Then, the same backbone is reused to synthesize realistic RGB videos conditioned on these motion sequences, effectively "dressing" the motion with texture and lighting. This two-stage process mirrors a reasoning-rendering paradigm: first infer dynamics, then render appearance. Our experiments show this decoupled approach is exceptionally efficient: adapting SVD, we match prior SOTA models with less than 6% of their compute. Scaling to LTX, our MAD-LTX model outperforms all open-source competitors, and supports a comprehensive suite of text, ego, and object controls. Project page: https://vita-epfl.github.io/MAD-World-Model/

  </details>



- **PrivLEX: Detecting legal concepts in images through Vision-Language Models**  
  Darya Baranouskaya, Andrea Cavallaro  
  _2026-01-14_ · https://arxiv.org/abs/2601.09449v1  
  <details><summary>Abstract</summary>

  We present PrivLEX, a novel image privacy classifier that grounds its decisions in legally defined personal data concepts. PrivLEX is the first interpretable privacy classifier aligned with legal concepts that leverages the recognition capabilities of Vision-Language Models (VLMs). PrivLEX relies on zero-shot VLM concept detection to provide interpretable classification through a label-free Concept Bottleneck Model, without requiring explicit concept labels during training. We demonstrate PrivLEX's ability to identify personal data concepts that are present in images. We further analyse the sensitivity of such concepts as perceived by human annotators of image privacy datasets.

  </details>



- **Do Transformers Understand Ancient Roman Coin Motifs Better than CNNs?**  
  David Reid, Ognjen Arandjelovic  
  _2026-01-14_ · https://arxiv.org/abs/2601.09433v1  
  <details><summary>Abstract</summary>

  Automated analysis of ancient coins has the potential to help researchers extract more historical insights from large collections of coins and to help collectors understand what they are buying or selling. Recent research in this area has shown promise in focusing on identification of semantic elements as they are commonly depicted on ancient coins, by using convolutional neural networks (CNNs). This paper is the first to apply the recently proposed Vision Transformer (ViT) deep learning architecture to the task of identification of semantic elements on coins, using fully automatic learning from multi-modal data (images and unstructured text). This article summarises previous research in the area, discusses the training and implementation of ViT and CNN models for ancient coins analysis and provides an evaluation of their performance. The ViT models were found to outperform the newly trained CNN models in accuracy.

  </details>



- **Draw it like Euclid: Teaching transformer models to generate CAD profiles using ruler and compass construction steps**  
  Siyi Li, Joseph G. Lambourne, Longfei Zhang, Pradeep Kumar Jayaraman, Karl. D. D. Willis  
  _2026-01-14_ · https://arxiv.org/abs/2601.09428v1  
  <details><summary>Abstract</summary>

  We introduce a new method of generating Computer Aided Design (CAD) profiles via a sequence of simple geometric constructions including curve offsetting, rotations and intersections. These sequences start with geometry provided by a designer and build up the points and curves of the final profile step by step. We demonstrate that adding construction steps between the designer's input geometry and the final profile improves generation quality in a similar way to the introduction of a chain of thought in language models. Similar to the constraints in a parametric CAD model, the construction sequences reduce the degrees of freedom in the modeled shape to a small set of parameter values which can be adjusted by the designer, allowing parametric editing with the constructed geometry evaluated to floating point precision. In addition we show that applying reinforcement learning to the construction sequences gives further improvements over a wide range of metrics, including some which were not explicitly optimized.

  </details>



- **Variable Basis Mapping for Real-Time Volumetric Visualization**  
  Qibiao Li, Yuxuan Wang, Youcheng Cai, Huangsheng Du, Ligang Liu  
  _2026-01-14_ · https://arxiv.org/abs/2601.09417v1  
  <details><summary>Abstract</summary>

  Real-time visualization of large-scale volumetric data remains challenging, as direct volume rendering and voxel-based methods suffer from prohibitively high computational cost. We propose Variable Basis Mapping (VBM), a framework that transforms volumetric fields into 3D Gaussian Splatting (3DGS) representations through wavelet-domain analysis. First, we precompute a compact Wavelet-to-Gaussian Transition Bank that provides optimal Gaussian surrogates for canonical wavelet atoms across multiple scales. Second, we perform analytical Gaussian construction that maps discrete wavelet coefficients directly to 3DGS parameters using a closed-form, mathematically principled rule. Finally, a lightweight image-space fine-tuning stage further refines the representation to improve rendering fidelity. Experiments on diverse datasets demonstrate that VBM significantly accelerates convergence and enhances rendering quality, enabling real-time volumetric visualization.

  </details>



- **Radiomics-Integrated Deep Learning with Hierarchical Loss for Osteosarcoma Histology Classification**  
  Yaxi Chen, Zi Ye, Shaheer U. Saeed, Oliver Yu, Simin Ni, Jie Huang, Yipeng Hu  
  _2026-01-14_ · https://arxiv.org/abs/2601.09416v1  
  <details><summary>Abstract</summary>

  Osteosarcoma (OS) is an aggressive primary bone malignancy. Accurate histopathological assessment of viable versus non-viable tumor regions after neoadjuvant chemotherapy is critical for prognosis and treatment planning, yet manual evaluation remains labor-intensive, subjective, and prone to inter-observer variability. Recent advances in digital pathology have enabled automated necrosis quantification. Evaluating on test data, independently sampled on patient-level, revealed that the deep learning model performance dropped significantly from the tile-level generalization ability reported in previous studies. First, this work proposes the use of radiomic features as additional input in model training. We show that, despite that they are derived from the images, such a multimodal input effectively improved the classification performance, in addition to its added benefits in interpretability. Second, this work proposes to optimize two binary classification tasks with hierarchical classes (i.e. tumor-vs-non-tumor and viable-vs-non-viable), as opposed to the alternative ``flat'' three-class classification task (i.e. non-tumor, non-viable tumor, viable tumor), thereby enabling a hierarchical loss. We show that such a hierarchical loss, with trainable weightings between the two tasks, the per-class performance can be improved significantly. Using the TCIA OS Tumor Assessment dataset, we experimentally demonstrate the benefits from each of the proposed new approaches and their combination, setting a what we consider new state-of-the-art performance on this open dataset for this application. Code and trained models: https://github.com/YaxiiC/RadiomicsOS.git.

  </details>



- **Detail Loss in Super-Resolution Models Based on the Laplacian Pyramid and Repeated Upscaling and Downscaling Process**  
  Sangjun Han, Youngmi Hur  
  _2026-01-14_ · https://arxiv.org/abs/2601.09410v1  
  <details><summary>Abstract</summary>

  With advances in artificial intelligence, image processing has gained significant interest. Image super-resolution is a vital technology closely related to real-world applications, as it enhances the quality of existing images. Since enhancing fine details is crucial for the super-resolution task, pixels that contribute to high-frequency information should be emphasized. This paper proposes two methods to enhance high-frequency details in super-resolution images: a Laplacian pyramid-based detail loss and a repeated upscaling and downscaling process. Total loss with our detail loss guides a model by separately generating and controlling super-resolution and detail images. This approach allows the model to focus more effectively on high-frequency components, resulting in improved super-resolution images. Additionally, repeated upscaling and downscaling amplify the effectiveness of the detail loss by extracting diverse information from multiple low-resolution features. We conduct two types of experiments. First, we design a CNN-based model incorporating our methods. This model achieves state-of-the-art results, surpassing all currently available CNN-based and even some attention-based models. Second, we apply our methods to existing attention-based models on a small scale. In all our experiments, attention-based models adding our detail loss show improvements compared to the originals. These results demonstrate our approaches effectively enhance super-resolution images across different model structures.

  </details>


