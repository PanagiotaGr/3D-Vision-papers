# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-02-27 07:10 UTC_

Total papers shown: **50**


---

- **SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation**  
  Vaibhav Agrawal, Rishubh Parihar, Pradhaan Bhat, Ravi Kiran Sarvadevabhatla, R. Venkatesh Babu  
  _2026-02-26_ · https://arxiv.org/abs/2602.23359v1  
  <details><summary>Abstract</summary>

  We identify occlusion reasoning as a fundamental yet overlooked aspect for 3D layout-conditioned generation. It is essential for synthesizing partially occluded objects with depth-consistent geometry and scale. While existing methods can generate realistic scenes that follow input layouts, they often fail to model precise inter-object occlusions. We propose SeeThrough3D, a model for 3D layout conditioned generation that explicitly models occlusions. We introduce an occlusion-aware 3D scene representation (OSCR), where objects are depicted as translucent 3D boxes placed within a virtual environment and rendered from desired camera viewpoint. The transparency encodes hidden object regions, enabling the model to reason about occlusions, while the rendered viewpoint provides explicit camera control during generation. We condition a pretrained flow based text-to-image image generation model by introducing a set of visual tokens derived from our rendered 3D representation. Furthermore, we apply masked self-attention to accurately bind each object bounding box to its corresponding textual description, enabling accurate generation of multiple objects without object attribute mixing. To train the model, we construct a synthetic dataset with diverse multi-object scenes with strong inter-object occlusions. SeeThrough3D generalizes effectively to unseen object categories and enables precise 3D layout control with realistic occlusions and consistent camera control.

  </details>



- **A Dataset is Worth 1 MB**  
  Elad Kimchi Shoshani, Leeyam Gabay, Yedid Hoshen  
  _2026-02-26_ · https://arxiv.org/abs/2602.23358v1  
  <details><summary>Abstract</summary>

  A dataset server must often distribute the same large payload to many clients, incurring massive communication costs. Since clients frequently operate on diverse hardware and software frameworks, transmitting a pre-trained model is often infeasible; instead, agents require raw data to train their own task-specific models locally. While dataset distillation attempts to compress training signals, current methods struggle to scale to high-resolution data and rarely achieve sufficiently small files. In this paper, we propose Pseudo-Labels as Data (PLADA), a method that completely eliminates pixel transmission. We assume agents are preloaded with a large, generic, unlabeled reference dataset (e.g., ImageNet-1K, ImageNet-21K) and communicate a new task by transmitting only the class labels for specific images. To address the distribution mismatch between the reference and target datasets, we introduce a pruning mechanism that filters the reference dataset to retain only the labels of the most semantically relevant images for the target task. This selection process simultaneously maximizes training efficiency and minimizes transmission payload. Experiments on 10 diverse datasets demonstrate that our approach can transfer task knowledge with a payload of less than 1 MB while retaining high classification accuracy, offering a promising solution for efficient dataset serving.

  </details>



- **Evaluating Zero-Shot and One-Shot Adaptation of Small Language Models in Leader-Follower Interaction**  
  Rafael R. Baptista, André de Lima Salgado, Ricardo V. Godoy, Marcelo Becker, Thiago Boaventura, Gustavo J. G. Lahr  
  _2026-02-26_ · https://arxiv.org/abs/2602.23312v1  
  <details><summary>Abstract</summary>

  Leader-follower interaction is an important paradigm in human-robot interaction (HRI). Yet, assigning roles in real time remains challenging for resource-constrained mobile and assistive robots. While large language models (LLMs) have shown promise for natural communication, their size and latency limit on-device deployment. Small language models (SLMs) offer a potential alternative, but their effectiveness for role classification in HRI has not been systematically evaluated. In this paper, we present a benchmark of SLMs for leader-follower communication, introducing a novel dataset derived from a published database and augmented with synthetic samples to capture interaction-specific dynamics. We investigate two adaptation strategies: prompt engineering and fine-tuning, studied under zero-shot and one-shot interaction modes, compared with an untrained baseline. Experiments with Qwen2.5-0.5B reveal that zero-shot fine-tuning achieves robust classification performance (86.66% accuracy) while maintaining low latency (22.2 ms per sample), significantly outperforming baseline and prompt-engineered approaches. However, results also indicate a performance degradation in one-shot modes, where increased context length challenges the model's architectural capacity. These findings demonstrate that fine-tuned SLMs provide an effective solution for direct role assignment, while highlighting critical trade-offs between dialogue complexity and classification reliability on the edge.

  </details>



- **ManifoldGD: Training-Free Hierarchical Manifold Guidance for Diffusion-Based Dataset Distillation**  
  Ayush Roy, Wei-Yang Alex Lee, Rudrasis Chakraborty, Vishnu Suresh Lokhande  
  _2026-02-26_ · https://arxiv.org/abs/2602.23295v1  
  <details><summary>Abstract</summary>

  In recent times, large datasets hinder efficient model training while also containing redundant concepts. Dataset distillation aims to synthesize compact datasets that preserve the knowledge of large-scale training sets while drastically reducing storage and computation. Recent advances in diffusion models have enabled training-free distillation by leveraging pre-trained generative priors; however, existing guidance strategies remain limited. Current score-based methods either perform unguided denoising or rely on simple mode-based guidance toward instance prototype centroids (IPC centroids), which often are rudimentary and suboptimal. We propose Manifold-Guided Distillation (ManifoldGD), a training-free diffusion-based framework that integrates manifold consistent guidance at every denoising timestep. Our method employs IPCs computed via a hierarchical, divisive clustering of VAE latent features, yielding a multi-scale coreset of IPCs that captures both coarse semantic modes and fine intra-class variability. Using a local neighborhood of the extracted IPC centroids, we create the latent manifold for each diffusion denoising timestep. At each denoising step, we project the mode-alignment vector onto the local tangent space of the estimated latent manifold, thus constraining the generation trajectory to remain manifold-faithful while preserving semantic consistency. This formulation improves representativeness, diversity, and image fidelity without requiring any model retraining. Empirical results demonstrate consistent gains over existing training-free and training-based baselines in terms of FID, l2 distance among real and synthetic dataset embeddings, and classification accuracy, establishing ManifoldGD as the first geometry-aware training-free data distillation framework.

  </details>



- **LineGraph2Road: Structural Graph Reasoning on Line Graphs for Road Network Extraction**  
  Zhengyang Wei, Renzhi Jing, Yiyi He, Jenny Suckale  
  _2026-02-26_ · https://arxiv.org/abs/2602.23290v1  
  <details><summary>Abstract</summary>

  The accurate and automatic extraction of roads from satellite imagery is critical for applications in navigation and urban planning, significantly reducing the need for manual annotation. Many existing methods decompose this task into keypoint extraction and connectedness prediction, but often struggle to capture long-range dependencies and complex topologies. Here, we propose LineGraph2Road, a framework that improves connectedness prediction by formulating it as binary classification over edges in a constructed global but sparse Euclidean graph, where nodes are keypoints extracted from segmentation masks and edges connect node pairs within a predefined distance threshold, representing potential road segments. To better learn structural link representation, we transform the original graph into its corresponding line graph and apply a Graph Transformer on it for connectedness prediction. This formulation overcomes the limitations of endpoint-embedding fusion on set-isomorphic links, enabling rich link representations and effective relational reasoning over the global structure. Additionally, we introduce an overpass/underpass head to resolve multi-level crossings and a coupled NMS strategy to preserve critical connections. We evaluate LineGraph2Road on three benchmarks: City-scale, SpaceNet, and Global-scale, and show that it achieves state-of-the-art results on two key metrics, TOPO-F1 and APLS. It also captures fine visual details critical for real-world deployment. We will make our code publicly available.

  </details>



- **Large Multimodal Models as General In-Context Classifiers**  
  Marco Garosi, Matteo Farina, Alessandro Conti, Massimiliano Mancini, Elisa Ricci  
  _2026-02-26_ · https://arxiv.org/abs/2602.23229v1  
  <details><summary>Abstract</summary>

  Which multimodal model should we use for classification? Previous studies suggest that the answer lies in CLIP-like contrastive Vision-Language Models (VLMs), due to their remarkable performance in zero-shot classification. In contrast, Large Multimodal Models (LMM) are more suitable for complex tasks. In this work, we argue that this answer overlooks an important capability of LMMs: in-context learning. We benchmark state-of-the-art LMMs on diverse datasets for closed-world classification and find that, although their zero-shot performance is lower than CLIP's, LMMs with a few in-context examples can match or even surpass contrastive VLMs with cache-based adapters, their "in-context" equivalent. We extend this analysis to the open-world setting, where the generative nature of LMMs makes them more suitable for the task. In this challenging scenario, LMMs struggle whenever provided with imperfect context information. To address this issue, we propose CIRCLE, a simple training-free method that assigns pseudo-labels to in-context examples, iteratively refining them with the available context itself. Through extensive experiments, we show that CIRCLE establishes a robust baseline for open-world classification, surpassing VLM counterparts and highlighting the potential of LMMs to serve as unified classifiers, and a flexible alternative to specialized models.

  </details>



- **Motion-aware Event Suppression for Event Cameras**  
  Roberto Pellerito, Nico Messikommer, Giovanni Cioffi, Marco Cannici, Davide Scaramuzza  
  _2026-02-26_ · https://arxiv.org/abs/2602.23204v1  
  <details><summary>Abstract</summary>

  In this work, we introduce the first framework for Motion-aware Event Suppression, which learns to filter events triggered by IMOs and ego-motion in real time. Our model jointly segments IMOs in the current event stream while predicting their future motion, enabling anticipatory suppression of dynamic events before they occur. Our lightweight architecture achieves 173 Hz inference on consumer-grade GPUs with less than 1 GB of memory usage, outperforming previous state-of-the-art methods on the challenging EVIMO benchmark by 67\% in segmentation accuracy while operating at a 53\% higher inference rate. Moreover, we demonstrate significant benefits for downstream applications: our method accelerates Vision Transformer inference by 83\% via token pruning and improves event-based visual odometry accuracy, reducing Absolute Trajectory Error (ATE) by 13\%.

  </details>



- **Phys-3D: Physics-Constrained Real-Time Crowd Tracking and Counting on Railway Platforms**  
  Bin Zeng, Johannes Künzel, Anna Hilsmann, Peter Eisert  
  _2026-02-26_ · https://arxiv.org/abs/2602.23177v1  
  <details><summary>Abstract</summary>

  Accurate, real-time crowd counting on railway platforms is essential for safety and capacity management. We propose to use a single camera mounted in a train, scanning the platform while arriving. While hardware constraints are simple, counting remains challenging due to dense occlusions, camera motion, and perspective distortions during train arrivals. Most existing tracking-by-detection approaches assume static cameras or ignore physical consistency in motion modeling, leading to unreliable counting under dynamic conditions. We propose a physics-constrained tracking framework that unifies detection, appearance, and 3D motion reasoning in a real-time pipeline. Our approach integrates a transfer-learned YOLOv11m detector with EfficientNet-B0 appearance encoding within DeepSORT, while introducing a physics-constrained Kalman model (Phys-3D) that enforces physically plausible 3D motion dynamics through pinhole geometry. To address counting brittleness under occlusions, we implement a virtual counting band with persistence. On our platform benchmark, MOT-RailwayPlatformCrowdHead Dataset(MOT-RPCH), our method reduces counting error to 2.97%, demonstrating robust performance despite motion and occlusions. Our results show that incorporating first-principles geometry and motion priors enables reliable crowd counting in safety-critical transportation scenarios, facilitating effective train scheduling and platform safety management.

  </details>



- **AgentVista: Evaluating Multimodal Agents in Ultra-Challenging Realistic Visual Scenarios**  
  Zhaochen Su, Jincheng Gao, Hangyu Guo, Zhenhua Liu, Lueyang Zhang, Xinyu Geng, Shijue Huang, Peng Xia, Guanyu Jiang, Cheng Wang, et al.  
  _2026-02-26_ · https://arxiv.org/abs/2602.23166v1  
  <details><summary>Abstract</summary>

  Real-world multimodal agents solve multi-step workflows grounded in visual evidence. For example, an agent can troubleshoot a device by linking a wiring photo to a schematic and validating the fix with online documentation, or plan a trip by interpreting a transit map and checking schedules under routing constraints. However, existing multimodal benchmarks mainly evaluate single-turn visual reasoning or specific tool skills, and they do not fully capture the realism, visual subtlety, and long-horizon tool use that practical agents require. We introduce AgentVista, a benchmark for generalist multimodal agents that spans 25 sub-domains across 7 categories, pairing realistic and detail-rich visual scenarios with natural hybrid tool use. Tasks require long-horizon tool interactions across modalities, including web search, image search, page navigation, and code-based operations for both image processing and general programming. Comprehensive evaluation of state-of-the-art models exposes significant gaps in their ability to carry out long-horizon multimodal tool use. Even the best model in our evaluation, Gemini-3-Pro with tools, achieves only 27.3% overall accuracy, and hard instances can require more than 25 tool-calling turns. We expect AgentVista to accelerate the development of more capable and reliable multimodal agents for realistic and ultra-challenging problem solving.

  </details>



- **DyaDiT: A Multi-Modal Diffusion Transformer for Socially Favorable Dyadic Gesture Generation**  
  Yichen Peng, Jyun-Ting Song, Siyeol Jung, Ruofan Liu, Haiyang Liu, Xuangeng Chu, Ruicong Liu, Erwin Wu, Hideki Koike, Kris Kitani  
  _2026-02-26_ · https://arxiv.org/abs/2602.23165v1  
  <details><summary>Abstract</summary>

  Generating realistic conversational gestures are essential for achieving natural, socially engaging interactions with digital humans. However, existing methods typically map a single audio stream to a single speaker's motion, without considering social context or modeling the mutual dynamics between two people engaging in conversation. We present DyaDiT, a multi-modal diffusion transformer that generates contextually appropriate human motion from dyadic audio signals. Trained on Seamless Interaction Dataset, DyaDiT takes dyadic audio with optional social-context tokens to produce context-appropriate motion. It fuses information from both speakers to capture interaction dynamics, uses a motion dictionary to encode motion priors, and can optionally utilize the conversational partner's gestures to produce more responsive motion. We evaluate DyaDiT on standard motion generation metrics and conduct quantitative user studies, demonstrating that it not only surpasses existing methods on objective metrics but is also strongly preferred by users, highlighting its robustness and socially favorable motion generation. Code and models will be released upon acceptance.

  </details>



- **No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors**  
  Tao Liu, Gang Wan, Kan Ren, Shibo Wen  
  _2026-02-26_ · https://arxiv.org/abs/2602.23141v1  
  <details><summary>Abstract</summary>

  We propose a new unsupervised framework for online video stabilization. Unlike methods based on deep learning that require paired stable and unstable datasets, our approach instantiates the classical stabilization pipeline with three stages and incorporates a multithreaded buffering mechanism. This design addresses three longstanding challenges in end-to-end learning: limited data, poor controllability, and inefficiency on hardware with constrained resources. Existing benchmarks focus mainly on handheld videos with a forward view in visible light, which restricts the applicability of stabilization to domains such as UAV nighttime remote sensing. To fill this gap, we introduce a new multimodal UAV aerial video dataset (UAV-Test). Experiments show that our method consistently outperforms state-of-the-art online stabilizers in both quantitative metrics and visual quality, while achieving performance comparable to offline methods.

  </details>



- **Devling into Adversarial Transferability on Image Classification: Review, Benchmark, and Evaluation**  
  Xiaosen Wang, Zhijin Ge, Bohan Liu, Zheng Fang, Fengfan Zhou, Ruixuan Zhang, Shaokang Wang, Yuyang Luo  
  _2026-02-26_ · https://arxiv.org/abs/2602.23117v1  
  <details><summary>Abstract</summary>

  Adversarial transferability refers to the capacity of adversarial examples generated on the surrogate model to deceive alternate, unexposed victim models. This property eliminates the need for direct access to the victim model during an attack, thereby raising considerable security concerns in practical applications and attracting substantial research attention recently. In this work, we discern a lack of a standardized framework and criteria for evaluating transfer-based attacks, leading to potentially biased assessments of existing approaches. To rectify this gap, we have conducted an exhaustive review of hundreds of related works, organizing various transfer-based attacks into six distinct categories. Subsequently, we propose a comprehensive framework designed to serve as a benchmark for evaluating these attacks. In addition, we delineate common strategies that enhance adversarial transferability and highlight prevalent issues that could lead to unfair comparisons. Finally, we provide a brief review of transfer-based attacks beyond image classification.

  </details>



- **WARM-CAT: : Warm-Started Test-Time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning**  
  Xudong Yan, Songhe Feng, Jiaxin Wang, Xin Su, Yi Jin  
  _2026-02-26_ · https://arxiv.org/abs/2602.23114v1  
  <details><summary>Abstract</summary>

  Compositional Zero-Shot Learning (CZSL) aims to recognize novel attribute-object compositions based on the knowledge learned from seen ones. Existing methods suffer from performance degradation caused by the distribution shift of label space at test time, which stems from the inclusion of unseen compositions recombined from attributes and objects. To overcome the challenge, we propose a novel approach that accumulates comprehensive knowledge in both textual and visual modalities from unsupervised data to update multimodal prototypes at test time. Building on this, we further design an adaptive update weight to control the degree of prototype adjustment, enabling the model to flexibly adapt to distribution shift during testing. Moreover, a dynamic priority queue is introduced that stores high-confidence images to acquire visual prototypes from historical images for inference. Since the model tends to favor compositions already stored in the queue during testing, we warm-start the queue by initializing it with training images for visual prototypes of seen compositions and generating unseen visual prototypes using the mapping learned between seen and unseen textual prototypes. Considering the semantic consistency of multimodal knowledge, we align textual and visual prototypes by multimodal collaborative representation learning. To provide a more reliable evaluation for CZSL, we introduce a new benchmark dataset, C-Fashion, and refine the widely used but noisy MIT-States dataset. Extensive experiments indicate that our approach achieves state-of-the-art performance on four benchmark datasets under both closed-world and open-world settings. The source code and datasets are available at https://github.com/xud-yan/WARM-CAT .

  </details>



- **An Empirical Analysis of Cooperative Perception for Occlusion Risk Mitigation**  
  Aihong Wang, Tenghui Xie, Fuxi Wen, Jun Li  
  _2026-02-26_ · https://arxiv.org/abs/2602.23051v1  
  <details><summary>Abstract</summary>

  Occlusions present a significant challenge for connected and automated vehicles, as they can obscure critical road users from perception systems. Traditional risk metrics often fail to capture the cumulative nature of these threats over time adequately. In this paper, we propose a novel and universal risk assessment metric, the Risk of Tracking Loss (RTL), which aggregates instantaneous risk intensity throughout occluded periods. This provides a holistic risk profile that encompasses both high-intensity, short-term threats and prolonged exposure. Utilizing diverse and high-fidelity real-world datasets, a large-scale statistical analysis is conducted to characterize occlusion risk and validate the effectiveness of the proposed metric. The metric is applied to evaluate different vehicle-to-everything (V2X) deployment strategies. Our study shows that full V2X penetration theoretically eliminates this risk, the reduction is highly nonlinear; a substantial statistical benefit requires a high penetration threshold of 75-90%. To overcome this limitation, we propose a novel asymmetric communication framework that allows even non-connected vehicles to receive warnings. Experimental results demonstrate that this paradigm achieves better risk mitigation performance. We found that our approach at 25% penetration outperforms the traditional symmetric model at 75%, and benefits saturate at only 50% penetration. This work provides a crucial risk assessment metric and a cost-effective, strategic roadmap for accelerating the safety benefits of V2X deployment.

  </details>



- **D-FINE-seg: Object Detection and Instance Segmentation Framework with multi-backend deployment**  
  Argo Saakyan, Dmitry Solntsev  
  _2026-02-26_ · https://arxiv.org/abs/2602.23043v1  
  <details><summary>Abstract</summary>

  Transformer-based real-time object detectors achieve strong accuracy-latency trade-offs, and D-FINE is among the top-performing recent architectures. However, real-time instance segmentation with transformers is still less common. We present D-FINE-seg, an instance segmentation extension of D-FINE that adds: a lightweight mask head, segmentation-aware training, including box cropped BCE and dice mask losses, auxiliary and denoising mask supervision, and adapted Hungarian matching cost. On the TACO dataset, D-FINE-seg improves F1-score over Ultralytics YOLO26 under a unified TensorRT FP16 end-to-end benchmarking protocol, while maintaining competitive latency. Second contribution is an end-to-end pipeline for training, exporting, and optimized inference across ONNX, TensorRT, OpenVINO for both object detection and instance segmentation tasks. This framework is released as open-source under the Apache-2.0 license. GitHub repository - https://github.com/ArgoHA/D-FINE-seg.

  </details>



- **PackUV: Packed Gaussian UV Maps for 4D Volumetric Video**  
  Aashish Rai, Angela Xing, Anushka Agarwal, Xiaoyan Cong, Zekun Li, Tao Lu, Aayush Prakash, Srinath Sridhar  
  _2026-02-26_ · https://arxiv.org/abs/2602.23040v1  
  <details><summary>Abstract</summary>

  Volumetric videos offer immersive 4D experiences, but remain difficult to reconstruct, store, and stream at scale. Existing Gaussian Splatting based methods achieve high-quality reconstruction but break down on long sequences, temporal inconsistency, and fail under large motions and disocclusions. Moreover, their outputs are typically incompatible with conventional video coding pipelines, preventing practical applications. We introduce PackUV, a novel 4D Gaussian representation that maps all Gaussian attributes into a sequence of structured, multi-scale UV atlas, enabling compact, image-native storage. To fit this representation from multi-view videos, we propose PackUV-GS, a temporally consistent fitting method that directly optimizes Gaussian parameters in the UV domain. A flow-guided Gaussian labeling and video keyframing module identifies dynamic Gaussians, stabilizes static regions, and preserves temporal coherence even under large motions and disocclusions. The resulting UV atlas format is the first unified volumetric video representation compatible with standard video codecs (e.g., FFV1) without losing quality, enabling efficient streaming within existing multimedia infrastructure. To evaluate long-duration volumetric capture, we present PackUV-2B, the largest multi-view video dataset to date, featuring more than 50 synchronized cameras, substantial motion, and frequent disocclusions across 100 sequences and 2B (billion) frames. Extensive experiments demonstrate that our method surpasses existing baselines in rendering fidelity while scaling to sequences up to 30 minutes with consistent quality.

  </details>



- **Small Object Detection Model with Spatial Laplacian Pyramid Attention and Multi-Scale Features Enhancement in Aerial Images**  
  Zhangjian Ji, Huijia Yan, Shaotong Qiao, Kai Feng, Wei Wei  
  _2026-02-26_ · https://arxiv.org/abs/2602.23031v1  
  <details><summary>Abstract</summary>

  Detecting objects in aerial images confronts some significant challenges, including small size, dense and non-uniform distribution of objects over high-resolution images, which makes detection inefficient. Thus, in this paper, we proposed a small object detection algorithm based on a Spatial Laplacian Pyramid Attention and Multi-Scale Feature Enhancement in aerial images. Firstly, in order to improve the feature representation of ResNet-50 on small objects, we presented a novel Spatial Laplacian Pyramid Attention (SLPA) module, which is integrated after each stage of ResNet-50 to identify and emphasize important local regions. Secondly, to enhance the model's semantic understanding and features representation, we designed a Multi-Scale Feature Enhancement Module (MSFEM), which is incorporated into the lateral connections of C5 layer for building Feature Pyramid Network (FPN). Finally, the features representation quality of traditional feature pyramid network will be affected because the features are not aligned when the upper and lower layers are fused. In order to handle it, we utilized deformable convolutions to align the features in the fusion processing of the upper and lower levels of the Feature Pyramid Network, which can help enhance the model's ability to detect and recognize small objects. The extensive experimental results on two benchmark datasets: VisDrone and DOTA demonstrate that our improved model performs better for small object detection in aerial images compared to the original algorithm.

  </details>



- **DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis**  
  Xinglong Luo, Ao Luo, Zhengning Wang, Yueqi Yang, Chaoyu Feng, Lei Lei, Bing Zeng, Shuaicheng Liu  
  _2026-02-26_ · https://arxiv.org/abs/2602.23022v1  
  <details><summary>Abstract</summary>

  Image alignment is a fundamental task in computer vision with broad applications. Existing methods predominantly employ optical flow-based image warping. However, this technique is susceptible to common challenges such as occlusions and illumination variations, leading to degraded alignment visual quality and compromised accuracy in downstream tasks. In this paper, we present DMAligner, a diffusion-based framework for image alignment through alignment-oriented view synthesis. DMAligner is crafted to tackle the challenges in image alignment from a new perspective, employing a generation-based solution that showcases strong capabilities and avoids the problems associated with flow-based image warping. Specifically, we propose a Dynamics-aware Diffusion Training approach for learning conditional image generation, synthesizing a novel view for image alignment. This incorporates a Dynamics-aware Mask Producing (DMP) module to adaptively distinguish dynamic foreground regions from static backgrounds, enabling the diffusion model to more effectively handle challenges that classical methods struggle to solve. Furthermore, we develop the Dynamic Scene Image Alignment (DSIA) dataset using Blender, which includes 1,033 indoor and outdoor scenes with over 30K image pairs tailored for image alignment. Extensive experimental results demonstrate the superiority of the proposed approach on DSIA benchmarks, as well as on a series of widely-used video datasets for qualitative comparisons. Our code is available at https://github.com/boomluo02/DMAligner.

  </details>



- **SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling**  
  Camile Lendering, Erkut Akdag, Egor Bondarev  
  _2026-02-26_ · https://arxiv.org/abs/2602.23013v1  
  <details><summary>Abstract</summary>

  Detecting visual anomalies in industrial inspection often requires training with only a few normal images per category. Recent few-shot methods achieve strong results employing foundation-model features, but typically rely on memory banks, auxiliary datasets, or multi-modal tuning of vision-language models. We therefore question whether such complexity is necessary given the feature representations of vision foundation models. To answer this question, we introduce SubspaceAD, a training-free method, that operates in two simple stages. First, patch-level features are extracted from a small set of normal images by a frozen DINOv2 backbone. Second, a Principal Component Analysis (PCA) model is fit to these features to estimate the low-dimensional subspace of normal variations. At inference, anomalies are detected via the reconstruction residual with respect to this subspace, producing interpretable and statistically grounded anomaly scores. Despite its simplicity, SubspaceAD achieves state-of-the-art performance across one-shot and few-shot settings without training, prompt tuning, or memory banks. In the one-shot anomaly detection setting, SubspaceAD achieves image-level and pixel-level AUROC of 98.0% and 97.6% on the MVTec-AD dataset, and 93.3% and 98.3% on the VisA dataset, respectively, surpassing prior state-of-the-art results. Code and demo are available at https://github.com/CLendering/SubspaceAD.

  </details>



- **HELMLAB: An Analytical, Data-Driven Color Space for Perceptual Distance in UI Design Systems**  
  Gorkem Yildiz  
  _2026-02-26_ · https://arxiv.org/abs/2602.23010v1  
  <details><summary>Abstract</summary>

  We present HELMLAB, a 72-parameter analytical color space for UI design systems. The forward transform maps CIE XYZ to a perceptually-organized Lab representation through learned matrices, per-channel power compression, Fourier hue correction, and embedded Helmholtz-Kohlrausch lightness adjustment. A post-pipeline neutral correction guarantees that achromatic colors map to a=b=0 (chroma < 10^-6), and a rigid rotation of the chromatic plane improves hue-angle alignment without affecting the distance metric, which is invariant under isometries. On the COMBVD dataset (3,813 color pairs), HELMLAB achieves a STRESS of 23.22, a 20.4% reduction from CIEDE2000 (29.18). Cross-validation on He et al. 2022 and MacAdam 1974 shows competitive cross-dataset performance. The transform is invertible with round-trip errors below 10^-14. Gamut mapping, design-token export, and dark/light mode adaptation utilities are included for use in web and mobile design systems.

  </details>



- **An automatic counting algorithm for the quantification and uncertainty analysis of the number of microglial cells trainable in small and heterogeneous datasets**  
  L. Martino, M. M. Garcia, P. S. Paradas, E. Curbelo  
  _2026-02-26_ · https://arxiv.org/abs/2602.22974v1  
  <details><summary>Abstract</summary>

  Counting immunopositive cells on biological tissues generally requires either manual annotation or (when available) automatic rough systems, for scanning signal surface and intensity in whole slide imaging. In this work, we tackle the problem of counting microglial cells in lumbar spinal cord cross-sections of rats by omitting cell detection and focusing only on the counting task. Manual cell counting is, however, a time-consuming task and additionally entails extensive personnel training. The classic automatic color-based methods roughly inform about the total labeled area and intensity (protein quantification) but do not specifically provide information on cell number. Since the images to be analyzed have a high resolution but a huge amount of pixels contain just noise or artifacts, we first perform a pre-processing generating several filtered images {(providing a tailored, efficient feature extraction)}. Then, we design an automatic kernel counter that is a non-parametric and non-linear method. The proposed scheme can be easily trained in small datasets since, in its basic version, it relies only on one hyper-parameter. However, being non-parametric and non-linear, the proposed algorithm is flexible enough to express all the information contained in rich and heterogeneous datasets as well (providing the maximum overfit if required). Furthermore, the proposed kernel counter also provides uncertainty estimation of the given prediction, and can directly tackle the case of receiving several expert opinions over the same image. Different numerical experiments with artificial and real datasets show very promising results. Related Matlab code is also provided.

  </details>



- **Certified Circuits: Stability Guarantees for Mechanistic Circuits**  
  Alaa Anani, Tobias Lorenz, Bernt Schiele, Mario Fritz, Jonas Fischer  
  _2026-02-26_ · https://arxiv.org/abs/2602.22968v1  
  <details><summary>Abstract</summary>

  Understanding how neural networks arrive at their predictions is essential for debugging, auditing, and deployment. Mechanistic interpretability pursues this goal by identifying circuits - minimal subnetworks responsible for specific behaviors. However, existing circuit discovery methods are brittle: circuits depend strongly on the chosen concept dataset and often fail to transfer out-of-distribution, raising doubts whether they capture concept or dataset-specific artifacts. We introduce Certified Circuits, which provide provable stability guarantees for circuit discovery. Our framework wraps any black-box discovery algorithm with randomized data subsampling to certify that circuit component inclusion decisions are invariant to bounded edit-distance perturbations of the concept dataset. Unstable neurons are abstained from, yielding circuits that are more compact and more accurate. On ImageNet and OOD datasets, certified circuits achieve up to 91% higher accuracy while using 45% fewer neurons, and remain reliable where baselines degrade. Certified Circuits puts circuit discovery on formal ground by producing mechanistic explanations that are provably stable and better aligned with the target concept. Code will be released soon!

  </details>



- **Can Agents Distinguish Visually Hard-to-Separate Diseases in a Zero-Shot Setting? A Pilot Study**  
  Zihao Zhao, Frederik Hauke, Juliana De Castilhos, Sven Nebelung, Daniel Truhn  
  _2026-02-26_ · https://arxiv.org/abs/2602.22959v1  
  <details><summary>Abstract</summary>

  The rapid progress of multimodal large language models (MLLMs) has led to increasing interest in agent-based systems. While most prior work in medical imaging concentrates on automating routine clinical workflows, we study an underexplored yet clinically significant setting: distinguishing visually hard-to-separate diseases in a zero-shot setting. We benchmark representative agents on two imaging-only proxy diagnostic tasks, (1) melanoma vs. atypical nevus and (2) pulmonary edema vs. pneumonia, where visual features are highly confounded despite substantial differences in clinical management. We introduce a multi-agent framework based on contrastive adjudication. Experimental results show improved diagnostic performance (an 11-percentage-point gain in accuracy on dermoscopy data) and reduced unsupported claims on qualitative samples, although overall performance remains insufficient for clinical deployment. We acknowledge the inherent uncertainty in human annotations and the absence of clinical context, which further limit the translation to real-world settings. Within this controlled setting, this pilot study provides preliminary insights into zero-shot agent performance in visually confounded scenarios.

  </details>



- **MM-NeuroOnco: A Multimodal Benchmark and Instruction Dataset for MRI-Based Brain Tumor Diagnosis**  
  Feng Guo, Jiaxiang Liu, Yang Li, Qianqian Shi, Mingkun Xu  
  _2026-02-26_ · https://arxiv.org/abs/2602.22955v1  
  <details><summary>Abstract</summary>

  Accurate brain tumor diagnosis requires models to not only detect lesions but also generate clinically interpretable reasoning grounded in imaging manifestations, yet existing public datasets remain limited in annotation richness and diagnostic semantics. To bridge this gap, we introduce MM-NeuroOnco, a large-scale multimodal benchmark and instruction-tuning dataset for brain tumor MRI understanding, consisting of 24,726 MRI slices from 20 data sources paired with approximately 200,000 semantically enriched multimodal instructions spanning diverse tumor subtypes and imaging modalities. To mitigate the scarcity and high cost of diagnostic semantic annotations, we develop a multi-model collaborative pipeline for automated medical information completion and quality control, enabling the generation of diagnosis-related semantics beyond mask-only annotations. Building upon this dataset, we further construct MM-NeuroOnco-Bench, a manually annotated evaluation benchmark with a rejection-aware setting to reduce biases inherent in closed-ended question formats. Evaluation across ten representative models shows that even the strongest baseline, Gemini 3 Flash, achieves only 41.88% accuracy on diagnosis-related questions, highlighting the substantial challenges of multimodal brain tumor diagnostic understanding. Leveraging MM-NeuroOnco, we further propose NeuroOnco-GPT, which achieves a 27% absolute accuracy improvement on diagnostic questions following fine-tuning. This result demonstrates the effectiveness of our dataset and benchmark in advancing clinically grounded multimodal diagnostic reasoning. Code and dataset are publicly available at: https://github.com/gfnnnb/MM-NeuroOnco

  </details>



- **OpenFS: Multi-Hand-Capable Fingerspelling Recognition with Implicit Signing-Hand Detection and Frame-Wise Letter-Conditioned Synthesis**  
  Junuk Cha, Jihyeon Kim, Han-Mu Park  
  _2026-02-26_ · https://arxiv.org/abs/2602.22949v1  
  <details><summary>Abstract</summary>

  Fingerspelling is a component of sign languages in which words are spelled out letter by letter using specific hand poses. Automatic fingerspelling recognition plays a crucial role in bridging the communication gap between Deaf and hearing communities, yet it remains challenging due to the signing-hand ambiguity issue, the lack of appropriate training losses, and the out-of-vocabulary (OOV) problem. Prior fingerspelling recognition methods rely on explicit signing-hand detection, which often leads to recognition failures, and on a connectionist temporal classification (CTC) loss, which exhibits the peaky behavior problem. To address these issues, we develop OpenFS, an open-source approach for fingerspelling recognition and synthesis. We propose a multi-hand-capable fingerspelling recognizer that supports both single- and multi-hand inputs and performs implicit signing-hand detection by incorporating a dual-level positional encoding and a signing-hand focus (SF) loss. The SF loss encourages cross-attention to focus on the signing hand, enabling implicit signing-hand detection during recognition. Furthermore, without relying on the CTC loss, we introduce a monotonic alignment (MA) loss that enforces the output letter sequence to follow the temporal order of the input pose sequence through cross-attention regularization. In addition, we propose a frame-wise letter-conditioned generator that synthesizes realistic fingerspelling pose sequences for OOV words. This generator enables the construction of a new synthetic benchmark, called FSNeo. Through comprehensive experiments, we demonstrate that our approach achieves state-of-the-art performance in recognition and validate the effectiveness of the proposed recognizer and generator. Codes and data are available in: https://github.com/JunukCha/OpenFS.

  </details>



- **Velocity and stroke rate reconstruction of canoe sprint team boats based on panned and zoomed video recordings**  
  Julian Ziegler, Daniel Matthes, Finn Gerdts, Patrick Frenzel, Torsten Warnke, Matthias Englert, Tina Koevari, Mirco Fuchs  
  _2026-02-26_ · https://arxiv.org/abs/2602.22941v1  
  <details><summary>Abstract</summary>

  Pacing strategies, defined by velocity and stroke rate profiles, are essential for peak performance in canoe sprint. While GPS is the gold standard for analysis, its limited availability necessitates automated video-based solutions. This paper presents an extended framework for reconstructing performance metrics from panned and zoomed video recordings across all sprint disciplines (K1-K4, C1-C2) and distances (200m-500m). Our method utilizes YOLOv8 for buoy and athlete detection, leveraging the known buoy grid to estimate homographies. We generalized the estimation of the boat position by means of learning a boat-specific athlete offset using a U-net based boat tip calibration. Further, we implement a robust tracking scheme using optical flow to adapt to multi-athlete boat types. Finally, we introduce methods to extract stroke rate information from either pose estimations or the athlete bounding boxes themselves. Evaluation against GPS data from elite competitions yields a velocity RRMSE of 0.020 +- 0.011 (rho = 0.956) and a stroke rate RRMSE of 0.022 +- 0.024 (rho = 0.932). The methods provide coaches with highly accurate, automated feedback without requiring on-boat sensors or manual annotation.

  </details>



- **MSJoE: Jointly Evolving MLLM and Sampler for Efficient Long-Form Video Understanding**  
  Wenhui Tan, Xiaoyi Yu, Jiaze Li, Yijing Chen, Jianzhong Ju, Zhenbo Luo, Ruihua Song, Jian Luan  
  _2026-02-26_ · https://arxiv.org/abs/2602.22932v1  
  <details><summary>Abstract</summary>

  Efficiently understanding long-form videos remains a fundamental challenge for multimodal large language models (MLLMs). In this paper, we present MLLM-Sampler Joint Evolution (MSJoE), a novel framework that jointly evolves the MLLM and a lightweight key-frame sampler for efficient long-form video understanding. MSJoE builds upon a key assumption that only a small subset of key-frames is truly informative for answering each question to a video. Specifically, MSJoE first reasons out several queries, which describe diverse visual perspectives relevant to the question. Then, these queries interact with a frozen CLIP model to produce a query-frame similarity matrix. Finally, a lightweight sampler predicts key-frame sampling weights from this matrix, selecting a compact set of informative frames, which are then fed into the MLLM for answer generation. Both the MLLM and sampler are jointly optimized through reinforcement learning, enabling co-adaptation of query-reasoning, frame-sampling, and key-frame understanding. A new long-video QA dataset containing 2.8K videos with 7K question-answer pairs is collected to support the training process. Extensive experiments on VideoMME, LongVideoBench, LVBench, and MLVU show that MSJoE achieves 8.0\% accuracy gain upon the base MLLM, and 1.1\% higher accuracy than strongest baseline method.

  </details>



- **WaterVideoQA: ASV-Centric Perception and Rule-Compliant Reasoning via Multi-Modal Agents**  
  Runwei Guan, Shaofeng Liang, Ningwei Ouyang, Weichen Fei, Shanliang Yao, Wei Dai, Chenhao Ge, Penglei Sun, Xiaohui Zhu, Tao Huang, et al.  
  _2026-02-26_ · https://arxiv.org/abs/2602.22923v1  
  <details><summary>Abstract</summary>

  While autonomous navigation has achieved remarkable success in passive perception (e.g., object detection and segmentation), it remains fundamentally constrained by a void in knowledge-driven, interactive environmental cognition. In the high-stakes domain of maritime navigation, the ability to bridge the gap between raw visual perception and complex cognitive reasoning is not merely an enhancement but a critical prerequisite for Autonomous Surface Vessels to execute safe and precise maneuvers. To this end, we present WaterVideoQA, the first large-scale, comprehensive Video Question Answering benchmark specifically engineered for all-waterway environments. This benchmark encompasses 3,029 video clips across six distinct waterway categories, integrating multifaceted variables such as volatile lighting and dynamic weather to rigorously stress-test ASV capabilities across a five-tier hierarchical cognitive framework. Furthermore, we introduce NaviMind, a pioneering multi-agent neuro-symbolic system designed for open-ended maritime reasoning. By synergizing Adaptive Semantic Routing, Situation-Aware Hierarchical Reasoning, and Autonomous Self-Reflective Verification, NaviMind transitions ASVs from superficial pattern matching to regulation-compliant, interpretable decision-making. Experimental results demonstrate that our framework significantly transcends existing baselines, establishing a new paradigm for intelligent, trustworthy interaction in dynamic maritime environments.

  </details>



- **Bayesian Preference Elicitation: Human-In-The-Loop Optimization of An Active Prosthesis**  
  Sophia Taddei, Wouter Koppen, Eligia Alfio, Stefano Nuzzo, Louis Flynn, Maria Alejandra Diaz, Sebastian Rojas Gonzalez, Tom Dhaene, Kevin De Pauw, Ivo Couckuyt, et al.  
  _2026-02-26_ · https://arxiv.org/abs/2602.22922v1  
  <details><summary>Abstract</summary>

  Tuning active prostheses for people with amputation is time-consuming and relies on metrics that may not fully reflect user needs. We introduce a human-in-the-loop optimization (HILO) approach that leverages direct user preferences to personalize a standard four-parameter prosthesis controller efficiently. Our method employs preference-based Multiobjective Bayesian Optimization that uses a state-or-the-art acquisition function especially designed for preference learning, and includes two algorithmic variants: a discrete version (\textit{EUBO-LineCoSpar}), and a continuous version (\textit{BPE4Prost}). Simulation results on benchmark functions and real-application trials demonstrate efficient convergence, robust preference elicitation, and measurable biomechanical improvements, illustrating the potential of preference-driven tuning for user-centered prosthesis control.

  </details>



- **OSDaR-AR: Enhancing Railway Perception Datasets via Multi-modal Augmented Reality**  
  Federico Nesti, Gianluca D'Amico, Mauro Marinoni, Giorgio Buttazzo  
  _2026-02-26_ · https://arxiv.org/abs/2602.22920v1  
  <details><summary>Abstract</summary>

  Although deep learning has significantly advanced the perception capabilities of intelligent transportation systems, railway applications continue to suffer from a scarcity of high-quality, annotated data for safety-critical tasks like obstacle detection. While photorealistic simulators offer a solution, they often struggle with the ``sim-to-real" gap; conversely, simple image-masking techniques lack the spatio-temporal coherence required to obtain augmented single- and multi-frame scenes with the correct appearance and dimensions. This paper introduces a multi-modal augmented reality framework designed to bridge this gap by integrating photorealistic virtual objects into real-world railway sequences from the OSDaR23 dataset. Utilizing Unreal Engine 5 features, our pipeline leverages LiDAR point-clouds and INS/GNSS data to ensure accurate object placement and temporal stability across RGB frames. This paper also proposes a segmentation-based refinement strategy for INS/GNSS data to significantly improve the realism of the augmented sequences, as confirmed by the comparative study presented in the paper. Carefully designed augmented sequences are collected to produce OSDaR-AR, a public dataset designed to support the development of next-generation railway perception systems. The dataset is available at the following page: https://syndra.retis.santannapisa.it/osdarar.html

  </details>



- **Towards Multimodal Domain Generalization with Few Labels**  
  Hongzhao Li, Hao Dong, Hualei Wan, Shupan Li, Mingliang Xu, Muhammad Haris Khan  
  _2026-02-26_ · https://arxiv.org/abs/2602.22917v1  
  <details><summary>Abstract</summary>

  Multimodal models ideally should generalize to unseen domains while remaining data-efficient to reduce annotation costs. To this end, we introduce and study a new problem, Semi-Supervised Multimodal Domain Generalization (SSMDG), which aims to learn robust multimodal models from multi-source data with few labeled samples. We observe that existing approaches fail to address this setting effectively: multimodal domain generalization methods cannot exploit unlabeled data, semi-supervised multimodal learning methods ignore domain shifts, and semi-supervised domain generalization methods are confined to single-modality inputs. To overcome these limitations, we propose a unified framework featuring three key components: Consensus-Driven Consistency Regularization, which obtains reliable pseudo-labels through confident fused-unimodal consensus; Disagreement-Aware Regularization, which effectively utilizes ambiguous non-consensus samples; and Cross-Modal Prototype Alignment, which enforces domain- and modality-invariant representations while promoting robustness under missing modalities via cross-modal translation. We further establish the first SSMDG benchmarks, on which our method consistently outperforms strong baselines in both standard and missing-modality scenarios. Our benchmarks and code are available at https://github.com/lihongzhao99/SSMDG.

  </details>



- **OmniGAIA: Towards Native Omni-Modal AI Agents**  
  Xiaoxi Li, Wenxiang Jiao, Jiarui Jin, Shijian Wang, Guanting Dong, Jiajie Jin, Hao Wang, Yinuo Wang, Ji-Rong Wen, Yuan Lu, et al.  
  _2026-02-26_ · https://arxiv.org/abs/2602.22897v1  
  <details><summary>Abstract</summary>

  Human intelligence naturally intertwines omni-modal perception -- spanning vision, audio, and language -- with complex reasoning and tool usage to interact with the world. However, current multi-modal LLMs are primarily confined to bi-modal interactions (e.g., vision-language), lacking the unified cognitive capabilities required for general AI assistants. To bridge this gap, we introduce OmniGAIA, a comprehensive benchmark designed to evaluate omni-modal agents on tasks necessitating deep reasoning and multi-turn tool execution across video, audio, and image modalities. Constructed via a novel omni-modal event graph approach, OmniGAIA synthesizes complex, multi-hop queries derived from real-world data that require cross-modal reasoning and external tool integration. Furthermore, we propose OmniAtlas, a native omni-modal foundation agent under tool-integrated reasoning paradigm with active omni-modal perception. Trained on trajectories synthesized via a hindsight-guided tree exploration strategy and OmniDPO for fine-grained error correction, OmniAtlas effectively enhances the tool-use capabilities of existing open-source models. This work marks a step towards next-generation native omni-modal AI assistants for real-world scenarios.

  </details>



- **DySL-VLA: Efficient Vision-Language-Action Model Inference via Dynamic-Static Layer-Skipping for Robot Manipulation**  
  Zebin Yang, Yijiahao Qi, Tong Xie, Bo Yu, Shaoshan Liu, Meng Li  
  _2026-02-26_ · https://arxiv.org/abs/2602.22896v1  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models have shown remarkable success in robotic tasks like manipulation by fusing a language model's reasoning with a vision model's 3D understanding. However, their high computational cost remains a major obstacle for real-world applications that require real-time performance. We observe that the actions within a task have varying levels of importance: critical steps demand high precision, while less important ones can tolerate more variance. Leveraging this insight, we propose DySL-VLA, a novel framework that addresses computational cost by dynamically skipping VLA layers based on each action's importance. DySL-VLA categorizes its layers into two types: informative layers, which are consistently executed, and incremental layers, which can be selectively skipped. To intelligently skip layers without sacrificing accuracy, we invent a prior-post skipping guidance mechanism to determine when to initiate layer-skipping. We also propose a skip-aware two-stage knowledge distillation algorithm to efficiently train a standard VLA into a DySL-VLA. Our experiments indicate that DySL-VLA achieves 2.1% improvement in success length over Deer-VLA on the Calvin dataset, while simultaneously reducing trainable parameters by a factor of 85.7 and providing a 3.75x speedup relative to the RoboFlamingo baseline at iso-accuracy. Our code is available on https://github.com/PKU-SEC-Lab/DYSL_VLA.

  </details>



- **SO3UFormer: Learning Intrinsic Spherical Features for Rotation-Robust Panoramic Segmentation**  
  Qinfeng Zhu, Yunxi Jiang, Lei Fan  
  _2026-02-26_ · https://arxiv.org/abs/2602.22867v1  
  <details><summary>Abstract</summary>

  Panoramic semantic segmentation models are typically trained under a strict gravity-aligned assumption. However, real-world captures often deviate from this canonical orientation due to unconstrained camera motions, such as the rotational jitter of handheld devices or the dynamic attitude shifts of aerial platforms. This discrepancy causes standard spherical Transformers to overfit global latitude cues, leading to performance collapse under 3D reorientations. To address this, we introduce SO3UFormer, a rotation-robust architecture designed to learn intrinsic spherical features that are less sensitive to the underlying coordinate frame. Our approach rests on three geometric pillars: (1) an intrinsic feature formulation that decouples the representation from the gravity vector by removing absolute latitude encoding; (2) quadrature-consistent spherical attention that accounts for non-uniform sampling densities; and (3) a gauge-aware relative positional mechanism that encodes local angular geometry using tangent-plane projected angles and discrete gauge pooling, avoiding reliance on global axes. We further use index-based spherical resampling together with a logit-level SO(3)-consistency regularizer during training. To rigorously benchmark robustness, we introduce Pose35, a dataset variant of Stanford2D3D perturbed by random rotations within $\pm 35^\circ$. Under the extreme test of arbitrary full SO(3) rotations, existing SOTAs fail catastrophically: the baseline SphereUFormer drops from 67.53 mIoU to 25.26 mIoU. In contrast, SO3UFormer demonstrates remarkable stability, achieving 72.03 mIoU on Pose35 and retaining 70.67 mIoU under full SO(3) rotations.

  </details>



- **A data- and compute-efficient chest X-ray foundation model beyond aggressive scaling**  
  Chong Wang, Yabin Zhang, Yunhe Gao, Maya Varma, Clemence Mottez, Faidra Patsatzi, Jiaming Liu, Jin Long, Jean-Benoit Delbrouck, Sergios Gatidis, et al.  
  _2026-02-26_ · https://arxiv.org/abs/2602.22843v1  
  <details><summary>Abstract</summary>

  Foundation models for medical imaging are typically pretrained on increasingly large datasets, following a "scale-at-all-costs" paradigm. However, this strategy faces two critical challenges: large-scale medical datasets often contain substantial redundancy and severe class imbalance that bias representation learning toward over-represented patterns, and indiscriminate training regardless of heterogeneity in data quality incurs considerable computational inefficiency. Here we demonstrate that active, principled data curation during pretraining can serve as a viable, cost-effective alternative to brute-force dataset enlargement. We introduce CheXficient, a chest X-ray (CXR) foundation model that selectively prioritizes informative training samples. CheXficient is pretrained on only 22.7% of 1,235,004 paired CXR images and reports while consuming under 27.3% of the total compute budget, yet achieving comparable or superior performance to its full-data counterpart and other large-scale pretrained models. We assess CheXficient across 20 individual benchmarks spanning 5 task types, including non-adapted off-the-shelf evaluations (zero-shot findings classification and crossmodal retrieval) and adapted downstream tasks (disease prediction, semantic segmentation, and radiology report generation). Further analyses show that CheXficient systematically prioritizes under-represented training samples, improving generalizability on long-tailed or rare conditions. Overall, our work offers practical insights into the data and computation demands for efficient pretraining and downstream adaptation of medical vision-language foundation models.

  </details>



- **CMSA-Net: Causal Multi-scale Aggregation with Adaptive Multi-source Reference for Video Polyp Segmentation**  
  Tong Wang, Yaolei Qi, Siwen Wang, Imran Razzak, Guanyu Yang, Yutong Xie  
  _2026-02-26_ · https://arxiv.org/abs/2602.22821v1  
  <details><summary>Abstract</summary>

  Video polyp segmentation (VPS) is an important task in computer-aided colonoscopy, as it helps doctors accurately locate and track polyps during examinations. However, VPS remains challenging because polyps often look similar to surrounding mucosa, leading to weak semantic discrimination. In addition, large changes in polyp position and scale across video frames make stable and accurate segmentation difficult. To address these challenges, we propose a robust VPS framework named CMSA-Net. The proposed network introduces a Causal Multi-scale Aggregation (CMA) module to effectively gather semantic information from multiple historical frames at different scales. By using causal attention, CMA ensures that temporal feature propagation follows strict time order, which helps reduce noise and improve feature reliability. Furthermore, we design a Dynamic Multi-source Reference (DMR) strategy that adaptively selects informative and reliable reference frames based on semantic separability and prediction confidence. This strategy provides strong multi-frame guidance while keeping the model efficient for real-time inference. Extensive experiments on the SUN-SEG dataset demonstrate that CMSA-Net achieves state-of-the-art performance, offering a favorable balance between segmentation accuracy and real-time clinical applicability.

  </details>



- **Face Time Traveller : Travel Through Ages Without Losing Identity**  
  Purbayan Kar, Ayush Ghadiya, Vishal Chudasama, Pankaj Wasnik, C. V. Jawahar  
  _2026-02-26_ · https://arxiv.org/abs/2602.22819v1  
  <details><summary>Abstract</summary>

  Face aging, an ill-posed problem shaped by environmental and genetic factors, is vital in entertainment, forensics, and digital archiving, where realistic age transformations must preserve both identity and visual realism. However, existing works relying on numerical age representations overlook the interplay of biological and contextual cues. Despite progress in recent face aging models, they struggle with identity preservation in wide age transformations, also static attention and optimization-heavy inversion in diffusion limit adaptability, fine-grained control and background consistency. To address these challenges, we propose Face Time Traveller (FaceTT), a diffusion-based framework that achieves high-fidelity, identity-consistent age transformation. Here, we introduce a Face-Attribute-Aware Prompt Refinement strategy that encodes intrinsic (biological) and extrinsic (environmental) aging cues for context-aware conditioning. A tuning-free Angular Inversion method is proposed that efficiently maps real faces into the diffusion latent space for fast and accurate reconstruction. Moreover, an Adaptive Attention Control mechanism is introduced that dynamically balances cross-attention for semantic aging cues and self-attention for structural and identity preservation. Extensive experiments on benchmark datasets and in-the-wild testset demonstrate that FaceTT achieves superior identity retention, background preservation and aging realism over state-of-the-art (SOTA) methods.

  </details>



- **LeRobot: An Open-Source Library for End-to-End Robot Learning**  
  Remi Cadene, Simon Aliberts, Francesco Capuano, Michel Aractingi, Adil Zouitine, Pepijn Kooijmans, Jade Choghari, Martino Russi, Caroline Pascal, Steven Palma, et al.  
  _2026-02-26_ · https://arxiv.org/abs/2602.22818v1  
  <details><summary>Abstract</summary>

  Robotics is undergoing a significant transformation powered by advances in high-level control techniques based on machine learning, giving rise to the field of robot learning. Recent progress in robot learning has been accelerated by the increasing availability of affordable teleoperation systems, large-scale openly available datasets, and scalable learning-based methods. However, development in the field of robot learning is often slowed by fragmented, closed-source tools designed to only address specific sub-components within the robotics stack. In this paper, we present \texttt{lerobot}, an open-source library that integrates across the entire robot learning stack, from low-level middleware communication for motor controls to large-scale dataset collection, storage and streaming. The library is designed with a strong focus on real-world robotics, supporting accessible hardware platforms while remaining extensible to new embodiments. It also supports efficient implementations for various state-of-the-art robot learning algorithms from multiple prominent paradigms, as well as a generalized asynchronous inference stack. Unlike traditional pipelines which heavily rely on hand-crafted techniques, \texttt{lerobot} emphasizes scalable learning approaches that improve directly with more data and compute. Designed for accessibility, scalability, and openness, \texttt{lerobot} lowers the barrier to entry for researchers and practitioners to robotics while providing a platform for reproducible, state-of-the-art robot learning.

  </details>



- **PhotoAgent: Agentic Photo Editing with Exploratory Visual Aesthetic Planning**  
  Mingde Yao, Zhiyuan You, Tam-King Man, Menglu Wang, Tianfan Xue  
  _2026-02-26_ · https://arxiv.org/abs/2602.22809v1  
  <details><summary>Abstract</summary>

  With the recent fast development of generative models, instruction-based image editing has shown great potential in generating high-quality images. However, the quality of editing highly depends on carefully designed instructions, placing the burden of task decomposition and sequencing entirely on the user. To achieve autonomous image editing, we present PhotoAgent, a system that advances image editing through explicit aesthetic planning. Specifically, PhotoAgent formulates autonomous image editing as a long-horizon decision-making problem. It reasons over user aesthetic intent, plans multi-step editing actions via tree search, and iteratively refines results through closed-loop execution with memory and visual feedback, without requiring step-by-step user prompts. To support reliable evaluation in real-world scenarios, we introduce UGC-Edit, an aesthetic evaluation benchmark consisting of 7,000 photos and a learned aesthetic reward model. We also construct a test set containing 1,017 photos to systematically assess autonomous photo editing performance. Extensive experiments demonstrate that PhotoAgent consistently improves both instruction adherence and visual quality compared with baseline methods. The project page is https://github.com/mdyao/PhotoAgent.

  </details>



- **GSTurb: Gaussian Splatting for Atmospheric Turbulence Mitigation**  
  Hanliang Du, Zhangji Lu, Zewei Cai, Qijian Tang, Qifeng Yu, Xiaoli Liu  
  _2026-02-26_ · https://arxiv.org/abs/2602.22800v1  
  <details><summary>Abstract</summary>

  Atmospheric turbulence causes significant image degradation due to pixel displacement (tilt) and blur, particularly in long-range imaging applications. In this paper, we propose a novel framework for atmospheric turbulence mitigation, GSTurb, which integrates optical flow-guided tilt correction and Gaussian splatting for modeling non-isoplanatic blur. The framework employs Gaussian parameters to represent tilt and blur, and optimizes them across multiple frames to enhance restoration. Experimental results on the ATSyn-static dataset demonstrate the effectiveness of our method, achieving a peak PSNR of 27.67 dB and SSIM of 0.8735. Compared to the state-of-the-art method, GSTurb improves PSNR by 1.3 dB (a 4.5% increase) and SSIM by 0.048 (a 5.8% increase). Additionally, on real datasets, including the TSRWGAN Real-World and CLEAR datasets, GSTurb outperforms existing methods, showing significant improvements in both qualitative and quantitative performance. These results highlight that combining optical flow-guided tilt correction with Gaussian splatting effectively enhances image restoration under both synthetic and real-world turbulence conditions. The code for this method will be available at https://github.com/DuhlLiamz/3DGS_turbulence/tree/main.

  </details>



- **Beyond Detection: Multi-Scale Hidden-Code for Natural Image Deepfake Recovery and Factual Retrieval**  
  Yuan-Chih Chen, Chun-Shien Lu  
  _2026-02-26_ · https://arxiv.org/abs/2602.22759v1  
  <details><summary>Abstract</summary>

  Recent advances in image authenticity have primarily focused on deepfake detection and localization, leaving recovery of tampered contents for factual retrieval relatively underexplored. We propose a unified hidden-code recovery framework that enables both retrieval and restoration from post-hoc and in-generation watermarking paradigms. Our method encodes semantic and perceptual information into a compact hidden-code representation, refined through multi-scale vector quantization, and enhances contextual reasoning via conditional Transformer modules. To enable systematic evaluation for natural images, we construct ImageNet-S, a benchmark that provides paired image-label factual retrieval tasks. Extensive experiments on ImageNet-S demonstrate that our method exhibits promising retrieval and reconstruction performance while remaining fully compatible with diverse watermarking pipelines. This framework establishes a foundation for general-purpose image recovery beyond detection and localization.

  </details>



- **SPATIALALIGN: Aligning Dynamic Spatial Relationships in Video Generation**  
  Fengming Liu, Tat-Jen Cham, Chuanxia Zheng  
  _2026-02-26_ · https://arxiv.org/abs/2602.22745v1  
  <details><summary>Abstract</summary>

  Most text-to-video (T2V) generators prioritize aesthetic quality, but often ignoring the spatial constraints in the generated videos. In this work, we present SPATIALALIGN, a self-improvement framework that enhances T2V models capabilities to depict Dynamic Spatial Relationships (DSR) specified in text prompts. We present a zeroth-order regularized Direct Preference Optimization (DPO) to fine-tune T2V models towards better alignment with DSR. Specifically, we design DSR-SCORE, a geometry-based metric that quantitatively measures the alignment between generated videos and the specified DSRs in prompts, which is a step forward from prior works that rely on VLM for evaluation. We also conduct a dataset of text-video pairs with diverse DSRs to facilitate the study. Extensive experiments demonstrate that our fine-tuned model significantly out performs the baseline in spatial relationships. The code will be released in Link.

  </details>



- **SUPERGLASSES: Benchmarking Vision Language Models as Intelligent Agents for AI Smart Glasses**  
  Zhuohang Jiang, Xu Yuan, Haohao Qu, Shanru Lin, Kanglong Liu, Wenqi Fan, Qing Li  
  _2026-02-26_ · https://arxiv.org/abs/2602.22683v1  
  <details><summary>Abstract</summary>

  The rapid advancement of AI-powered smart glasses, one of the hottest wearable devices, has unlocked new frontiers for multimodal interaction, with Visual Question Answering (VQA) over external knowledge sources emerging as a core application. Existing Vision Language Models (VLMs) adapted to smart glasses are typically trained and evaluated on traditional multimodal datasets; however, these datasets lack the variety and realism needed to reflect smart glasses usage scenarios and diverge from their specific challenges, where accurately identifying the object of interest must precede any external knowledge retrieval. To bridge this gap, we introduce SUPERGLASSES, the first comprehensive VQA benchmark built on real-world data entirely collected by smart glasses devices. SUPERGLASSES comprises 2,422 egocentric image-question pairs spanning 14 image domains and 8 query categories, enriched with full search trajectories and reasoning annotations. We evaluate 26 representative VLMs on this benchmark, revealing significant performance gaps. To address the limitations of existing models, we further propose SUPERLENS, a multimodal smart glasses agent that enables retrieval-augmented answer generation by integrating automatic object detection, query decoupling, and multimodal web search. Our agent achieves state-of-the-art performance, surpassing GPT-4o by 2.19 percent, and highlights the need for task-specific solutions in smart glasses VQA scenarios.

  </details>



- **SPMamba-YOLO: An Underwater Object Detection Network Based on Multi-Scale Feature Enhancement and Global Context Modeling**  
  Guanghao Liao, Zhen Liu, Liyuan Cao, Yonghui Yang, Qi Li  
  _2026-02-26_ · https://arxiv.org/abs/2602.22674v1  
  <details><summary>Abstract</summary>

  Underwater object detection is a critical yet challenging research problem owing to severe light attenuation, color distortion, background clutter, and the small scale of underwater targets. To address these challenges, we propose SPMamba-YOLO, a novel underwater object detection network that integrates multi-scale feature enhancement with global context modeling. Specifically, a Spatial Pyramid Pooling Enhanced Layer Aggregation Network (SPPELAN) module is introduced to strengthen multi-scale feature aggregation and expand the receptive field, while a Pyramid Split Attention (PSA) mechanism enhances feature discrimination by emphasizing informative regions and suppressing background interference. In addition, a Mamba-based state space modeling module is incorporated to efficiently capture long-range dependencies and global contextual information, thereby improving detection robustness in complex underwater environments. Extensive experiments on the URPC2022 dataset demonstrate that SPMamba-YOLO outperforms the YOLOv8n baseline by more than 4.9\% in mAP@0.5, particularly for small and densely distributed underwater objects, while maintaining a favorable balance between detection accuracy and computational cost.

  </details>



- **Rethinking the Practicality of Vision-language-action Model: A Comprehensive Benchmark and An Improved Baseline**  
  Wenxuan Song, Jiayi Chen, Xiaoquan Sun, Huashuo Lei, Yikai Qin, Wei Zhao, Pengxiang Ding, Han Zhao, Tongxin Wang, Pengxu Hou, et al.  
  _2026-02-26_ · https://arxiv.org/abs/2602.22663v1  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models have emerged as a generalist robotic agent. However, existing VLAs are hindered by excessive parameter scales, prohibitive pre-training requirements, and limited applicability to diverse embodiments. To improve the practicality of VLAs, we propose a comprehensive benchmark and an improved baseline. First, we propose CEBench, a new benchmark spanning diverse embodiments in both simulation and the real world with consideration of domain randomization. We collect 14.4k simulated trajectories and 1.6k real-world expert-curated trajectories to support training on CEBench. Second, using CEBench as our testbed, we study three critical aspects of VLAs' practicality and offer several key findings. Informed by these findings, we introduce LLaVA-VLA, a lightweight yet powerful VLA designed for practical deployment on consumer-grade GPUs. Architecturally, it integrates a compact VLM backbone with multi-view perception, proprioceptive tokenization, and action chunking. To eliminate reliance on costly pre-training, LLaVA-VLA adopts a two-stage training paradigm including post-training and fine-tuning. Furthermore, LLaVA-VLA extends the action space to unify navigation and manipulation. Experiments across embodiments demonstrate the capabilities of generalization and versatility of LLaVA-VLA , while real-world mobile manipulation experiments establish it as the first end-to-end VLA model for mobile manipulation. We will open-source all datasets, codes, and checkpoints upon acceptance to foster reproducibility and future research.

  </details>



- **Scaling Audio-Visual Quality Assessment Dataset via Crowdsourcing**  
  Renyu Yang, Jian Jin, Lili Meng, Meiqin Liu, Yilin Wang, Balu Adsumilli, Weisi Lin  
  _2026-02-26_ · https://arxiv.org/abs/2602.22659v1  
  <details><summary>Abstract</summary>

  Audio-visual quality assessment (AVQA) research has been stalled by limitations of existing datasets: they are typically small in scale, with insufficient diversity in content and quality, and annotated only with overall scores. These shortcomings provide limited support for model development and multimodal perception research. We propose a practical approach for AVQA dataset construction. First, we design a crowdsourced subjective experiment framework for AVQA, breaks the constraints of in-lab settings and achieves reliable annotation across varied environments. Second, a systematic data preparation strategy is further employed to ensure broad coverage of both quality levels and semantic scenarios. Third, we extend the dataset with additional annotations, enabling research on multimodal perception mechanisms and their relation to content. Finally, we validate this approach through YT-NTU-AVQ, the largest and most diverse AVQA dataset to date, consisting of 1,620 user-generated audio and video (A/V) sequences. The dataset and platform code are available at https://github.com/renyu12/YT-NTU-AVQ

  </details>



- **Interactive Medical-SAM2 GUI: A Napari-based semi-automatic annotation tool for medical images**  
  Woojae Hong, Jong Ha Hwang, Jiyong Chung, Joongyeon Choi, Hyunngun Kim, Yong Hwy Kim  
  _2026-02-26_ · https://arxiv.org/abs/2602.22649v1  
  <details><summary>Abstract</summary>

  Interactive Medical-SAM2 GUI is an open-source desktop application for semi-automatic annotation of 2D and 3D medical images. Built on the Napari multi-dimensional viewer, box/point prompting is integrated with SAM2-style propagation by treating a 3D volume as a slice sequence, enabling mask propagation from sparse prompts using Medical-SAM2 on top of SAM2. Voxel-level annotation remains essential for developing and validating medical imaging algorithms, yet manual labeling is slow and expensive for 3D scans, and existing integrations frequently emphasize per-slice interaction without providing a unified, cohort-oriented workflow for navigation, propagation, interactive correction, and quantitative export in a single local pipeline. To address this practical limitation, a local-first Napari workflow is provided for efficient 3D annotation across multiple studies using standard DICOM series and/or NIfTI volumes. Users can annotate cases sequentially under a single root folder with explicit proceed/skip actions, initialize objects via box-first prompting (including first/last-slice initialization for single-object propagation), refine predictions with point prompts, and finalize labels through prompt-first correction prior to saving. During export, per-object volumetry and 3D volume rendering are supported, and image geometry is preserved via SimpleITK. The GUI is implemented in Python using Napari and PyTorch, with optional N4 bias-field correction, and is intended exclusively for research annotation workflows. The code is released on the project page: https://github.com/SKKU-IBE/Medical-SAM2GUI/.

  </details>



- **DP-aware AdaLN-Zero: Taming Conditioning-Induced Heavy-Tailed Gradients in Differentially Private Diffusion**  
  Tao Huang, Jiayang Meng, Xu Yang, Chen Hou, Hong Chen  
  _2026-02-26_ · https://arxiv.org/abs/2602.22610v1  
  <details><summary>Abstract</summary>

  Condition injection enables diffusion models to generate context-aware outputs, which is essential for many time-series tasks. However, heterogeneous conditional contexts (e.g., observed history, missingness patterns or outlier covariates) can induce heavy-tailed per-example gradients. Under Differentially Private Stochastic Gradient Descent (DP-SGD), these rare conditioning-driven heavy-tailed gradients disproportionately trigger global clipping, resulting in outlier-dominated updates, larger clipping bias, and degraded utility under a fixed privacy budget. In this paper, we propose DP-aware AdaLN-Zero, a drop-in sensitivity-aware conditioning mechanism for conditional diffusion transformers that limits conditioning-induced gain without modifying the DP-SGD mechanism. DP-aware AdaLN-Zero jointly constrains conditioning representation magnitude and AdaLN modulation parameters via bounded re-parameterization, suppressing extreme gradient tail events before gradient clipping and noise injection. Empirically, DP-SGD equipped with DP-aware AdaLN-Zero improves interpolation/imputation and forecasting under matched privacy settings. We observe consistent gains on a real-world power dataset and two public ETT benchmarks over vanilla DP-SGD. Moreover, gradient diagnostics attribute these improvements to conditioning-specific tail reshaping and reduced clipping distortion, while preserving expressiveness in non-private training. Overall, these results show that sensitivity-aware conditioning can substantially improve private conditional diffusion training without sacrificing standard performance.

  </details>



- **LoR-LUT: Learning Compact 3D Lookup Tables via Low-Rank Residuals**  
  Ziqi Zhao, Abhijit Mishra, Shounak Roychowdhury  
  _2026-02-26_ · https://arxiv.org/abs/2602.22607v1  
  <details><summary>Abstract</summary>

  We present LoR-LUT, a unified low-rank formulation for compact and interpretable 3D lookup table (LUT) generation. Unlike conventional 3D-LUT-based techniques that rely on fusion of basis LUTs, which are usually dense tensors, our unified approach extends the current framework by jointly using residual corrections, which are in fact low-rank tensors, together with a set of basis LUTs. The approach described here improves the existing perceptual quality of an image, which is primarily due to the technique's novel use of residual corrections. At the same time, we achieve the same level of trilinear interpolation complexity, using a significantly smaller number of network, residual corrections, and LUT parameters. The experimental results obtained from LoR-LUT, which is trained on the MIT-Adobe FiveK dataset, reproduce expert-level retouching characteristics with high perceptual fidelity and a sub-megabyte model size. Furthermore, we introduce an interactive visualization tool, termed LoR-LUT Viewer, which transforms an input image into the LUT-adjusted output image, via a number of slidebars that control different parameters. The tool provides an effective way to enhance interpretability and user confidence in the visual results. Overall, our proposed formulation offers a compact, interpretable, and efficient direction for future LUT-based image enhancement and style transfer.

  </details>



- **BetterScene: 3D Scene Synthesis with Representation-Aligned Generative Model**  
  Yuci Han, Charles Toth, John E. Anderson, William J. Shuart, Alper Yilmaz  
  _2026-02-26_ · https://arxiv.org/abs/2602.22596v1  
  <details><summary>Abstract</summary>

  We present BetterScene, an approach to enhance novel view synthesis (NVS) quality for diverse real-world scenes using extremely sparse, unconstrained photos. BetterScene leverages the production-ready Stable Video Diffusion (SVD) model pretrained on billions of frames as a strong backbone, aiming to mitigate artifacts and recover view-consistent details at inference time. Conventional methods have developed similar diffusion-based solutions to address these challenges of novel view synthesis. Despite significant improvements, these methods typically rely on off-the-shelf pretrained diffusion priors and fine-tune only the UNet module while keeping other components frozen, which still leads to inconsistent details and artifacts even when incorporating geometry-aware regularizations like depth or semantic conditions. To address this, we investigate the latent space of the diffusion model and introduce two components: (1) temporal equivariance regularization and (2) vision foundation model-aligned representation, both applied to the variational autoencoder (VAE) module within the SVD pipeline. BetterScene integrates a feed-forward 3D Gaussian Splatting (3DGS) model to render features as inputs for the SVD enhancer and generate continuous, artifact-free, consistent novel views. We evaluate on the challenging DL3DV-10K dataset and demonstrate superior performance compared to state-of-the-art methods.

  </details>


