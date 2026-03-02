# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-03-02 07:14 UTC_

Total papers shown: **30**


---

- **Histopathology Image Normalization via Latent Manifold Compaction**  
  Xiaolong Zhang, Jianwei Zhang, Selim Sevim, Emek Demir, Ece Eksi, Xubo Song  
  _2026-02-27_ · https://arxiv.org/abs/2602.24251v1  
  <details><summary>Abstract</summary>

  Batch effects arising from technical variations in histopathology staining protocols, scanners, and acquisition pipelines pose a persistent challenge for computational pathology, hindering cross-batch generalization and limiting reliable deployment of models across clinical sites. In this work, we introduce Latent Manifold Compaction (LMC), an unsupervised representation learning framework that performs image harmonization by learning batch-invariant embeddings from a single source dataset through explicit compaction of stain-induced latent manifolds. This allows LMC to generalize to target domain data unseen during training. Evaluated on three challenging public and in-house benchmarks, LMC substantially reduces batch-induced separations across multiple datasets and consistently outperforms state-of-the-art normalization methods in downstream cross-batch classification and detection tasks, enabling superior generalization.

  </details>



- **SafeGen-LLM: Enhancing Safety Generalization in Task Planning for Robotic Systems**  
  Jialiang Fan, Weizhe Xu, Mengyu Liu, Oleg Sokolsky, Insup Lee, Fangxin Kong  
  _2026-02-27_ · https://arxiv.org/abs/2602.24235v1  
  <details><summary>Abstract</summary>

  Safety-critical task planning in robotic systems remains challenging: classical planners suffer from poor scalability, Reinforcement Learning (RL)-based methods generalize poorly, and base Large Language Models (LLMs) cannot guarantee safety. To address this gap, we propose safety-generalizable large language models, named SafeGen-LLM. SafeGen-LLM can not only enhance the safety satisfaction of task plans but also generalize well to novel safety properties in various domains. We first construct a multi-domain Planning Domain Definition Language 3 (PDDL3) benchmark with explicit safety constraints. Then, we introduce a two-stage post-training framework: Supervised Fine-Tuning (SFT) on a constraint-compliant planning dataset to learn planning syntax and semantics, and Group Relative Policy Optimization (GRPO) guided by fine-grained reward machines derived from formal verification to enforce safety alignment and by curriculum learning to better handle complex tasks. Extensive experiments show that SafeGen-LLM achieves strong safety generalization and outperforms frontier proprietary baselines across multi-domain planning tasks and multiple input formats (e.g., PDDLs and natural language).

  </details>



- **Enhancing Spatial Understanding in Image Generation via Reward Modeling**  
  Zhenyu Tang, Chaoran Feng, Yufan Deng, Jie Wu, Xiaojie Li, Rui Wang, Yunpeng Chen, Daquan Zhou  
  _2026-02-27_ · https://arxiv.org/abs/2602.24233v1  
  <details><summary>Abstract</summary>

  Recent progress in text-to-image generation has greatly advanced visual fidelity and creativity, but it has also imposed higher demands on prompt complexity-particularly in encoding intricate spatial relationships. In such cases, achieving satisfactory results often requires multiple sampling attempts. To address this challenge, we introduce a novel method that strengthens the spatial understanding of current image generation models. We first construct the SpatialReward-Dataset with over 80k preference pairs. Building on this dataset, we build SpatialScore, a reward model designed to evaluate the accuracy of spatial relationships in text-to-image generation, achieving performance that even surpasses leading proprietary models on spatial evaluation. We further demonstrate that this reward model effectively enables online reinforcement learning for the complex spatial generation. Extensive experiments across multiple benchmarks show that our specialized reward model yields significant and consistent gains in spatial understanding for image generation.

  </details>



- **A multimodal slice discovery framework for systematic failure detection and explanation in medical image classification**  
  Yixuan Liu, Kanwal K. Bhatia, Ahmed E. Fetit  
  _2026-02-27_ · https://arxiv.org/abs/2602.24183v1  
  <details><summary>Abstract</summary>

  Despite advances in machine learning-based medical image classifiers, the safety and reliability of these systems remain major concerns in practical settings. Existing auditing approaches mainly rely on unimodal features or metadata-based subgroup analyses, which are limited in interpretability and often fail to capture hidden systematic failures. To address these limitations, we introduce the first automated auditing framework that extends slice discovery methods to multimodal representations specifically for medical applications. Comprehensive experiments were conducted under common failure scenarios using the MIMIC-CXR-JPG dataset, demonstrating the framework's strong capability in both failure discovery and explanation generation. Our results also show that multimodal information generally allows more comprehensive and effective auditing of classifiers, while unimodal variants beyond image-only inputs exhibit strong potential in scenarios where resources are constrained.

  </details>



- **Fixed Anchors Are Not Enough: Dynamic Retrieval and Persistent Homology for Dataset Distillation**  
  Muquan Li, Hang Gou, Yingyi Ma, Rongzheng Wang, Ke Qin, Tao He  
  _2026-02-27_ · https://arxiv.org/abs/2602.24144v1  
  <details><summary>Abstract</summary>

  Decoupled dataset distillation (DD) compresses large corpora into a few synthetic images by matching a frozen teacher's statistics. However, current residual-matching pipelines rely on static real patches, creating a fit-complexity gap and a pull-to-anchor effect that reduce intra-class diversity and hurt generalization. To address these issues, we introduce RETA -- a Retrieval and Topology Alignment framework for decoupled DD. First, Dynamic Retrieval Connection (DRC) selects a real patch from a prebuilt pool by minimizing a fit-complexity score in teacher feature space; the chosen patch is injected via a residual connection to tighten feature fit while controlling injected complexity. Second, Persistent Topology Alignment (PTA) regularizes synthesis with persistent homology: we build a mutual k-NN feature graph, compute persistence images of components and loops, and penalize topology discrepancies between real and synthetic sets, mitigating pull-to-anchor effect. Across CIFAR-100, Tiny-ImageNet, ImageNet-1K, and multiple ImageNet subsets, RETA consistently outperforms various baselines under comparable time and memory, especially reaching 64.3% top-1 accuracy on ImageNet-1K with ResNet-18 at 50 images per class, +3.1% over the best prior.

  </details>



- **Robust Skills, Brittle Grounding: Diagnosing Restricted Generalization in Vision-Language Action Policies via Multi-Object Picking**  
  David Emukpere, Romain Deffayet, Jean-Michel Renders  
  _2026-02-27_ · https://arxiv.org/abs/2602.24143v1  
  <details><summary>Abstract</summary>

  Vision-language action (VLA) policies often report strong manipulation benchmark performance with relatively few demonstrations, but it remains unclear whether this reflects robust language-to-object grounding or reliance on object--location correlations that do not transfer beyond the training distribution. We present a controlled multi-object picking study that progressively increases object placement variability up to full workspace randomization and evaluates held-out object--location pairings that break familiar associations without increasing spatial difficulty. Across these stress tests and data scaling, we find that for representative VLA policies, including SmolVLA and $π_{0.5}$, execution of the manipulation primitive remains substantially more reliable than instruction-conditioned task success in harder regimes, suggesting that manipulation skill acquisition is decoupled from instruction following. We recommend augmenting manipulation benchmarks with task ladders and decomposed metrics that separately measure primitive execution and instruction-conditioned success to better diagnose instruction-grounded generalization.

  </details>



- **Multimodal Optimal Transport for Unsupervised Temporal Segmentation in Surgical Robotics**  
  Omar Mohamed, Edoardo Fazzari, Ayah Al-Naji, Hamdan Alhadhrami, Khalfan Hableel, Saif Alkindi, Cesare Stefanini  
  _2026-02-27_ · https://arxiv.org/abs/2602.24138v1  
  <details><summary>Abstract</summary>

  Recognizing surgical phases and steps from video is a fundamental problem in computer-assisted interventions. Recent approaches increasingly rely on large-scale pre-training on thousands of labeled surgical videos, followed by zero-shot transfer to specific procedures. While effective, this strategy incurs substantial computational and data collection costs. In this work, we question whether such heavy pre-training is truly necessary. We propose Text-Augmented Action Segmentation Optimal Transport (TASOT), an unsupervised method for surgical phase and step recognition that extends Action Segmentation Optimal Transport (ASOT) by incorporating textual information generated directly from the videos. TASOT formulates temporal action segmentation as a multimodal optimal transport problem, where the matching cost is defined as a weighted combination of visual and text-based costs. The visual term captures frame-level appearance similarity, while the text term provides complementary semantic cues, and both are jointly regularized through a temporally consistent unbalanced Gromov-Wasserstein formulation. This design enables effective alignment between video frames and surgical actions without surgical-specific pretraining or external web-scale supervision. We evaluate TASOT on multiple benchmark surgical datasets and observe consistent and substantial improvements over existing zero-shot methods, including StrasBypass70 (+23.7), BernBypass70 (+4.5), Cholec80 (+16.5), and AutoLaparo (+19.6). These results demonstrate that fine-grained surgical understanding can be achieved by exploiting information already present in standard visual and textual representations, without resorting to increasingly complex pre-training pipelines. The code will be available at https://github.com/omar8ahmed9/TASOT.

  </details>



- **EvalMVX: A Unified Benchmarking for Neural 3D Reconstruction under Diverse Multiview Setups**  
  Zaiyan Yang, Jieji Ren, Xiangyi Wang, zonglin li, Xu Cao, Heng Guo, Zhanyu Ma, Boxin Shi  
  _2026-02-27_ · https://arxiv.org/abs/2602.24065v1  
  <details><summary>Abstract</summary>

  Recent advancements in neural surface reconstruction have significantly enhanced 3D reconstruction. However, current real world datasets mainly focus on benchmarking multiview stereo (MVS) based on RGB inputs. Multiview photometric stereo (MVPS) and multiview shape from polarization (MVSfP), though indispensable on high-fidelity surface reconstruction and sparse inputs, have not been quantitatively assessed together with MVS. To determine the working range of different MVX (MVS, MVSfP, and MVPS) techniques, we propose EvalMVX, a real-world dataset containing $25$ objects, each captured with a polarized camera under $20$ varying views and $17$ light conditions including OLAT and natural illumination, leading to $8,500$ images. Each object includes aligned ground-truth 3D mesh, facilitating quantitative benchmarking of MVX methods simultaneously. Based on our EvalMVX, we evaluate $13$ MVX methods published in recent years, record the best-performing methods, and identify open problems under diverse geometric details and reflectance types. We hope EvalMVX and the benchmarking results can inspire future research on multiview 3D reconstruction.

  </details>



- **Ordinal Diffusion Models for Color Fundus Images**  
  Gustav Schmidt, Philipp Berens, Sarah Müller  
  _2026-02-27_ · https://arxiv.org/abs/2602.24013v1  
  <details><summary>Abstract</summary>

  It has been suggested that generative image models such as diffusion models can improve performance on clinically relevant tasks by offering deep learning models supplementary training data. However, most conditional diffusion models treat disease stages as independent classes, ignoring the continuous nature of disease progression. This mismatch is problematic in medical imaging because continuous pathological processes are typically only observed through coarse, discrete but ordered labels as in ophthalmology for diabetic retinopathy (DR). We propose an ordinal latent diffusion model for generating color fundus images that explicitly incorporates the ordered structure of DR severity into the generation process. Instead of categorical conditioning, we used a scalar disease representation, enabling a smooth transition between adjacent stages. We evaluated our approach using visual realism metrics and classification-based clinical consistency analysis on the EyePACS dataset. Compared to a standard conditional diffusion model, our model reduced the Fréchet inception distance for four of the five DR stages and increased the quadratic weighted $κ$ from 0.79 to 0.87. Furthermore, interpolation experiments showed that the model captured a continuous spectrum of disease progression learned from ordered, coarse class labels.

  </details>



- **Venus: Benchmarking and Empowering Multimodal Large Language Models for Aesthetic Guidance and Cropping**  
  Tianxiang Du, Hulingxiao He, Yuxin Peng  
  _2026-02-27_ · https://arxiv.org/abs/2602.23980v1  
  <details><summary>Abstract</summary>

  The widespread use of smartphones has made photography ubiquitous, yet a clear gap remains between ordinary users and professional photographers, who can identify aesthetic issues and provide actionable shooting guidance during capture. We define this capability as aesthetic guidance (AG) -- an essential but largely underexplored domain in computational aesthetics. Existing multimodal large language models (MLLMs) primarily offer overly positive feedback, failing to identify issues or provide actionable guidance. Without AG capability, they cannot effectively identify distracting regions or optimize compositional balance, thus also struggling in aesthetic cropping, which aims to refine photo composition through reframing after capture. To address this, we introduce AesGuide, the first large-scale AG dataset and benchmark with 10,748 photos annotated with aesthetic scores, analyses, and guidance. Building upon it, we propose Venus, a two-stage framework that first empowers MLLMs with AG capability through progressively complex aesthetic questions and then activates their aesthetic cropping power via CoT-based rationales. Extensive experiments show that Venus substantially improves AG capability and achieves state-of-the-art (SOTA) performance in aesthetic cropping, enabling interpretable and interactive aesthetic refinement across both stages of photo creation. Code is available at https://github.com/PKU-ICST-MIPL/Venus_CVPR2026.

  </details>



- **MSVBench: Towards Human-Level Evaluation of Multi-Shot Video Generation**  
  Haoyuan Shi, Yunxin Li, Nanhao Deng, Zhenran Xu, Xinyu Chen, Longyue Wang, Baotian Hu, Min Zhang  
  _2026-02-27_ · https://arxiv.org/abs/2602.23969v1  
  <details><summary>Abstract</summary>

  The evolution of video generation toward complex, multi-shot narratives has exposed a critical deficit in current evaluation methods. Existing benchmarks remain anchored to single-shot paradigms, lacking the comprehensive story assets and cross-shot metrics required to assess long-form coherence and appeal. To bridge this gap, we introduce MSVBench, the first comprehensive benchmark featuring hierarchical scripts and reference images tailored for Multi-Shot Video generation. We propose a hybrid evaluation framework that synergizes the high-level semantic reasoning of Large Multimodal Models (LMMs) with the fine-grained perceptual rigor of domain-specific expert models. Evaluating 20 video generation methods across diverse paradigms, we find that current models--despite strong visual fidelity--primarily behave as visual interpolators rather than true world models. We further validate the reliability of our benchmark by demonstrating a state-of-the-art Spearman's rank correlation of 94.4% with human judgments. Finally, MSVBench extends beyond evaluation by providing a scalable supervisory signal. Fine-tuning a lightweight model on its pipeline-refined reasoning traces yields human-aligned performance comparable to commercial models like Gemini-2.5-Flash.

  </details>



- **SpikeTrack: A Spike-driven Framework for Efficient Visual Tracking**  
  Qiuyang Zhang, Jiujun Cheng, Qichao Mao, Cong Liu, Yu Fang, Yuhong Li, Mengying Ge, Shangce Gao  
  _2026-02-27_ · https://arxiv.org/abs/2602.23963v1  
  <details><summary>Abstract</summary>

  Spiking Neural Networks (SNNs) promise energy-efficient vision, but applying them to RGB visual tracking remains difficult: Existing SNN tracking frameworks either do not fully align with spike-driven computation or do not fully leverage neurons' spatiotemporal dynamics, leading to a trade-off between efficiency and accuracy. To address this, we introduce SpikeTrack, a spike-driven framework for energy-efficient RGB object tracking. SpikeTrack employs a novel asymmetric design that uses asymmetric timestep expansion and unidirectional information flow, harnessing spatiotemporal dynamics while cutting computation. To ensure effective unidirectional information transfer between branches, we design a memory-retrieval module inspired by neural inference mechanisms. This module recurrently queries a compact memory initialized by the template to retrieve target cues and sharpen target perception over time. Extensive experiments demonstrate that SpikeTrack achieves the state-of-the-art among SNN-based trackers and remains competitive with advanced ANN trackers. Notably, it surpasses TransT on LaSOT dataset while consuming only 1/26 of its energy. To our knowledge, SpikeTrack is the first spike-driven framework to make RGB tracking both accurate and energy efficient. The code and models are available at https://github.com/faicaiwawa/SpikeTrack.

  </details>



- **Extending 2D foundational DINOv3 representations to 3D segmentation of neonatal brain MR images**  
  Annayah Usman, Behraj Khan, Tahir Qasim Syed  
  _2026-02-27_ · https://arxiv.org/abs/2602.23962v1  
  <details><summary>Abstract</summary>

  Precise volumetric delineation of hippocampal structures is essential for quantifying neurodevelopmental trajectories in pre-term and term infants, where subtle morphological variations may carry prognostic significance. While foundation encoders trained on large-scale visual data offer discriminative representations, their 2D formulation is a limitation with respect to the $3$D organization of brain anatomy. We propose a volumetric segmentation strategy that reconciles this tension through a structured window-based disassembly-reassembly mechanism: the global MRI volume is decomposed into non-overlapping 3D windows or sub-cubes, each processed via a separate decoding arm built upon frozen high-fidelity features, and subsequently reassembled prior to a ground-truth correspendence using a dense-prediction head. This architecture preserves constant a decoder memory footprint while forcing predictions to lie within an anatomically consistent geometry. Evaluated on the ALBERT dataset for hippocampal segmentation, the proposed approach achieves a Dice score of 0.65 for a single 3D window. The method demonstrates that volumetric anatomical structure could be recovered from frozen 2D foundation representations through structured compositional decoding, and offers a principled and generalizable extension for foundation models for 3D medical applications.

  </details>



- **Clinically-aligned ischemic stroke segmentation and ASPECTS scoring on NCCT imaging using a slice-gated loss on foundation representations**  
  Hiba Azeem, Behraj Khan, Tahir Qasim Syed  
  _2026-02-27_ · https://arxiv.org/abs/2602.23961v1  
  <details><summary>Abstract</summary>

  Rapid infarct assessment on non-contrast CT (NCCT) is essential for acute ischemic stroke management. Most deep learning methods perform pixel-wise segmentation without modeling the structured anatomical reasoning underlying ASPECTS scoring, where basal ganglia (BG) and supraganglionic (SG) levels are clinically interpreted in a coupled manner. We propose a clinically aligned framework that combines a frozen DINOv3 backbone with a lightweight decoder and introduce a Territory-Aware Gated Loss (TAGL) to enforce BG-SG consistency during training. This anatomically informed supervision adds no inference-time complexity. Our method achieves a Dice score of 0.6385 on AISD, outperforming prior CNN and foundation-model baselines. On a proprietary ASPECTS dataset, TAGL improves mean Dice from 0.698 to 0.767. These results demonstrate that integrating foundation representations with structured clinical priors improves NCCT stroke segmentation and ASPECTS delineation.

  </details>



- **GDA-YOLO11: Amodal Instance Segmentation for Occlusion-Robust Robotic Fruit Harvesting**  
  Caner Beldek, Emre Sariyildiz, Son Lam Phung, Gursel Alici  
  _2026-02-27_ · https://arxiv.org/abs/2602.23953v1  
  <details><summary>Abstract</summary>

  Occlusion remains a critical challenge in robotic fruit harvesting, as undetected or inaccurately localised fruits often results in substantial crop losses. To mitigate this issue, we propose a harvesting framework using a new amodal segmentation model, GDA-YOLO11, which incorporates architectural improvements and an updated asymmetric mask loss. The proposed model is trained on a modified version of a public citrus dataset and evaluated on both the base dataset and occlusion-sensitive subsets with varying occlusion levels. Within the framework, full fruit masks, including invisible regions, are inferred by GDA-YOLO11, and picking points are subsequently estimated using the Euclidean distance transform. These points are then projected into 3D coordinates for robotic harvesting execution. Experiments were conducted using real citrus fruits in a controlled environment simulating occlusion scenarios. Notably, to the best of our knowledge, this study provides the first practical demonstration of amodal instance segmentation in robotic fruit harvesting. GDA-YOLO11 achieves a precision of 0.844, recall of 0.846, mAP@50 of 0.914, and mAP@50:95 of 0.636, outperforming YOLO11n by 5.1%, 1.3%, and 1.0% in precision, mAP@50, and mAP@50:95, respectively. The framework attains harvesting success rates of 92.59%, 85.18%, 48.14%, and 22.22% at zero to high occlusion levels, improving success by 3.5% under medium and high occlusion. These findings demonstrate that GDA-YOLO11 enhances occlusion robust segmentation and streamlines perception-to-action integration, paving the way for more reliable autonomous systems in agriculture.

  </details>



- **Micro-expression Recognition Based on Dual-branch Feature Extraction and Fusion**  
  Mingjie Zhang, Bo Li, Wanting Liu, Hongyan Cui, Yue Li, Qingwen Li, Hong Li, Ge Gao  
  _2026-02-27_ · https://arxiv.org/abs/2602.23950v1  
  <details><summary>Abstract</summary>

  Micro-expressions, characterized by transience and subtlety, pose challenges to existing optical flow-based recognition methods. To address this, this paper proposes a dual-branch micro-expression feature extraction network integrated with parallel attention. Key contributions include: 1) a residual network designed to alleviate gradient anishing and network degradation; 2) an Inception network constructed to enhance model representation and suppress interference from irrelevant regions; 3) an adaptive feature fusion module developed to integrate dual-branch features. Experiments on the CASME II dataset demonstrate that the proposed method achieves 74.67% accuracy, outperforming LBP-TOP (by 11.26%), MSMMT (by 3.36%), and other comparative methods.

  </details>



- **PointCoT: A Multi-modal Benchmark for Explicit 3D Geometric Reasoning**  
  Dongxu Zhang, Yiding Sun, Pengcheng Li, Yumou Liu, Hongqiang Lin, Haoran Xu, Xiaoxuan Mu, Liang Lin, Wenbiao Yan, Ning Yang, et al.  
  _2026-02-27_ · https://arxiv.org/abs/2602.23945v1  
  <details><summary>Abstract</summary>

  While Multimodal Large Language Models (MLLMs) demonstrate proficiency in 2D scenes, extending their perceptual intelligence to 3D point cloud understanding remains a significant challenge. Current approaches focus primarily on aligning 3D features with pre-trained models. However, they typically treat geometric reasoning as an implicit mapping process. These methods bypass intermediate logical steps and consequently suffer from geometric hallucinations. They confidently generate plausible responses that fail to ground in precise structural details. To bridge this gap, we present PointCoT, a novel framework that empowers MLLMs with explicit Chain-of-Thought (CoT) reasoning for 3D data. We advocate for a \textit{Look, Think, then Answer} paradigm. In this approach, the model is supervised to generate geometry-grounded rationales before predicting final answers. To facilitate this, we construct Point-Reason-Instruct, a large-scale benchmark comprising $\sim$86k instruction-tuning samples with hierarchical CoT annotations. By leveraging a dual-stream multi-modal architecture, our method synergizes semantic appearance with geometric truth. Extensive experiments demonstrate that PointCoT achieves state-of-the-art performance on complex reasoning tasks.

  </details>



- **Learning to Build: Autonomous Robotic Assembly of Stable Structures Without Predefined Plans**  
  Jingwen Wang, Johannes Kirschner, Paul Rolland, Luis Salamanca, Stefana Parascho  
  _2026-02-27_ · https://arxiv.org/abs/2602.23934v1  
  <details><summary>Abstract</summary>

  This paper presents a novel autonomous robotic assembly framework for constructing stable structures without relying on predefined architectural blueprints. Instead of following fixed plans, construction tasks are defined through targets and obstacles, allowing the system to adapt more flexibly to environmental uncertainty and variations during the building process. A reinforcement learning (RL) policy, trained using deep Q-learning with successor features, serves as the decision-making component. As a proof of concept, we evaluate the approach on a benchmark of 15 2D robotic assembly tasks of discrete block construction. Experiments using a real-world closed-loop robotic setup demonstrate the feasibility of the method and its ability to handle construction noise. The results suggest that our framework offers a promising direction for more adaptable and robust robotic construction in real-world environments.

  </details>



- **The Geometry of Transfer: Unlocking Medical Vision Manifolds for Training-Free Model Ranking**  
  Jiaqi Tang, Shaoyang Zhang, Xiaoqi Wang, Jiaying Zhou, Yang Liu, Qingchao Chen  
  _2026-02-27_ · https://arxiv.org/abs/2602.23916v1  
  <details><summary>Abstract</summary>

  The advent of large-scale self-supervised learning (SSL) has produced a vast zoo of medical foundation models. However, selecting optimal medical foundation models for specific segmentation tasks remains a computational bottleneck. Existing Transferability Estimation (TE) metrics, primarily designed for classification, rely on global statistical assumptions and fail to capture the topological complexity essential for dense prediction. We propose a novel Topology-Driven Transferability Estimation framework that evaluates manifold tractability rather than statistical overlap. Our approach introduces three components: (1) Global Representation Topology Divergence (GRTD), utilizing Minimum Spanning Trees to quantify feature-label structural isomorphism; (2) Local Boundary-Aware Topological Consistency (LBTC), which assesses manifold separability specifically at critical anatomical boundaries; and (3) Task-Adaptive Fusion, which dynamically integrates global and local metrics based on the semantic cardinality of the target task. Validated on the large-scale OpenMind benchmark across diverse anatomical targets and SSL foundation models, our approach significantly outperforms state-of-the-art baselines by around \textbf{31\%} relative improvement in the weighted Kendall, providing a robust, training-free proxy for efficient model selection without the cost of fine-tuning. The code will be made publicly available upon acceptance.

  </details>



- **SegMate: Asymmetric Attention-Based Lightweight Architecture for Efficient Multi-Organ Segmentation**  
  Andrei-Alexandru Bunea, Dan-Matei Popovici, Radu Tudor Ionescu  
  _2026-02-27_ · https://arxiv.org/abs/2602.23903v1  
  <details><summary>Abstract</summary>

  State-of-the-art models for medical image segmentation achieve excellent accuracy but require substantial computational resources, limiting deployment in resource-constrained clinical settings. We present SegMate, an efficient 2.5D framework that achieves state-of-the-art accuracy, while considerably reducing computational requirements. Our efficient design is the result of meticulously integrating asymmetric architectures, attention mechanisms, multi-scale feature fusion, slice-based positional conditioning, and multi-task optimization. We demonstrate the efficiency-accuracy trade-off of our framework across three modern backbones (EfficientNetV2-M, MambaOut-Tiny, FastViT-T12). We perform experiments on three datasets: TotalSegmentator, SegTHOR and AMOS22. Compared with the vanilla models, SegMate reduces computation (GFLOPs) by up to 2.5x and memory footprint (VRAM) by up to 2.1x, while generally registering performance gains of around 1%. On TotalSegmentator, we achieve a Dice score of 93.51% with only 295MB peak GPU memory. Zero-shot cross-dataset evaluations on SegTHOR and AMOS22 demonstrate strong generalization, with Dice scores of up to 86.85% and 89.35%, respectively. We release our open-source code at https://github.com/andreibunea99/SegMate.

  </details>



- **Ref-Adv: Exploring MLLM Visual Reasoning in Referring Expression Tasks**  
  Qihua Dong, Kuo Yang, Lin Ju, Handong Zhao, Yitian Zhang, Yizhou Wang, Huimin Zeng, Jianglin Lu, Yun Fu  
  _2026-02-27_ · https://arxiv.org/abs/2602.23898v1  
  <details><summary>Abstract</summary>

  Referring Expression Comprehension (REC) links language to region level visual perception. Standard benchmarks (RefCOCO, RefCOCO+, RefCOCOg) have progressed rapidly with multimodal LLMs but remain weak tests of visual reasoning and grounding: (i) many expressions are very short, leaving little reasoning demand; (ii) images often contain few distractors, making the target easy to find; and (iii) redundant descriptors enable shortcut solutions that bypass genuine text understanding and visual reasoning. We introduce Ref-Adv, a modern REC benchmark that suppresses shortcuts by pairing linguistically nontrivial expressions with only the information necessary to uniquely identify the target. The dataset contains referring expressions on real images, curated with hard distractors and annotated with reasoning facets including negation. We conduct comprehensive ablations (word order perturbations and descriptor deletion sufficiency) to show that solving Ref-Adv requires reasoning beyond simple cues, and we evaluate a broad suite of contemporary multimodal LLMs on Ref-Adv. Despite strong results on RefCOCO, RefCOCO+, and RefCOCOg, models drop markedly on Ref-Adv, revealing reliance on shortcuts and gaps in visual reasoning and grounding. We provide an in depth failure analysis and aim for Ref-Adv to guide future work on visual reasoning and grounding in MLLMs.

  </details>



- **NAU-QMUL: Utilizing BERT and CLIP for Multi-modal AI-Generated Image Detection**  
  Xiaoyu Guo, Arkaitz Zubiaga  
  _2026-02-27_ · https://arxiv.org/abs/2602.23863v1  
  <details><summary>Abstract</summary>

  With the aim of detecting AI-generated images and identifying the specific models responsible for their generation, we propose a multi-modal multi-task model. The model leverages pre-trained BERT and CLIP Vision encoders for text and image feature extraction, respectively, and employs cross-modal feature fusion with a tailored multi-task loss function. Additionally, a pseudo-labeling-based data augmentation strategy was utilized to expand the training dataset with high-confidence samples. The model achieved fifth place in both Tasks A and B of the `CT2: AI-Generated Image Detection' competition, with F1 scores of 83.16\% and 48.88\%, respectively. These findings highlight the effectiveness of the proposed architecture and its potential for advancing AI-generated content detection in real-world scenarios. The source code for our method is published on https://github.com/xxxxxxxxy/AIGeneratedImageDetection.

  </details>



- **Revisiting Integration of Image and Metadata for DICOM Series Classification: Cross-Attention and Dictionary Learning**  
  Tuan Truong, Melanie Dohmen, Sara Lorio, Matthias Lenga  
  _2026-02-27_ · https://arxiv.org/abs/2602.23833v1  
  <details><summary>Abstract</summary>

  Automated identification of DICOM image series is essential for large-scale medical image analysis, quality control, protocol harmonization, and reliable downstream processing. However, DICOM series classification remains challenging due to heterogeneous slice content, variable series length, and entirely missing, incomplete or inconsistent DICOM metadata. We propose an end-to-end multimodal framework for DICOM series classification that jointly models image content and acquisition metadata while explicitly accounting for all these challenges. (i) Images and metadata are encoded with modality-aware modules and fused using a bi-directional cross-modal attention mechanism. (ii) Metadata is processed by a sparse, missingness-aware encoder based on learnable feature dictionaries and value-conditioned modulation. By design, the approach does not require any form of imputation. (iii) Variability in series length and image data dimensions is handled via a 2.5D visual encoder and attention operating on equidistantly sampled slices. We evaluate the proposed approach on the publicly available Duke Liver MRI dataset and a large multi-institutional in-house cohort, assessing both in-domain performance and out-of-domain generalization. Across all evaluation settings, the proposed method consistently outperforms relevant image only, metadata-only and multimodal 2D/3D baselines. The results demonstrate that explicitly modeling metadata sparsity and cross-modal interactions improves robustness for DICOM series classification.

  </details>



- **APPO: Attention-guided Perception Policy Optimization for Video Reasoning**  
  Henghui Du, Chang Zhou, Xi Chen, Di Hu  
  _2026-02-27_ · https://arxiv.org/abs/2602.23823v1  
  <details><summary>Abstract</summary>

  Complex video reasoning, actually, relies excessively on fine-grained perception rather than on expert (e.g., Ph.D, Science)-level reasoning. Through extensive empirical observation, we have recognized the critical impact of perception. In particular, when perception ability is almost fixed, enhancing reasoning from Qwen3-8B to OpenAI-o3 yields only 0.7% performance improvement. Conversely, even minimal change in perception model scale (from 7B to 32B) boosts performance by 1.4%, indicating enhancing perception, rather than reasoning, is more critical to improve performance. Therefore, exploring how to enhance perception ability through reasoning without the need for expensive fine-grained annotation information is worthwhile. To achieve this goal, we specially propose APPO, the Attention-guided Perception Policy Optimization algorithm that leverages token-level dense rewards to improve model's fine-grained perception. The core idea behind APPO is to optimize those tokens from different responses that primarily focus on the same crucial video frame (called intra-group perception tokens). Experimental results on diverse video benchmarks and models with different scales (3/7B) demonstrate APPO consistently outperforms GRPO and DAPO (0.5%~4%). We hope our work provides a promising approach to effectively enhance model's perception abilities through reasoning in a low-cost manner, serving diverse scenarios and demands.

  </details>



- **Action-Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation**  
  Chongyang Xu, Haipeng Li, Shen Cheng, Jingyu Hu, Haoqiang Fan, Ziliang Feng, Shuaicheng Liu  
  _2026-02-27_ · https://arxiv.org/abs/2602.23814v1  
  <details><summary>Abstract</summary>

  Bimanual manipulation requires policies that can reason about 3D geometry, anticipate how it evolves under action, and generate smooth, coordinated motions. However, existing methods typically rely on 2D features with limited spatial awareness, or require explicit point clouds that are difficult to obtain reliably in real-world settings. At the same time, recent 3D geometric foundation models show that accurate and diverse 3D structure can be reconstructed directly from RGB images in a fast and robust manner. We leverage this opportunity and propose a framework that builds bimanual manipulation directly on a pre-trained 3D geometric foundation model. Our policy fuses geometry-aware latents, 2D semantic features, and proprioception into a unified state representation, and uses diffusion model to jointly predict a future action chunk and a future 3D latent that decodes into a dense pointmap. By explicitly predicting how the 3D scene will evolve together with the action sequence, the policy gains strong spatial understanding and predictive capability using only RGB observations. We evaluate our method both in simulation on the RoboTwin benchmark and in real-world robot executions. Our approach consistently outperforms 2D-based and point-cloud-based baselines, achieving state-of-the-art performance in manipulation success, inter-arm coordination, and 3D spatial prediction accuracy. Code is available at https://github.com/Chongyang-99/GAP.git.

  </details>



- **See, Act, Adapt: Active Perception for Unsupervised Cross-Domain Visual Adaptation via Personalized VLM-Guided Agent**  
  Tianci Tang, Tielong Cai, Hongwei Wang, Gaoang Wang  
  _2026-02-27_ · https://arxiv.org/abs/2602.23806v1  
  <details><summary>Abstract</summary>

  Pre-trained perception models excel in generic image domains but degrade significantly in novel environments like indoor scenes. The conventional remedy is fine-tuning on downstream data which incurs catastrophic forgetting of prior knowledge and demands costly, scene-specific annotations. We propose a paradigm shift through Sea$^2$ (See, Act, Adapt): rather than adapting the perception modules themselves, we adapt how they are deployed through an intelligent pose-control agent. Sea$^2$ keeps all perception modules frozen, requiring no downstream labels during training, and uses only scalar perceptual feedback to navigate the agent toward informative viewpoints. Specially, we transform a vision-language model (VLM) into a low-level pose controller through a two-stage training pipeline: first fine-tuning it on rule-based exploration trajectories that systematically probe indoor scenes, and then refining the policy via unsupervised reinforcement learning that constructs rewards from the perception module's outputs and confidence. Unlike prior active perception methods that couple exploration with specific models or collect data for retraining them, Sea$^2$ directly leverages off-the-shelf perception models for various tasks without the need for retraining. We conducted experiments on three visual perception tasks, including visual grounding, segmentation and 3D box estimation, with performance improvements of 13.54%, 15.92% and 27.68% respectively on dataset ReplicaCAD.

  </details>



- **BiM-GeoAttn-Net: Linear-Time Depth Modeling with Geometry-Aware Attention for 3D Aortic Dissection CTA Segmentation**  
  Yuan Zhang, Lei Liu, Jialin Zhang, Ya-Nan Zhang, Ling Wang, Nan Mu  
  _2026-02-27_ · https://arxiv.org/abs/2602.23803v1  
  <details><summary>Abstract</summary>

  Accurate segmentation of aortic dissection (AD) lumens in CT angiography (CTA) is essential for quantitative morphological assessment and clinical decision-making. However, reliable 3D delineation remains challenging due to limited long-range context modeling, which compromises inter-slice coherence, and insufficient structural discrimination under low-contrast conditions. To address these limitations, we propose BiM-GeoAttn-Net, a lightweight framework that integrates linear-time depth-wise state-space modeling with geometry-aware vessel refinement. Our approach is featured by Bidirectional Depth Mamba (BiM) to efficiently capture cross-slice dependencies and Geometry-Aware Vessel Attention (GeoAttn) module that employs orientation-sensitive anisotropic filtering to refine tubular structures and sharpen ambiguous boundaries. Extensive experiments on a multi-source AD CTA dataset demonstrate that BiM-GeoAttn-Net achieves a Dice score of 93.35% and an HD95 of 12.36 mm, outperforming representative CNN-, Transformer-, and SSM-based baselines in overlap metrics while maintaining competitive boundary accuracy. These results suggest that coupling linear-time depth modeling with geometry-aware refinement provides an effective, computationally efficient solution for robust 3D AD segmentation.

  </details>



- **FluoCLIP: Stain-Aware Focus Quality Assessment in Fluorescence Microscopy**  
  Hyejin Park, Jiwon Yoon, Sumin Park, Suree Kim, Sinae Jang, Eunsoo Lee, Dongmin Kang, Dongbo Min  
  _2026-02-27_ · https://arxiv.org/abs/2602.23791v1  
  <details><summary>Abstract</summary>

  Accurate focus quality assessment (FQA) in fluorescence microscopy remains challenging, as the stain-dependent optical properties of fluorescent dyes cause abrupt and heterogeneous focus shifts. However, existing datasets and models overlook this variability, treating focus quality as a stain-agnostic problem. In this work, we formulate the task of stain-aware FQA, emphasizing that focus behavior in fluorescence microscopy must be modeled as a function of staining characteristics. Through quantitative analysis of existing datasets (FocusPath, BBBC006) and our newly curated FluoMix, we demonstrate that focus-rank relationships vary substantially across stains, underscoring the need for stain-aware modeling in fluorescence microscopy. To support this new formulation, we propose FluoMix, the first dataset for stain-aware FQA that encompasses multiple tissues, fluorescent stains, and focus variations. Building on this dataset, we propose FluoCLIP, a two-stage vision-language framework that leverages CLIP's alignment capability to interpret focus quality in the context of biological staining. In the stain-grounding phase, FluoCLIP learns general stain representations by aligning textual stain tokens with visual features, while in the stain-guided ranking phase, it optimizes stain-specific rank prompts for ordinal focus prediction. Together, our formulation, dataset, and framework establish the first foundation for stain-aware FQA, and FluoCLIP achieves strong generalization across diverse fluorescence microscopy conditions.

  </details>



- **VideoPulse: Neonatal heart rate and peripheral capillary oxygen saturation (SpO2) estimation from contact free video**  
  Deependra Dewagiri, Kamesh Anuradha, Pabadhi Liyanage, Helitha Kulatunga, Pamuditha Somarathne, Udaya S. K. P. Miriya Thanthrige, Nishani Lucas, Anusha Withana, Joshua P. Kulasingham  
  _2026-02-27_ · https://arxiv.org/abs/2602.23771v1  
  <details><summary>Abstract</summary>

  Remote photoplethysmography (rPPG) enables contact free monitoring of vital signs and is especially valuable for neonates, since conventional methods often require sustained skin contact with adhesive probes that can irritate fragile skin and increase infection control burden. We present VideoPulse, a neonatal dataset and an end to end pipeline that estimates neonatal heart rate and peripheral capillary oxygen saturation (SpO2) from facial video. VideoPulse contains 157 recordings totaling 2.6 hours from 52 neonates with diverse face orientations. Our pipeline performs face alignment and artifact aware supervision using denoised pulse oximeter signals, then applies 3D CNN backbones for heart rate and SpO2 regression with label distribution smoothing and weighted regression for SpO2. Predictions are produced in 2 second windows. On the NBHR neonatal dataset, we obtain heart rate MAE 2.97 bpm using 2 second windows (2.80 bpm at 6 second windows) and SpO2 MAE 1.69 percent. Under cross dataset evaluation, the NBHR trained heart rate model attains 5.34 bpm MAE on VideoPulse, and fine tuning an NBHR pretrained SpO2 model on VideoPulse yields MAE 1.68 percent. These results indicate that short unaligned neonatal video segments can support accurate heart rate and SpO2 estimation, enabling low cost non invasive monitoring in neonatal intensive care.

  </details>



- **OPTIAGENT: A Physics-Driven Agentic Framework for Automated Optical Design**  
  Yuyu Geng, Lei Sun, Yao Gao, Xinxin Hu, Zhonghua Yi, Xiaolong Qian, Weijian Hu, Jian Bai, Kaiwei Wang  
  _2026-02-27_ · https://arxiv.org/abs/2602.23761v1  
  <details><summary>Abstract</summary>

  Optical design is the process of configuring optical elements to precisely manipulate light for high-fidelity imaging. It is inherently a highly non-convex optimization problem that relies heavily on human heuristic expertise and domain-specific knowledge. While Large Language Models (LLMs) possess extensive optical knowledge, their capabilities in leveraging the knowledge in designing lens system remain significantly constrained. This work represents the first attempt to employ LLMs in the field of optical design. We bridge the expertise gap by enabling users without formal optical training to successfully develop functional lens systems. Concretely, we curate a comprehensive dataset, named OptiDesignQA, which encompasses both classical lens systems sourced from standard optical textbooks and novel configurations generated by automated design algorithms for training and evaluation. Furthermore, we inject domain-specific optical expertise into the LLM through a hybrid objective of full-system synthesis and lens completion. To align the model with optical principles, we employ Group Relative Policy Optimization Done Right (DrGRPO) guided by Optical Lexicographic Reward for physics-driven policy alignment. This reward system incorporates structural format rewards, physical feasibility rewards, light-manipulation accuracy, and LLM-based heuristics. Finally, our model integrates with specialized optical optimization routines for end-to-end fine-tuning and precision refinement. We benchmark our proposed method against both traditional optimization-based automated design algorithms and LLM counterparts, and experimental results show the superiority of our method.

  </details>


