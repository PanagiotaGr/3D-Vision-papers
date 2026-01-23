# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-01-23 06:52 UTC_

Total papers shown: **50**


---

- **Point Bridge: 3D Representations for Cross Domain Policy Learning**  
  Siddhant Haldar, Lars Johannsmeier, Lerrel Pinto, Abhishek Gupta, Dieter Fox, Yashraj Narang, Ajay Mandlekar  
  _2026-01-22_ · https://arxiv.org/abs/2601.16212v1  
  <details><summary>Abstract</summary>

  Robot foundation models are beginning to deliver on the promise of generalist robotic agents, yet progress remains constrained by the scarcity of large-scale real-world manipulation datasets. Simulation and synthetic data generation offer a scalable alternative, but their usefulness is limited by the visual domain gap between simulation and reality. In this work, we present Point Bridge, a framework that leverages unified, domain-agnostic point-based representations to unlock synthetic datasets for zero-shot sim-to-real policy transfer, without explicit visual or object-level alignment. Point Bridge combines automated point-based representation extraction via Vision-Language Models (VLMs), transformer-based policy learning, and efficient inference-time pipelines to train capable real-world manipulation agents using only synthetic data. With additional co-training on small sets of real demonstrations, Point Bridge further improves performance, substantially outperforming prior vision-based sim-and-real co-training methods. It achieves up to 44% gains in zero-shot sim-to-real transfer and up to 66% with limited real data across both single-task and multitask settings. Videos of the robot are best viewed at: https://pointbridge3d.github.io/

  </details>



- **Rethinking Composed Image Retrieval Evaluation: A Fine-Grained Benchmark from Image Editing**  
  Tingyu Song, Yanzhao Zhang, Mingxin Li, Zhuoning Guo, Dingkun Long, Pengjun Xie, Siyue Zhang, Yilun Zhao, Shu Wu  
  _2026-01-22_ · https://arxiv.org/abs/2601.16125v1  
  <details><summary>Abstract</summary>

  Composed Image Retrieval (CIR) is a pivotal and complex task in multimodal understanding. Current CIR benchmarks typically feature limited query categories and fail to capture the diverse requirements of real-world scenarios. To bridge this evaluation gap, we leverage image editing to achieve precise control over modification types and content, enabling a pipeline for synthesizing queries across a broad spectrum of categories. Using this pipeline, we construct EDIR, a novel fine-grained CIR benchmark. EDIR encompasses 5,000 high-quality queries structured across five main categories and fifteen subcategories. Our comprehensive evaluation of 13 multimodal embedding models reveals a significant capability gap; even state-of-the-art models (e.g., RzenEmbed and GME) struggle to perform consistently across all subcategories, highlighting the rigorous nature of our benchmark. Through comparative analysis, we further uncover inherent limitations in existing benchmarks, such as modality biases and insufficient categorical coverage. Furthermore, an in-domain training experiment demonstrates the feasibility of our benchmark. This experiment clarifies the task challenges by distinguishing between categories that are solvable with targeted data and those that expose intrinsic limitations of current model architectures.

  </details>



- **synthocr-gen: A synthetic ocr dataset generator for low-resource languages- breaking the data barrier**  
  Haq Nawaz Malik, Kh Mohmad Shafi, Tanveer Ahmad Reshi  
  _2026-01-22_ · https://arxiv.org/abs/2601.16113v1  
  <details><summary>Abstract</summary>

  Optical Character Recognition (OCR) for low-resource languages remains a significant challenge due to the scarcity of large-scale annotated training datasets. Languages such as Kashmiri, with approximately 7 million speakers and a complex Perso-Arabic script featuring unique diacritical marks, currently lack support in major OCR systems including Tesseract, TrOCR, and PaddleOCR. Manual dataset creation for such languages is prohibitively expensive, time-consuming, and error-prone, often requiring word by word transcription of printed or handwritten text. We present SynthOCR-Gen, an open-source synthetic OCR dataset generator specifically designed for low-resource languages. Our tool addresses the fundamental bottleneck in OCR development by transforming digital Unicode text corpora into ready-to-use training datasets. The system implements a comprehensive pipeline encompassing text segmentation (character, word, n-gram, sentence, and line levels), Unicode normalization with script purity enforcement, multi-font rendering with configurable distribution, and 25+ data augmentation techniques simulating real-world document degradations including rotation, blur, noise, and scanner artifacts. We demonstrate the efficacy of our approach by generating a 600,000-sample word-segmented Kashmiri OCR dataset, which we release publicly on HuggingFace. This work provides a practical pathway for bringing low-resource languages into the era of vision-language AI models, and the tool is openly available for researchers and practitioners working with underserved writing systems worldwide.

  </details>



- **DTP: A Simple yet Effective Distracting Token Pruning Framework for Vision-Language Action Models**  
  Chenyang Li, Jieyuan Liu, Bin Li, Bo Gao, Yilin Yuan, Yangfan He, Yuchen Li, Jingqun Tang  
  _2026-01-22_ · https://arxiv.org/abs/2601.16065v1  
  <details><summary>Abstract</summary>

  Vision-Language Action (VLA) models have shown remarkable progress in robotic manipulation by leveraging the powerful perception abilities of Vision-Language Models (VLMs) to understand environments and directly output actions. However, by default, VLA models may overly attend to image tokens in the task-irrelevant region, which we describe as 'distracting tokens'. This behavior can disturb the model from the generation of the desired action tokens in each step, affecting the success rate of tasks. In this paper, we introduce a simple yet effective plug-and-play Distracting Token Pruning (DTP) framework, which dynamically detects and prunes these distracting image tokens. By correcting the model's visual attention patterns, we aim to improve the task success rate, as well as exploring the performance upper boundaries of the model without altering its original architecture or adding additional inputs. Experiments on the SIMPLER Benchmark (Li et al., 2024) show that our method consistently achieving relative improvements in task success rates across different types of novel VLA models, demonstrating generalizability to transformer-based VLAs. Further analysis reveals a negative correlation between the task success rate and the amount of attentions in the task-irrelevant region for all models tested, highlighting a common phenomenon of VLA models that could guide future research. We also publish our code at: https://anonymous.4open.science/r/CBD3.

  </details>



- **Phi-SegNet: Phase-Integrated Supervision for Medical Image Segmentation**  
  Shams Nafisa Ali, Taufiq Hasan  
  _2026-01-22_ · https://arxiv.org/abs/2601.16064v1  
  <details><summary>Abstract</summary>

  Deep learning has substantially advanced medical image segmentation, yet achieving robust generalization across diverse imaging modalities and anatomical structures remains a major challenge. A key contributor to this limitation lies in how existing architectures, ranging from CNNs to Transformers and their hybrids, primarily encode spatial information while overlooking frequency-domain representations that capture rich structural and textural cues. Although few recent studies have begun exploring spectral information at the feature level, supervision-level integration of frequency cues-crucial for fine-grained object localization-remains largely untapped. To this end, we propose Phi-SegNet, a CNN-based architecture that incorporates phase-aware information at both architectural and optimization levels. The network integrates Bi-Feature Mask Former (BFMF) modules that blend neighboring encoder features to reduce semantic gaps, and Reverse Fourier Attention (RFA) blocks that refine decoder outputs using phase-regularized features. A dedicated phase-aware loss aligns these features with structural priors, forming a closed feedback loop that emphasizes boundary precision. Evaluated on five public datasets spanning X-ray, US, histopathology, MRI, and colonoscopy, Phi-SegNet consistently achieved state-of-the-art performance, with an average relative improvement of 1.54+/-1.26% in IoU and 0.98+/-0.71% in F1-score over the next best-performing model. In cross-dataset generalization scenarios involving unseen datasets from the known domain, Phi-SegNet also exhibits robust and superior performance, highlighting its adaptability and modality-agnostic design. These findings demonstrate the potential of leveraging spectral priors in both feature representation and supervision, paving the way for generalized segmentation frameworks that excel in fine-grained object localization.

  </details>



- **ProGiDiff: Prompt-Guided Diffusion-Based Medical Image Segmentation**  
  Yuan Lin, Murong Xu, Marc Hölle, Chinmay Prabhakar, Andreas Maier, Vasileios Belagiannis, Bjoern Menze, Suprosanna Shit  
  _2026-01-22_ · https://arxiv.org/abs/2601.16060v1  
  <details><summary>Abstract</summary>

  Widely adopted medical image segmentation methods, although efficient, are primarily deterministic and remain poorly amenable to natural language prompts. Thus, they lack the capability to estimate multiple proposals, human interaction, and cross-modality adaptation. Recently, text-to-image diffusion models have shown potential to bridge the gap. However, training them from scratch requires a large dataset-a limitation for medical image segmentation. Furthermore, they are often limited to binary segmentation and cannot be conditioned on a natural language prompt. To this end, we propose a novel framework called ProGiDiff that leverages existing image generation models for medical image segmentation purposes. Specifically, we propose a ControlNet-style conditioning mechanism with a custom encoder, suitable for image conditioning, to steer a pre-trained diffusion model to output segmentation masks. It naturally extends to a multi-class setting simply by prompting the target organ. Our experiment on organ segmentation from CT images demonstrates strong performance compared to previous methods and could greatly benefit from an expert-in-the-loop setting to leverage multiple proposals. Importantly, we demonstrate that the learned conditioning mechanism can be easily transferred through low-rank, few-shot adaptation to segment MR images.

  </details>



- **Keyframe-Based Feed-Forward Visual Odometry**  
  Weichen Dai, Wenhan Su, Da Kong, Yuhang Ming, Wanzeng Kong  
  _2026-01-22_ · https://arxiv.org/abs/2601.16020v1  
  <details><summary>Abstract</summary>

  The emergence of visual foundation models has revolutionized visual odometry~(VO) and SLAM, enabling pose estimation and dense reconstruction within a single feed-forward network. However, unlike traditional pipelines that leverage keyframe methods to enhance efficiency and accuracy, current foundation model based methods, such as VGGT-Long, typically process raw image sequences indiscriminately. This leads to computational redundancy and degraded performance caused by low inter-frame parallax, which provides limited contextual stereo information. Integrating traditional geometric heuristics into these methods is non-trivial, as their performance depends on high-dimensional latent representations rather than explicit geometric metrics. To bridge this gap, we propose a novel keyframe-based feed-forward VO. Instead of relying on hand-crafted rules, our approach employs reinforcement learning to derive an adaptive keyframe policy in a data-driven manner, aligning selection with the intrinsic characteristics of the underlying foundation model. We train our agent on TartanAir dataset and conduct extensive evaluations across several real-world datasets. Experimental results demonstrate that the proposed method achieves consistent and substantial improvements over state-of-the-art feed-forward VO methods.

  </details>



- **PhysicsMind: Sim and Real Mechanics Benchmarking for Physical Reasoning and Prediction in Foundational VLMs and World Models**  
  Chak-Wing Mak, Guanyu Zhu, Boyi Zhang, Hongji Li, Xiaowei Chi, Kevin Zhang, Yichen Wu, Yangfan He, Chun-Kai Fan, Wentao Lu, et al.  
  _2026-01-22_ · https://arxiv.org/abs/2601.16007v1  
  <details><summary>Abstract</summary>

  Modern foundational Multimodal Large Language Models (MLLMs) and video world models have advanced significantly in mathematical, common-sense, and visual reasoning, but their grasp of the underlying physics remains underexplored. Existing benchmarks attempting to measure this matter rely on synthetic, Visual Question Answer templates or focus on perceptual video quality that is tangential to measuring how well the video abides by physical laws. To address this fragmentation, we introduce PhysicsMind, a unified benchmark with both real and simulation environments that evaluates law-consistent reasoning and generation over three canonical principles: Center of Mass, Lever Equilibrium, and Newton's First Law. PhysicsMind comprises two main tasks: i) VQA tasks, testing whether models can reason and determine physical quantities and values from images or short videos, and ii) Video Generation(VG) tasks, evaluating if predicted motion trajectories obey the same center-of-mass, torque, and inertial constraints as the ground truth. A broad range of recent models and video generation models is evaluated on PhysicsMind and found to rely on appearance heuristics while often violating basic mechanics. These gaps indicate that current scaling and training are still insufficient for robust physical understanding, underscoring PhysicsMind as a focused testbed for physics-aware multimodal models. Our data will be released upon acceptance.

  </details>



- **PUMA: Perception-driven Unified Foothold Prior for Mobility Augmented Quadruped Parkour**  
  Liang Wang, Kanzhong Yao, Yang Liu, Weikai Qin, Jun Wu, Zhe Sun, Qiuguo Zhu  
  _2026-01-22_ · https://arxiv.org/abs/2601.15995v1  
  <details><summary>Abstract</summary>

  Parkour tasks for quadrupeds have emerged as a promising benchmark for agile locomotion. While human athletes can effectively perceive environmental characteristics to select appropriate footholds for obstacle traversal, endowing legged robots with similar perceptual reasoning remains a significant challenge. Existing methods often rely on hierarchical controllers that follow pre-computed footholds, thereby constraining the robot's real-time adaptability and the exploratory potential of reinforcement learning. To overcome these challenges, we present PUMA, an end-to-end learning framework that integrates visual perception and foothold priors into a single-stage training process. This approach leverages terrain features to estimate egocentric polar foothold priors, composed of relative distance and heading, guiding the robot in active posture adaptation for parkour tasks. Extensive experiments conducted in simulation and real-world environments across various discrete complex terrains, demonstrate PUMA's exceptional agility and robustness in challenging scenarios.

  </details>



- **A Multi-View Pipeline and Benchmark Dataset for 3D Hand Pose Estimation in Surgery**  
  Valery Fischer, Alan Magdaleno, Anna-Katharina Calek, Nicola Cavalcanti, Nathan Hoffman, Christoph Germann, Joschua Wüthrich, Max Krähenmann, Mazda Farshad, Philipp Fürnstahl, et al.  
  _2026-01-22_ · https://arxiv.org/abs/2601.15918v1  
  <details><summary>Abstract</summary>

  Purpose: Accurate 3D hand pose estimation supports surgical applications such as skill assessment, robot-assisted interventions, and geometry-aware workflow analysis. However, surgical environments pose severe challenges, including intense and localized lighting, frequent occlusions by instruments or staff, and uniform hand appearance due to gloves, combined with a scarcity of annotated datasets for reliable model training. Method: We propose a robust multi-view pipeline for 3D hand pose estimation in surgical contexts that requires no domain-specific fine-tuning and relies solely on off-the-shelf pretrained models. The pipeline integrates reliable person detection, whole-body pose estimation, and state-of-the-art 2D hand keypoint prediction on tracked hand crops, followed by a constrained 3D optimization. In addition, we introduce a novel surgical benchmark dataset comprising over 68,000 frames and 3,000 manually annotated 2D hand poses with triangulated 3D ground truth, recorded in a replica operating room under varying levels of scene complexity. Results: Quantitative experiments demonstrate that our method consistently outperforms baselines, achieving a 31% reduction in 2D mean joint error and a 76% reduction in 3D mean per-joint position error. Conclusion: Our work establishes a strong baseline for 3D hand pose estimation in surgery, providing both a training-free pipeline and a comprehensive annotated dataset to facilitate future research in surgical computer vision.

  </details>



- **The Latency Wall: Benchmarking Off-the-Shelf Emotion Recognition for Real-Time Virtual Avatars**  
  Yarin Benyamin  
  _2026-01-22_ · https://arxiv.org/abs/2601.15914v1  
  <details><summary>Abstract</summary>

  In the realm of Virtual Reality (VR) and Human-Computer Interaction (HCI), real-time emotion recognition shows promise for supporting individuals with Autism Spectrum Disorder (ASD) in improving social skills. This task requires a strict latency-accuracy trade-off, with motion-to-photon (MTP) latency kept below 140 ms to maintain contingency. However, most off-the-shelf Deep Learning models prioritize accuracy over the strict timing constraints of commodity hardware. As a first step toward accessible VR therapy, we benchmark State-of-the-Art (SOTA) models for Zero-Shot Facial Expression Recognition (FER) on virtual characters using the UIBVFED dataset. We evaluate Medium and Nano variants of YOLO (v8, v11, and v12) for face detection, alongside general-purpose Vision Transformers including CLIP, SigLIP, and ViT-FER.Our results on CPU-only inference demonstrate that while face detection on stylized avatars is robust (100% accuracy), a "Latency Wall" exists in the classification stage. The YOLOv11n architecture offers the optimal balance for detection (~54 ms). However, general-purpose Transformers like CLIP and SigLIP fail to achieve viable accuracy (<23%) or speed (>150 ms) for real-time loops. This study highlights the necessity for lightweight, domain-specific architectures to enable accessible, real-time AI in therapeutic settings.

  </details>



- **ThermoSplat: Cross-Modal 3D Gaussian Splatting with Feature Modulation and Geometry Decoupling**  
  Zhaoqi Su, Shihai Chen, Xinyan Lin, Liqin Huang, Zhipeng Su, Xiaoqiang Lu  
  _2026-01-22_ · https://arxiv.org/abs/2601.15897v1  
  <details><summary>Abstract</summary>

  Multi-modal scene reconstruction integrating RGB and thermal infrared data is essential for robust environmental perception across diverse lighting and weather conditions. However, extending 3D Gaussian Splatting (3DGS) to multi-spectral scenarios remains challenging. Current approaches often struggle to fully leverage the complementary information of multi-modal data, typically relying on mechanisms that either tend to neglect cross-modal correlations or leverage shared representations that fail to adaptively handle the complex structural correlations and physical discrepancies between spectrums. To address these limitations, we propose ThermoSplat, a novel framework that enables deep spectral-aware reconstruction through active feature modulation and adaptive geometry decoupling. First, we introduce a Cross-Modal FiLM Modulation mechanism that dynamically conditions shared latent features on thermal structural priors, effectively guiding visible texture synthesis with reliable cross-modal geometric cues. Second, to accommodate modality-specific geometric inconsistencies, we propose a Modality-Adaptive Geometric Decoupling scheme that learns independent opacity offsets and executes an independent rasterization pass for the thermal branch. Additionally, a hybrid rendering pipeline is employed to integrate explicit Spherical Harmonics with implicit neural decoding, ensuring both semantic consistency and high-frequency detail preservation. Extensive experiments on the RGBT-Scenes dataset demonstrate that ThermoSplat achieves state-of-the-art rendering quality across both visible and thermal spectrums.

  </details>



- **PMPBench: A Paired Multi-Modal Pan-Cancer Benchmark for Medical Image Synthesis**  
  Yifan Chen, Fei Yin, Hao Chen, Jia Wu, Chao Li  
  _2026-01-22_ · https://arxiv.org/abs/2601.15884v1  
  <details><summary>Abstract</summary>

  Contrast medium plays a pivotal role in radiological imaging, as it amplifies lesion conspicuity and improves detection for the diagnosis of tumor-related diseases. However, depending on the patient's health condition or the medical resources available, the use of contrast medium is not always feasible. Recent work has explored AI-based image translation to synthesize contrast-enhanced images directly from non-contrast scans, aims to reduce side effects and streamlines clinical workflows. Progress in this direction has been constrained by data limitations: (1) existing public datasets focus almost exclusively on brain-related paired MR modalities; (2) other collections include partially paired data but suffer from missing modalities/timestamps and imperfect spatial alignment; (3) explicit labeling of CT vs. CTC or DCE phases is often absent; (4) substantial resources remain private. To bridge this gap, we introduce the first public, fully paired, pan-cancer medical imaging dataset spanning 11 human organs. The MR data include complete dynamic contrast-enhanced (DCE) sequences covering all three phases (DCE1-DCE3), while the CT data provide paired non-contrast and contrast-enhanced acquisitions (CTC). The dataset is curated for anatomical correspondence, enabling rigorous evaluation of 1-to-1, N-to-1, and N-to-N translation settings (e.g., predicting DCE phases from non-contrast inputs). Built upon this resource, we establish a comprehensive benchmark. We report results from representative baselines of contemporary image-to-image translation. We release the dataset and benchmark to catalyze research on safe, effective contrast synthesis, with direct relevance to multi-organ oncology imaging workflows. Our code and dataset are publicly available at https://github.com/YifanChen02/PMPBench.

  </details>



- **Towards Realistic Remote Sensing Dataset Distillation with Discriminative Prototype-guided Diffusion**  
  Yonghao Xu, Pedram Ghamisi, Qihao Weng  
  _2026-01-22_ · https://arxiv.org/abs/2601.15829v1  
  <details><summary>Abstract</summary>

  Recent years have witnessed the remarkable success of deep learning in remote sensing image interpretation, driven by the availability of large-scale benchmark datasets. However, this reliance on massive training data also brings two major challenges: (1) high storage and computational costs, and (2) the risk of data leakage, especially when sensitive categories are involved. To address these challenges, this study introduces the concept of dataset distillation into the field of remote sensing image interpretation for the first time. Specifically, we train a text-to-image diffusion model to condense a large-scale remote sensing dataset into a compact and representative distilled dataset. To improve the discriminative quality of the synthesized samples, we propose a classifier-driven guidance by injecting a classification consistency loss from a pre-trained model into the diffusion training process. Besides, considering the rich semantic complexity of remote sensing imagery, we further perform latent space clustering on training samples to select representative and diverse prototypes as visual style guidance, while using a visual language model to provide aggregated text descriptions. Experiments on three high-resolution remote sensing scene classification benchmarks show that the proposed method can distill realistic and diverse samples for downstream model training. Code and pre-trained models are available online (https://github.com/YonghaoXu/DPD).

  </details>



- **Beyond Off-the-Shelf Models: A Lightweight and Accessible Machine Learning Pipeline for Ecologists Working with Image Data**  
  Clare Chemery, Hendrik Edelhoff, Ludwig Bothmann  
  _2026-01-22_ · https://arxiv.org/abs/2601.15813v1  
  <details><summary>Abstract</summary>

  We introduce a lightweight experimentation pipeline designed to lower the barrier for applying machine learning (ML) methods for classifying images in ecological research. We enable ecologists to experiment with ML models independently, thus they can move beyond off-the-shelf models and generate insights tailored to local datasets and specific classification tasks and target variables. Our tool combines a simple command-line interface for preprocessing, training, and evaluation with a graphical interface for annotation, error analysis, and model comparison. This design enables ecologists to build and iterate on compact, task-specific classifiers without requiring advanced ML expertise. As a proof of concept, we apply the pipeline to classify red deer (Cervus elaphus) by age and sex from 3392 camera trap images collected in the Veldenstein Forest, Germany. Using 4352 cropped images containing individual deer labeled by experts, we trained and evaluated multiple backbone architectures with a wide variety of parameters and data augmentation strategies. Our best-performing models achieved 90.77% accuracy for age classification and 96.15% for sex classification. These results demonstrate that reliable demographic classification is feasible even with limited data to answer narrow, well-defined ecological problems. More broadly, the framework provides ecologists with an accessible tool for developing ML models tailored to specific research questions, paving the way for broader adoption of ML in wildlife monitoring and demographic analysis.

  </details>



- **Assessing Situational and Spatial Awareness of VLMs with Synthetically Generated Video**  
  Pascal Benschop, Justin Dauwels, Jan van Gemert  
  _2026-01-22_ · https://arxiv.org/abs/2601.15780v1  
  <details><summary>Abstract</summary>

  Spatial reasoning in vision language models (VLMs) remains fragile when semantics hinge on subtle temporal or geometric cues. We introduce a synthetic benchmark that probes two complementary skills: situational awareness (recognizing whether an interaction is harmful or benign) and spatial awareness (tracking who does what to whom, and reasoning about relative positions and motion). Through minimal video pairs, we test three challenges: distinguishing violence from benign activity, binding assailant roles across viewpoints, and judging fine-grained trajectory alignment. While we evaluate recent VLMs in a training-free setting, the benchmark is applicable to any video classification model. Results show performance only slightly above chance across tasks. A simple aid, stable color cues, partly reduces assailant role confusions but does not resolve the underlying weakness. By releasing data and code, we aim to provide reproducible diagnostics and seed exploration of lightweight spatial priors to complement large-scale pretraining.

  </details>



- **Diffusion Model-Based Data Augmentation for Enhanced Neuron Segmentation**  
  Liuyun Jiang, Yanchao Zhang, Jinyue Guo, Yizhuo Lu, Ruining Zhou, Hua Han  
  _2026-01-22_ · https://arxiv.org/abs/2601.15779v1  
  <details><summary>Abstract</summary>

  Neuron segmentation in electron microscopy (EM) aims to reconstruct the complete neuronal connectome; however, current deep learning-based methods are limited by their reliance on large-scale training data and extensive, time-consuming manual annotations. Traditional methods augment the training set through geometric and photometric transformations; however, the generated samples remain highly correlated with the original images and lack structural diversity. To address this limitation, we propose a diffusion-based data augmentation framework capable of generating diverse and structurally plausible image-label pairs for neuron segmentation. Specifically, the framework employs a resolution-aware conditional diffusion model with multi-scale conditioning and EM resolution priors to enable voxel-level image synthesis from 3D masks. It further incorporates a biology-guided mask remodeling module that produces augmented masks with enhanced structural realism. Together, these components effectively enrich the training set and improve segmentation performance. On the AC3 and AC4 datasets under low-annotation regimes, our method improves the ARAND metric by 32.1% and 30.7%, respectively, when combined with two different post-processing methods. Our code is available at https://github.com/HeadLiuYun/NeuroDiff.

  </details>



- **Atlas-Assisted Segment Anything Model for Fetal Brain MRI (FeTal-SAM)**  
  Qi Zeng, Weide Liu, Bo Li, Ryne Didier, P. Ellen Grant, Davood Karimi  
  _2026-01-22_ · https://arxiv.org/abs/2601.15759v1  
  <details><summary>Abstract</summary>

  This paper presents FeTal-SAM, a novel adaptation of the Segment Anything Model (SAM) tailored for fetal brain MRI segmentation. Traditional deep learning methods often require large annotated datasets for a fixed set of labels, making them inflexible when clinical or research needs change. By integrating atlas-based prompts and foundation-model principles, FeTal-SAM addresses two key limitations in fetal brain MRI segmentation: (1) the need to retrain models for varying label definitions, and (2) the lack of insight into whether segmentations are driven by genuine image contrast or by learned spatial priors. We leverage multi-atlas registration to generate spatially aligned label templates that serve as dense prompts, alongside a bounding-box prompt, for SAM's segmentation decoder. This strategy enables binary segmentation on a per-structure basis, which is subsequently fused to reconstruct the full 3D segmentation volumes. Evaluations on two datasets, the dHCP dataset and an in-house dataset demonstrate FeTal-SAM's robust performance across gestational ages. Notably, it achieves Dice scores comparable to state-of-the-art baselines which were trained for each dataset and label definition for well-contrasted structures like cortical plate and cerebellum, while maintaining the flexibility to segment any user-specified anatomy. Although slightly lower accuracy is observed for subtle, low-contrast structures (e.g., hippocampus, amygdala), our results highlight FeTal-SAM's potential to serve as a general-purpose segmentation model without exhaustive retraining. This method thus constitutes a promising step toward clinically adaptable fetal brain MRI analysis tools.

  </details>



- **Sub-Region-Aware Modality Fusion and Adaptive Prompting for Multi-Modal Brain Tumor Segmentation**  
  Shadi Alijani, Fereshteh Aghaee Meibodi, Homayoun Najjaran  
  _2026-01-22_ · https://arxiv.org/abs/2601.15734v1  
  <details><summary>Abstract</summary>

  The successful adaptation of foundation models to multi-modal medical imaging is a critical yet unresolved challenge. Existing models often struggle to effectively fuse information from multiple sources and adapt to the heterogeneous nature of pathological tissues. To address this, we introduce a novel framework for adapting foundation models to multi-modal medical imaging, featuring two key technical innovations: sub-region-aware modality attention and adaptive prompt engineering. The attention mechanism enables the model to learn the optimal combination of modalities for each tumor sub-region, while the adaptive prompting strategy leverages the inherent capabilities of foundation models to refine segmentation accuracy. We validate our framework on the BraTS 2020 brain tumor segmentation dataset, demonstrating that our approach significantly outperforms baseline methods, particularly in the challenging necrotic core sub-region. Our work provides a principled and effective approach to multi-modal fusion and prompting, paving the way for more accurate and robust foundation model-based solutions in medical imaging.

  </details>



- **VideoThinker: Building Agentic VideoLLMs with LLM-Guided Tool Reasoning**  
  Chenglin Li, Qianglong Chen, Feng Han, Yikun Wang, Xingxi Yin, Yan Gong, Ruilin Li, Yin Zhang, Jiaqi Wang  
  _2026-01-22_ · https://arxiv.org/abs/2601.15724v1  
  <details><summary>Abstract</summary>

  Long-form video understanding remains a fundamental challenge for current Video Large Language Models. Most existing models rely on static reasoning over uniformly sampled frames, which weakens temporal localization and leads to substantial information loss in long videos. Agentic tools such as temporal retrieval, spatial zoom, and temporal zoom offer a natural way to overcome these limitations by enabling adaptive exploration of key moments. However, constructing agentic video understanding data requires models that already possess strong long-form video comprehension, creating a circular dependency. We address this challenge with VideoThinker, an agentic Video Large Language Model trained entirely on synthetic tool interaction trajectories. Our key idea is to convert videos into rich captions and employ a powerful agentic language model to generate multi-step tool use sequences in caption space. These trajectories are subsequently grounded back to video by replacing captions with the corresponding frames, yielding a large-scale interleaved video and tool reasoning dataset without requiring any long-form understanding from the underlying model. Training on this synthetic agentic dataset equips VideoThinker with dynamic reasoning capabilities, adaptive temporal exploration, and multi-step tool use. Remarkably, VideoThinker significantly outperforms both caption-only language model agents and strong video model baselines across long-video benchmarks, demonstrating the effectiveness of tool augmented synthetic data and adaptive retrieval and zoom reasoning for long-form video understanding.

  </details>



- **Zero-Shot Product Attribute Labeling with Vision-Language Models: A Three-Tier Evaluation Framework**  
  Shubham Shukla, Kunal Sonalkar  
  _2026-01-22_ · https://arxiv.org/abs/2601.15711v1  
  <details><summary>Abstract</summary>

  Fine-grained attribute prediction is essential for fashion retail applications including catalog enrichment, visual search, and recommendation systems. Vision-Language Models (VLMs) offer zero-shot prediction without task-specific training, yet their systematic evaluation on multi-attribute fashion tasks remains underexplored. A key challenge is that fashion attributes are often conditional. For example, "outer fabric" is undefined when no outer garment is visible. This requires models to detect attribute applicability before attempting classification. We introduce a three-tier evaluation framework that decomposes this challenge: (1) overall task performance across all classes (including NA class: suggesting attribute is not applicable) for all attributes, (2) attribute applicability detection, and (3) fine-grained classification when attributes are determinable. Using DeepFashion-MultiModal, which explicitly defines NA (meaning attribute doesn't exist or is not visible) within attribute label spaces, we benchmark nine VLMs spanning flagship (GPT-5, Gemini 2.5 Pro), efficient (GPT-5 Mini, Gemini 2.5 Flash), and ultra-efficient tiers (GPT-5 Nano, Gemini 2.5 Flash-Lite) against classifiers trained on pretrained Fashion-CLIP embeddings on 5,000 images across 18 attributes. Our findings reveal that: (1) zero-shot VLMs achieve 64.0% macro-F1, a threefold improvement over logistic regression on pretrained Fashion-CLIP embeddings; (2) VLMs excel at fine-grained classification (Tier 3: 70.8% F1) but struggle with applicability detection (Tier 2: 34.1% NA-F1), identifying a key bottleneck; (3) efficient models achieve over 90% of flagship performance at lower cost, offering practical deployment paths. This diagnostic framework enables practitioners to pinpoint whether errors stem from visibility detection or classification, guiding targeted improvements for production systems.

  </details>



- **Enhanced LULC Segmentation via Lightweight Model Refinements on ALOS-2 SAR Data**  
  Ali Caglayan, Nevrez Imamoglu, Toru Kouyama  
  _2026-01-22_ · https://arxiv.org/abs/2601.15705v1  
  <details><summary>Abstract</summary>

  This work focuses on national-scale land-use/land-cover (LULC) semantic segmentation using ALOS-2 single-polarization (HH) SAR data over Japan, together with a companion binary water detection task. Building on SAR-W-MixMAE self-supervised pretraining [1], we address common SAR dense-prediction failure modes, boundary over-smoothing, missed thin/slender structures, and rare-class degradation under long-tailed labels, without increasing pipeline complexity. We introduce three lightweight refinements: (i) injecting high-resolution features into multi-scale decoding, (ii) a progressive refine-up head that alternates convolutional refinement and stepwise upsampling, and (iii) an $α$-scale factor that tempers class reweighting within a focal+dice objective. The resulting model yields consistent improvements on the Japan-wide ALOS-2 LULC benchmark, particularly for under-represented classes, and improves water detection across standard evaluation metrics.

  </details>



- **Performance-guided Reinforced Active Learning for Object Detection**  
  Zhixuan Liang, Xingyu Zeng, Rui Zhao, Ping Luo  
  _2026-01-22_ · https://arxiv.org/abs/2601.15688v1  
  <details><summary>Abstract</summary>

  Active learning (AL) strategies aim to train high-performance models with minimal labeling efforts, only selecting the most informative instances for annotation. Current approaches to evaluating data informativeness predominantly focus on the data's distribution or intrinsic information content and do not directly correlate with downstream task performance, such as mean average precision (mAP) in object detection. Thus, we propose Performance-guided (i.e. mAP-guided) Reinforced Active Learning for Object Detection (MGRAL), a novel approach that leverages the concept of expected model output changes as informativeness. To address the combinatorial explosion challenge of batch sample selection and the non-differentiable correlation between model performance and selected batches, MGRAL skillfully employs a reinforcement learning-based sampling agent that optimizes selection using policy gradient with mAP improvement as reward. Moreover, to reduce the computational overhead of mAP estimation with unlabeled samples, MGRAL utilizes an unsupervised way with fast look-up tables, ensuring feasible deployment. We evaluate MGRAL's active learning performance on detection tasks over PASCAL VOC and COCO benchmarks. Our approach demonstrates the highest AL curve with convincing visualizations, establishing a new paradigm in reinforcement learning-driven active object detection.

  </details>



- **Consistency-Regularized GAN for Few-Shot SAR Target Recognition**  
  Yikui Zhai, Shikuang Liu, Wenlve Zhou, Hongsheng Zhang, Zhiheng Zhou, Xiaolin Tian, C. L. Philip Chen  
  _2026-01-22_ · https://arxiv.org/abs/2601.15681v1  
  <details><summary>Abstract</summary>

  Few-shot recognition in synthetic aperture radar (SAR) imagery remains a critical bottleneck for real-world applications due to extreme data scarcity. A promising strategy involves synthesizing a large dataset with a generative adversarial network (GAN), pre-training a model via self-supervised learning (SSL), and then fine-tuning on the few labeled samples. However, this approach faces a fundamental paradox: conventional GANs themselves require abundant data for stable training, contradicting the premise of few-shot learning. To resolve this, we propose the consistency-regularized generative adversarial network (Cr-GAN), a novel framework designed to synthesize diverse, high-fidelity samples even when trained under these severe data limitations. Cr-GAN introduces a dual-branch discriminator that decouples adversarial training from representation learning. This architecture enables a channel-wise feature interpolation strategy to create novel latent features, complemented by a dual-domain cycle consistency mechanism that ensures semantic integrity. Our Cr-GAN framework is adaptable to various GAN architectures, and its synthesized data effectively boosts multiple SSL algorithms. Extensive experiments on the MSTAR and SRSDD datasets validate our approach, with Cr-GAN achieving a highly competitive accuracy of 71.21% and 51.64%, respectively, in the 8-shot setting, significantly outperforming leading baselines, while requiring only ~5 of the parameters of state-of-the-art diffusion models. Code is available at: https://github.com/yikuizhai/Cr-GAN.

  </details>



- **Skywork UniPic 3.0: Unified Multi-Image Composition via Sequence Modeling**  
  Hongyang Wei, Hongbo Liu, Zidong Wang, Yi Peng, Baixin Xu, Size Wu, Xuying Zhang, Xianglong He, Zexiang Liu, Peiyu Wang, et al.  
  _2026-01-22_ · https://arxiv.org/abs/2601.15664v1  
  <details><summary>Abstract</summary>

  The recent surge in popularity of Nano-Banana and Seedream 4.0 underscores the community's strong interest in multi-image composition tasks. Compared to single-image editing, multi-image composition presents significantly greater challenges in terms of consistency and quality, yet existing models have not disclosed specific methodological details for achieving high-quality fusion. Through statistical analysis, we identify Human-Object Interaction (HOI) as the most sought-after category by the community. We therefore systematically analyze and implement a state-of-the-art solution for multi-image composition with a primary focus on HOI-centric tasks. We present Skywork UniPic 3.0, a unified multimodal framework that integrates single-image editing and multi-image composition. Our model supports an arbitrary (1~6) number and resolution of input images, as well as arbitrary output resolutions (within a total pixel budget of 1024x1024). To address the challenges of multi-image composition, we design a comprehensive data collection, filtering, and synthesis pipeline, achieving strong performance with only 700K high-quality training samples. Furthermore, we introduce a novel training paradigm that formulates multi-image composition as a sequence-modeling problem, transforming conditional generation into unified sequence synthesis. To accelerate inference, we integrate trajectory mapping and distribution matching into the post-training stage, enabling the model to produce high-fidelity samples in just 8 steps and achieve a 12.5x speedup over standard synthesis sampling. Skywork UniPic 3.0 achieves state-of-the-art performance on single-image editing benchmark and surpasses both Nano-Banana and Seedream 4.0 on multi-image composition benchmark, thereby validating the effectiveness of our data pipeline and training paradigm. Code, models and dataset are publicly available.

  </details>



- **Explainable Deepfake Detection with RL Enhanced Self-Blended Images**  
  Ning Jiang, Dingheng Zeng, Yanhong Liu, Haiyang Yi, Shijie Yu, Minghe Weng, Haifeng Shen, Ying Li  
  _2026-01-22_ · https://arxiv.org/abs/2601.15624v1  
  <details><summary>Abstract</summary>

  Most prior deepfake detection methods lack explainable outputs. With the growing interest in multimodal large language models (MLLMs), researchers have started exploring their use in interpretable deepfake detection. However, a major obstacle in applying MLLMs to this task is the scarcity of high-quality datasets with detailed forgery attribution annotations, as textual annotation is both costly and challenging - particularly for high-fidelity forged images or videos. Moreover, multiple studies have shown that reinforcement learning (RL) can substantially enhance performance in visual tasks, especially in improving cross-domain generalization. To facilitate the adoption of mainstream MLLM frameworks in deepfake detection with reduced annotation cost, and to investigate the potential of RL in this context, we propose an automated Chain-of-Thought (CoT) data generation framework based on Self-Blended Images, along with an RL-enhanced deepfake detection framework. Extensive experiments validate the effectiveness of our CoT data construction pipeline, tailored reward mechanism, and feedback-driven synthetic data generation approach. Our method achieves performance competitive with state-of-the-art (SOTA) approaches across multiple cross-dataset benchmarks. Implementation details are available at https://github.com/deon1219/rlsbi.

  </details>



- **AION: Aerial Indoor Object-Goal Navigation Using Dual-Policy Reinforcement Learning**  
  Zichen Yan, Yuchen Hou, Shenao Wang, Yichao Gao, Rui Huang, Lin Zhao  
  _2026-01-22_ · https://arxiv.org/abs/2601.15614v1  
  <details><summary>Abstract</summary>

  Object-Goal Navigation (ObjectNav) requires an agent to autonomously explore an unknown environment and navigate toward target objects specified by a semantic label. While prior work has primarily studied zero-shot ObjectNav under 2D locomotion, extending it to aerial platforms with 3D locomotion capability remains underexplored. Aerial robots offer superior maneuverability and search efficiency, but they also introduce new challenges in spatial perception, dynamic control, and safety assurance. In this paper, we propose AION for vision-based aerial ObjectNav without relying on external localization or global maps. AION is an end-to-end dual-policy reinforcement learning (RL) framework that decouples exploration and goal-reaching behaviors into two specialized policies. We evaluate AION on the AI2-THOR benchmark and further assess its real-time performance in IsaacSim using high-fidelity drone models. Experimental results show that AION achieves superior performance across comprehensive evaluation metrics in exploration, navigation efficiency, and safety. The video can be found at https://youtu.be/TgsUm6bb7zg.

  </details>



- **FUGC: Benchmarking Semi-Supervised Learning Methods for Cervical Segmentation**  
  Jieyun Bai, Yitong Tang, Zihao Zhou, Mahdi Islam, Musarrat Tabassum, Enrique Almar-Munoz, Hongyu Liu, Hui Meng, Nianjiang Lv, Bo Deng, et al.  
  _2026-01-22_ · https://arxiv.org/abs/2601.15572v1  
  <details><summary>Abstract</summary>

  Accurate segmentation of cervical structures in transvaginal ultrasound (TVS) is critical for assessing the risk of spontaneous preterm birth (PTB), yet the scarcity of labeled data limits the performance of supervised learning approaches. This paper introduces the Fetal Ultrasound Grand Challenge (FUGC), the first benchmark for semi-supervised learning in cervical segmentation, hosted at ISBI 2025. FUGC provides a dataset of 890 TVS images, including 500 training images, 90 validation images, and 300 test images. Methods were evaluated using the Dice Similarity Coefficient (DSC), Hausdorff Distance (HD), and runtime (RT), with a weighted combination of 0.4/0.4/0.2. The challenge attracted 10 teams with 82 participants submitting innovative solutions. The best-performing methods for each individual metric achieved 90.26\% mDSC, 38.88 mHD, and 32.85 ms RT, respectively. FUGC establishes a standardized benchmark for cervical segmentation, demonstrates the efficacy of semi-supervised methods with limited labeled data, and provides a foundation for AI-assisted clinical PTB risk assessment.

  </details>



- **VIOLA: Towards Video In-Context Learning with Minimal Annotations**  
  Ryo Fujii, Hideo Saito, Ryo Hachiuma  
  _2026-01-22_ · https://arxiv.org/abs/2601.15549v1  
  <details><summary>Abstract</summary>

  Generalizing Multimodal Large Language Models (MLLMs) to novel video domains is essential for real-world deployment but remains challenging due to the scarcity of labeled data. While In-Context Learning (ICL) offers a training-free adaptation path, standard methods rely on large annotated pools, which are often impractical in specialized environments like industrial or surgical settings since they require the experts' annotations. To bridge this gap, we introduce VIOLA (Video In-cOntext Learning with minimal Annotation), a label-efficient framework that synergizes minimal expert supervision with abundant unlabeled data. First, to maximize the efficiency of a strict annotation budget, we propose density-uncertainty-weighted sampling. Unlike standard diversity or uncertainty strategies that risk selecting visual outliers, our method leverages density estimation to identify samples that are simultaneously diverse, representative, and informative. Second, to utilize the remaining unlabeled data without noise propagation, we construct a hybrid pool and introduce confidence-aware retrieval and confidence-aware prompting. These mechanisms explicitly model label reliability, retrieving demonstrations based on a composite score of similarity and confidence while enabling the MLLM to adaptively distinguish between verified ground truths and noisy pseudo-labels. Extensive experiments across nine diverse benchmarks using four MLLMs demonstrate that our framework significantly outperforms various baselines in low-resource settings, achieving robust adaptation with minimal annotation costs.

  </details>



- **A Machine Vision Approach to Preliminary Skin Lesion Assessments**  
  Ali Khreis, Ro'Yah Radaideh, Quinn McGill  
  _2026-01-21_ · https://arxiv.org/abs/2601.15539v1  
  <details><summary>Abstract</summary>

  Early detection of malignant skin lesions is critical for improving patient outcomes in aggressive, metastatic skin cancers. This study evaluates a comprehensive system for preliminary skin lesion assessment that combines the clinically established ABCD rule of dermoscopy (analyzing Asymmetry, Borders, Color, and Dermoscopic Structures) with machine learning classification. Using a 1,000-image subset of the HAM10000 dataset, the system implements an automated, rule-based pipeline to compute a Total Dermoscopy Score (TDS) for each lesion. This handcrafted approach is compared against various machine learning solutions, including traditional classifiers (Logistic Regression, Random Forest, and SVM) and deep learning models. While the rule-based system provides high clinical interpretability, results indicate a performance bottleneck when reducing complex morphology to five numerical features. Experimental findings show that transfer learning with EfficientNet-B0 failed significantly due to domain shift between natural and medical images. In contrast, a custom three-layer Convolutional Neural Network (CNN) trained from scratch achieved 78.5% accuracy and 86.5% recall on median-filtered images, representing a 19-point accuracy improvement over traditional methods. The results demonstrate that direct pixel-level learning captures diagnostic patterns beyond handcrafted features and that purpose-built lightweight architectures can outperform large pretrained models for small, domain-specific medical datasets.

  </details>



- **Controllable Layered Image Generation for Real-World Editing**  
  Jinrui Yang, Qing Liu, Yijun Li, Mengwei Ren, Letian Zhang, Zhe Lin, Cihang Xie, Yuyin Zhou  
  _2026-01-21_ · https://arxiv.org/abs/2601.15507v1  
  <details><summary>Abstract</summary>

  Recent image generation models have shown impressive progress, yet they often struggle to yield controllable and consistent results when users attempt to edit specific elements within an existing image. Layered representations enable flexible, user-driven content creation, but existing approaches often fail to produce layers with coherent compositing relationships, and their object layers typically lack realistic visual effects such as shadows and reflections. To overcome these limitations, we propose LASAGNA, a novel, unified framework that generates an image jointly with its composing layers--a photorealistic background and a high-quality transparent foreground with compelling visual effects. Unlike prior work, LASAGNA efficiently learns correct image composition from a wide range of conditioning inputs--text prompts, foreground, background, and location masks--offering greater controllability for real-world applications. To enable this, we introduce LASAGNA-48K, a new dataset composed of clean backgrounds and RGBA foregrounds with physically grounded visual effects. We also propose LASAGNABENCH, the first benchmark for layer editing. We demonstrate that LASAGNA excels in generating highly consistent and coherent results across multiple image layers simultaneously, enabling diverse post-editing applications that accurately preserve identity and visual effects. LASAGNA-48K and LASAGNABENCH will be publicly released to foster open research in the community. The project page is https://rayjryang.github.io/LASAGNA-Page/.

  </details>



- **Hybrid Vision Transformer_GAN Attribute Neutralizer for Mitigating Bias in Chest X_Ray Diagnosis**  
  Jobeal Solomon, Ali Mohammed Mansoor Alsahag, Seyed Sahand Mohammadi Ziabari  
  _2026-01-21_ · https://arxiv.org/abs/2601.15490v1  
  <details><summary>Abstract</summary>

  Bias in chest X-ray classifiers frequently stems from sex- and age-related shortcuts, leading to systematic underdiagnosis of minority subgroups. Previous pixel-space attribute neutralizers, which rely on convolutional encoders, lessen but do not fully remove this attribute leakage at clinically usable edit strengths. This study evaluates whether substituting the U-Net convolutional encoder with a Vision Transformer backbone in the Attribute-Neutral Framework can reduce demographic attribute leakage while preserving diagnostic accuracy. A data-efficient Image Transformer Small (DeiT-S) neutralizer was trained on the ChestX-ray14 dataset. Its edited images, generated across eleven edit-intensity levels, were evaluated with an independent AI judge for attribute leakage and with a convolutional neural network (ConvNet) for disease prediction. At a moderate edit level (alpha = 0.5), the Vision Transformer (ViT) neutralizer reduces patient sex-recognition area under the curve (AUC) to approximately 0.80, about 10 percentage points below the original framework's convolutional U-Net encoder, despite being trained for only half as many epochs. Meanwhile, macro receiver operating characteristic area under the curve (ROC AUC) across 15 findings stays within five percentage points of the unedited baseline, and the worst-case subgroup AUC remains near 0.70. These results indicate that global self-attention vision models can further suppress attribute leakage without sacrificing clinical utility, suggesting a practical route toward fairer chest X-ray AI.

  </details>



- **Neural Collision Detection for Multi-arm Laparoscopy Surgical Robots Through Learning-from-Simulation**  
  Sarvin Ghiasi, Majid Roshanfar, Jake Barralet, Liane S. Feldman, Amir Hooshiar  
  _2026-01-21_ · https://arxiv.org/abs/2601.15459v1  
  <details><summary>Abstract</summary>

  This study presents an integrated framework for enhancing the safety and operational efficiency of robotic arms in laparoscopic surgery by addressing key challenges in collision detection and minimum distance estimation. By combining analytical modeling, real-time simulation, and machine learning, the framework offers a robust solution for ensuring safe robotic operations. An analytical model was developed to estimate the minimum distances between robotic arms based on their joint configurations, offering precise theoretical calculations that serve as both a validation tool and a benchmark. To complement this, a 3D simulation environment was created to model two 7-DOF Kinova robotic arms, generating a diverse dataset of configurations for collision detection and distance estimation. Using these insights, a deep neural network model was trained with joint actuators of robot arms and relative positions as inputs, achieving a mean absolute error of 282.2 mm and an R-squared value of 0.85. The close alignment between predicted and actual distances highlights the network's accuracy and its ability to generalize spatial relationships. This work demonstrates the effectiveness of combining analytical precision with machine learning algorithms to enhance the precision and reliability of robotic systems.

  </details>



- **Evaluating Multimodal Large Language Models for Heterogeneous Face Recognition**  
  Hatef Otroshi Shahreza, Anjith George, Sébastien Marcel  
  _2026-01-21_ · https://arxiv.org/abs/2601.15406v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have recently demonstrated strong performance on a wide range of vision-language tasks, raising interest in their potential use for biometric applications. In this paper, we conduct a systematic evaluation of state-of-the-art MLLMs for heterogeneous face recognition (HFR), where enrollment and probe images are from different sensing modalities, including visual (VIS), near infrared (NIR), short-wave infrared (SWIR), and thermal camera. We benchmark multiple open-source MLLMs across several cross-modality scenarios, including VIS-NIR, VIS-SWIR, and VIS-THERMAL face recognition. The recognition performance of MLLMs is evaluated using biometric protocols and based on different metrics, including Acquire Rate, Equal Error Rate (EER), and True Accept Rate (TAR). Our results reveal substantial performance gaps between MLLMs and classical face recognition systems, particularly under challenging cross-spectral conditions, in spite of recent advances in MLLMs. Our findings highlight the limitations of current MLLMs for HFR and also the importance of rigorous biometric evaluation when considering their deployment in face recognition systems.

  </details>



- **GeMM-GAN: A Multimodal Generative Model Conditioned on Histopathology Images and Clinical Descriptions for Gene Expression Profile Generation**  
  Francesca Pia Panaccione, Carlo Sgaravatti, Pietro Pinoli  
  _2026-01-21_ · https://arxiv.org/abs/2601.15392v1  
  <details><summary>Abstract</summary>

  Biomedical research increasingly relies on integrating diverse data modalities, including gene expression profiles, medical images, and clinical metadata. While medical images and clinical metadata are routinely collected in clinical practice, gene expression data presents unique challenges for widespread research use, mainly due to stringent privacy regulations and costly laboratory experiments. To address these limitations, we present GeMM-GAN, a novel Generative Adversarial Network conditioned on histopathology tissue slides and clinical metadata, designed to synthesize realistic gene expression profiles. GeMM-GAN combines a Transformer Encoder for image patches with a final Cross Attention mechanism between patches and text tokens, producing a conditioning vector to guide a generative model in generating biologically coherent gene expression profiles. We evaluate our approach on the TCGA dataset and demonstrate that our framework outperforms standard generative models and generates more realistic and functionally meaningful gene expression profiles, improving by more than 11\% the accuracy on downstream disease type prediction compared to current state-of-the-art generative models. Code will be available at: https://github.com/francescapia/GeMM-GAN

  </details>



- **LuxRemix: Lighting Decomposition and Remixing for Indoor Scenes**  
  Ruofan Liang, Norman Müller, Ethan Weber, Duncan Zauss, Nandita Vijaykumar, Peter Kontschieder, Christian Richardt  
  _2026-01-21_ · https://arxiv.org/abs/2601.15283v1  
  <details><summary>Abstract</summary>

  We present a novel approach for interactive light editing in indoor scenes from a single multi-view scene capture. Our method leverages a generative image-based light decomposition model that factorizes complex indoor scene illumination into its constituent light sources. This factorization enables independent manipulation of individual light sources, specifically allowing control over their state (on/off), chromaticity, and intensity. We further introduce multi-view lighting harmonization to ensure consistent propagation of the lighting decomposition across all scene views. This is integrated into a relightable 3D Gaussian splatting representation, providing real-time interactive control over the individual light sources. Our results demonstrate highly photorealistic lighting decomposition and relighting outcomes across diverse indoor scenes. We evaluate our method on both synthetic and real-world datasets and provide a quantitative and qualitative comparison to state-of-the-art techniques. For video results and interactive demos, see https://luxremix.github.io.

  </details>



- **Rethinking Video Generation Model for the Embodied World**  
  Yufan Deng, Zilin Pan, Hongyu Zhang, Xiaojie Li, Ruoqing Hu, Yufei Ding, Yiming Zou, Yan Zeng, Daquan Zhou  
  _2026-01-21_ · https://arxiv.org/abs/2601.15282v1  
  <details><summary>Abstract</summary>

  Video generation models have significantly advanced embodied intelligence, unlocking new possibilities for generating diverse robot data that capture perception, reasoning, and action in the physical world. However, synthesizing high-quality videos that accurately reflect real-world robotic interactions remains challenging, and the lack of a standardized benchmark limits fair comparisons and progress. To address this gap, we introduce a comprehensive robotics benchmark, RBench, designed to evaluate robot-oriented video generation across five task domains and four distinct embodiments. It assesses both task-level correctness and visual fidelity through reproducible sub-metrics, including structural consistency, physical plausibility, and action completeness. Evaluation of 25 representative models highlights significant deficiencies in generating physically realistic robot behaviors. Furthermore, the benchmark achieves a Spearman correlation coefficient of 0.96 with human evaluations, validating its effectiveness. While RBench provides the necessary lens to identify these deficiencies, achieving physical realism requires moving beyond evaluation to address the critical shortage of high-quality training data. Driven by these insights, we introduce a refined four-stage data pipeline, resulting in RoVid-X, the largest open-source robotic dataset for video generation with 4 million annotated video clips, covering thousands of tasks and enriched with comprehensive physical property annotations. Collectively, this synergistic ecosystem of evaluation and data establishes a robust foundation for rigorous assessment and scalable training of video models, accelerating the evolution of embodied AI toward general intelligence.

  </details>



- **DrivIng: A Large-Scale Multimodal Driving Dataset with Full Digital Twin Integration**  
  Dominik Rößle, Xujun Xie, Adithya Mohan, Venkatesh Thirugnana Sambandham, Daniel Cremers, Torsten Schön  
  _2026-01-21_ · https://arxiv.org/abs/2601.15260v1  
  <details><summary>Abstract</summary>

  Perception is a cornerstone of autonomous driving, enabling vehicles to understand their surroundings and make safe, reliable decisions. Developing robust perception algorithms requires large-scale, high-quality datasets that cover diverse driving conditions and support thorough evaluation. Existing datasets often lack a high-fidelity digital twin, limiting systematic testing, edge-case simulation, sensor modification, and sim-to-real evaluations. To address this gap, we present DrivIng, a large-scale multimodal dataset with a complete geo-referenced digital twin of a ~18 km route spanning urban, suburban, and highway segments. Our dataset provides continuous recordings from six RGB cameras, one LiDAR, and high-precision ADMA-based localization, captured across day, dusk, and night. All sequences are annotated at 10 Hz with 3D bounding boxes and track IDs across 12 classes, yielding ~1.2 million annotated instances. Alongside the benefits of a digital twin, DrivIng enables a 1-to-1 transfer of real traffic into simulation, preserving agent interactions while enabling realistic and flexible scenario testing. To support reproducible research and robust validation, we benchmark DrivIng with state-of-the-art perception models and publicly release the dataset, digital twin, HD map, and codebase.

  </details>



- **PROGRESSLM: Towards Progress Reasoning in Vision-Language Models**  
  Jianshu Zhang, Chengxuan Qian, Haosen Sun, Haoran Lu, Dingcheng Wang, Letian Xue, Han Liu  
  _2026-01-21_ · https://arxiv.org/abs/2601.15224v1  
  <details><summary>Abstract</summary>

  Estimating task progress requires reasoning over long-horizon dynamics rather than recognizing static visual content. While modern Vision-Language Models (VLMs) excel at describing what is visible, it remains unclear whether they can infer how far a task has progressed from partial observations. To this end, we introduce Progress-Bench, a benchmark for systematically evaluating progress reasoning in VLMs. Beyond benchmarking, we further explore a human-inspired two-stage progress reasoning paradigm through both training-free prompting and training-based approach based on curated dataset ProgressLM-45K. Experiments on 14 VLMs show that most models are not yet ready for task progress estimation, exhibiting sensitivity to demonstration modality and viewpoint changes, as well as poor handling of unanswerable cases. While training-free prompting that enforces structured progress reasoning yields limited and model-dependent gains, the training-based ProgressLM-3B achieves consistent improvements even at a small model scale, despite being trained on a task set fully disjoint from the evaluation tasks. Further analyses reveal characteristic error patterns and clarify when and why progress reasoning succeeds or fails.

  </details>



- **ScenDi: 3D-to-2D Scene Diffusion Cascades for Urban Generation**  
  Hanlei Guo, Jiahao Shao, Xinya Chen, Xiyang Tan, Sheng Miao, Yujun Shen, Yiyi Liao  
  _2026-01-21_ · https://arxiv.org/abs/2601.15221v1  
  <details><summary>Abstract</summary>

  Recent advancements in 3D object generation using diffusion models have achieved remarkable success, but generating realistic 3D urban scenes remains challenging. Existing methods relying solely on 3D diffusion models tend to suffer a degradation in appearance details, while those utilizing only 2D diffusion models typically compromise camera controllability. To overcome this limitation, we propose ScenDi, a method for urban scene generation that integrates both 3D and 2D diffusion models. We first train a 3D latent diffusion model to generate 3D Gaussians, enabling the rendering of images at a relatively low resolution. To enable controllable synthesis, this 3DGS generation process can be optionally conditioned by specifying inputs such as 3d bounding boxes, road maps, or text prompts. Then, we train a 2D video diffusion model to enhance appearance details conditioned on rendered images from the 3D Gaussians. By leveraging the coarse 3D scene as guidance for 2D video diffusion, ScenDi generates desired scenes based on input conditions and successfully adheres to accurate camera trajectories. Experiments on two challenging real-world datasets, Waymo and KITTI-360, demonstrate the effectiveness of our approach.

  </details>



- **BBoxMaskPose v2: Expanding Mutual Conditioning to 3D**  
  Miroslav Purkrabek, Constantin Kolomiiets, Jiri Matas  
  _2026-01-21_ · https://arxiv.org/abs/2601.15200v1  
  <details><summary>Abstract</summary>

  Most 2D human pose estimation benchmarks are nearly saturated, with the exception of crowded scenes. We introduce PMPose, a top-down 2D pose estimator that incorporates the probabilistic formulation and the mask-conditioning. PMPose improves crowded pose estimation without sacrificing performance on standard scenes. Building on this, we present BBoxMaskPose v2 (BMPv2) integrating PMPose and an enhanced SAM-based mask refinement module. BMPv2 surpasses state-of-the-art by 1.5 average precision (AP) points on COCO and 6 AP points on OCHuman, becoming the first method to exceed 50 AP on OCHuman. We demonstrate that BMP's 2D prompting of 3D model improves 3D pose estimation in crowded scenes and that advances in 2D pose quality directly benefit 3D estimation. Results on the new OCHuman-Pose dataset show that multi-person performance is more affected by pose prediction accuracy than by detection. The code, models, and data are available on https://MiraPurkrabek.github.io/BBox-Mask-Pose/.

  </details>



- **BayesianVLA: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries**  
  Shijie Lian, Bin Yu, Xiaopeng Lin, Laurence T. Yang, Zhaolong Shen, Changti Wu, Yuzhuo Miao, Cong Huang, Kai Chen  
  _2026-01-21_ · https://arxiv.org/abs/2601.15197v2  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models have shown promise in robot manipulation but often struggle to generalize to new instructions or complex multi-task scenarios. We identify a critical pathology in current training paradigms where goal-driven data collection creates a dataset bias. In such datasets, language instructions are highly predictable from visual observations alone, causing the conditional mutual information between instructions and actions to vanish, a phenomenon we term Information Collapse. Consequently, models degenerate into vision-only policies that ignore language constraints and fail in out-of-distribution (OOD) settings. To address this, we propose BayesianVLA, a novel framework that enforces instruction following via Bayesian decomposition. By introducing learnable Latent Action Queries, we construct a dual-branch architecture to estimate both a vision-only prior $p(a \mid v)$ and a language-conditioned posterior $π(a \mid v, \ell)$. We then optimize the policy to maximize the conditional Pointwise Mutual Information (PMI) between actions and instructions. This objective effectively penalizes the vision shortcut and rewards actions that explicitly explain the language command. Without requiring new data, BayesianVLA significantly improves generalization. Extensive experiments across on SimplerEnv and RoboCasa demonstrate substantial gains, including an 11.3% improvement on the challenging OOD SimplerEnv benchmark, validating the ability of our approach to robustly ground language in action.

  </details>



- **Large-Scale Multidimensional Knowledge Profiling of Scientific Literature**  
  Zhucun Xue, Jiangning Zhang, Juntao Jiang, Jinzhuo Liu, Haoyang He, Teng Hu, Xiaobin Hu, Guangming Yao, Yi Yuan, Yong Liu  
  _2026-01-21_ · https://arxiv.org/abs/2601.15170v1  
  <details><summary>Abstract</summary>

  The rapid expansion of research across machine learning, vision, and language has produced a volume of publications that is increasingly difficult to synthesize. Traditional bibliometric tools rely mainly on metadata and offer limited visibility into the semantic content of papers, making it hard to track how research themes evolve over time or how different areas influence one another. To obtain a clearer picture of recent developments, we compile a unified corpus of more than 100,000 papers from 22 major conferences between 2020 and 2025 and construct a multidimensional profiling pipeline to organize and analyze their textual content. By combining topic clustering, LLM-assisted parsing, and structured retrieval, we derive a comprehensive representation of research activity that supports the study of topic lifecycles, methodological transitions, dataset and model usage patterns, and institutional research directions. Our analysis highlights several notable shifts, including the growth of safety, multimodal reasoning, and agent-oriented studies, as well as the gradual stabilization of areas such as neural machine translation and graph-based methods. These findings provide an evidence-based view of how AI research is evolving and offer a resource for understanding broader trends and identifying emerging directions. Code and dataset: https://github.com/xzc-zju/Profiling_Scientific_Literature

  </details>



- **AI-Based Culvert-Sewer Inspection**  
  Christina Thrainer  
  _2026-01-21_ · https://arxiv.org/abs/2601.15366v1  
  <details><summary>Abstract</summary>

  Culverts and sewer pipes are critical components of drainage systems, and their failure can lead to serious risks to public safety and the environment. In this thesis, we explore methods to improve automated defect segmentation in culverts and sewer pipes. Collecting and annotating data in this field is cumbersome and requires domain knowledge. Having a large dataset for structural defect detection is therefore not feasible. Our proposed methods are tested under conditions with limited annotated data to demonstrate applicability to real-world scenarios. Overall, this thesis proposes three methods to significantly enhance defect segmentation and handle data scarcity. This can be addressed either by enhancing the training data or by adjusting a models architecture. First, we evaluate preprocessing strategies, including traditional data augmentation and dynamic label injection. These techniques significantly improve segmentation performance, increasing both Intersection over Union (IoU) and F1 score. Second, we introduce FORTRESS, a novel architecture that combines depthwise separable convolutions, adaptive Kolmogorov-Arnold Networks (KAN), and multi-scale attention mechanisms. FORTRESS achieves state-of-the-art performance on the culvert sewer pipe defect dataset, while significantly reducing the number of trainable parameters, as well as its computational cost. Finally, we investigate few-shot semantic segmentation and its applicability to defect detection. Few-shot learning aims to train models with only limited data available. By employing a bidirectional prototypical network with attention mechanisms, the model achieves richer feature representations and achieves satisfactory results across evaluation metrics.

  </details>



- **BREPS: Bounding-Box Robustness Evaluation of Promptable Segmentation**  
  Andrey Moskalenko, Danil Kuznetsov, Irina Dudko, Anastasiia Iasakova, Nikita Boldyrev, Denis Shepelev, Andrei Spiridonov, Andrey Kuznetsov, Vlad Shakhuro  
  _2026-01-21_ · https://arxiv.org/abs/2601.15123v1  
  <details><summary>Abstract</summary>

  Promptable segmentation models such as SAM have established a powerful paradigm, enabling strong generalization to unseen objects and domains with minimal user input, including points, bounding boxes, and text prompts. Among these, bounding boxes stand out as particularly effective, often outperforming points while significantly reducing annotation costs. However, current training and evaluation protocols typically rely on synthetic prompts generated through simple heuristics, offering limited insight into real-world robustness. In this paper, we investigate the robustness of promptable segmentation models to natural variations in bounding box prompts. First, we conduct a controlled user study and collect thousands of real bounding box annotations. Our analysis reveals substantial variability in segmentation quality across users for the same model and instance, indicating that SAM-like models are highly sensitive to natural prompt noise. Then, since exhaustive testing of all possible user inputs is computationally prohibitive, we reformulate robustness evaluation as a white-box optimization problem over the bounding box prompt space. We introduce BREPS, a method for generating adversarial bounding boxes that minimize or maximize segmentation error while adhering to naturalness constraints. Finally, we benchmark state-of-the-art models across 10 datasets, spanning everyday scenes to medical imaging. Code - https://github.com/emb-ai/BREPS.

  </details>



- **Training-Free and Interpretable Hateful Video Detection via Multi-stage Adversarial Reasoning**  
  Shuonan Yang, Yuchen Zhang, Zeyu Fu  
  _2026-01-21_ · https://arxiv.org/abs/2601.15115v1  
  <details><summary>Abstract</summary>

  Hateful videos pose serious risks by amplifying discrimination, inciting violence, and undermining online safety. Existing training-based hateful video detection methods are constrained by limited training data and lack of interpretability, while directly prompting large vision-language models often struggle to deliver reliable hate detection. To address these challenges, this paper introduces MARS, a training-free Multi-stage Adversarial ReaSoning framework that enables reliable and interpretable hateful content detection. MARS begins with the objective description of video content, establishing a neutral foundation for subsequent analysis. Building on this, it develops evidence-based reasoning that supports potential hateful interpretations, while in parallel incorporating counter-evidence reasoning to capture plausible non-hateful perspectives. Finally, these perspectives are synthesized into a conclusive and explainable decision. Extensive evaluation on two real-world datasets shows that MARS achieves up to 10% improvement under certain backbones and settings compared to other training-free approaches and outperforms state-of-the-art training-based methods on one dataset. In addition, MARS produces human-understandable justifications, thereby supporting compliance oversight and enhancing the transparency of content moderation workflows. The code is available at https://github.com/Multimodal-Intelligence-Lab-MIL/MARS.

  </details>



- **Three-dimensional visualization of X-ray micro-CT with large-scale datasets: Efficiency and accuracy for real-time interaction**  
  Yipeng Yin, Rao Yao, Qingying Li, Dazhong Wang, Hong Zhou, Zhijun Fang, Jianing Chen, Longjie Qian, Mingyue Wu  
  _2026-01-21_ · https://arxiv.org/abs/2601.15098v1  
  <details><summary>Abstract</summary>

  As Micro-CT technology continues to refine its characterization of material microstructures, industrial CT ultra-precision inspection is generating increasingly large datasets, necessitating solutions to the trade-off between accuracy and efficiency in the 3D characterization of defects during ultra-precise detection. This article provides a unique perspective on recent advances in accurate and efficient 3D visualization using Micro-CT, tracing its evolution from medical imaging to industrial non-destructive testing (NDT). Among the numerous CT reconstruction and volume rendering methods, this article selectively reviews and analyzes approaches that balance accuracy and efficiency, offering a comprehensive analysis to help researchers quickly grasp highly efficient and accurate 3D reconstruction methods for microscopic features. By comparing the principles of computed tomography with advancements in microstructural technology, this article examines the evolution of CT reconstruction algorithms from analytical methods to deep learning techniques, as well as improvements in volume rendering algorithms, acceleration, and data reduction. Additionally, it explores advanced lighting models for high-accuracy, photorealistic, and efficient volume rendering. Furthermore, this article envisions potential directions in CT reconstruction and volume rendering. It aims to guide future research in quickly selecting efficient and precise methods and developing new ideas and approaches for real-time online monitoring of internal material defects through virtual-physical interaction, for applying digital twin model to structural health monitoring (SHM).

  </details>



- **The Pictorial Cortex: Zero-Shot Cross-Subject fMRI-to-Image Reconstruction via Compositional Latent Modeling**  
  Jingyang Huo, Yikai Wang, Yanwei Fu, Jianfeng Feng  
  _2026-01-21_ · https://arxiv.org/abs/2601.15071v1  
  <details><summary>Abstract</summary>

  Decoding visual experiences from human brain activity remains a central challenge at the intersection of neuroscience, neuroimaging, and artificial intelligence. A critical obstacle is the inherent variability of cortical responses: neural activity elicited by the same visual stimulus differs across individuals and trials due to anatomical, functional, cognitive, and experimental factors, making fMRI-to-image reconstruction non-injective. In this paper, we tackle a challenging yet practically meaningful problem: zero-shot cross-subject fMRI-to-image reconstruction, where the visual experience of a previously unseen individual must be reconstructed without subject-specific training. To enable principled evaluation, we present a unified cortical-surface dataset -- UniCortex-fMRI, assembled from multiple visual-stimulus fMRI datasets to provide broad coverage of subjects and stimuli. Our UniCortex-fMRI is particularly processed by standardized data formats to make it possible to explore this possibility in the zero-shot scenario of cross-subject fMRI-to-image reconstruction. To tackle the modeling challenge, we propose PictorialCortex, which models fMRI activity using a compositional latent formulation that structures stimulus-driven representations under subject-, dataset-, and trial-related variability. PictorialCortex operates in a universal cortical latent space and implements this formulation through a latent factorization-composition module, reinforced by paired factorization and re-factorizing consistency regularization. During inference, surrogate latents synthesized under multiple seen-subject conditions are aggregated to guide diffusion-based image synthesis for unseen subjects. Extensive experiments show that PictorialCortex improves zero-shot cross-subject visual reconstruction, highlighting the benefits of compositional latent modeling and multi-dataset training.

  </details>



- **SpooFL: Spoofing Federated Learning**  
  Isaac Baglin, Xiatian Zhu, Simon Hadfield  
  _2026-01-21_ · https://arxiv.org/abs/2601.15055v1  
  <details><summary>Abstract</summary>

  Traditional defenses against Deep Leakage (DL) attacks in Federated Learning (FL) primarily focus on obfuscation, introducing noise, transformations or encryption to degrade an attacker's ability to reconstruct private data. While effective to some extent, these methods often still leak high-level information such as class distributions or feature representations, and are frequently broken by increasingly powerful denoising attacks. We propose a fundamentally different perspective on FL defense: framing it as a spoofing problem.We introduce SpooFL (Figure 1), a spoofing-based defense that deceives attackers into believing they have recovered the true training data, while actually providing convincing but entirely synthetic samples from an unrelated task. Unlike prior synthetic-data defenses that share classes or distributions with the private data and thus still leak semantic information, SpooFL uses a state-of-the-art generative model trained on an external dataset with no class overlap. As a result, attackers are misled into recovering plausible yet completely irrelevant samples, preventing meaningful data leakage while preserving FL training integrity. We implement the first example of such a spoofing defense, and evaluate our method against state-of-the-art DL defenses and demonstrate that it successfully misdirects attackers without compromising model performance significantly.

  </details>



- **Federated Transformer-GNN for Privacy-Preserving Brain Tumor Localization with Modality-Level Explainability**  
  Andrea Protani, Riccardo Taiello, Marc Molina Van Den Bosch, Luigi Serio  
  _2026-01-21_ · https://arxiv.org/abs/2601.15042v1  
  <details><summary>Abstract</summary>

  Deep learning models for brain tumor analysis require large and diverse datasets that are often siloed across healthcare institutions due to privacy regulations. We present a federated learning framework for brain tumor localization that enables multi-institutional collaboration without sharing sensitive patient data. Our method extends a hybrid Transformer-Graph Neural Network architecture derived from prior decoder-free supervoxel GNNs and is deployed within CAFEIN\textsuperscript{\textregistered}, CERN's federated learning platform designed for healthcare environments. We provide an explainability analysis through Transformer attention mechanisms that reveals which MRI modalities drive the model predictions. Experiments on the BraTS dataset demonstrate a key finding: while isolated training on individual client data triggers early stopping well before reaching full training capacity, federated learning enables continued model improvement by leveraging distributed data, ultimately matching centralized performance. This result provides strong justification for federated learning when dealing with complex tasks and high-dimensional input data, as aggregating knowledge from multiple institutions significantly benefits the learning process. Our explainability analysis, validated through rigorous statistical testing on the full test set (paired t-tests with Bonferroni correction), reveals that deeper network layers significantly increase attention to T2 and FLAIR modalities ($p<0.001$, Cohen's $d$=1.50), aligning with clinical practice.

  </details>


