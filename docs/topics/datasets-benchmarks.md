# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-01-25 06:48 UTC_

Total papers shown: **22**


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


