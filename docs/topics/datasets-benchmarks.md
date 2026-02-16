# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-02-16 07:19 UTC_

Total papers shown: **25**


---

- **Conversational Image Segmentation: Grounding Abstract Concepts with Scalable Supervision**  
  Aadarsh Sahoo, Georgia Gkioxari  
  _2026-02-13_ · https://arxiv.org/abs/2602.13195v1  
  <details><summary>Abstract</summary>

  Conversational image segmentation grounds abstract, intent-driven concepts into pixel-accurate masks. Prior work on referring image grounding focuses on categorical and spatial queries (e.g., "left-most apple") and overlooks functional and physical reasoning (e.g., "where can I safely store the knife?"). We address this gap and introduce Conversational Image Segmentation (CIS) and ConverSeg, a benchmark spanning entities, spatial relations, intent, affordances, functions, safety, and physical reasoning. We also present ConverSeg-Net, which fuses strong segmentation priors with language understanding, and an AI-powered data engine that generates prompt-mask pairs without human supervision. We show that current language-guided segmentation models are inadequate for CIS, while ConverSeg-Net trained on our data engine achieves significant gains on ConverSeg and maintains strong performance on existing language-guided segmentation benchmarks. Project webpage: https://glab-caltech.github.io/converseg/

  </details>



- **Universal Transformation of One-Class Classifiers for Unsupervised Anomaly Detection**  
  Declan McIntosh, Alexandra Branzan Albu  
  _2026-02-13_ · https://arxiv.org/abs/2602.13091v1  
  <details><summary>Abstract</summary>

  Detecting anomalies in images and video is an essential task for multiple real-world problems, including industrial inspection, computer-assisted diagnosis, and environmental monitoring. Anomaly detection is typically formulated as a one-class classification problem, where the training data consists solely of nominal values, leaving methods built on this assumption susceptible to training label noise. We present a dataset folding method that transforms an arbitrary one-class classifier-based anomaly detector into a fully unsupervised method. This is achieved by making a set of key weak assumptions: that anomalies are uncommon in the training dataset and generally heterogeneous. These assumptions enable us to utilize multiple independently trained instances of a one-class classifier to filter the training dataset for anomalies. This transformation requires no modifications to the underlying anomaly detector; the only changes are algorithmically selected data subsets used for training. We demonstrate that our method can transform a wide variety of one-class classifier anomaly detectors for both images and videos into unsupervised ones. Our method creates the first unsupervised logical anomaly detectors by transforming existing methods. We also demonstrate that our method achieves state-of-the-art performance for unsupervised anomaly detection on the MVTec AD, ViSA, and MVTec Loco AD datasets. As improvements to one-class classifiers are made, our method directly transfers those improvements to the unsupervised domain, linking the domains.

  </details>



- **Implicit-Scale 3D Reconstruction for Multi-Food Volume Estimation from Monocular Images**  
  Yuhao Chen, Gautham Vinod, Siddeshwar Raghavan, Talha Ibn Mahmud, Bruce Coburn, Jinge Ma, Fengqing Zhu, Jiangpeng He  
  _2026-02-13_ · https://arxiv.org/abs/2602.13041v1  
  <details><summary>Abstract</summary>

  We present Implicit-Scale 3D Reconstruction from Monocular Multi-Food Images, a benchmark dataset designed to advance geometry-based food portion estimation in realistic dining scenarios. Existing dietary assessment methods largely rely on single-image analysis or appearance-based inference, including recent vision-language models, which lack explicit geometric reasoning and are sensitive to scale ambiguity. This benchmark reframes food portion estimation as an implicit-scale 3D reconstruction problem under monocular observations. To reflect real-world conditions, explicit physical references and metric annotations are removed; instead, contextual objects such as plates and utensils are provided, requiring algorithms to infer scale from implicit cues and prior knowledge. The dataset emphasizes multi-food scenes with diverse object geometries, frequent occlusions, and complex spatial arrangements. The benchmark was adopted as a challenge at the MetaFood 2025 Workshop, where multiple teams proposed reconstruction-based solutions. Experimental results show that while strong vision--language baselines achieve competitive performance, geometry-based reconstruction methods provide both improved accuracy and greater robustness, with the top-performing approach achieving 0.21 MAPE in volume estimation and 5.7 L1 Chamfer Distance in geometric accuracy.

  </details>



- **Resource-Efficient Gesture Recognition through Convexified Attention**  
  Daniel Schwartz, Dario Salvucci, Yusuf Osmanlioglu, Richard Vallett, Genevieve Dion, Ali Shokoufandeh  
  _2026-02-13_ · https://arxiv.org/abs/2602.13030v1  
  <details><summary>Abstract</summary>

  Wearable e-textile interfaces require gesture recognition capabilities but face severe constraints in power consumption, computational capacity, and form factor that make traditional deep learning impractical. While lightweight architectures like MobileNet improve efficiency, they still demand thousands of parameters, limiting deployment on textile-integrated platforms. We introduce a convexified attention mechanism for wearable applications that dynamically weights features while preserving convexity through nonexpansive simplex projection and convex loss functions. Unlike conventional attention mechanisms using non-convex softmax operations, our approach employs Euclidean projection onto the probability simplex combined with multi-class hinge loss, ensuring global convergence guarantees. Implemented on a textile-based capacitive sensor with four connection points, our approach achieves 100.00\% accuracy on tap gestures and 100.00\% on swipe gestures -- consistent across 10-fold cross-validation and held-out test evaluation -- while requiring only 120--360 parameters, a 97\% reduction compared to conventional approaches. With sub-millisecond inference times (290--296$μ$s) and minimal storage requirements ($<$7KB), our method enables gesture interfaces directly within e-textiles without external processing. Our evaluation, conducted in controlled laboratory conditions with a single-user dataset, demonstrates feasibility for basic gesture interactions. Real-world deployment would require validation across multiple users, environmental conditions, and more complex gesture vocabularies. These results demonstrate how convex optimization can enable efficient on-device machine learning for textile interfaces.

  </details>



- **Human-Aligned MLLM Judges for Fine-Grained Image Editing Evaluation: A Benchmark, Framework, and Analysis**  
  Runzhou Liu, Hailey Weingord, Sejal Mittal, Prakhar Dungarwal, Anusha Nandula, Bo Ni, Samyadeep Basu, Hongjie Chen, Nesreen K. Ahmed, Li Li, et al.  
  _2026-02-13_ · https://arxiv.org/abs/2602.13028v1  
  <details><summary>Abstract</summary>

  Evaluating image editing models remains challenging due to the coarse granularity and limited interpretability of traditional metrics, which often fail to capture aspects important to human perception and intent. Such metrics frequently reward visually plausible outputs while overlooking controllability, edit localization, and faithfulness to user instructions. In this work, we introduce a fine-grained Multimodal Large Language Model (MLLM)-as-a-Judge framework for image editing that decomposes common evaluation notions into twelve fine-grained interpretable factors spanning image preservation, edit quality, and instruction fidelity. Building on this formulation, we present a new human-validated benchmark that integrates human judgments, MLLM-based evaluations, model outputs, and traditional metrics across diverse image editing tasks. Through extensive human studies, we show that the proposed MLLM judges align closely with human evaluations at a fine granularity, supporting their use as reliable and scalable evaluators. We further demonstrate that traditional image editing metrics are often poor proxies for these factors, failing to distinguish over-edited or semantically imprecise outputs, whereas our judges provide more intuitive and informative assessments in both offline and online settings. Together, this work introduces a benchmark, a principled factorization, and empirical evidence positioning fine-grained MLLM judges as a practical foundation for studying, comparing, and improving image editing approaches.

  </details>



- **Learning Image-based Tree Crown Segmentation from Enhanced Lidar-based Pseudo-labels**  
  Julius Pesonen, Stefan Rua, Josef Taher, Niko Koivumäki, Xiaowei Yu, Eija Honkavaara  
  _2026-02-13_ · https://arxiv.org/abs/2602.13022v1  
  <details><summary>Abstract</summary>

  Mapping individual tree crowns is essential for tasks such as maintaining urban tree inventories and monitoring forest health, which help us understand and care for our environment. However, automatically separating the crowns from each other in aerial imagery is challenging due to factors such as the texture and partial tree crown overlaps. In this study, we present a method to train deep learning models that segment and separate individual trees from RGB and multispectral images, using pseudo-labels derived from aerial laser scanning (ALS) data. Our study shows that the ALS-derived pseudo-labels can be enhanced using a zero-shot instance segmentation model, Segment Anything Model 2 (SAM 2). Our method offers a way to obtain domain-specific training annotations for optical image-based models without any manual annotation cost, leading to segmentation models which outperform any available models which have been targeted for general domain deployment on the same task.

  </details>



- **Towards Universal Video MLLMs with Attribute-Structured and Quality-Verified Instructions**  
  Yunheng Li, Hengrui Zhang, Meng-Hao Guo, Wenzhao Gao, Shaoyong Jia, Shaohui Jiao, Qibin Hou, Ming-Ming Cheng  
  _2026-02-13_ · https://arxiv.org/abs/2602.13013v1  
  <details><summary>Abstract</summary>

  Universal video understanding requires modeling fine-grained visual and audio information over time in diverse real-world scenarios. However, the performance of existing models is primarily constrained by video-instruction data that represents complex audiovisual content as single, incomplete descriptions, lacking fine-grained organization and reliable annotation. To address this, we introduce: (i) ASID-1M, an open-source collection of one million structured, fine-grained audiovisual instruction annotations with single- and multi-attribute supervision; (ii) ASID-Verify, a scalable data curation pipeline for annotation, with automatic verification and refinement that enforces semantic and temporal consistency between descriptions and the corresponding audiovisual content; and (iii) ASID-Captioner, a video understanding model trained via Supervised Fine-Tuning (SFT) on the ASID-1M. Experiments across seven benchmarks covering audiovisual captioning, attribute-wise captioning, caption-based QA, and caption-based temporal grounding show that ASID-Captioner improves fine-grained caption quality while reducing hallucinations and improving instruction following. It achieves state-of-the-art performance among open-source models and is competitive with Gemini-3-Pro.

  </details>



- **MASAR: Motion-Appearance Synergy Refinement for Joint Detection and Trajectory Forecasting**  
  Mohammed Amine Bencheikh Lehocine, Julian Schmidt, Frank Moosmann, Dikshant Gupta, Fabian Flohr  
  _2026-02-13_ · https://arxiv.org/abs/2602.13003v1  
  <details><summary>Abstract</summary>

  Classical autonomous driving systems connect perception and prediction modules via hand-crafted bounding-box interfaces, limiting information flow and propagating errors to downstream tasks. Recent research aims to develop end-to-end models that jointly address perception and prediction; however, they often fail to fully exploit the synergy between appearance and motion cues, relying mainly on short-term visual features. We follow the idea of "looking backward to look forward", and propose MASAR, a novel fully differentiable framework for joint 3D detection and trajectory forecasting compatible with any transformer-based 3D detector. MASAR employs an object-centric spatio-temporal mechanism that jointly encodes appearance and motion features. By predicting past trajectories and refining them using guidance from appearance cues, MASAR captures long-term temporal dependencies that enhance future trajectory forecasting. Experiments conducted on the nuScenes dataset demonstrate MASAR's effectiveness, showing improvements of over 20% in minADE and minFDE while maintaining robust detection performance. Code and models are available at https://github.com/aminmed/MASAR.

  </details>



- **INHerit-SG: Incremental Hierarchical Semantic Scene Graphs with RAG-Style Retrieval**  
  YukTungSamuel Fang, Zhikang Shi, Jiabin Qiu, Zixuan Chen, Jieqi Shi, Hao Xu, Jing Huo, Yang Gao  
  _2026-02-13_ · https://arxiv.org/abs/2602.12971v1  
  <details><summary>Abstract</summary>

  Driven by advancements in foundation models, semantic scene graphs have emerged as a prominent paradigm for high-level 3D environmental abstraction in robot navigation. However, existing approaches are fundamentally misaligned with the needs of embodied tasks. As they rely on either offline batch processing or implicit feature embeddings, the maps can hardly support interpretable human-intent reasoning in complex environments. To address these limitations, we present INHerit-SG. We redefine the map as a structured, RAG-ready knowledge base where natural-language descriptions are introduced as explicit semantic anchors to better align with human intent. An asynchronous dual-process architecture, together with a Floor-Room-Area-Object hierarchy, decouples geometric segmentation from time-consuming semantic reasoning. An event-triggered map update mechanism reorganizes the graph only when meaningful semantic events occur. This strategy enables our graph to maintain long-term consistency with relatively low computational overhead. For retrieval, we deploy multi-role Large Language Models (LLMs) to decompose queries into atomic constraints and handle logical negations, and employ a hard-to-soft filtering strategy to ensure robust reasoning. This explicit interpretability improves the success rate and reliability of complex retrievals, enabling the system to adapt to a broader spectrum of human interaction tasks. We evaluate INHerit-SG on a newly constructed dataset, HM3DSem-SQR, and in real-world environments. Experiments demonstrate that our system achieves state-of-the-art performance on complex queries, and reveal its scalability for downstream navigation tasks. Project Page: https://fangyuktung.github.io/INHeritSG.github.io/

  </details>



- **Beyond Benchmarks of IUGC: Rethinking Requirements of Deep Learning Methods for Intrapartum Ultrasound Biometry from Fetal Ultrasound Videos**  
  Jieyun Bai, Zihao Zhou, Yitong Tang, Jie Gan, Zhuonan Liang, Jianan Fan, Lisa B. Mcguire, Jillian L. Clarke, Weidong Cai, Jacaueline Spurway, et al.  
  _2026-02-13_ · https://arxiv.org/abs/2602.12922v1  
  <details><summary>Abstract</summary>

  A substantial proportion (45\%) of maternal deaths, neonatal deaths, and stillbirths occur during the intrapartum phase, with a particularly high burden in low- and middle-income countries. Intrapartum biometry plays a critical role in monitoring labor progression; however, the routine use of ultrasound in resource-limited settings is hindered by a shortage of trained sonographers. To address this challenge, the Intrapartum Ultrasound Grand Challenge (IUGC), co-hosted with MICCAI 2024, was launched. The IUGC introduces a clinically oriented multi-task automatic measurement framework that integrates standard plane classification, fetal head-pubic symphysis segmentation, and biometry, enabling algorithms to exploit complementary task information for more accurate estimation. Furthermore, the challenge releases the largest multi-center intrapartum ultrasound video dataset to date, comprising 774 videos (68,106 frames) collected from three hospitals, providing a robust foundation for model training and evaluation. In this study, we present a comprehensive overview of the challenge design, review the submissions from eight participating teams, and analyze their methods from five perspectives: preprocessing, data augmentation, learning strategy, model architecture, and post-processing. In addition, we perform a systematic analysis of the benchmark results to identify key bottlenecks, explore potential solutions, and highlight open challenges for future research. Although encouraging performance has been achieved, our findings indicate that the field remains at an early stage, and further in-depth investigation is required before large-scale clinical deployment. All benchmark solutions and the complete dataset have been publicly released to facilitate reproducible research and promote continued advances in automatic intrapartum ultrasound biometry.

  </details>



- **EPRBench: A High-Quality Benchmark Dataset for Event Stream Based Visual Place Recognition**  
  Xiao Wang, Xingxing Xiong, Jinfeng Gao, Xufeng Lou, Bo Jiang, Si-bao Chen, Yaowei Wang, Yonghong Tian  
  _2026-02-13_ · https://arxiv.org/abs/2602.12919v1  
  <details><summary>Abstract</summary>

  Event stream-based Visual Place Recognition (VPR) is an emerging research direction that offers a compelling solution to the instability of conventional visible-light cameras under challenging conditions such as low illumination, overexposure, and high-speed motion. Recognizing the current scarcity of dedicated datasets in this domain, we introduce EPRBench, a high-quality benchmark specifically designed for event stream-based VPR. EPRBench comprises 10K event sequences and 65K event frames, collected using both handheld and vehicle-mounted setups to comprehensively capture real-world challenges across diverse viewpoints, weather conditions, and lighting scenarios. To support semantic-aware and language-integrated VPR research, we provide LLM-generated scene descriptions, subsequently refined through human annotation, establishing a solid foundation for integrating LLMs into event-based perception pipelines. To facilitate systematic evaluation, we implement and benchmark 15 state-of-the-art VPR algorithms on EPRBench, offering a strong baseline for future algorithmic comparisons. Furthermore, we propose a novel multi-modal fusion paradigm for VPR: leveraging LLMs to generate textual scene descriptions from raw event streams, which then guide spatially attentive token selection, cross-modal feature fusion, and multi-scale representation learning. This framework not only achieves highly accurate place recognition but also produces interpretable reasoning processes alongside its predictions, significantly enhancing model transparency and explainability. The dataset and source code will be released on https://github.com/Event-AHU/Neuromorphic_ReID

  </details>



- **Adding internal audio sensing to internal vision enables human-like in-hand fabric recognition with soft robotic fingertips**  
  Iris Andrussow, Jans Solano, Benjamin A. Richardson, Georg Martius, Katherine J. Kuchenbecker  
  _2026-02-13_ · https://arxiv.org/abs/2602.12918v1  
  <details><summary>Abstract</summary>

  Distinguishing the feel of smooth silk from coarse cotton is a trivial everyday task for humans. When exploring such fabrics, fingertip skin senses both spatio-temporal force patterns and texture-induced vibrations that are integrated to form a haptic representation of the explored material. It is challenging to reproduce this rich, dynamic perceptual capability in robots because tactile sensors typically cannot achieve both high spatial resolution and high temporal sampling rate. In this work, we present a system that can sense both types of haptic information, and we investigate how each type influences robotic tactile perception of fabrics. Our robotic hand's middle finger and thumb each feature a soft tactile sensor: one is the open-source Minsight sensor that uses an internal camera to measure fingertip deformation and force at 50 Hz, and the other is our new sensor Minsound that captures vibrations through an internal MEMS microphone with a bandwidth from 50 Hz to 15 kHz. Inspired by the movements humans make to evaluate fabrics, our robot actively encloses and rubs folded fabric samples between its two sensitive fingers. Our results test the influence of each sensing modality on overall classification performance, showing high utility for the audio-based sensor. Our transformer-based method achieves a maximum fabric classification accuracy of 97 % on a dataset of 20 common fabrics. Incorporating an external microphone away from Minsound increases our method's robustness in loud ambient noise conditions. To show that this audio-visual tactile sensing approach generalizes beyond the training data, we learn general representations of fabric stretchiness, thickness, and roughness.

  </details>



- **Robustness of Object Detection of Autonomous Vehicles in Adverse Weather Conditions**  
  Fox Pettersen, Hong Zhu  
  _2026-02-13_ · https://arxiv.org/abs/2602.12902v1  
  <details><summary>Abstract</summary>

  As self-driving technology advances toward widespread adoption, determining safe operational thresholds across varying environmental conditions becomes critical for public safety. This paper proposes a method for evaluating the robustness of object detection ML models in autonomous vehicles under adverse weather conditions. It employs data augmentation operators to generate synthetic data that simulates different severance degrees of the adverse operation conditions at progressive intensity levels to find the lowest intensity of the adverse conditions at which the object detection model fails. The robustness of the object detection model is measured by the average first failure coefficients (AFFC) over the input images in the benchmark. The paper reports an experiment with four object detection models: YOLOv5s, YOLOv11s, Faster R-CNN, and Detectron2, utilising seven data augmentation operators that simulate weather conditions fog, rain, and snow, and lighting conditions of dark, bright, flaring, and shadow. The experiment data show that the method is feasible, effective, and efficient to evaluate and compare the robustness of object detection models in various adverse operation conditions. In particular, the Faster R-CNN model achieved the highest robustness with an overall average AFFC of 71.9% over all seven adverse conditions, while YOLO variants showed the AFFC values of 43%. The method is also applied to assess the impact of model training that targets adverse operation conditions using synthetic data on model robustness. It is observed that such training can improve robustness in adverse conditions but may suffer from diminishing returns and forgetting phenomena (i.e., decline in robustness) if overtrained.

  </details>



- **RADAR: Revealing Asymmetric Development of Abilities in MLLM Pre-training**  
  Yunshuang Nie, Bingqian Lin, Minzhe Niu, Kun Xiang, Jianhua Han, Guowei Huang, Xingyue Quan, Hang Xu, Bokui Chen, Xiaodan Liang  
  _2026-02-13_ · https://arxiv.org/abs/2602.12892v1  
  <details><summary>Abstract</summary>

  Pre-trained Multi-modal Large Language Models (MLLMs) provide a knowledge-rich foundation for post-training by leveraging their inherent perception and reasoning capabilities to solve complex tasks. However, the lack of an efficient evaluation framework impedes the diagnosis of their performance bottlenecks. Current evaluation primarily relies on testing after supervised fine-tuning, which introduces laborious additional training and autoregressive decoding costs. Meanwhile, common pre-training metrics cannot quantify a model's perception and reasoning abilities in a disentangled manner. Furthermore, existing evaluation benchmarks are typically limited in scale or misaligned with pre-training objectives. Thus, we propose RADAR, an efficient ability-centric evaluation framework for Revealing Asymmetric Development of Abilities in MLLM pRe-training. RADAR involves two key components: (1) Soft Discrimination Score, a novel metric for robustly tracking ability development without fine-tuning, based on quantifying nuanced gradations of the model preference for the correct answer over distractors; and (2) Multi-Modal Mixture Benchmark, a new 15K+ sample benchmark for comprehensively evaluating pre-trained MLLMs' perception and reasoning abilities in a 0-shot manner, where we unify authoritative benchmark datasets and carefully collect new datasets, extending the evaluation scope and addressing the critical gaps in current benchmarks. With RADAR, we comprehensively reveal the asymmetric development of perceptual and reasoning capabilities in pretrained MLLMs across diverse factors, including data volume, model size, and pretraining strategy. Our RADAR underscores the need for a decomposed perspective on pre-training ability bottlenecks, informing targeted interventions to advance MLLMs efficiently. Our code is publicly available at https://github.com/Nieysh/RADAR.

  </details>



- **RoadscapesQA: A Multitask, Multimodal Dataset for Visual Question Answering on Indian Roads**  
  Vijayasri Iyer, Maahin Rathinagiriswaran, Jyothikamalesh S  
  _2026-02-13_ · https://arxiv.org/abs/2602.12877v1  
  <details><summary>Abstract</summary>

  Understanding road scenes is essential for autonomous driving, as it enables systems to interpret visual surroundings to aid in effective decision-making. We present Roadscapes, a multitask multimodal dataset consisting of upto 9,000 images captured in diverse Indian driving environments, accompanied by manually verified bounding boxes. To facilitate scalable scene understanding, we employ rule-based heuristics to infer various scene attributes, which are subsequently used to generate question-answer (QA) pairs for tasks such as object grounding, reasoning, and scene understanding. The dataset includes a variety of scenes from urban and rural India, encompassing highways, service roads, village paths, and congested city streets, captured in both daytime and nighttime settings. Roadscapes has been curated to advance research on visual scene understanding in unstructured environments. In this paper, we describe the data collection and annotation process, present key dataset statistics, and provide initial baselines for image QA tasks using vision-language models.

  </details>



- **X-VORTEX: Spatio-Temporal Contrastive Learning for Wake Vortex Trajectory Forecasting**  
  Zhan Qu, Michael Färber  
  _2026-02-13_ · https://arxiv.org/abs/2602.12869v1  
  <details><summary>Abstract</summary>

  Wake vortices are strong, coherent air turbulences created by aircraft, and they pose a major safety and capacity challenge for air traffic management. Tracking how vortices move, weaken, and dissipate over time from LiDAR measurements is still difficult because scans are sparse, vortex signatures fade as the flow breaks down under atmospheric turbulence and instabilities, and point-wise annotation is prohibitively expensive. Existing approaches largely treat each scan as an independent, fully supervised segmentation problem, which overlooks temporal structure and does not scale to the vast unlabeled archives collected in practice. We present X-VORTEX, a spatio-temporal contrastive learning framework grounded in Augmentation Overlap Theory that learns physics-aware representations from unlabeled LiDAR point cloud sequences. X-VORTEX addresses two core challenges: sensor sparsity and time-varying vortex dynamics. It constructs paired inputs from the same underlying flight event by combining a weakly perturbed sequence with a strongly augmented counterpart produced via temporal subsampling and spatial masking, encouraging the model to align representations across missing frames and partial observations. Architecturally, a time-distributed geometric encoder extracts per-scan features and a sequential aggregator models the evolving vortex state across variable-length sequences. We evaluate on a real-world dataset of over one million LiDAR scans. X-VORTEX achieves superior vortex center localization while using only 1% of the labeled data required by supervised baselines, and the learned representations support accurate trajectory forecasting.

  </details>



- **Thinking Like a Radiologist: A Dataset for Anatomy-Guided Interleaved Vision Language Reasoning in Chest X-ray Interpretation**  
  Yichen Zhao, Zelin Peng, Piao Yang, Xiaokang Yang, Wei Shen  
  _2026-02-13_ · https://arxiv.org/abs/2602.12843v1  
  <details><summary>Abstract</summary>

  Radiological diagnosis is a perceptual process in which careful visual inspection and language reasoning are repeatedly interleaved. Most medical large vision language models (LVLMs) perform visual inspection only once and then rely on text-only chain-of-thought (CoT) reasoning, which operates purely in the linguistic space and is prone to hallucination. Recent methods attempt to mitigate this issue by introducing visually related coordinates, such as bounding boxes. However, these remain a pseudo-visual solution: coordinates are still text and fail to preserve rich visual details like texture and density. Motivated by the interleaved nature of radiological diagnosis, we introduce MMRad-IVL-22K, the first large-scale dataset designed for natively interleaved visual language reasoning in chest X-ray interpretation. MMRad-IVL-22K reflects a repeated cycle of reasoning and visual inspection workflow of radiologists, in which visual rationales complement textual descriptions and ground each step of the reasoning process. MMRad-IVL-22K comprises 21,994 diagnostic traces, enabling systematic scanning across 35 anatomical regions. Experimental results on advanced closed-source LVLMs demonstrate that report generation guided by multimodal CoT significantly outperforms that guided by text-only CoT in clinical accuracy and report quality (e.g., 6\% increase in the RadGraph metric), confirming that high-fidelity interleaved vision language evidence is a non-substitutable component of reliable medical AI. Furthermore, benchmarking across seven state-of-the-art open-source LVLMs demonstrates that models fine-tuned on MMRad-IVL-22K achieve superior reasoning consistency and report quality compared with both general-purpose and medical-specific LVLMs. The project page is available at https://github.com/qiuzyc/thinking_like_a_radiologist.

  </details>



- **3DLAND: 3D Lesion Abdominal Anomaly Localization Dataset**  
  Mehran Advand, Zahra Dehghanian, Navid Faraji, Reza Barati, Seyed Amir Ahmad Safavi-Naini, Hamid R. Rabiee  
  _2026-02-13_ · https://arxiv.org/abs/2602.12820v1  
  <details><summary>Abstract</summary>

  Existing medical imaging datasets for abdominal CT often lack three-dimensional annotations, multi-organ coverage, or precise lesion-to-organ associations, hindering robust representation learning and clinical applications. To address this gap, we introduce 3DLAND, a large-scale benchmark dataset comprising over 6,000 contrast-enhanced CT volumes with over 20,000 high-fidelity 3D lesion annotations linked to seven abdominal organs: liver, kidneys, pancreas, spleen, stomach, and gallbladder. Our streamlined three-phase pipeline integrates automated spatial reasoning, prompt-optimized 2D segmentation, and memory-guided 3D propagation, validated by expert radiologists with surface dice scores exceeding 0.75. By providing diverse lesion types and patient demographics, 3DLAND enables scalable evaluation of anomaly detection, localization, and cross-organ transfer learning for medical AI. Our dataset establishes a new benchmark for evaluating organ-aware 3D segmentation models, paving the way for advancements in healthcare-oriented AI. To facilitate reproducibility and further research, the 3DLAND dataset and implementation code are publicly available at https://mehrn79.github.io/3DLAND.

  </details>



- **Bootstrapping MLLM for Weakly-Supervised Class-Agnostic Object Counting**  
  Xiaowen Zhang, Zijie Yue, Yong Luo, Cairong Zhao, Qijun Chen, Miaojing Shi  
  _2026-02-13_ · https://arxiv.org/abs/2602.12774v1  
  <details><summary>Abstract</summary>

  Object counting is a fundamental task in computer vision, with broad applicability in many real-world scenarios. Fully-supervised counting methods require costly point-level annotations per object. Few weakly-supervised methods leverage only image-level object counts as supervision and achieve fairly promising results. They are, however, often limited to counting a single category, e.g. person. In this paper, we propose WS-COC, the first MLLM-driven weakly-supervised framework for class-agnostic object counting. Instead of directly fine-tuning MLLMs to predict object counts, which can be challenging due to the modality gap, we incorporate three simple yet effective strategies to bootstrap the counting paradigm in both training and testing: First, a divide-and-discern dialogue tuning strategy is proposed to guide the MLLM to determine whether the object count falls within a specific range and progressively break down the range through multi-round dialogue. Second, a compare-and-rank count optimization strategy is introduced to train the MLLM to optimize the relative ranking of multiple images according to their object counts. Third, a global-and-local counting enhancement strategy aggregates and fuses local and global count predictions to improve counting performance in dense scenes. Extensive experiments on FSC-147, CARPK, PUCPR+, and ShanghaiTech show that WS-COC matches or even surpasses many state-of-art fully-supervised methods while significantly reducing annotation costs. Code is available at https://github.com/viscom-tongji/WS-COC.

  </details>



- **Towards complete digital twins in cultural heritage with ART3mis 3D artifacts annotator**  
  Dimitrios Karamatskos, Vasileios Arampatzakis, Vasileios Sevetlidis, Stavros Nousias, Athanasios Kalogeras, Christos Koulamas, Aris Lalos, George Pavlidis  
  _2026-02-13_ · https://arxiv.org/abs/2602.12761v1  
  <details><summary>Abstract</summary>

  Archaeologists, as well as specialists and practitioners in cultural heritage, require applications with additional functions, such as the annotation and attachment of metadata to specific regions of the 3D digital artifacts, to go beyond the simplistic three-dimensional (3D) visualization. Different strategies addressed this issue, most of which are excellent in their particular area of application, but their capacity is limited to their design's purpose; they lack generalization and interoperability. This paper introduces ART3mis, a general-purpose, user-friendly, feature-rich, interactive web-based textual annotation tool for 3D objects. Moreover, it enables the communication, distribution, and reuse of information as it complies with the W3C Web Annotation Data Model. It is primarily designed to help cultural heritage conservators, restorers, and curators who lack technical expertise in 3D imaging and graphics, handle, segment, and annotate 3D digital replicas of artifacts with ease.

  </details>



- **Lung nodule classification on CT scan patches using 3D convolutional neural networks**  
  Volodymyr Sydorskyi  
  _2026-02-13_ · https://arxiv.org/abs/2602.12750v1  
  <details><summary>Abstract</summary>

  Lung cancer remains one of the most common and deadliest forms of cancer worldwide. The likelihood of successful treatment depends strongly on the stage at which the disease is diagnosed. Therefore, early detection of lung cancer represents a critical medical challenge. However, this task poses significant difficulties for thoracic radiologists due to the large number of studies to review, the presence of multiple nodules within the lungs, and the small size of many nodules, which complicates visual assessment. Consequently, the development of automated systems that incorporate highly accurate and computationally efficient lung nodule detection and classification modules is essential. This study introduces three methodological improvements for lung nodule classification: (1) an advanced CT scan cropping strategy that focuses the model on the target nodule while reducing computational cost; (2) target filtering techniques for removing noisy labels; (3) novel augmentation methods to improve model robustness. The integration of these techniques enables the development of a robust classification subsystem within a comprehensive Clinical Decision Support System for lung cancer detection, capable of operating across diverse acquisition protocols, scanner types, and upstream models (segmentation or detection). The multiclass model achieved a Macro ROC AUC of 0.9176 and a Macro F1-score of 0.7658, while the binary model reached a Binary ROC AUC of 0.9383 and a Binary F1-score of 0.8668 on the LIDC-IDRI dataset. These results outperform several previously reported approaches and demonstrate state-of-the-art performance for this task.

  </details>



- **Synthetic Craquelure Generation for Unsupervised Painting Restoration**  
  Jana Cuch-Guillén, Antonio Agudo, Raül Pérez-Gonzalo  
  _2026-02-13_ · https://arxiv.org/abs/2602.12742v1  
  <details><summary>Abstract</summary>

  Cultural heritage preservation increasingly demands non-invasive digital methods for painting restoration, yet identifying and restoring fine craquelure patterns from complex brushstrokes remains challenging due to scarce pixel-level annotations. We propose a fully annotation-free framework driven by a domain-specific synthetic craquelure generator, which simulates realistic branching and tapered fissure geometry using Bézier trajectories. Our approach couples a classical morphological detector with a learning-based refinement module: a SegFormer backbone adapted via Low-Rank Adaptation (LoRA). Uniquely, we employ a detector-guided strategy, injecting the morphological map as an input spatial prior, while a masked hybrid loss and logit adjustment constrain the training to focus specifically on refining candidate crack regions. The refined masks subsequently guide an Anisotropic Diffusion inpainting stage to reconstruct missing content. Experimental results demonstrate that our pipeline significantly outperforms state-of-the-art photographic restoration models in zero-shot settings, while faithfully preserving the original paint brushwork.

  </details>



- **ART3mis: Ray-Based Textual Annotation on 3D Cultural Objects**  
  Vasileios Arampatzakis, Vasileios Sevetlidis, Fotis Arnaoutoglou, Athanasios Kalogeras, Christos Koulamas, Aris Lalos, Chairi Kiourt, George Ioannakis, Anestis Koutsoudis, George Pavlidis  
  _2026-02-13_ · https://arxiv.org/abs/2602.12725v1  
  <details><summary>Abstract</summary>

  Beyond simplistic 3D visualisations, archaeologists, as well as cultural heritage experts and practitioners, need applications with advanced functionalities. Such as the annotation and attachment of metadata onto particular regions of the 3D digital objects. Various approaches have been presented to tackle this challenge, most of which achieve excellent results in the domain of their application. However, they are often confined to that specific domain and particular problem. In this paper, we present ART3mis - a general-purpose, user-friendly, interactive textual annotation tool for 3D objects. Primarily attuned to aid cultural heritage conservators, restorers and curators with no technical skills in 3D imaging and graphics, the tool allows for the easy handling, segmenting and annotating of 3D digital replicas of artefacts. ART3mis applies a user-driven, direct-on-surface approach. It can handle detailed 3D cultural objects in real-time and store textual annotations for multiple complex regions in JSON data format.

  </details>



- **Constrained PSO Six-Parameter Fuzzy PID Tuning Method for Balanced Optimization of Depth Tracking Performance in Underwater Vehicles**  
  Yanxi Ding, Tingyue Jia  
  _2026-02-13_ · https://arxiv.org/abs/2602.12700v1  
  <details><summary>Abstract</summary>

  Depth control of underwater vehicles in engineering applications must simultaneously satisfy requirements for rapid tracking, low overshoot, and actuator constraints. Traditional fuzzy PID tuning often relies on empirical methods, making it difficult to achieve a stable and reproducible equilibrium solution between performance enhancement and control cost. This paper proposes a constrained particle swarm optimization (PSO) method for tuning six-parameter fuzzy PID controllers. By adjusting the benchmark PID parameters alongside the fuzzy controller's input quantization factor and output proportional gain, it achieves synergistic optimization of the overall tuning strength and dynamic response characteristics of the fuzzy PID system. To ensure engineering feasibility of the optimization results, a time-weighted absolute error integral, adjustment time, relative overshoot control energy, and saturation occupancy rate are introduced. Control energy constraints are applied to construct a constraint-driven comprehensive evaluation system, suppressing pseudo-improvements achieved solely by increasing control inputs. Simulation results demonstrate that, while maintaining consistent control energy and saturation levels, the proposed method significantly enhances deep tracking performance: the time-weighted absolute error integral decreases from 0.2631 to 0.1473, the settling time shortens from 2.301 s to 1.613 s, and the relative overshoot reduces from 0.1494 to 0.01839. Control energy varied from 7980 to 7935, satisfying the energy constraint, while saturation occupancy decreased from 0.004 to 0.003. These results validate the effectiveness and engineering significance of the proposed constrained six-parameter joint tuning strategy for depth control in underwater vehicle navigation scenarios.

  </details>



- **SignScene: Visual Sign Grounding for Mapless Navigation**  
  Nicky Zimmerman, Joel Loo, Benjamin Koh, Zishuo Wang, David Hsu  
  _2026-02-13_ · https://arxiv.org/abs/2602.12686v1  
  <details><summary>Abstract</summary>

  Navigational signs enable humans to navigate unfamiliar environments without maps. This work studies how robots can similarly exploit signs for mapless navigation in the open world. A central challenge lies in interpreting signs: real-world signs are diverse and complex, and their abstract semantic contents need to be grounded in the local 3D scene. We formalize this as sign grounding, the problem of mapping semantic instructions on signs to corresponding scene elements and navigational actions. Recent Vision-Language Models (VLMs) offer the semantic common-sense and reasoning capabilities required for this task, but are sensitive to how spatial information is represented. We propose SignScene, a sign-centric spatial-semantic representation that captures navigation-relevant scene elements and sign information, and presents them to VLMs in a form conducive to effective reasoning. We evaluate our grounding approach on a dataset of 114 queries collected across nine diverse environment types, achieving 88% grounding accuracy and significantly outperforming baselines. Finally, we demonstrate that it enables real-world mapless navigation on a Spot robot using only signs.

  </details>


