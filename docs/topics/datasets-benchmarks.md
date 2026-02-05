# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-02-05 07:14 UTC_

Total papers shown: **50**


---

- **Toward Reliable and Explainable Nail Disease Classification: Leveraging Adversarial Training and Grad-CAM Visualization**  
  Farzia Hossain, Samanta Ghosh, Shahida Begum, B. M. Shahria Alam, Mohammad Tahmid Noor, Md Parvez Mia, Nishat Tasnim Niloy  
  _2026-02-04_ · https://arxiv.org/abs/2602.04820v1  
  <details><summary>Abstract</summary>

  Human nail diseases are gradually observed over all age groups, especially among older individuals, often going ignored until they become severe. Early detection and accurate diagnosis of such conditions are important because they sometimes reveal our body's health problems. But it is challenging due to the inferred visual differences between disease types. This paper presents a machine learning-based model for automated classification of nail diseases based on a publicly available dataset, which contains 3,835 images scaling six categories. In 224x224 pixels, all images were resized to ensure consistency. To evaluate performance, four well-known CNN models-InceptionV3, DenseNet201, EfficientNetV2, and ResNet50 were trained and analyzed. Among these, InceptionV3 outperformed the others with an accuracy of 95.57%, while DenseNet201 came next with 94.79%. To make the model stronger and less likely to make mistakes on tricky or noisy images, we used adversarial training. To help understand how the model makes decisions, we used SHAP to highlight important features in the predictions. This system could be a helpful support for doctors, making nail disease diagnosis more accurate and faster.

  </details>



- **XtraLight-MedMamba for Classification of Neoplastic Tubular Adenomas**  
  Aqsa Sultana, Rayan Afsar, Ahmed Rahu, Surendra P. Singh, Brian Shula, Brandon Combs, Derrick Forchetti, Vijayan K. Asari  
  _2026-02-04_ · https://arxiv.org/abs/2602.04819v1  
  <details><summary>Abstract</summary>

  Accurate risk stratification of precancerous polyps during routine colonoscopy screenings is essential for lowering the risk of developing colorectal cancer (CRC). However, assessment of low-grade dysplasia remains limited by subjective histopathologic interpretation. Advancements in digital pathology and deep learning provide new opportunities to identify subtle and fine morphologic patterns associated with malignant progression that may be imperceptible to the human eye. In this work, we propose XtraLight-MedMamba, an ultra-lightweight state-space-based deep learning framework for classifying neoplastic tubular adenomas from whole-slide images (WSIs). The architecture is a blend of ConvNext based shallow feature extractor with parallel vision mamba to efficiently model both long- and short-range dependencies and image generalization. An integration of Spatial and Channel Attention Bridge (SCAB) module enhances multiscale feature extraction, while Fixed Non-Negative Orthogonal Classifier (FNOClassifier) enables substantial parameter reduction and improved generalization. The model was evaluated on a curated dataset acquired from patients with low-grade tubular adenomas, stratified into case and control cohorts based on subsequent CRC development. XtraLight-MedMamba achieved an accuracy of 97.18% and an F1-score of 0.9767 using approximately 32,000 parameters, outperforming transformer-based and conventional Mamba architectures with significantly higher model complexity.

  </details>



- **VISTA-Bench: Do Vision-Language Models Really Understand Visualized Text as Well as Pure Text?**  
  Qing'an Liu, Juntong Feng, Yuhao Wang, Xinzhe Han, Yujie Cheng, Yue Zhu, Haiwen Diao, Yunzhi Zhuge, Huchuan Lu  
  _2026-02-04_ · https://arxiv.org/abs/2602.04802v1  
  <details><summary>Abstract</summary>

  Vision-Language Models (VLMs) have achieved impressive performance in cross-modal understanding across textual and visual inputs, yet existing benchmarks predominantly focus on pure-text queries. In real-world scenarios, language also frequently appears as visualized text embedded in images, raising the question of whether current VLMs handle such input requests comparably. We introduce VISTA-Bench, a systematic benchmark from multimodal perception, reasoning, to unimodal understanding domains. It evaluates visualized text understanding by contrasting pure-text and visualized-text questions under controlled rendering conditions. Extensive evaluation of over 20 representative VLMs reveals a pronounced modality gap: models that perform well on pure-text queries often degrade substantially when equivalent semantic content is presented as visualized text. This gap is further amplified by increased perceptual difficulty, highlighting sensitivity to rendering variations despite unchanged semantics. Overall, VISTA-Bench provides a principled evaluation framework to diagnose this limitation and to guide progress toward more unified language representations across tokenized text and pixels. The source dataset is available at https://github.com/QingAnLiu/VISTA-Bench.

  </details>



- **Mitigating Long-Tail Bias via Prompt-Controlled Diffusion Augmentation**  
  Buddhi Wijenayake, Nichula Wasalathilake, Roshan Godaliyadda, Vijitha Herath, Parakrama Ekanayake, Vishal M. Patel  
  _2026-02-04_ · https://arxiv.org/abs/2602.04749v1  
  <details><summary>Abstract</summary>

  Semantic segmentation of high-resolution remote-sensing imagery is critical for urban mapping and land-cover monitoring, yet training data typically exhibits severe long-tailed pixel imbalance. In the dataset LoveDA, this challenge is compounded by an explicit Urban/Rural split with distinct appearance and inconsistent class-frequency statistics across domains. We present a prompt-controlled diffusion augmentation framework that synthesizes paired label--image samples with explicit control of both domain and semantic composition. Stage~A uses a domain-aware, masked ratio-conditioned discrete diffusion model to generate layouts that satisfy user-specified class-ratio targets while respecting learned co-occurrence structure. Stage~B translates layouts into photorealistic, domain-consistent images using Stable Diffusion with ControlNet guidance. Mixing the resulting ratio and domain-controlled synthetic pairs with real data yields consistent improvements across multiple segmentation backbones, with gains concentrated on minority classes and improved Urban and Rural generalization, demonstrating controllable augmentation as a practical mechanism to mitigate long-tail bias in remote-sensing segmentation. Source codes, pretrained models, and synthetic datasets are available at \href{https://github.com/Buddhi19/SyntheticGen.git}{Github}

  </details>



- **Annotation Free Spacecraft Detection and Segmentation using Vision Language Models**  
  Samet Hicsonmez, Jose Sosa, Dan Pineau, Inder Pal Singh, Arunkumar Rathinam, Abd El Rahman Shabayek, Djamila Aouada  
  _2026-02-04_ · https://arxiv.org/abs/2602.04699v1  
  <details><summary>Abstract</summary>

  Vision Language Models (VLMs) have demonstrated remarkable performance in open-world zero-shot visual recognition. However, their potential in space-related applications remains largely unexplored. In the space domain, accurate manual annotation is particularly challenging due to factors such as low visibility, illumination variations, and object blending with planetary backgrounds. Developing methods that can detect and segment spacecraft and orbital targets without requiring extensive manual labeling is therefore of critical importance. In this work, we propose an annotation-free detection and segmentation pipeline for space targets using VLMs. Our approach begins by automatically generating pseudo-labels for a small subset of unlabeled real data with a pre-trained VLM. These pseudo-labels are then leveraged in a teacher-student label distillation framework to train lightweight models. Despite the inherent noise in the pseudo-labels, the distillation process leads to substantial performance gains over direct zero-shot VLM inference. Experimental evaluations on the SPARK-2024, SPEED+, and TANGO datasets on segmentation tasks demonstrate consistent improvements in average precision (AP) by up to 10 points. Code and models are available at https://github.com/giddyyupp/annotation-free-spacecraft-segmentation.

  </details>



- **DRMOT: A Dataset and Framework for RGBD Referring Multi-Object Tracking**  
  Sijia Chen, Lijuan Ma, Yanqiu Yu, En Yu, Liman Liu, Wenbing Tao  
  _2026-02-04_ · https://arxiv.org/abs/2602.04692v1  
  <details><summary>Abstract</summary>

  Referring Multi-Object Tracking (RMOT) aims to track specific targets based on language descriptions and is vital for interactive AI systems such as robotics and autonomous driving. However, existing RMOT models rely solely on 2D RGB data, making it challenging to accurately detect and associate targets characterized by complex spatial semantics (e.g., ``the person closest to the camera'') and to maintain reliable identities under severe occlusion, due to the absence of explicit 3D spatial information. In this work, we propose a novel task, RGBD Referring Multi-Object Tracking (DRMOT), which explicitly requires models to fuse RGB, Depth (D), and Language (L) modalities to achieve 3D-aware tracking. To advance research on the DRMOT task, we construct a tailored RGBD referring multi-object tracking dataset, named DRSet, designed to evaluate models' spatial-semantic grounding and tracking capabilities. Specifically, DRSet contains RGB images and depth maps from 187 scenes, along with 240 language descriptions, among which 56 descriptions incorporate depth-related information. Furthermore, we propose DRTrack, a MLLM-guided depth-referring tracking framework. DRTrack performs depth-aware target grounding from joint RGB-D-L inputs and enforces robust trajectory association by incorporating depth cues. Extensive experiments on the DRSet dataset demonstrate the effectiveness of our framework.

  </details>



- **A labeled dataset of simulated phlebotomy procedures for medical AI: polygon annotations for object detection and human-object interaction**  
  Raúl Jiménez Cruz, César Torres-Huitzil, Marco Franceschetti, Ronny Seiger, Luciano García-Bañuelos, Barbara Weber  
  _2026-02-04_ · https://arxiv.org/abs/2602.04624v1  
  <details><summary>Abstract</summary>

  This data article presents a dataset of 11,884 labeled images documenting a simulated blood extraction (phlebotomy) procedure performed on a training arm. Images were extracted from high-definition videos recorded under controlled conditions and curated to reduce redundancy using Structural Similarity Index Measure (SSIM) filtering. An automated face-anonymization step was applied to all videos prior to frame selection. Each image contains polygon annotations for five medically relevant classes: syringe, rubber band, disinfectant wipe, gloves, and training arm. The annotations were exported in a segmentation format compatible with modern object detection frameworks (e.g., YOLOv8), ensuring broad usability. This dataset is partitioned into training (70%), validation (15%), and test (15%) subsets and is designed to advance research in medical training automation and human-object interaction. It enables multiple applications, including phlebotomy tool detection, procedural step recognition, workflow analysis, conformance checking, and the development of educational systems that provide structured feedback to medical trainees. The data and accompanying label files are publicly available on Zenodo.

  </details>



- **ImmuVis: Hyperconvolutional Foundation Model for Imaging Mass Cytometry**  
  Marcin Możejko, Dawid Uchal, Krzysztof Gogolewski, Piotr Kupidura, Szymon Łukasik, Jakub Giezgała, Tomasz Nocoń, Kacper Pietrzyk, Robert Pieniuta, Mateusz Sulimowicz, et al.  
  _2026-02-04_ · https://arxiv.org/abs/2602.04585v1  
  <details><summary>Abstract</summary>

  We present ImmuVis, an efficient convolutional foundation model for imaging mass cytometry (IMC), a high-throughput multiplex imaging technology that handles molecular marker measurements as image channels and enables large-scale spatial tissue profiling. Unlike natural images, multiplex imaging lacks a fixed channel space, as real-world marker sets vary across studies, violating a core assumption of standard vision backbones. To address this, ImmuVis introduces marker-adaptive hyperconvolutions that generate convolutional kernels from learned marker embeddings, enabling a single model to operate on arbitrary measured marker subsets without retraining. We pretrain ImmuVis on the largest to-date dataset, IMC17M (28 cohorts, 24,405 images, 265 markers, over 17M patches), using self-supervised masked reconstruction. ImmuVis outperforms SOTA baselines and ablations in virtual staining and downstream classification tasks at substantially lower compute cost than transformer-based alternatives, and is the sole model that provides calibrated uncertainty via a heteroscedastic likelihood objective. These results position ImmuVis as a practical, efficient foundation model for real-world IMC modeling.

  </details>



- **SalFormer360: a transformer-based saliency estimation model for 360-degree videos**  
  Mahmoud Z. A. Wahba, Francesco Barbato, Sara Baldoni, Federica Battisti  
  _2026-02-04_ · https://arxiv.org/abs/2602.04584v1  
  <details><summary>Abstract</summary>

  Saliency estimation has received growing attention in recent years due to its importance in a wide range of applications. In the context of 360-degree video, it has been particularly valuable for tasks such as viewport prediction and immersive content optimization. In this paper, we propose SalFormer360, a novel saliency estimation model for 360-degree videos built on a transformer-based architecture. Our approach is based on the combination of an existing encoder architecture, SegFormer, and a custom decoder. The SegFormer model was originally developed for 2D segmentation tasks, and it has been fine-tuned to adapt it to 360-degree content. To further enhance prediction accuracy in our model, we incorporated Viewing Center Bias to reflect user attention in 360-degree environments. Extensive experiments on the three largest benchmark datasets for saliency estimation demonstrate that SalFormer360 outperforms existing state-of-the-art methods. In terms of Pearson Correlation Coefficient, our model achieves 8.4% higher performance on Sport360, 2.5% on PVS-HM, and 18.6% on VR-EyeTracking compared to previous state-of-the-art.

  </details>



- **Understanding Degradation with Vision Language Model**  
  Guanzhou Lan, Chenyi Liao, Yuqi Yang, Qianli Ma, Zhigang Wang, Dong Wang, Bin Zhao, Xuelong Li  
  _2026-02-04_ · https://arxiv.org/abs/2602.04565v1  
  <details><summary>Abstract</summary>

  Understanding visual degradations is a critical yet challenging problem in computer vision. While recent Vision-Language Models (VLMs) excel at qualitative description, they often fall short in understanding the parametric physics underlying image degradations. In this work, we redefine degradation understanding as a hierarchical structured prediction task, necessitating the concurrent estimation of degradation types, parameter keys, and their continuous physical values. Although these sub-tasks operate in disparate spaces, we prove that they can be unified under one autoregressive next-token prediction paradigm, whose error is bounded by the value-space quantization grid. Building on this insight, we introduce DU-VLM, a multimodal chain-of-thought model trained with supervised fine-tuning and reinforcement learning using structured rewards. Furthermore, we show that DU-VLM can serve as a zero-shot controller for pre-trained diffusion models, enabling high-fidelity image restoration without fine-tuning the generative backbone. We also introduce \textbf{DU-110k}, a large-scale dataset comprising 110,000 clean-degraded pairs with grounded physical annotations. Extensive experiments demonstrate that our approach significantly outperforms generalist baselines in both accuracy and robustness, exhibiting generalization to unseen distributions.

  </details>



- **SLUM-i: Semi-supervised Learning for Urban Mapping of Informal Settlements and Data Quality Benchmarking**  
  Muhammad Taha Mukhtar, Syed Musa Ali Kazmi, Khola Naseem, Muhammad Ali Chattha, Andreas Dengel, Sheraz Ahmed, Muhammad Naseer Bajwa, Muhammad Imran Malik  
  _2026-02-04_ · https://arxiv.org/abs/2602.04525v1  
  <details><summary>Abstract</summary>

  Rapid urban expansion has fueled the growth of informal settlements in major cities of low- and middle-income countries, with Lahore and Karachi in Pakistan and Mumbai in India serving as prominent examples. However, large-scale mapping of these settlements is severely constrained not only by the scarcity of annotations but by inherent data quality challenges, specifically high spectral ambiguity between formal and informal structures and significant annotation noise. We address this by introducing a benchmark dataset for Lahore, constructed from scratch, along with companion datasets for Karachi and Mumbai, which were derived from verified administrative boundaries, totaling 1,869 $\text{km}^2$ of area. To evaluate the global robustness of our framework, we extend our experiments to five additional established benchmarks, encompassing eight cities across three continents, and provide comprehensive data quality assessments of all datasets. We also propose a new semi-supervised segmentation framework designed to mitigate the class imbalance and feature degradation inherent in standard semi-supervised learning pipelines. Our method integrates a Class-Aware Adaptive Thresholding mechanism that dynamically adjusts confidence thresholds to prevent minority class suppression and a Prototype Bank System that enforces semantic consistency by anchoring predictions to historically learned high-fidelity feature representations. Extensive experiments across a total of eight cities spanning three continents demonstrate that our approach outperforms state-of-the-art semi-supervised baselines. Most notably, our method demonstrates superior domain transfer capability whereby a model trained on only 10% of source labels reaches a 0.461 mIoU on unseen geographies and outperforms the zero-shot generalization of fully supervised models.

  </details>



- **Temporal Slowness in Central Vision Drives Semantic Object Learning**  
  Timothy Schaumlöffel, Arthur Aubret, Gemma Roig, Jochen Triesch  
  _2026-02-04_ · https://arxiv.org/abs/2602.04462v1  
  <details><summary>Abstract</summary>

  Humans acquire semantic object representations from egocentric visual streams with minimal supervision. Importantly, the visual system processes with high resolution only the center of its field of view and learns similar representations for visual inputs occurring close in time. This emphasizes slowly changing information around gaze locations. This study investigates the role of central vision and slowness learning in the formation of semantic object representations from human-like visual experience. We simulate five months of human-like visual experience using the Ego4D dataset and generate gaze coordinates with a state-of-the-art gaze prediction model. Using these predictions, we extract crops that mimic central vision and train a time-contrastive Self-Supervised Learning model on them. Our results show that combining temporal slowness and central vision improves the encoding of different semantic facets of object representations. Specifically, focusing on central vision strengthens the extraction of foreground object features, while considering temporal slowness, especially during fixational eye movements, allows the model to encode broader semantic information about objects. These findings provide new insights into the mechanisms by which humans may develop semantic object representations from natural visual experience.

  </details>



- **Seg-ReSearch: Segmentation with Interleaved Reasoning and External Search**  
  Tianming Liang, Qirui Du, Jian-Fang Hu, Haichao Jiang, Zicheng Lin, Wei-Shi Zheng  
  _2026-02-04_ · https://arxiv.org/abs/2602.04454v1  
  <details><summary>Abstract</summary>

  Segmentation based on language has been a popular topic in computer vision. While recent advances in multimodal large language models (MLLMs) have endowed segmentation systems with reasoning capabilities, these efforts remain confined by the frozen internal knowledge of MLLMs, which limits their potential for real-world scenarios that involve up-to-date information or domain-specific concepts. In this work, we propose \textbf{Seg-ReSearch}, a novel segmentation paradigm that overcomes the knowledge bottleneck of existing approaches. By enabling interleaved reasoning and external search, Seg-ReSearch empowers segmentation systems to handle dynamic, open-world queries that extend beyond the frozen knowledge of MLLMs. To effectively train this capability, we introduce a hierarchical reward design that harmonizes initial guidance with progressive incentives, mitigating the dilemma between sparse outcome signals and rigid step-wise supervision. For evaluation, we construct OK-VOS, a challenging benchmark that explicitly requires outside knowledge for video object segmentation. Experiments on OK-VOS and two existing reasoning segmentation benchmarks demonstrate that our Seg-ReSearch improves state-of-the-art approaches by a substantial margin. Code and data will be released at https://github.com/iSEE-Laboratory/Seg-ReSearch.

  </details>



- **SynthVerse: A Large-Scale Diverse Synthetic Dataset for Point Tracking**  
  Weiguang Zhao, Haoran Xu, Xingyu Miao, Qin Zhao, Rui Zhang, Kaizhu Huang, Ning Gao, Peizhou Cao, Mingze Sun, Mulin Yu, et al.  
  _2026-02-04_ · https://arxiv.org/abs/2602.04441v1  
  <details><summary>Abstract</summary>

  Point tracking aims to follow visual points through complex motion, occlusion, and viewpoint changes, and has advanced rapidly with modern foundation models. Yet progress toward general point tracking remains constrained by limited high-quality data, as existing datasets often provide insufficient diversity and imperfect trajectory annotations. To this end, we introduce SynthVerse, a large-scale, diverse synthetic dataset specifically designed for point tracking. SynthVerse includes several new domains and object types missing from existing synthetic datasets, such as animated-film-style content, embodied manipulation, scene navigation, and articulated objects. SynthVerse substantially expands dataset diversity by covering a broader range of object categories and providing high-quality dynamic motions and interactions, enabling more robust training and evaluation for general point tracking. In addition, we establish a highly diverse point tracking benchmark to systematically evaluate state-of-the-art methods under broader domain shifts. Extensive experiments and analyses demonstrate that training with SynthVerse yields consistent improvements in generalization and reveal limitations of existing trackers under diverse settings.

  </details>



- **Med-MMFL: A Multimodal Federated Learning Benchmark in Healthcare**  
  Aavash Chhetri, Bibek Niroula, Pratik Shrestha, Yash Raj Shrestha, Lesley A Anderson, Prashnna K Gyawali, Loris Bazzani, Binod Bhattarai  
  _2026-02-04_ · https://arxiv.org/abs/2602.04416v1  
  <details><summary>Abstract</summary>

  Federated learning (FL) enables collaborative model training across decentralized medical institutions while preserving data privacy. However, medical FL benchmarks remain scarce, with existing efforts focusing mainly on unimodal or bimodal modalities and a limited range of medical tasks. This gap underscores the need for standardized evaluation to advance systematic understanding in medical MultiModal FL (MMFL). To this end, we introduce Med-MMFL, the first comprehensive MMFL benchmark for the medical domain, encompassing diverse modalities, tasks, and federation scenarios. Our benchmark evaluates six representative state-of-the-art FL algorithms, covering different aggregation strategies, loss formulations, and regularization techniques. It spans datasets with 2 to 4 modalities, comprising a total of 10 unique medical modalities, including text, pathology images, ECG, X-ray, radiology reports, and multiple MRI sequences. Experiments are conducted across naturally federated, synthetic IID, and synthetic non-IID settings to simulate real-world heterogeneity. We assess segmentation, classification, modality alignment (retrieval), and VQA tasks. To support reproducibility and fair comparison of future multimodal federated learning (MMFL) methods under realistic medical settings, we release the complete benchmark implementation, including data processing and partitioning pipelines, at https://github.com/bhattarailab/Med-MMFL-Benchmark .

  </details>



- **LCUDiff: Latent Capacity Upgrade Diffusion for Faithful Human Body Restoration**  
  Jue Gong, Zihan Zhou, Jingkai Wang, Shu Li, Libo Liu, Jianliang Lan, Yulun Zhang  
  _2026-02-04_ · https://arxiv.org/abs/2602.04406v1  
  <details><summary>Abstract</summary>

  Existing methods for restoring degraded human-centric images often struggle with insufficient fidelity, particularly in human body restoration (HBR). Recent diffusion-based restoration methods commonly adapt pre-trained text-to-image diffusion models, where the variational autoencoder (VAE) can significantly bottleneck restoration fidelity. We propose LCUDiff, a stable one-step framework that upgrades a pre-trained latent diffusion model from the 4-channel latent space to the 16-channel latent space. For VAE fine-tuning, channel splitting distillation (CSD) is used to keep the first four channels aligned with pre-trained priors while allocating the additional channels to effectively encode high-frequency details. We further design prior-preserving adaptation (PPA) to smoothly bridge the mismatch between 4-channel diffusion backbones and the higher-dimensional 16-channel latent. In addition, we propose a decoder router (DeR) for per-sample decoder routing using restoration-quality score annotations, which improves visual quality across diverse conditions. Experiments on synthetic and real-world datasets show competitive results with higher fidelity and fewer artifacts under mild degradations, while preserving one-step efficiency. The code and model will be at https://github.com/gobunu/LCUDiff.

  </details>



- **Finding NeMO: A Geometry-Aware Representation of Template Views for Few-Shot Perception**  
  Sebastian Jung, Leonard Klüpfel, Rudolph Triebel, Maximilian Durner  
  _2026-02-04_ · https://arxiv.org/abs/2602.04343v1  
  <details><summary>Abstract</summary>

  We present Neural Memory Object (NeMO), a novel object-centric representation that can be used to detect, segment and estimate the 6DoF pose of objects unseen during training using RGB images. Our method consists of an encoder that requires only a few RGB template views depicting an object to generate a sparse object-like point cloud using a learned UDF containing semantic and geometric information. Next, a decoder takes the object encoding together with a query image to generate a variety of dense predictions. Through extensive experiments, we show that our method can be used for few-shot object perception without requiring any camera-specific parameters or retraining on target data. Our proposed concept of outsourcing object information in a NeMO and using a single network for multiple perception tasks enhances interaction with novel objects, improving scalability and efficiency by enabling quick object onboarding without retraining or extensive pre-processing. We report competitive and state-of-the-art results on various datasets and perception tasks of the BOP benchmark, demonstrating the versatility of our approach. https://github.com/DLR-RM/nemo

  </details>



- **Explicit Uncertainty Modeling for Active CLIP Adaptation with Dual Prompt Tuning**  
  Qian-Wei Wang, Yaguang Song, Shu-Tao Xia  
  _2026-02-04_ · https://arxiv.org/abs/2602.04340v1  
  <details><summary>Abstract</summary>

  Pre-trained vision-language models such as CLIP exhibit strong transferability, yet adapting them to downstream image classification tasks under limited annotation budgets remains challenging. In active learning settings, the model must select the most informative samples for annotation from a large pool of unlabeled data. Existing approaches typically estimate uncertainty via entropy-based criteria or representation clustering, without explicitly modeling uncertainty from the model perspective. In this work, we propose a robust uncertainty modeling framework for active CLIP adaptation based on dual-prompt tuning. We introduce two learnable prompts in the textual branch of CLIP. The positive prompt enhances the discriminability of task-specific textual embeddings corresponding to light-weight tuned visual embeddings, improving classification reliability. Meanwhile, the negative prompt is trained in an reversed manner to explicitly model the probability that the predicted label is correct, providing a principled uncertainty signal for guiding active sample selection. Extensive experiments across different fine-tuning paradigms demonstrate that our method consistently outperforms existing active learning methods under the same annotation budget.

  </details>



- **Fine-tuning Pre-trained Vision-Language Models in a Human-Annotation-Free Manner**  
  Qian-Wei Wang, Guanghao Meng, Ren Cai, Yaguang Song, Shu-Tao Xia  
  _2026-02-04_ · https://arxiv.org/abs/2602.04337v1  
  <details><summary>Abstract</summary>

  Large-scale vision-language models (VLMs) such as CLIP exhibit strong zero-shot generalization, but adapting them to downstream tasks typically requires costly labeled data. Existing unsupervised self-training methods rely on pseudo-labeling, yet often suffer from unreliable confidence filtering, confirmation bias, and underutilization of low-confidence samples. We propose Collaborative Fine-Tuning (CoFT), an unsupervised adaptation framework that leverages unlabeled data through a dual-model, cross-modal collaboration mechanism. CoFT introduces a dual-prompt learning strategy with positive and negative textual prompts to explicitly model pseudo-label cleanliness in a sample-dependent manner, removing the need for hand-crafted thresholds or noise assumptions. The negative prompt also regularizes lightweight visual adaptation modules, improving robustness under noisy supervision. CoFT employs a two-phase training scheme, transitioning from parameter-efficient fine-tuning on high-confidence samples to full fine-tuning guided by collaboratively filtered pseudo-labels. Building on CoFT, CoFT+ further enhances adaptation via iterative fine-tuning, momentum contrastive learning, and LLM-generated prompts. Extensive experiments demonstrate consistent gains over existing unsupervised methods and even few-shot supervised baselines.

  </details>



- **Safe and Stylized Trajectory Planning for Autonomous Driving via Diffusion Model**  
  Shuo Pei, Yong Wang, Yuanchen Zhu, Chen Sun, Qin Li, Yanan Zhao, Huachun Tan  
  _2026-02-04_ · https://arxiv.org/abs/2602.04329v1  
  <details><summary>Abstract</summary>

  Achieving safe and stylized trajectory planning in complex real-world scenarios remains a critical challenge for autonomous driving systems. This paper proposes the SDD Planner, a diffusion-based framework designed to effectively reconcile safety constraints with driving styles in real time. The framework integrates two core modules: a Multi-Source Style-Aware Encoder, which employs distance-sensitive attention to fuse dynamic agent data and environmental contexts for heterogeneous safety-style perception; and a Style-Guided Dynamic Trajectory Generator, which adaptively modulates priority weights within the diffusion denoising process to generate user-preferred yet safe trajectories. Extensive experiments demonstrate that SDD Planner achieves state-of-the-art performance. On the StyleDrive benchmark, it improves the SM-PDMS metric by 3.9% over WoTE, the strongest baseline. Furthermore, on the NuPlan Test14 and Test14-hard benchmarks, SDD Planner ranks first with overall scores of 91.76 and 80.32, respectively, outperforming leading methods such as PLUTO. Real-vehicle closed-loop tests further confirm that SDD Planner maintains high safety standards while aligning with preset driving styles, validating its practical applicability for real-world deployment.

  </details>



- **Multiview Self-Representation Learning across Heterogeneous Views**  
  Jie Chen, Zhu Wang, Chuanbin Liu, Xi Peng  
  _2026-02-04_ · https://arxiv.org/abs/2602.04328v1  
  <details><summary>Abstract</summary>

  Features of the same sample generated by different pretrained models often exhibit inherently distinct feature distributions because of discrepancies in the model pretraining objectives or architectures. Learning invariant representations from large-scale unlabeled visual data with various pretrained models in a fully unsupervised transfer manner remains a significant challenge. In this paper, we propose a multiview self-representation learning (MSRL) method in which invariant representations are learned by exploiting the self-representation property of features across heterogeneous views. The features are derived from large-scale unlabeled visual data through transfer learning with various pretrained models and are referred to as heterogeneous multiview data. An individual linear model is stacked on top of its corresponding frozen pretrained backbone. We introduce an information-passing mechanism that relies on self-representation learning to support feature aggregation over the outputs of the linear model. Moreover, an assignment probability distribution consistency scheme is presented to guide multiview self-representation learning by exploiting complementary information across different views. Consequently, representation invariance across different linear models is enforced through this scheme. In addition, we provide a theoretical analysis of the information-passing mechanism, the assignment probability distribution consistency and the incremental views. Extensive experiments with multiple benchmark visual datasets demonstrate that the proposed MSRL method consistently outperforms several state-of-the-art approaches.

  </details>



- **JOintGS: Joint Optimization of Cameras, Bodies and 3D Gaussians for In-the-Wild Monocular Reconstruction**  
  Zihan Lou, Jinlong Fan, Sihan Ma, Yuxiang Yang, Jing Zhang  
  _2026-02-04_ · https://arxiv.org/abs/2602.04317v1  
  <details><summary>Abstract</summary>

  Reconstructing high-fidelity animatable 3D human avatars from monocular RGB videos remains challenging, particularly in unconstrained in-the-wild scenarios where camera parameters and human poses from off-the-shelf methods (e.g., COLMAP, HMR2.0) are often inaccurate. Splatting (3DGS) advances demonstrate impressive rendering quality and real-time performance, they critically depend on precise camera calibration and pose annotations, limiting their applicability in real-world settings. We present JOintGS, a unified framework that jointly optimizes camera extrinsics, human poses, and 3D Gaussian representations from coarse initialization through a synergistic refinement mechanism. Our key insight is that explicit foreground-background disentanglement enables mutual reinforcement: static background Gaussians anchor camera estimation via multi-view consistency; refined cameras improve human body alignment through accurate temporal correspondence; optimized human poses enhance scene reconstruction by removing dynamic artifacts from static constraints. We further introduce a temporal dynamics module to capture fine-grained pose-dependent deformations and a residual color field to model illumination variations. Extensive experiments on NeuMan and EMDB datasets demonstrate that JOintGS achieves superior reconstruction quality, with 2.1~dB PSNR improvement over state-of-the-art methods on NeuMan dataset, while maintaining real-time rendering. Notably, our method shows significantly enhanced robustness to noisy initialization compared to the baseline.Our source code is available at https://github.com/MiliLab/JOintGS.

  </details>



- **Light Up Your Face: A Physically Consistent Dataset and Diffusion Model for Face Fill-Light Enhancement**  
  Jue Gong, Zihan Zhou, Jingkai Wang, Xiaohong Liu, Yulun Zhang, Xiaokang Yang  
  _2026-02-04_ · https://arxiv.org/abs/2602.04300v1  
  <details><summary>Abstract</summary>

  Face fill-light enhancement (FFE) brightens underexposed faces by adding virtual fill light while keeping the original scene illumination and background unchanged. Most face relighting methods aim to reshape overall lighting, which can suppress the input illumination or modify the entire scene, leading to foreground-background inconsistency and mismatching practical FFE needs. To support scalable learning, we introduce LightYourFace-160K (LYF-160K), a large-scale paired dataset built with a physically consistent renderer that injects a disk-shaped area fill light controlled by six disentangled factors, producing 160K before-and-after pairs. We first pretrain a physics-aware lighting prompt (PALP) that embeds the 6D parameters into conditioning tokens, using an auxiliary planar-light reconstruction objective. Building on a pretrained diffusion backbone, we then train a fill-light diffusion (FiLitDiff), an efficient one-step model conditioned on physically grounded lighting codes, enabling controllable and high-fidelity fill lighting at low computational cost. Experiments on held-out paired sets demonstrate strong perceptual quality and competitive full-reference metrics, while better preserving background illumination. The dataset and model will be at https://github.com/gobunu/Light-Up-Your-Face.

  </details>



- **Decoupled Hierarchical Distillation for Multimodal Emotion Recognition**  
  Yong Li, Yuanzhi Wang, Yi Ding, Shiqing Zhang, Ke Lu, Cuntai Guan  
  _2026-02-04_ · https://arxiv.org/abs/2602.04260v1  
  <details><summary>Abstract</summary>

  Human multimodal emotion recognition (MER) seeks to infer human emotions by integrating information from language, visual, and acoustic modalities. Although existing MER approaches have achieved promising results, they still struggle with inherent multimodal heterogeneities and varying contributions from different modalities. To address these challenges, we propose a novel framework, Decoupled Hierarchical Multimodal Distillation (DHMD). DHMD decouples each modality's features into modality-irrelevant (homogeneous) and modality-exclusive (heterogeneous) components using a self-regression mechanism. The framework employs a two-stage knowledge distillation (KD) strategy: (1) coarse-grained KD via a Graph Distillation Unit (GD-Unit) in each decoupled feature space, where a dynamic graph facilitates adaptive distillation among modalities, and (2) fine-grained KD through a cross-modal dictionary matching mechanism, which aligns semantic granularities across modalities to produce more discriminative MER representations. This hierarchical distillation approach enables flexible knowledge transfer and effectively improves cross-modal feature alignment. Experimental results demonstrate that DHMD consistently outperforms state-of-the-art MER methods, achieving 1.3\%/2.4\% (ACC$_7$), 1.3\%/1.9\% (ACC$_2$) and 1.9\%/1.8\% (F1) relative improvement on CMU-MOSI/CMU-MOSEI dataset, respectively. Meanwhile, visualization results reveal that both the graph edges and dictionary activations in DHMD exhibit meaningful distribution patterns across modality-irrelevant/-exclusive feature spaces.

  </details>



- **ACIL: Active Class Incremental Learning for Image Classification**  
  Aditya R. Bhattacharya, Debanjan Goswami, Shayok Chakraborty  
  _2026-02-04_ · https://arxiv.org/abs/2602.04252v1  
  <details><summary>Abstract</summary>

  Continual learning (or class incremental learning) is a realistic learning scenario for computer vision systems, where deep neural networks are trained on episodic data, and the data from previous episodes are generally inaccessible to the model. Existing research in this domain has primarily focused on avoiding catastrophic forgetting, which occurs due to the continuously changing class distributions in each episode and the inaccessibility of the data from previous episodes. However, these methods assume that all the training samples in every episode are annotated; this not only incurs a huge annotation cost, but also results in a wastage of annotation effort, since most of the samples in a given episode will not be accessible to the model in subsequent episodes. Active learning algorithms identify the salient and informative samples from large amounts of unlabeled data and are instrumental in reducing the human annotation effort in inducing a deep neural network. In this paper, we propose ACIL, a novel active learning framework for class incremental learning settings. We exploit a criterion based on uncertainty and diversity to identify the exemplar samples that need to be annotated in each episode, and will be appended to the data in the next episode. Such a framework can drastically reduce annotation cost and can also avoid catastrophic forgetting. Our extensive empirical analyses on several vision datasets corroborate the promise and potential of our framework against relevant baselines.

  </details>



- **GeoLanG: Geometry-Aware Language-Guided Grasping with Unified RGB-D Multimodal Learning**  
  Rui Tang, Guankun Wang, Long Bai, Huxin Gao, Jiewen Lai, Chi Kit Ng, Jiazheng Wang, Fan Zhang, Hongliang Ren  
  _2026-02-04_ · https://arxiv.org/abs/2602.04231v1  
  <details><summary>Abstract</summary>

  Language-guided grasping has emerged as a promising paradigm for enabling robots to identify and manipulate target objects through natural language instructions, yet it remains highly challenging in cluttered or occluded scenes. Existing methods often rely on multi-stage pipelines that separate object perception and grasping, which leads to limited cross-modal fusion, redundant computation, and poor generalization in cluttered, occluded, or low-texture scenes. To address these limitations, we propose GeoLanG, an end-to-end multi-task framework built upon the CLIP architecture that unifies visual and linguistic inputs into a shared representation space for robust semantic alignment and improved generalization. To enhance target discrimination under occlusion and low-texture conditions, we explore a more effective use of depth information through the Depth-guided Geometric Module (DGGM), which converts depth into explicit geometric priors and injects them into the attention mechanism without additional computational overhead. In addition, we propose Adaptive Dense Channel Integration, which adaptively balances the contributions of multi-layer features to produce more discriminative and generalizable visual representations. Extensive experiments on the OCID-VLG dataset, as well as in both simulation and real-world hardware, demonstrate that GeoLanG enables precise and robust language-guided grasping in complex, cluttered environments, paving the way toward more reliable multimodal robotic manipulation in real-world human-centric settings.

  </details>



- **An Intuitionistic Fuzzy Logic Driven UNet architecture: Application to Brain Image segmentation**  
  Hanuman Verma, Kiho Im, Pranabesh Maji, Akshansh Gupta  
  _2026-02-04_ · https://arxiv.org/abs/2602.04227v1  
  <details><summary>Abstract</summary>

  Accurate segmentation of MRI brain images is essential for image analysis, diagnosis of neuro-logical disorders and medical image computing. In the deep learning approach, the convolutional neural networks (CNNs), especially UNet, are widely applied in medical image segmentation. However, it is difficult to deal with uncertainty due to the partial volume effect in brain images. To overcome this limitation, we propose an enhanced framework, named UNet with intuitionistic fuzzy logic (IF-UNet), which incorporates intuitionistic fuzzy logic into UNet. The model processes input data in terms of membership, nonmembership, and hesitation degrees, allowing it to better address tissue ambiguity resulting from partial volume effects and boundary uncertainties. The proposed architecture is evaluated on the Internet Brain Segmentation Repository (IBSR) dataset, and its performance is computed using accuracy, Dice coefficient, and intersection over union (IoU). Experimental results confirm that IF-UNet improves segmentation quality with handling uncertainty in brain images.

  </details>



- **VTok: A Unified Video Tokenizer with Decoupled Spatial-Temporal Latents**  
  Feng Wang, Yichun Shi, Ceyuan Yang, Qiushan Guo, Jingxiang Sun, Alan Yuille, Peng Wang  
  _2026-02-04_ · https://arxiv.org/abs/2602.04202v1  
  <details><summary>Abstract</summary>

  This work presents VTok, a unified video tokenization framework that can be used for both generation and understanding tasks. Unlike the leading vision-language systems that tokenize videos through a naive frame-sampling strategy, we propose to decouple the spatial and temporal representations of videos by retaining the spatial features of a single key frame while encoding each subsequent frame into a single residual token, achieving compact yet expressive video tokenization. Our experiments suggest that VTok effectively reduces the complexity of video representation from the product of frame count and per-frame token count to their sum, while the residual tokens sufficiently capture viewpoint and motion changes relative to the key frame. Extensive evaluations demonstrate the efficacy and efficiency of VTok: it achieves notably higher performance on a range of video understanding and text-to-video generation benchmarks compared with baselines using naive tokenization, all with shorter token sequences per video (e.g., 3.4% higher accuracy on our TV-Align benchmark and 1.9% higher VBench score). Remarkably, VTok produces more coherent motion and stronger guidance following in text-to-video generation, owing to its more consistent temporal encoding. We hope VTok can serve as a standardized video tokenization paradigm for future research in video understanding and generation.

  </details>



- **Natural Language Instructions for Scene-Responsive Human-in-the-Loop Motion Planning in Autonomous Driving using Vision-Language-Action Models**  
  Angel Martinez-Sanchez, Parthib Roy, Ross Greer  
  _2026-02-04_ · https://arxiv.org/abs/2602.04184v1  
  <details><summary>Abstract</summary>

  Instruction-grounded driving, where passenger language guides trajectory planning, requires vehicles to understand intent before motion. However, most prior instruction-following planners rely on simulation or fixed command vocabularies, limiting real-world generalization. doScenes, the first real-world dataset linking free-form instructions (with referentiality) to nuScenes ground-truth motion, enables instruction-conditioned planning. In this work, we adapt OpenEMMA, an open-source MLLM-based end-to-end driving framework that ingests front-camera views and ego-state and outputs 10-step speed-curvature trajectories, to this setting, presenting a reproducible instruction-conditioned baseline on doScenes and investigate the effects of human instruction prompts on predicted driving behavior. We integrate doScenes directives as passenger-style prompts within OpenEMMA's vision-language interface, enabling linguistic conditioning before trajectory generation. Evaluated on 849 annotated scenes using ADE, we observe that instruction conditioning substantially improves robustness by preventing extreme baseline failures, yielding a 98.7% reduction in mean ADE. When such outliers are removed, instructions still influence trajectory alignment, with well-phrased prompts improving ADE by up to 5.1%. We use this analysis to discuss what makes a "good" instruction for the OpenEMMA framework. We release the evaluation prompts and scripts to establish a reproducible baseline for instruction-aware planning. GitHub: https://github.com/Mi3-Lab/doScenes-VLM-Planning

  </details>



- **GenMRP: A Generative Multi-Route Planning Framework for Efficient and Personalized Real-Time Industrial Navigation**  
  Chengzhang Wang, Chao Chen, Jun Tao, Tengfei Liu, He Bai, Song Wang, Longfei Xu, Kaikui Liu, Xiangxiang Chu  
  _2026-02-04_ · https://arxiv.org/abs/2602.04174v1  
  <details><summary>Abstract</summary>

  Existing industrial-scale navigation applications contend with massive road networks, typically employing two main categories of approaches for route planning. The first relies on precomputed road costs for optimal routing and heuristic algorithms for generating alternatives, while the second, generative methods, has recently gained significant attention. However, the former struggles with personalization and route diversity, while the latter fails to meet the efficiency requirements of large-scale real-time scenarios. To address these limitations, we propose GenMRP, a generative framework for multi-route planning. To ensure generation efficiency, GenMRP first introduces a skeleton-to-capillary approach that dynamically constructs a relevant sub-network significantly smaller than the full road network. Within this sub-network, routes are generated iteratively. The first iteration identifies the optimal route, while the subsequent ones generate alternatives that balance quality and diversity using the newly proposed correctional boosting approach. Each iteration incorporates road features, user historical sequences, and previously generated routes into a Link Cost Model to update road costs, followed by route generation using the Dijkstra algorithm. Extensive experiments show that GenMRP achieves state-of-the-art performance with high efficiency in both offline and online environments. To facilitate further research, we have publicly released the training and evaluation dataset. GenMRP has been fully deployed in a real-world navigation app, demonstrating its effectiveness and benefits.

  </details>



- **MA3DSG: Multi-Agent 3D Scene Graph Generation for Large-Scale Indoor Environments**  
  Yirum Kim, Jaewoo Kim, Ue-Hwan Kim  
  _2026-02-04_ · https://arxiv.org/abs/2602.04152v1  
  <details><summary>Abstract</summary>

  Current 3D scene graph generation (3DSGG) approaches heavily rely on a single-agent assumption and small-scale environments, exhibiting limited scalability to real-world scenarios. In this work, we introduce Multi-Agent 3D Scene Graph Generation (MA3DSG) model, the first framework designed to tackle this scalability challenge using multiple agents. We develop a training-free graph alignment algorithm that efficiently merges partial query graphs from individual agents into a unified global scene graph. Leveraging extensive analysis and empirical insights, our approach enables conventional single-agent systems to operate collaboratively without requiring any learnable parameters. To rigorously evaluate 3DSGG performance, we propose MA3DSG-Bench-a benchmark that supports diverse agent configurations, domain sizes, and environmental conditions-providing a more general and extensible evaluation framework. This work lays a solid foundation for scalable, multi-agent 3DSGG research.

  </details>



- **JSynFlow: Japanese Synthesised Flowchart Visual Question Answering Dataset built with Large Language Models**  
  Hiroshi Sasaki  
  _2026-02-04_ · https://arxiv.org/abs/2602.04142v1  
  <details><summary>Abstract</summary>

  Vision and language models (VLMs) are expected to analyse complex documents, such as those containing flowcharts, through a question-answering (QA) interface. The ability to recognise and interpret these flowcharts is in high demand, as they provide valuable insights unavailable in text-only explanations. However, developing VLMs with precise flowchart understanding requires large-scale datasets of flowchart images and corresponding text, the creation of which is highly time-consuming. To address this challenge, we introduce JSynFlow, a synthesised visual QA dataset for Japanese flowcharts, generated using large language models (LLMs). Our dataset comprises task descriptions for various business occupations, the corresponding flowchart images rendered from domain-specific language (DSL) code, and related QA pairs. This paper details the dataset's synthesis procedure and demonstrates that fine-tuning with JSynFlow significantly improves VLM performance on flowchart-based QA tasks. Our dataset is publicly available at https://huggingface.co/datasets/jri-advtechlab/jsynflow.

  </details>



- **KGLAMP: Knowledge Graph-guided Language model for Adaptive Multi-robot Planning and Replanning**  
  Chak Lam Shek, Faizan M. Tariq, Sangjae Bae, David Isele, Piyush Gupta  
  _2026-02-04_ · https://arxiv.org/abs/2602.04129v1  
  <details><summary>Abstract</summary>

  Heterogeneous multi-robot systems are increasingly deployed in long-horizon missions that require coordination among robots with diverse capabilities. However, existing planning approaches struggle to construct accurate symbolic representations and maintain plan consistency in dynamic environments. Classical PDDL planners require manually crafted symbolic models, while LLM-based planners often ignore agent heterogeneity and environmental uncertainty. We introduce KGLAMP, a knowledge-graph-guided LLM planning framework for heterogeneous multi-robot teams. The framework maintains a structured knowledge graph encoding object relations, spatial reachability, and robot capabilities, which guides the LLM in generating accurate PDDL problem specifications. The knowledge graph serves as a persistent, dynamically updated memory that incorporates new observations and triggers replanning upon detecting inconsistencies, enabling symbolic plans to adapt to evolving world states. Experiments on the MAT-THOR benchmark show that KGLAMP improves performance by at least 25.5% over both LLM-only and PDDL-based variants.

  </details>



- **DMS2F-HAD: A Dual-branch Mamba-based Spatial-Spectral Fusion Network for Hyperspectral Anomaly Detection**  
  Aayushma Pant, Lakpa Tamang, Tsz-Kwan Lee, Sunil Aryal  
  _2026-02-04_ · https://arxiv.org/abs/2602.04102v1  
  <details><summary>Abstract</summary>

  Hyperspectral anomaly detection (HAD) aims to identify rare and irregular targets in high-dimensional hyperspectral images (HSIs), which are often noisy and unlabelled data. Existing deep learning methods either fail to capture long-range spectral dependencies (e.g., convolutional neural networks) or suffer from high computational cost (e.g., Transformers). To address these challenges, we propose DMS2F-HAD, a novel dual-branch Mamba-based model. Our architecture utilizes Mamba's linear-time modeling to efficiently learn distinct spatial and spectral features in specialized branches, which are then integrated by a dynamic gated fusion mechanism to enhance anomaly localization. Across fourteen benchmark HSI datasets, our proposed DMS2F-HAD not only achieves a state-of-the-art average AUC of 98.78%, but also demonstrates superior efficiency with an inference speed 4.6 times faster than comparable deep learning methods. The results highlight DMS2FHAD's strong generalization and scalability, positioning it as a strong candidate for practical HAD applications.

  </details>



- **VideoBrain: Learning Adaptive Frame Sampling for Long Video Understanding**  
  Junbo Zou, Ziheng Huang, Shengjie Zhang, Liwen Zhang, Weining Shen  
  _2026-02-04_ · https://arxiv.org/abs/2602.04094v1  
  <details><summary>Abstract</summary>

  Long-form video understanding remains challenging for Vision-Language Models (VLMs) due to the inherent tension between computational constraints and the need to capture information distributed across thousands of frames. Existing approaches either sample frames uniformly (risking information loss) or select keyframes in a single pass (with no recovery from poor choices). We propose VideoBrain, an end-to-end framework that enables VLMs to adaptively acquire visual information through learned sampling policies. Our approach features dual complementary agents: a CLIP-based agent for semantic retrieval across the video and a Uniform agent for dense temporal sampling within intervals. Unlike prior agent-based methods that rely on text-only LLMs orchestrating visual tools, our VLM directly perceives frames and reasons about information sufficiency. To prevent models from invoking agents indiscriminately to maximize rewards, we introduce a behavior-aware reward function coupled with a data classification pipeline that teaches the model when agent invocation is genuinely beneficial. Experiments on four long video benchmarks demonstrate that VideoBrain achieves +3.5% to +9.0% improvement over the baseline while using 30-40% fewer frames, with strong cross-dataset generalization to short video benchmarks.

  </details>



- **iSight: Towards expert-AI co-assessment for improved immunohistochemistry staining interpretation**  
  Jacob S. Leiby, Jialu Yao, Pan Lu, George Hu, Anna Davidian, Shunsuke Koga, Olivia Leung, Pravin Patel, Isabella Tondi Resta, Rebecca Rojansky, et al.  
  _2026-02-03_ · https://arxiv.org/abs/2602.04063v1  
  <details><summary>Abstract</summary>

  Immunohistochemistry (IHC) provides information on protein expression in tissue sections and is commonly used to support pathology diagnosis and disease triage. While AI models for H\&E-stained slides show promise, their applicability to IHC is limited due to domain-specific variations. Here we introduce HPA10M, a dataset that contains 10,495,672 IHC images from the Human Protein Atlas with comprehensive metadata included, and encompasses 45 normal tissue types and 20 major cancer types. Based on HPA10M, we trained iSight, a multi-task learning framework for automated IHC staining assessment. iSight combines visual features from whole-slide images with tissue metadata through a token-level attention mechanism, simultaneously predicting staining intensity, location, quantity, tissue type, and malignancy status. On held-out data, iSight achieved 85.5\% accuracy for location, 76.6\% for intensity, and 75.7\% for quantity, outperforming fine-tuned foundation models (PLIP, CONCH) by 2.5--10.2\%. In addition, iSight demonstrates well-calibrated predictions with expected calibration errors of 0.0150-0.0408. Furthermore, in a user study with eight pathologists evaluating 200 images from two datasets, iSight outperformed initial pathologist assessments on the held-out HPA dataset (79\% vs 68\% for location, 70\% vs 57\% for intensity, 68\% vs 52\% for quantity). Inter-pathologist agreement also improved after AI assistance in both held-out HPA (Cohen's $κ$ increased from 0.63 to 0.70) and Stanford TMAD datasets (from 0.74 to 0.76), suggesting expert--AI co-assessment can improve IHC interpretation. This work establishes a foundation for AI systems that can improve IHC diagnostic accuracy and highlights the potential for integrating iSight into clinical workflows to enhance the consistency and reliability of IHC assessment.

  </details>



- **AtlasPatch: An Efficient and Scalable Tool for Whole Slide Image Preprocessing in Computational Pathology**  
  Ahmed Alagha, Christopher Leclerc, Yousef Kotp, Omar Metwally, Calvin Moras, Peter Rentopoulos, Ghodsiyeh Rostami, Bich Ngoc Nguyen, Jumanah Baig, Abdelhakim Khellaf, et al.  
  _2026-02-03_ · https://arxiv.org/abs/2602.03998v1  
  <details><summary>Abstract</summary>

  Whole-slide image (WSI) preprocessing, typically comprising tissue detection followed by patch extraction, is foundational to AI-driven computational pathology workflows. This remains a major computational bottleneck as existing tools either rely on inaccurate heuristic thresholding for tissue detection, or adopt AI-based approaches trained on limited-diversity data that operate at the patch level, incurring substantial computational complexity. We present AtlasPatch, an efficient and scalable slide preprocessing framework for accurate tissue detection and high-throughput patch extraction with minimal computational overhead. AtlasPatch's tissue detection module is trained on a heterogeneous and semi-manually annotated dataset of ~30,000 WSI thumbnails, using efficient fine-tuning of the Segment-Anything model. The tool extrapolates tissue masks from thumbnails to full-resolution slides to extract patch coordinates at user-specified magnifications, with options to stream patches directly into common image encoders for embedding or store patch images, all efficiently parallelized across CPUs and GPUs. We assess AtlasPatch across segmentation precision, computational complexity, and downstream multiple-instance learning, matching state-of-the-art performance while operating at a fraction of their computational cost. AtlasPatch is open-source and available at https://github.com/AtlasAnalyticsLab/AtlasPatch.

  </details>



- **Efficient Long-Horizon Vision-Language-Action Models via Static-Dynamic Disentanglement**  
  Weikang Qiu, Tinglin Huang, Aosong Feng, Rex Ying  
  _2026-02-03_ · https://arxiv.org/abs/2602.03983v1  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models have recently emerged as a promising paradigm for generalist robotic control. Built upon vision-language model (VLM) architectures, VLAs predict actions conditioned on visual observations and language instructions, achieving strong performance and generalization across tasks. However, VLAs face two major challenges: limited long-horizon context and inefficient inference due to the quadratic attention complexity and large parameter counts. Our work is motivated by the observation that much of the visual information in a trajectory remains static across timesteps (e.g., the background). Leveraging this property, we propose SD-VLA, a framework that disentangles visual inputs into multi-level static and dynamic tokens, which enables (1) retaining a single copy of static tokens across frames to significantly reduce context length, and (2) reusing the key-value (KV) cache of static tokens through a lightweight recache gate that updates only when necessary. This design enables efficient multi-frame integration and efficient inference. In addition, we introduce a new benchmark that more effectively evaluates the long-horizon temporal dependency modeling ability of VLAs. Experimental results show that our approach outperforms baselines on this benchmark by 39.8% absolute improvement in success rate, and achieves a 3.9% gain on the SimplerEnv benchmark. Moreover, SD-VLA delivers a 2.26x inference speedup over the base VLA model on the same benchmark, enabling faster and more practical real-world deployment.

  </details>



- **FullStack-Agent: Enhancing Agentic Full-Stack Web Coding via Development-Oriented Testing and Repository Back-Translation**  
  Zimu Lu, Houxing Ren, Yunqiao Yang, Ke Wang, Zhuofan Zong, Mingjie Zhan, Hongsheng Li  
  _2026-02-03_ · https://arxiv.org/abs/2602.03798v1  
  <details><summary>Abstract</summary>

  Assisting non-expert users to develop complex interactive websites has become a popular task for LLM-powered code agents. However, existing code agents tend to only generate frontend web pages, masking the lack of real full-stack data processing and storage with fancy visual effects. Notably, constructing production-level full-stack web applications is far more challenging than only generating frontend web pages, demanding careful control of data flow, comprehensive understanding of constantly updating packages and dependencies, and accurate localization of obscure bugs in the codebase. To address these difficulties, we introduce FullStack-Agent, a unified agent system for full-stack agentic coding that consists of three parts: (1) FullStack-Dev, a multi-agent framework with strong planning, code editing, codebase navigation, and bug localization abilities. (2) FullStack-Learn, an innovative data-scaling and self-improving method that back-translates crawled and synthesized website repositories to improve the backbone LLM of FullStack-Dev. (3) FullStack-Bench, a comprehensive benchmark that systematically tests the frontend, backend and database functionalities of the generated website. Our FullStack-Dev outperforms the previous state-of-the-art method by 8.7%, 38.2%, and 15.9% on the frontend, backend, and database test cases respectively. Additionally, FullStack-Learn raises the performance of a 30B model by 9.7%, 9.5%, and 2.8% on the three sets of test cases through self-improvement, demonstrating the effectiveness of our approach. The code is released at https://github.com/mnluzimu/FullStack-Agent.

  </details>



- **SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?**  
  Azmine Toushik Wasi, Wahid Faisal, Abdur Rahman, Mahfuz Ahmed Anik, Munem Shahriar, Mohsin Mahmud Topu, Sadia Tasnim Meem, Rahatun Nesa Priti, Sabrina Afroz Mitu, Md. Iqramul Hoque, et al.  
  _2026-02-03_ · https://arxiv.org/abs/2602.03916v1  
  <details><summary>Abstract</summary>

  Spatial reasoning is a fundamental aspect of human cognition, yet it remains a major challenge for contemporary vision-language models (VLMs). Prior work largely relied on synthetic or LLM-generated environments with limited task designs and puzzle-like setups, failing to capture the real-world complexity, visual noise, and diverse spatial relationships that VLMs encounter. To address this, we introduce SpatiaLab, a comprehensive benchmark for evaluating VLMs' spatial reasoning in realistic, unconstrained contexts. SpatiaLab comprises 1,400 visual question-answer pairs across six major categories: Relative Positioning, Depth & Occlusion, Orientation, Size & Scale, Spatial Navigation, and 3D Geometry, each with five subcategories, yielding 30 distinct task types. Each subcategory contains at least 25 questions, and each main category includes at least 200 questions, supporting both multiple-choice and open-ended evaluation. Experiments across diverse state-of-the-art VLMs, including open- and closed-source models, reasoning-focused, and specialized spatial reasoning models, reveal a substantial gap in spatial reasoning capabilities compared with humans. In the multiple-choice setup, InternVL3.5-72B achieves 54.93% accuracy versus 87.57% for humans. In the open-ended setting, all models show a performance drop of around 10-25%, with GPT-5-mini scoring highest at 40.93% versus 64.93% for humans. These results highlight key limitations in handling complex spatial relationships, depth perception, navigation, and 3D geometry. By providing a diverse, real-world evaluation framework, SpatiaLab exposes critical challenges and opportunities for advancing VLMs' spatial reasoning, offering a benchmark to guide future research toward robust, human-aligned spatial understanding. SpatiaLab is available at: https://spatialab-reasoning.github.io/.

  </details>



- **RAWDet-7: A Multi-Scenario Benchmark for Object Detection and Description on Quantized RAW Images**  
  Mishal Fatima, Shashank Agnihotri, Kanchana Vaishnavi Gandikota, Michael Moeller, Margret Keuper  
  _2026-02-03_ · https://arxiv.org/abs/2602.03760v1  
  <details><summary>Abstract</summary>

  Most vision models are trained on RGB images processed through ISP pipelines optimized for human perception, which can discard sensor-level information useful for machine reasoning. RAW images preserve unprocessed scene data, enabling models to leverage richer cues for both object detection and object description, capturing fine-grained details, spatial relationships, and contextual information often lost in processed images. To support research in this domain, we introduce RAWDet-7, a large-scale dataset of ~25k training and 7.6k test RAW images collected across diverse cameras, lighting conditions, and environments, densely annotated for seven object categories following MS-COCO and LVIS conventions. In addition, we provide object-level descriptions derived from the corresponding high-resolution sRGB images, facilitating the study of object-level information preservation under RAW image processing and low-bit quantization. The dataset allows evaluation under simulated 4-bit, 6-bit, and 8-bit quantization, reflecting realistic sensor constraints, and provides a benchmark for studying detection performance, description quality & detail, and generalization in low-bit RAW image processing. Dataset & code upon acceptance.

  </details>



- **Edge-Optimized Vision-Language Models for Underground Infrastructure Assessment**  
  Johny J. Lopez, Md Meftahul Ferdaus, Mahdi Abdelguerfi  
  _2026-02-03_ · https://arxiv.org/abs/2602.03742v1  
  <details><summary>Abstract</summary>

  Autonomous inspection of underground infrastructure, such as sewer and culvert systems, is critical to public safety and urban sustainability. Although robotic platforms equipped with visual sensors can efficiently detect structural deficiencies, the automated generation of human-readable summaries from these detections remains a significant challenge, especially on resource-constrained edge devices. This paper presents a novel two-stage pipeline for end-to-end summarization of underground deficiencies, combining our lightweight RAPID-SCAN segmentation model with a fine-tuned Vision-Language Model (VLM) deployed on an edge computing platform. The first stage employs RAPID-SCAN (Resource-Aware Pipeline Inspection and Defect Segmentation using Compact Adaptive Network), achieving 0.834 F1-score with only 0.64M parameters for efficient defect segmentation. The second stage utilizes a fine-tuned Phi-3.5 VLM that generates concise, domain-specific summaries in natural language from the segmentation outputs. We introduce a curated dataset of inspection images with manually verified descriptions for VLM fine-tuning and evaluation. To enable real-time performance, we employ post-training quantization with hardware-specific optimization, achieving significant reductions in model size and inference latency without compromising summarization quality. We deploy and evaluate our complete pipeline on a mobile robotic platform, demonstrating its effectiveness in real-world inspection scenarios. Our results show the potential of edge-deployable integrated AI systems to bridge the gap between automated defect detection and actionable insights for infrastructure maintenance, paving the way for more scalable and autonomous inspection solutions.

  </details>



- **RegionReasoner: Region-Grounded Multi-Round Visual Reasoning**  
  Wenfang Sun, Hao Chen, Yingjun Du, Yefeng Zheng, Cees G. M. Snoek  
  _2026-02-03_ · https://arxiv.org/abs/2602.03733v1  
  <details><summary>Abstract</summary>

  Large vision-language models have achieved remarkable progress in visual reasoning, yet most existing systems rely on single-step or text-only reasoning, limiting their ability to iteratively refine understanding across multiple visual contexts. To address this limitation, we introduce a new multi-round visual reasoning benchmark with training and test sets spanning both detection and segmentation tasks, enabling systematic evaluation under iterative reasoning scenarios. We further propose RegionReasoner, a reinforcement learning framework that enforces grounded reasoning by requiring each reasoning trace to explicitly cite the corresponding reference bounding boxes, while maintaining semantic coherence via a global-local consistency reward. This reward extracts key objects and nouns from both global scene captions and region-level captions, aligning them with the reasoning trace to ensure consistency across reasoning steps. RegionReasoner is optimized with structured rewards combining grounding fidelity and global-local semantic alignment. Experiments on detection and segmentation tasks show that RegionReasoner-7B, together with our newly introduced benchmark RegionDial-Bench, considerably improves multi-round reasoning accuracy, spatial grounding precision, and global-local consistency, establishing a strong baseline for this emerging research direction.

  </details>



- **Referring Industrial Anomaly Segmentation**  
  Pengfei Yue, Xiaokang Jiang, Yilin Lu, Jianghang Lin, Shengchuan Zhang, Liujuan Cao  
  _2026-02-03_ · https://arxiv.org/abs/2602.03673v1  
  <details><summary>Abstract</summary>

  Industrial Anomaly Detection (IAD) is vital for manufacturing, yet traditional methods face significant challenges: unsupervised approaches yield rough localizations requiring manual thresholds, while supervised methods overfit due to scarce, imbalanced data. Both suffer from the "One Anomaly Class, One Model" limitation. To address this, we propose Referring Industrial Anomaly Segmentation (RIAS), a paradigm leveraging language to guide detection. RIAS generates precise masks from text descriptions without manual thresholds and uses universal prompts to detect diverse anomalies with a single model. We introduce the MVTec-Ref dataset to support this, designed with diverse referring expressions and focusing on anomaly patterns, notably with 95% small anomalies. We also propose the Dual Query Token with Mask Group Transformer (DQFormer) benchmark, enhanced by Language-Gated Multi-Level Aggregation (LMA) to improve multi-scale segmentation. Unlike traditional methods using redundant queries, DQFormer employs only "Anomaly" and "Background" tokens for efficient visual-textual integration. Experiments demonstrate RIAS's effectiveness in advancing IAD toward open-set capabilities. Code: https://github.com/swagger-coder/RIAS-MVTec-Ref.

  </details>



- **MM-SCALE: Grounded Multimodal Moral Reasoning via Scalar Judgment and Listwise Alignment**  
  Eunkyu Park, Wesley Hanwen Deng, Cheyon Jin, Matheus Kunzler Maldaner, Jordan Wheeler, Jason I. Hong, Hong Shen, Adam Perer, Ken Holstein, Motahhare Eslami, et al.  
  _2026-02-03_ · https://arxiv.org/abs/2602.03665v1  
  <details><summary>Abstract</summary>

  Vision-Language Models (VLMs) continue to struggle to make morally salient judgments in multimodal and socially ambiguous contexts. Prior works typically rely on binary or pairwise supervision, which often fail to capture the continuous and pluralistic nature of human moral reasoning. We present MM-SCALE (Multimodal Moral Scale), a large-scale dataset for aligning VLMs with human moral preferences through 5-point scalar ratings and explicit modality grounding. Each image-scenario pair is annotated with moral acceptability scores and grounded reasoning labels by humans using an interface we tailored for data collection, enabling listwise preference optimization over ranked scenario sets. By moving from discrete to scalar supervision, our framework provides richer alignment signals and finer calibration of multimodal moral reasoning. Experiments show that VLMs fine-tuned on MM-SCALE achieve higher ranking fidelity and more stable safety calibration than those trained with binary signals.

  </details>



- **SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection**  
  Wei Zhang, Xiang Liu, Ningjing Liu, Mingxin Liu, Wei Liao, Chunyan Xu, Xue Yang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03634v1  
  <details><summary>Abstract</summary>

  A consistent trend throughout the research of oriented object detection has been the pursuit of maintaining comparable performance with fewer and weaker annotations. This is particularly crucial in the remote sensing domain, where the dense object distribution and a wide variety of categories contribute to prohibitively high costs. Based on the supervision level, existing oriented object detection algorithms can be broadly grouped into fully supervised, semi-supervised, and weakly supervised methods. Within the scope of this work, we further categorize them to include sparsely supervised and partially weakly-supervised methods. To address the challenges of large-scale labeling, we introduce the first Sparse Partial Weakly-Supervised Oriented Object Detection framework, designed to efficiently leverage only a few sparse weakly-labeled data and plenty of unlabeled data. Our framework incorporates three key innovations: (1) We design a Sparse-annotation-Orientation-and-Scale-aware Student (SOS-Student) model to separate unlabeled objects from the background in a sparsely-labeled setting, and learn orientation and scale information from orientation-agnostic or scale-agnostic weak annotations. (2) We construct a novel Multi-level Pseudo-label Filtering strategy that leverages the distribution of model predictions, which is informed by the model's multi-layer predictions. (3) We propose a unique sparse partitioning approach, ensuring equal treatment for each category. Extensive experiments on the DOTA and DIOR datasets show that our framework achieves a significant performance gain over traditional oriented object detection methods mentioned above, offering a highly cost-effective solution. Our code is publicly available at https://github.com/VisionXLab/SPWOOD.

  </details>



- **KTV: Keyframes and Key Tokens Selection for Efficient Training-Free Video LLMs**  
  Baiyang Song, Jun Peng, Yuxin Zhang, Guangyao Chen, Feidiao Yang, Jianyuan Guo  
  _2026-02-03_ · https://arxiv.org/abs/2602.03615v1  
  <details><summary>Abstract</summary>

  Training-free video understanding leverages the strong image comprehension capabilities of pre-trained vision language models (VLMs) by treating a video as a sequence of static frames, thus obviating the need for costly video-specific training. However, this paradigm often suffers from severe visual redundancy and high computational overhead, especially when processing long videos. Crucially, existing keyframe selection strategies, especially those based on CLIP similarity, are prone to biases and may inadvertently overlook critical frames, resulting in suboptimal video comprehension. To address these significant challenges, we propose \textbf{KTV}, a novel two-stage framework for efficient and effective training-free video understanding. In the first stage, KTV performs question-agnostic keyframe selection by clustering frame-level visual features, yielding a compact, diverse, and representative subset of frames that mitigates temporal redundancy. In the second stage, KTV applies key visual token selection, pruning redundant or less informative tokens from each selected keyframe based on token importance and redundancy, which significantly reduces the number of tokens fed into the LLM. Extensive experiments on the Multiple-Choice VideoQA task demonstrate that KTV outperforms state-of-the-art training-free baselines while using significantly fewer visual tokens, \emph{e.g.}, only 504 visual tokens for a 60-min video with 10800 frames, achieving $44.8\%$ accuracy on the MLVU-Test benchmark. In particular, KTV also exceeds several training-based approaches on certain benchmarks.

  </details>



- **High-Resolution Underwater Camouflaged Object Detection: GBU-UCOD Dataset and Topology-Aware and Frequency-Decoupled Networks**  
  Wenji Wu, Shuo Ye, Yiyu Liu, Jiguang He, Zhuo Wang, Zitong Yu  
  _2026-02-03_ · https://arxiv.org/abs/2602.03591v1  
  <details><summary>Abstract</summary>

  Underwater Camouflaged Object Detection (UCOD) is a challenging task due to the extreme visual similarity between targets and backgrounds across varying marine depths. Existing methods often struggle with topological fragmentation of slender creatures in the deep sea and the subtle feature extraction of transparent organisms. In this paper, we propose DeepTopo-Net, a novel framework that integrates topology-aware modeling with frequency-decoupled perception. To address physical degradation, we design the Water-Conditioned Adaptive Perceptor (WCAP), which employs Riemannian metric tensors to dynamically deform convolutional sampling fields. Furthermore, the Abyssal-Topology Refinement Module (ATRM) is developed to maintain the structural connectivity of spindly targets through skeletal priors. Specifically, we first introduce GBU-UCOD, the first high-resolution (2K) benchmark tailored for marine vertical zonation, filling the data gap for hadal and abyssal zones. Extensive experiments on MAS3K, RMAS, and our proposed GBU-UCOD datasets demonstrate that DeepTopo-Net achieves state-of-the-art performance, particularly in preserving the morphological integrity of complex underwater patterns. The datasets and codes will be released at https://github.com/Wuwenji18/GBU-UCOD.

  </details>



- **SlowFocus: Enhancing Fine-grained Temporal Understanding in Video LLM**  
  Ming Nie, Dan Ding, Chunwei Wang, Yuanfan Guo, Jianhua Han, Hang Xu, Li Zhang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03589v1  
  <details><summary>Abstract</summary>

  Large language models (LLMs) have demonstrated exceptional capabilities in text understanding, which has paved the way for their expansion into video LLMs (Vid-LLMs) to analyze video data. However, current Vid-LLMs struggle to simultaneously retain high-quality frame-level semantic information (i.e., a sufficient number of tokens per frame) and comprehensive video-level temporal information (i.e., an adequate number of sampled frames per video). This limitation hinders the advancement of Vid-LLMs towards fine-grained video understanding. To address this issue, we introduce the SlowFocus mechanism, which significantly enhances the equivalent sampling frequency without compromising the quality of frame-level visual tokens. SlowFocus begins by identifying the query-related temporal segment based on the posed question, then performs dense sampling on this segment to extract local high-frequency features. A multi-frequency mixing attention module is further leveraged to aggregate these local high-frequency details with global low-frequency contexts for enhanced temporal comprehension. Additionally, to tailor Vid-LLMs to this innovative mechanism, we introduce a set of training strategies aimed at bolstering both temporal grounding and detailed temporal reasoning capabilities. Furthermore, we establish FineAction-CGR, a benchmark specifically devised to assess the ability of Vid-LLMs to process fine-grained temporal understanding tasks. Comprehensive experiments demonstrate the superiority of our mechanism across both existing public video understanding benchmarks and our proposed FineAction-CGR.

  </details>



- **AffordanceGrasp-R1:Leveraging Reasoning-Based Affordance Segmentation with Reinforcement Learning for Robotic Grasping**  
  Dingyi Zhou, Mu He, Zhuowei Fang, Xiangtong Yao, Yinlong Liu, Alois Knoll, Hu Cao  
  _2026-02-03_ · https://arxiv.org/abs/2602.03547v1  
  <details><summary>Abstract</summary>

  We introduce AffordanceGrasp-R1, a reasoning-driven affordance segmentation framework for robotic grasping that combines a chain-of-thought (CoT) cold-start strategy with reinforcement learning to enhance deduction and spatial grounding. In addition, we redesign the grasping pipeline to be more context-aware by generating grasp candidates from the global scene point cloud and subsequently filtering them using instruction-conditioned affordance masks. Extensive experiments demonstrate that AffordanceGrasp-R1 consistently outperforms state-of-the-art (SOTA) methods on benchmark datasets, and real-world robotic grasping evaluations further validate its robustness and generalization under complex language-conditioned manipulation scenarios.

  </details>


