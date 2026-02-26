# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-02-26 07:14 UTC_

Total papers shown: **50**


---

- **Behavioral Cloning for Robotic Connector Assembly: An Empirical Study**  
  Andreas Kernbach, Daniel Bargmann, Werner Kraus, Marco F. Huber  
  _2026-02-25_ · https://arxiv.org/abs/2602.22100v1  
  <details><summary>Abstract</summary>

  Automating the assembly of wire harnesses is challenging in automotive, electrical cabinet, and aircraft production, particularly due to deformable cables and a high variance in connector geometries. In addition, connectors must be inserted with limited force to avoid damage, while their poses can vary significantly. While humans can do this task intuitively by combining visual and haptic feedback, programming an industrial robot for such a task in an adaptable manner remains difficult. This work presents an empirical study investigating the suitability of behavioral cloning for learning an action prediction model for connector insertion that fuses force-torque sensing with a fixed position camera. We compare several network architectures and other design choices using a dataset of up to 300 successful human demonstrations collected via teleoperation of a UR5e robot with a SpaceMouse under varying connector poses. The resulting system is then evaluated against five different connector geometries under varying connector poses, achieving an overall insertion success rate of over 90 %.

  </details>



- **Overview of the CXR-LT 2026 Challenge: Multi-Center Long-Tailed and Zero Shot Chest X-ray Classification**  
  Hexin Dong, Yi Lin, Pengyu Zhou, Fengnian Zhao, Alan Clint Legasto, Mingquan Lin, Hao Chen, Yuzhe Yang, George Shih, Yifan Peng  
  _2026-02-25_ · https://arxiv.org/abs/2602.22092v1  
  <details><summary>Abstract</summary>

  Chest X-ray (CXR) interpretation is hindered by the long-tailed distribution of pathologies and the open-world nature of clinical environments. Existing benchmarks often rely on closed-set classes from single institutions, failing to capture the prevalence of rare diseases or the appearance of novel findings. To address this, we present the CXR-LT 2026 challenge. This third iteration of the benchmark introduces a multi-center dataset comprising over 145,000 images from PadChest and NIH Chest X-ray datasets. The challenge defines two core tasks: (1) Robust Multi-Label Classification on 30 known classes and (2) Open-World Generalization to 6 unseen (out-of-distribution) rare disease classes. We report the results of the top-performing teams, evaluating them via mean Average Precision (mAP), AUROC, and F1-score. The winning solutions achieved an mAP of 0.5854 on Task 1 and 0.4315 on Task 2, demonstrating that large-scale vision-language pre-training significantly mitigates the performance drop typically associated with zero-shot diagnosis.

  </details>



- **Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos**  
  Matthew Strong, Wei-Jer Chang, Quentin Herau, Jiezhi Yang, Yihan Hu, Chensheng Peng, Wei Zhan  
  _2026-02-25_ · https://arxiv.org/abs/2602.22091v1  
  <details><summary>Abstract</summary>

  Ego-centric driving videos available online provide an abundant source of visual data for autonomous driving, yet their lack of annotations makes it difficult to learn representations that capture both semantic structure and 3D geometry. Recent advances in large feedforward spatial models demonstrate that point maps and ego-motion can be inferred in a single forward pass, suggesting a promising direction for scalable driving perception. We therefore propose a label-free, teacher-guided framework for learning autonomous driving representations directly from unposed videos. Unlike prior self-supervised approaches that focus primarily on frame-to-frame consistency, we posit that safe and reactive driving depends critically on temporal context. To this end, we leverage a feedforward architecture equipped with a lightweight autoregressive module, trained using multi-modal supervisory signals that guide the model to jointly predict current and future point maps, camera poses, semantic segmentation, and motion masks. Multi-modal teachers provide sequence-level pseudo-supervision, enabling LFG to learn a unified pseudo-4D representation from raw YouTube videos without poses, labels, or LiDAR. The resulting encoder not only transfers effectively to downstream autonomous driving planning on the NAVSIM benchmark, surpassing multi-camera and LiDAR baselines with only a single monocular camera, but also yields strong performance when evaluated on a range of semantic, geometric, and qualitative motion prediction tasks. These geometry and motion-aware features position LFG as a compelling video-centric foundation model for autonomous driving.

  </details>



- **AutoSew: A Geometric Approach to Stitching Prediction with Graph Neural Networks**  
  Pablo Ríos-Navarro, Elena Garces, Jorge Lopez-Moreno  
  _2026-02-25_ · https://arxiv.org/abs/2602.22052v1  
  <details><summary>Abstract</summary>

  Automating garment assembly from sewing patterns remains a significant challenge due to the lack of standardized annotation protocols and the frequent absence of semantic cues. Existing methods often rely on panel labels or handcrafted heuristics, which limit their applicability to real-world, non-conforming patterns. We present AutoSew, a fully automatic, geometry-based approach for predicting stitch correspondences directly from 2D pattern contours. AutoSew formulates the problem as a graph matching task, leveraging a Graph Neural Network to capture local and global geometric context, and employing a differentiable optimal transport solver to infer stitching relationships-including multi-edge connections. To support this task, we update the GarmentCodeData dataset modifying over 18k patterns with realistic multi-edge annotations, reflecting industrial assembly scenarios. AutoSew achieves 96% F1-score and successfully assembles 73.3% of test garments without error, outperforming existing methods while relying solely on geometric input. Our results demonstrate that geometry alone can robustly guide stitching prediction, enabling scalable garment assembly without manual input.

  </details>



- **RT-RMOT: A Dataset and Framework for RGB-Thermal Referring Multi-Object Tracking**  
  Yanqiu Yu, Zhifan Jin, Sijia Chen, Tongfei Chu, En Yu, Liman Liu, Wenbing Tao  
  _2026-02-25_ · https://arxiv.org/abs/2602.22033v1  
  <details><summary>Abstract</summary>

  Referring Multi-Object Tracking has attracted increasing attention due to its human-friendly interactive characteristics, yet it exhibits limitations in low-visibility conditions, such as nighttime, smoke, and other challenging scenarios. To overcome this limitation, we propose a new RGB-Thermal RMOT task, named RT-RMOT, which aims to fuse RGB appearance features with the illumination robustness of the thermal modality to enable all-day referring multi-object tracking. To promote research on RT-RMOT, we construct the first Referring Multi-Object Tracking dataset under RGB-Thermal modality, named RefRT. It contains 388 language descriptions, 1,250 tracked targets, and 166,147 Language-RGB-Thermal (L-RGB-T) triplets. Furthermore, we propose RTrack, a framework built upon a multimodal large language model (MLLM) that integrates RGB, thermal, and textual features. Since the initial framework still leaves room for improvement, we introduce a Group Sequence Policy Optimization (GSPO) strategy to further exploit the model's potential. To alleviate training instability during RL fine-tuning, we introduce a Clipped Advantage Scaling (CAS) strategy to suppress gradient explosion. In addition, we design Structured Output Reward and Comprehensive Detection Reward to balance exploration and exploitation, thereby improving the completeness and accuracy of target perception. Extensive experiments on the RefRT dataset demonstrate the effectiveness of the proposed RTrack framework.

  </details>



- **RGB-Event HyperGraph Prompt for Kilometer Marker Recognition based on Pre-trained Foundation Models**  
  Xiaoyu Xian, Shiao Wang, Xiao Wang, Daxin Tian, Yan Tian  
  _2026-02-25_ · https://arxiv.org/abs/2602.22026v1  
  <details><summary>Abstract</summary>

  Metro trains often operate in highly complex environments, characterized by illumination variations, high-speed motion, and adverse weather conditions. These factors pose significant challenges for visual perception systems, especially those relying solely on conventional RGB cameras. To tackle these difficulties, we explore the integration of event cameras into the perception system, leveraging their advantages in low-light conditions, high-speed scenarios, and low power consumption. Specifically, we focus on Kilometer Marker Recognition (KMR), a critical task for autonomous metro localization under GNSS-denied conditions. In this context, we propose a robust baseline method based on a pre-trained RGB OCR foundation model, enhanced through multi-modal adaptation. Furthermore, we construct the first large-scale RGB-Event dataset, EvMetro5K, containing 5,599 pairs of synchronized RGB-Event samples, split into 4,479 training and 1,120 testing samples. Extensive experiments on EvMetro5K and other widely used benchmarks demonstrate the effectiveness of our approach for KMR. Both the dataset and source code will be released on https://github.com/Event-AHU/EvMetro5K_benchmark

  </details>



- **RobustVisRAG: Causality-Aware Vision-Based Retrieval-Augmented Generation under Visual Degradations**  
  I-Hsiang Chen, Yu-Wei Liu, Tse-Yu Wu, Yu-Chien Chiang, Jen-Chien Yang, Wei-Ting Chen  
  _2026-02-25_ · https://arxiv.org/abs/2602.22013v1  
  <details><summary>Abstract</summary>

  Vision-based Retrieval-Augmented Generation (VisRAG) leverages vision-language models (VLMs) to jointly retrieve relevant visual documents and generate grounded answers based on multimodal evidence. However, existing VisRAG models degrade in performance when visual inputs suffer from distortions such as blur, noise, low light, or shadow, where semantic and degradation factors become entangled within pretrained visual encoders, leading to errors in both retrieval and generation stages. To address this limitation, we introduce RobustVisRAG, a causality-guided dual-path framework that improves VisRAG robustness while preserving efficiency and zero-shot generalization. RobustVisRAG uses a non-causal path to capture degradation signals through unidirectional attention and a causal path to learn purified semantics guided by these signals. Together with the proposed Non-Causal Distortion Modeling and Causal Semantic Alignment objectives, the framework enforces a clear separation between semantics and degradations, enabling stable retrieval and generation under challenging visual conditions. To evaluate robustness under realistic conditions, we introduce the Distortion-VisRAG dataset, a large-scale benchmark containing both synthetic and real-world degraded documents across seven domains, with 12 synthetic and 5 real distortion types that comprehensively reflect practical visual degradations. Experimental results show that RobustVisRAG improves retrieval, generation, and end-to-end performance by 7.35%, 6.35%, and 12.40%, respectively, on real-world degradations, while maintaining comparable accuracy on clean inputs.

  </details>



- **PanoEnv: Exploring 3D Spatial Intelligence in Panoramic Environments with Reinforcement Learning**  
  Zekai Lin, Xu Zheng  
  _2026-02-25_ · https://arxiv.org/abs/2602.21992v1  
  <details><summary>Abstract</summary>

  360 panoramic images are increasingly used in virtual reality, autonomous driving, and robotics for holistic scene understanding. However, current Vision-Language Models (VLMs) struggle with 3D spatial reasoning on Equirectangular Projection (ERP) images due to geometric distortion and limited 3D supervision. We introduce PanoEnv, a large-scale VQA benchmark built from synthetic 3D environments, containing 14.8K questions across five categories (e.g., relative position, volume comparison) grounded in accurate 3D annotations including depth, segmentation, and bounding boxes. Benchmarking 14 state-of-the-art VLMs reveals limited 3D understanding, achieving only 49.34% overall accuracy and 8.36% on open-ended (OE) questions. To enhance 3D reasoning, we propose a reinforcement learning post-training framework based on Group Relative Policy Optimization (GRPO) with a ground-truth-guided reward that incorporates five geometry-aware strategies such as distance tolerance and spatial consistency. A two-stage curriculum further mitigates catastrophic forgetting: Stage 1 trains on structured tasks (true/false and multiple choice), and Stage 2 fine-tunes on mixed open-ended data to improve generalization. Our 7B model achieves new state-of-the-art performance, improving overall accuracy to 52.93% (+3.59%) and open-ended accuracy to 14.83% while maintaining structured-task performance. It also achieves top semantic evaluation scores (Q-Score 6.24, P-Score 5.95), surpassing 32B models. These results demonstrate that PanoEnv-QA and our curriculum-based RL framework effectively instill 3D spatial intelligence in VLMs for omnidirectional perception.

  </details>



- **PatchDenoiser: Parameter-efficient multi-scale patch learning and fusion denoiser for medical images**  
  Jitindra Fartiyal, Pedro Freire, Sergei K. Turitsyn, Sergei G. Solovski  
  _2026-02-25_ · https://arxiv.org/abs/2602.21987v1  
  <details><summary>Abstract</summary>

  Medical images are essential for diagnosis, treatment planning, and research, but their quality is often degraded by noise from low-dose acquisition, patient motion, or scanner limitations, affecting both clinical interpretation and downstream analysis. Traditional filtering approaches often over-smooth and lose fine anatomical details, while deep learning methods, including CNNs, GANs, and transformers, may struggle to preserve such details or require large, computationally expensive models, limiting clinical practicality. We propose PatchDenoiser, a lightweight, energy-efficient multi-scale patch-based denoising framework. It decomposes denoising into local texture extraction and global context aggregation, fused via a spatially aware patch fusion strategy. This design enables effective noise suppression while preserving fine structural and anatomical details. PatchDenoiser is ultra-lightweight, with far fewer parameters and lower computational complexity than CNN-, GAN-, and transformer-based denoisers. On the 2016 Mayo Low-Dose CT dataset, PatchDenoiser consistently outperforms state-of-the-art CNN- and GAN-based methods in PSNR and SSIM. It is robust to variations in slice thickness, reconstruction kernels, and HU windows, generalizes across scanners without fine-tuning, and reduces parameters by ~9x and energy consumption per inference by ~27x compared with conventional CNN denoisers. PatchDenoiser thus provides a practical, scalable, and computationally efficient solution for medical image denoising, balancing performance, robustness, and clinical deployability.

  </details>



- **Global-Local Dual Perception for MLLMs in High-Resolution Text-Rich Image Translation**  
  Junxin Lu, Tengfei Song, Zhanglin Wu, Pengfei Li, Xiaowei Liang, Hui Yang, Kun Chen, Ning Xie, Yunfei Lu, Jing Zhao, et al.  
  _2026-02-25_ · https://arxiv.org/abs/2602.21956v1  
  <details><summary>Abstract</summary>

  Text Image Machine Translation (TIMT) aims to translate text embedded in images in the source-language into target-language, requiring synergistic integration of visual perception and linguistic understanding. Existing TIMT methods, whether cascaded pipelines or end-to-end multimodal large language models (MLLMs),struggle with high-resolution text-rich images due to cluttered layouts, diverse fonts, and non-textual distractions, resulting in text omission, semantic drift, and contextual inconsistency. To address these challenges, we propose GLoTran, a global-local dual visual perception framework for MLLM-based TIMT. GLoTran integrates a low-resolution global image with multi-scale region-level text image slices under an instruction-guided alignment strategy, conditioning MLLMs to maintain scene-level contextual consistency while faithfully capturing fine-grained textual details. Moreover, to realize this dual-perception paradigm, we construct GLoD, a large-scale text-rich TIMT dataset comprising 510K high-resolution global-local image-text pairs covering diverse real-world scenarios. Extensive experiments demonstrate that GLoTran substantially improves translation completeness and accuracy over state-of-the-art MLLMs, offering a new paradigm for fine-grained TIMT under high-resolution and text-rich conditions.

  </details>



- **The Swarm Intelligence Freeway-Urban Trajectories (SWIFTraj) Dataset - Part II: A Graph-Based Approach for Trajectory Connection**  
  Xinkai Ji, Pan Liu, Yu Han  
  _2026-02-25_ · https://arxiv.org/abs/2602.21954v1  
  <details><summary>Abstract</summary>

  In Part I of this companion paper series, we introduced SWIFTraj, a new open-source vehicle trajectory dataset collected using a unmanned aerial vehicle (UAV) swarm. The dataset has two distinctive features. First, by connecting trajectories across consecutive UAV videos, it provides long-distance continuous trajectories, with the longest exceeding 4.5 km. Second, it covers an integrated traffic network consisting of both freeways and their connected urban roads. Obtaining such long-distance continuous trajectories from a UAV swarm is challenging, due to the need for accurate time alignment across multiple videos and the irregular spatial distribution of UAVs. To address these challenges, this paper proposes a novel graph-based approach for connecting vehicle trajectories captured by a UAV swarm. An undirected graph is constructed to represent flexible UAV layouts, and an automatic time alignment method based on trajectory matching cost minimization is developed to estimate optimal time offsets across videos. To associate trajectories of the same vehicle observed in different videos, a vehicle matching table is established using the Hungarian algorithm. The proposed approach is evaluated using both simulated and real-world data. Results from real-world experiments show that the time alignment error is within three video frames, corresponding to approximately 0.1 s, and that the vehicle matching achieves an F1-score of about 0.99. These results demonstrate the effectiveness of the proposed method in addressing key challenges in UAV-based trajectory connection and highlight its potential for large-scale vehicle trajectory collection.

  </details>



- **MindDriver: Introducing Progressive Multimodal Reasoning for Autonomous Driving**  
  Lingjun Zhang, Yujian Yuan, Changjie Wu, Xinyuan Chang, Xin Cai, Shuang Zeng, Linzhe Shi, Sijin Wang, Hang Zhang, Mu Xu  
  _2026-02-25_ · https://arxiv.org/abs/2602.21952v1  
  <details><summary>Abstract</summary>

  Vision-Language Models (VLM) exhibit strong reasoning capabilities, showing promise for end-to-end autonomous driving systems. Chain-of-Thought (CoT), as VLM's widely used reasoning strategy, is facing critical challenges. Existing textual CoT has a large gap between text semantic space and trajectory physical space. Although the recent approach utilizes future image to replace text as CoT process, it lacks clear planning-oriented objective guidance to generate images with accurate scene evolution. To address these, we innovatively propose MindDriver, a progressive multimodal reasoning framework that enables VLM to imitate human-like progressive thinking for autonomous driving. MindDriver presents semantic understanding, semantic-to-physical space imagination, and physical-space trajectory planning. To achieve aligned reasoning processes in MindDriver, we develop a feedback-guided automatic data annotation pipeline to generate aligned multimodal reasoning training data. Furthermore, we develop a progressive reinforcement fine-tuning method to optimize the alignment through progressive high- level reward-based learning. MindDriver demonstrates superior performance in both nuScences open-loop and Bench2Drive closed-loop evaluation. Codes are available at https://github.com/hotdogcheesewhite/MindDriver.

  </details>



- **Learning to Fuse and Reconstruct Multi-View Graphs for Diabetic Retinopathy Grading**  
  Haoran Li, Yuxin Lin, Huan Wang, Xiaoling Luo, Qi Zhu, Jiahua Shi, Huaming Chen, Bo Du, Johan Barthelemy, Zongyan Xue, et al.  
  _2026-02-25_ · https://arxiv.org/abs/2602.21944v1  
  <details><summary>Abstract</summary>

  Diabetic retinopathy (DR) is one of the leading causes of vision loss worldwide, making early and accurate DR grading critical for timely intervention. Recent clinical practices leverage multi-view fundus images for DR detection with a wide coverage of the field of view (FOV), motivating deep learning methods to explore the potential of multi-view learning for DR grading. However, existing methods often overlook the inter-view correlations when fusing multi-view fundus images, failing to fully exploit the inherent consistency across views originating from the same patient. In this work, we present MVGFDR, an end-to-end Multi-View Graph Fusion framework for DR grading. Different from existing methods that directly fuse visual features from multiple views, MVGFDR is equipped with a novel Multi-View Graph Fusion (MVGF) module to explicitly disentangle the shared and view-specific visual features. Specifically, MVGF comprises three key components: (1) Multi-view Graph Initialization, which constructs visual graphs via residual-guided connections and employs Discrete Cosine Transform (DCT) coefficients as frequency-domain anchors; (2) Multi-view Graph Fusion, which integrates selective nodes across multi-view graphs based on frequency-domain relevance to capture complementary view-specific information; and (3) Masked Cross-view Reconstruction, which leverages masked reconstruction of shared information across views to facilitate view-invariant representation learning. Extensive experimental results on MFIDDR, by far the largest multi-view fundus image dataset, demonstrate the superiority of our proposed approach over existing state-of-the-art approaches in diabetic retinopathy grading.

  </details>



- **Mobile-Ready Automated Triage of Diabetic Retinopathy Using Digital Fundus Images**  
  Aadi Joshi, Manav S. Sharma, Vijay Uttam Rathod, Ashlesha Sawant, Prajakta Musale, Asmita B. Kalamkar  
  _2026-02-25_ · https://arxiv.org/abs/2602.21943v1  
  <details><summary>Abstract</summary>

  Diabetic Retinopathy (DR) is a major cause of vision impairment worldwide. However, manual diagnosis is often time-consuming and prone to errors, leading to delays in screening. This paper presents a lightweight automated deep learning framework for efficient assessment of DR severity from digital fundus images. We use a MobileNetV3 architecture with a Consistent Rank Logits (CORAL) head to model the ordered progression of disease while maintaining computational efficiency for resource-constrained environments. The model is trained and validated on a combined dataset of APTOS 2019 and IDRiD images using a preprocessing pipeline including circular cropping and illumination normalization. Extensive experiments including 3-fold cross-validation and ablation studies demonstrate strong performance. The model achieves a Quadratic Weighted Kappa (QWK) score of 0.9019 and an accuracy of 80.03 percent. Additionally, we address real-world deployment challenges through model calibration to reduce overconfidence and optimization for mobile devices. The proposed system provides a scalable and practical tool for early-stage diabetic retinopathy screening.

  </details>



- **A Framework for Cross-Domain Generalization in Coronary Artery Calcium Scoring Across Gated and Non-Gated Computed Tomography**  
  Mahmut S. Gokmen, Moneera N. Haque, Steve W. Leung, Caroline N. Leach, Seth Parker, Stephen B. Hobbs, Vincent L. Sorrell, W. Brent Seales, V. K. Cody Bumgardner  
  _2026-02-25_ · https://arxiv.org/abs/2602.21935v1  
  <details><summary>Abstract</summary>

  Coronary artery calcium (CAC) scoring is a key predictor of cardiovascular risk, but it relies on ECG-gated CT scans, restricting its use to specialized cardiac imaging settings. We introduce an automated framework for CAC detection and lesion-specific Agatston scoring that operates across both gated and non-gated CT scans. At its core is CARD-ViT, a self-supervised Vision Transformer trained exclusively on gated CT data using DINO. Without any non-gated training data, our framework achieves 0.707 accuracy and a Cohen's kappa of 0.528 on the Stanford non-gated dataset, matching models trained directly on non-gated scans. On gated test sets, the framework achieves 0.910 accuracy with Cohen's kappa scores of 0.871 and 0.874 across independent datasets, demonstrating robust risk stratification. These results demonstrate the feasibility of cross-domain CAC scoring from gated to non-gated domains, supporting scalable cardiovascular screening in routine chest imaging without additional scans or annotations.

  </details>



- **Learning in the Null Space: Small Singular Values for Continual Learning**  
  Cuong Anh Pham, Praneeth Vepakomma, Samuel Horváth  
  _2026-02-25_ · https://arxiv.org/abs/2602.21919v1  
  <details><summary>Abstract</summary>

  Alleviating catastrophic forgetting while enabling further learning is a primary challenge in continual learning (CL). Orthogonal-based training methods have gained attention for their efficiency and strong theoretical properties, and many existing approaches enforce orthogonality through gradient projection. In this paper, we revisit orthogonality and exploit the fact that small singular values correspond to directions that are nearly orthogonal to the input space of previous tasks. Building on this principle, we introduce NESS (Null-space Estimated from Small Singular values), a CL method that applies orthogonality directly in the weight space rather than through gradient manipulation. Specifically, NESS constructs an approximate null space using the smallest singular values of each layer's input representation and parameterizes task-specific updates via a compact low-rank adaptation (LoRA-style) formulation constrained to this subspace. The subspace basis is fixed to preserve the null-space constraint, and only a single trainable matrix is learned for each task. This design ensures that the resulting updates remain approximately in the null space of previous inputs while enabling adaptation to new tasks. Our theoretical analysis and experiments on three benchmark datasets demonstrate competitive performance, low forgetting, and stable accuracy across tasks, highlighting the role of small singular values in continual learning. The code is available at https://github.com/pacman-ctm/NESS.

  </details>



- **Protein Graph Neural Networks for Heterogeneous Cryo-EM Reconstruction**  
  Jonathan Krook, Axel Janson, Joakim andén, Melanie Weber, Ozan Öktem  
  _2026-02-25_ · https://arxiv.org/abs/2602.21915v1  
  <details><summary>Abstract</summary>

  We present a geometry-aware method for heterogeneous single-particle cryogenic electron microscopy (cryo-EM) reconstruction that predicts atomic backbone conformations. To incorporate protein-structure priors, we represent the backbone as a graph and use a graph neural network (GNN) autodecoder that maps per-image latent variables to 3D displacements of a template conformation. The objective combines a data-discrepancy term based on a differentiable cryo-EM forward model with geometric regularization, and it supports unknown orientations via ellipsoidal support lifting (ESL) pose estimation. On synthetic datasets derived from molecular dynamics trajectories, the proposed GNN achieves higher accuracy compared to a multilayer perceptron (MLP) of comparable size, highlighting the benefits of a geometry-informed inductive bias.

  </details>



- **TIRAuxCloud: A Thermal Infrared Dataset for Day and Night Cloud Detection**  
  Alexis Apostolakis, Vasileios Botsos, Niklas Wölki, Andrea Spichtinger, Nikolaos Ioannis Bountos, Ioannis Papoutsis, Panayiotis Tsanakas  
  _2026-02-25_ · https://arxiv.org/abs/2602.21905v1  
  <details><summary>Abstract</summary>

  Clouds are a major obstacle in Earth observation, limiting the usability and reliability of critical remote sensing applications such as fire disaster response, urban heat island monitoring, and snow and ice cover mapping. Therefore, the ability to detect clouds 24/7 is of paramount importance. While visible and near-infrared bands are effective for daytime cloud detection, their dependence on solar illumination makes them unsuitable for nighttime monitoring. In contrast, thermal infrared (TIR) imagery plays a crucial role in detecting clouds at night, when sunlight is absent. Due to their generally lower temperatures, clouds emit distinct thermal signatures that are detectable in TIR bands. Despite this, accurate nighttime cloud detection remains challenging due to limited spectral information and the typically lower spatial resolution of TIR imagery. To address these challenges, we present TIRAuxCloud, a multi-modal dataset centered around thermal spectral data to facilitate cloud segmentation under both daytime and nighttime conditions. The dataset comprises a unique combination of multispectral data (TIR, optical, and near-infrared bands) from Landsat and VIIRS, aligned with auxiliary information layers. Elevation, land cover, meteorological variables, and cloud-free reference images are included to help reduce surface-cloud ambiguity and cloud formation uncertainty. To overcome the scarcity of manual cloud labels, we include a large set of samples with automated cloud masks and a smaller manually annotated subset to further evaluate and improve models. Comprehensive benchmarks are presented to establish performance baselines through supervised and transfer learning, demonstrating the dataset's value in advancing the development of innovative methods for day and night time cloud detection.

  </details>



- **UNet-Based Keypoint Regression for 3D Cone Localization in Autonomous Racing**  
  Mariia Baidachna, James Carty, Aidan Ferguson, Joseph Agrane, Varad Kulkarni, Aubrey Agub, Michael Baxendale, Aaron David, Rachel Horton, Elliott Atkinson  
  _2026-02-25_ · https://arxiv.org/abs/2602.21904v1  
  <details><summary>Abstract</summary>

  Accurate cone localization in 3D space is essential in autonomous racing for precise navigation around the track. Approaches that rely on traditional computer vision algorithms are sensitive to environmental variations, and neural networks are often trained on limited data and are infeasible to run in real time. We present a UNet-based neural network for keypoint detection on cones, leveraging the largest custom-labeled dataset we have assembled. Our approach enables accurate cone position estimation and the potential for color prediction. Our model achieves substantial improvements in keypoint accuracy over conventional methods. Furthermore, we leverage our predicted keypoints in the perception pipeline and evaluate the end-to-end autonomous system. Our results show high-quality performance across all metrics, highlighting the effectiveness of this approach and its potential for adoption in competitive autonomous racing systems.

  </details>



- **How to Take a Memorable Picture? Empowering Users with Actionable Feedback**  
  Francesco Laiti, Davide Talon, Jacopo Staiano, Elisa Ricci  
  _2026-02-25_ · https://arxiv.org/abs/2602.21877v1  
  <details><summary>Abstract</summary>

  Image memorability, i.e., how likely an image is to be remembered, has traditionally been studied in computer vision either as a passive prediction task, with models regressing a scalar score, or with generative methods altering the visual input to boost the image likelihood of being remembered. Yet, none of these paradigms supports users at capture time, when the crucial question is how to improve a photo memorability. We introduce the task of Memorability Feedback (MemFeed), where an automated model should provide actionable, human-interpretable guidance to users with the goal to enhance an image future recall. We also present MemCoach, the first approach designed to provide concrete suggestions in natural language for memorability improvement (e.g., "emphasize facial expression," "bring the subject forward"). Our method, based on Multimodal Large Language Models (MLLMs), is training-free and employs a teacher-student steering strategy, aligning the model internal activations toward more memorable patterns learned from a teacher model progressing along least-to-most memorable samples. To enable systematic evaluation on this novel task, we further introduce MemBench, a new benchmark featuring sequence-aligned photoshoots with annotated memorability scores. Our experiments, considering multiple MLLMs, demonstrate the effectiveness of MemCoach, showing consistently improved performance over several zero-shot models. The results indicate that memorability can not only be predicted but also taught and instructed, shifting the focus from mere prediction to actionable feedback for human creators.

  </details>



- **Understanding Annotation Error Propagation and Learning an Adaptive Policy for Expert Intervention in Barrett's Video Segmentation**  
  Lokesha Rasanjalee, Jin Lin Tan, Dileepa Pitawela, Rajvinder Singh, Hsiang-Ting Chen  
  _2026-02-25_ · https://arxiv.org/abs/2602.21855v1  
  <details><summary>Abstract</summary>

  Accurate annotation of endoscopic videos is essential yet time-consuming, particularly for challenging datasets such as dysplasia in Barrett's esophagus, where the affected regions are irregular and lack clear boundaries. Semi-automatic tools like Segment Anything Model 2 (SAM2) can ease this process by propagating annotations across frames, but small errors often accumulate and reduce accuracy, requiring expert review and correction. To address this, we systematically study how annotation errors propagate across different prompt types, namely masks, boxes, and points, and propose Learning-to-Re-Prompt (L2RP), a cost-aware framework that learns when and where to seek expert input. By tuning a human-cost parameter, our method balances annotation effort and segmentation accuracy. Experiments on a private Barrett's dysplasia dataset and the public SUN-SEG benchmark demonstrate improved temporal consistency and superior performance over baseline strategies.

  </details>



- **UniVBench: Towards Unified Evaluation for Video Foundation Models**  
  Jianhui Wei, Xiaotian Zhang, Yichen Li, Yuan Wang, Yan Zhang, Ziyi Chen, Zhihang Tang, Wei Xu, Zuozhu Liu  
  _2026-02-25_ · https://arxiv.org/abs/2602.21835v1  
  <details><summary>Abstract</summary>

  Video foundation models aim to integrate video understanding, generation, editing, and instruction following within a single framework, making them a central direction for next-generation multimodal systems. However, existing evaluation benchmarks remain fragmented and limited in scope, as they each target a single task, rely on task-specific metrics, and typically use short or simple video clips. As a result, they do not capture the unified capabilities that these models are designed to deliver. To address this gap, we introduce UniVBench, a benchmark purpose-built for evaluating video foundation models across four core abilities: video understanding, video generation, video editing, and a newly proposed task, video reconstruction, which assesses how faithfully a model can reproduce video content it has encountered. Our benchmark substantially expands the complexity of evaluation by incorporating 200 high-quality, diverse and multi-shot videos, each paired with detailed captions, multi-format editing instructions, and reference images. All videos are human-created and carefully validated, offering richer cinematic information than prior benchmarks. In addition, we develop a unified agentic evaluation system (UniV-Eval) that standardizes prompting, instruction parsing, and scoring across all tasks, enabling fair, scalable, and reproducible comparisons of unified video models. By grounding evaluation in instruction-based multi-shot video tasks, UniVBench provides the first framework for measuring the integrated capabilities that video foundation models aim to achieve. Extensive human annotations ensure our evaluation aligns with human judgment, enabling rigorous assessment and accelerating progress toward robust video intelligence.

  </details>



- **StoryMovie: A Dataset for Semantic Alignment of Visual Stories with Movie Scripts and Subtitles**  
  Daniel Oliveira, David Martins de Matos  
  _2026-02-25_ · https://arxiv.org/abs/2602.21829v1  
  <details><summary>Abstract</summary>

  Visual storytelling models that correctly ground entities in images may still hallucinate semantic relationships, generating incorrect dialogue attribution, character interactions, or emotional states. We introduce StoryMovie, a dataset of 1,757 stories aligned with movie scripts and subtitles through LCS matching. Our alignment pipeline synchronizes screenplay dialogue with subtitle timestamps, enabling dialogue attribution by linking character names from scripts to temporal positions from subtitles. Using this aligned content, we generate stories that maintain visual grounding tags while incorporating authentic character names, dialogue, and relationship dynamics. We fine-tune Qwen Storyteller3 on this dataset, building on prior work in visual grounding and entity re-identification. Evaluation using DeepSeek V3 as judge shows that Storyteller3 achieves an 89.9% win rate against base Qwen2.5-VL 7B on subtitle alignment. Compared to Storyteller, trained without script grounding, Storyteller3 achieves 48.5% versus 38.0%, confirming that semantic alignment progressively improves dialogue attribution beyond visual grounding alone.

  </details>



- **Joint Shadow Generation and Relighting via Light-Geometry Interaction Maps**  
  Shan Wang, Peixia Li, Chenchen Xu, Ziang Cheng, Jiayu Yang, Hongdong Li, Pulak Purkait  
  _2026-02-25_ · https://arxiv.org/abs/2602.21820v1  
  <details><summary>Abstract</summary>

  We propose Light-Geometry Interaction (LGI) maps, a novel representation that encodes light-aware occlusion from monocular depth. Unlike ray tracing, which requires full 3D reconstruction, LGI captures essential light-shadow interactions reliably and accurately, computed from off-the-shelf 2.5D depth map predictions. LGI explicitly ties illumination direction to geometry, providing a physics-inspired prior that constrains generative models. Without such prior, these models often produce floating shadows, inconsistent illumination, and implausible shadow geometry. Building on this representation, we propose a unified pipeline for joint shadow generation and relighting - unlike prior methods that treat them as disjoint tasks - capturing the intrinsic coupling of illumination and shadowing essential for modeling indirect effects. By embedding LGI into a bridge-matching generative backbone, we reduce ambiguity and enforce physically consistent light-shadow reasoning. To enable effective training, we curated the first large-scale benchmark dataset for joint shadow and relighting, covering reflections, transparency, and complex interreflections. Experiments show significant gains in realism and consistency across synthetic and real images. LGI thus bridges geometry-inspired rendering with generative modeling, enabling efficient, physically consistent shadow generation and relighting.

  </details>



- **Beyond Static Artifacts: A Forensic Benchmark for Video Deepfake Reasoning in Vision Language Models**  
  Zheyuan Gu, Qingsong Zhao, Yusong Wang, Zhaohong Huang, Xinqi Li, Cheng Yuan, Jiaowei Shao, Chi Zhang, Xuelong Li  
  _2026-02-25_ · https://arxiv.org/abs/2602.21779v1  
  <details><summary>Abstract</summary>

  Current Vision-Language Models (VLMs) for deepfake detection excel at identifying spatial artifacts but overlook a critical dimension: temporal inconsistencies in video forgeries. Adapting VLMs to reason about these dynamic cues remains a distinct challenge. To bridge this gap, we propose Forensic Answer-Questioning (FAQ), a large-scale benchmark that formulates temporal deepfake analysis as a multiple-choice task. FAQ introduces a three-level hierarchy to progressively evaluate and equip VLMs with forensic capabilities: (1) Facial Perception, testing the ability to identify static visual artifacts; (2) Temporal Deepfake Grounding, requiring the localization of dynamic forgery artifacts across frames; and (3) Forensic Reasoning, challenging models to synthesize evidence for final authenticity verdicts. We evaluate a range of VLMs on FAQ and generate a corresponding instruction-tuning set, FAQ-IT. Extensive experiments show that models fine-tuned on FAQ-IT achieve advanced performance on both in-domain and cross-dataset detection benchmarks. Ablation studies further validate the impact of our key design choices, confirming that FAQ is the driving force behind the temporal reasoning capabilities of these VLMs.

  </details>



- **From Statics to Dynamics: Physics-Aware Image Editing with Latent Transition Priors**  
  Liangbing Zhao, Le Zhuo, Sayak Paul, Hongsheng Li, Mohamed Elhoseiny  
  _2026-02-25_ · https://arxiv.org/abs/2602.21778v1  
  <details><summary>Abstract</summary>

  Instruction-based image editing has achieved remarkable success in semantic alignment, yet state-of-the-art models frequently fail to render physically plausible results when editing involves complex causal dynamics, such as refraction or material deformation. We attribute this limitation to the dominant paradigm that treats editing as a discrete mapping between image pairs, which provides only boundary conditions and leaves transition dynamics underspecified. To address this, we reformulate physics-aware editing as predictive physical state transitions and introduce PhysicTran38K, a large-scale video-based dataset comprising 38K transition trajectories across five physical domains, constructed via a two-stage filtering and constraint-aware annotation pipeline. Building on this supervision, we propose PhysicEdit, an end-to-end framework equipped with a textual-visual dual-thinking mechanism. It combines a frozen Qwen2.5-VL for physically grounded reasoning with learnable transition queries that provide timestep-adaptive visual guidance to a diffusion backbone. Experiments show that PhysicEdit improves over Qwen-Image-Edit by 5.9% in physical realism and 10.1% in knowledge-grounded editing, setting a new state-of-the-art for open-source methods, while remaining competitive with leading proprietary models.

  </details>



- **SAPNet++: Evolving Point-Prompted Instance Segmentation with Semantic and Spatial Awareness**  
  Zhaoyang Wei, Xumeng Han, Xuehui Yu, Xue Yang, Guorong Li, Zhenjun Han, Jianbin Jiao  
  _2026-02-25_ · https://arxiv.org/abs/2602.21762v1  
  <details><summary>Abstract</summary>

  Single-point annotation is increasingly prominent in visual tasks for labeling cost reduction. However, it challenges tasks requiring high precision, such as the point-prompted instance segmentation (PPIS) task, which aims to estimate precise masks using single-point prompts to train a segmentation network. Due to the constraints of point annotations, granularity ambiguity and boundary uncertainty arise the difficulty distinguishing between different levels of detail (eg. whole object vs. parts) and the challenge of precisely delineating object boundaries. Previous works have usually inherited the paradigm of mask generation along with proposal selection to achieve PPIS. However, proposal selection relies solely on category information, failing to resolve the ambiguity of different granularity. Furthermore, mask generators offer only finite discrete solutions that often deviate from actual masks, particularly at boundaries. To address these issues, we propose the Semantic-Aware Point-Prompted Instance Segmentation Network (SAPNet). It integrates Point Distance Guidance and Box Mining Strategy to tackle group and local issues caused by the point's granularity ambiguity. Additionally, we incorporate completeness scores within proposals to add spatial granularity awareness, enhancing multiple instance learning (MIL) in proposal selection termed S-MIL. The Multi-level Affinity Refinement conveys pixel and semantic clues, narrowing boundary uncertainty during mask refinement. These modules culminate in SAPNet++, mitigating point prompt's granularity ambiguity and boundary uncertainty and significantly improving segmentation performance. Extensive experiments on four challenging datasets validate the effectiveness of our methods, highlighting the potential to advance PPIS.

  </details>



- **Structure-to-Image: Zero-Shot Depth Estimation in Colonoscopy via High-Fidelity Sim-to-Real Adaptation**  
  Juan Yang, Yuyan Zhang, Han Jia, Bing Hu, Wanzhong Song  
  _2026-02-25_ · https://arxiv.org/abs/2602.21740v1  
  <details><summary>Abstract</summary>

  Monocular depth estimation (MDE) for colonoscopy is hampered by the domain gap between simulated and real-world images. Existing image-to-image translation methods, which use depth as a posterior constraint, often produce structural distortions and specular highlights by failing to balance realism with structure consistency. To address this, we propose a Structure-to-Image paradigm that transforms the depth map from a passive constraint into an active generative foundation. We are the first to introduce phase congruency to colonoscopic domain adaptation and design a cross-level structure constraint to co-optimize geometric structures and fine-grained details like vascular textures. In zero-shot evaluations conducted on a publicly available phantom dataset, the MDE model that was fine-tuned on our generated data achieved a maximum reduction of 44.18% in RMSE compared to competing methods. Our code is available at https://github.com/YyangJJuan/PC-S2I.git.

  </details>



- **Innovative Tooth Segmentation Using Hierarchical Features and Bidirectional Sequence Modeling**  
  Xinxin Zhao, Jian Jiang, Yan Tian, Liqin Wu, Zhaocheng Xu, Teddy Yang, Yunuo Zou, Xun Wang  
  _2026-02-25_ · https://arxiv.org/abs/2602.21712v1  
  <details><summary>Abstract</summary>

  Tooth image segmentation is a cornerstone of dental digitization. However, traditional image encoders relying on fixed-resolution feature maps often lead to discontinuous segmentation and poor discrimination between target regions and background, due to insufficient modeling of environmental and global context. Moreover, transformer-based self-attention introduces substantial computational overhead because of its quadratic complexity (O(n^2)), making it inefficient for high-resolution dental images. To address these challenges, we introduce a three-stage encoder with hierarchical feature representation to capture scale-adaptive information in dental images. By jointly leveraging low-level details and high-level semantics through cross-scale feature fusion, the model effectively preserves fine structural information while maintaining strong contextual awareness. Furthermore, a bidirectional sequence modeling strategy is incorporated to enhance global spatial context understanding without incurring high computational cost. We validate our method on two dental datasets, with experimental results demonstrating its superiority over existing approaches. On the OralVision dataset, our model achieves a 1.1% improvement in mean intersection over union (mIoU).

  </details>



- **SurGo-R1: Benchmarking and Modeling Contextual Reasoning for Operative Zone in Surgical Video**  
  Guanyi Qin, Xiaozhen Wang, Zhu Zhuo, Chang Han Low, Yuancan Xiao, Yibing Fu, Haofeng Liu, Kai Wang, Chunjiang Li, Yueming Jin  
  _2026-02-25_ · https://arxiv.org/abs/2602.21706v1  
  <details><summary>Abstract</summary>

  Minimally invasive surgery has dramatically improved patient operative outcomes, yet identifying safe operative zones remains challenging in critical phases, requiring surgeons to integrate visual cues, procedural phase, and anatomical context under high cognitive load. Existing AI systems offer binary safety verification or static detection, ignoring the phase-dependent nature of intraoperative reasoning. We introduce ResGo, a benchmark of laparoscopic frames annotated with Go Zone bounding boxes and clinician-authored rationales covering phase, exposure quality reasoning, next action and risk reminder. We introduce evaluation metrics that treat correct grounding under incorrect phase as failures, revealing that most vision-language models cannot handle such tasks and perform poorly. We then present SurGo-R1, a model optimized via RLHF with a multi-turn phase-then-go architecture where the model first identifies the surgical phase, then generates reasoning and Go Zone coordinates conditioned on that context. On unseen procedures, SurGo-R1 achieves 76.6% phase accuracy, 32.7 mIoU, and 54.8% hardcore accuracy, a 6.6$\times$ improvement over the mainstream generalist VLMs. Code, model and benchmark will be available at https://github.com/jinlab-imvr/SurGo-R1

  </details>



- **SF3D-RGB: Scene Flow Estimation from Monocular Camera and Sparse LiDAR**  
  Rajai Alhimdiat, Ramy Battrawy, René Schuster, Didier Stricker, Wesam Ashour  
  _2026-02-25_ · https://arxiv.org/abs/2602.21699v1  
  <details><summary>Abstract</summary>

  Scene flow estimation is an extremely important task in computer vision to support the perception of dynamic changes in the scene. For robust scene flow, learning-based approaches have recently achieved impressive results using either image-based or LiDAR-based modalities. However, these methods have tended to focus on the use of a single modality. To tackle these problems, we present a deep learning architecture, SF3D-RGB, that enables sparse scene flow estimation using 2D monocular images and 3D point clouds (e.g., acquired by LiDAR) as inputs. Our architecture is an end-to-end model that first encodes information from each modality into features and fuses them together. Then, the fused features enhance a graph matching module for better and more robust mapping matrix computation to generate an initial scene flow. Finally, a residual scene flow module further refines the initial scene flow. Our model is designed to strike a balance between accuracy and efficiency. Furthermore, experiments show that our proposed method outperforms single-modality methods and achieves better scene flow accuracy on real-world datasets while using fewer parameters compared to other state-of-the-art methods with fusion.

  </details>



- **E-comIQ-ZH: A Human-Aligned Dataset and Benchmark for Fine-Grained Evaluation of E-commerce Posters with Chain-of-Thought**  
  Meiqi Sun, Mingyu Li, Junxiong Zhu  
  _2026-02-25_ · https://arxiv.org/abs/2602.21698v1  
  <details><summary>Abstract</summary>

  Generative AI is widely used to create commercial posters. However, rapid advances in generation have outpaced automated quality assessment. Existing models emphasize generic esthetics or low level distortions and lack the functional criteria required for e-commerce design. It is especially challenging for Chinese content, where complex characters often produce subtle but critical textual artifacts that are overlooked by existing methods. To address this, we introduce E-comIQ-ZH, a framework for evaluating Chinese e-commerce posters. We build the first dataset E-comIQ-18k to feature multi dimensional scores and expert calibrated Chain of Thought (CoT) rationales. Using this dataset, we train E-comIQ-M, a specialized evaluation model that aligns with human expert judgment. Our framework enables E-comIQ-Bench, the first automated and scalable benchmark for the generation of Chinese e-commerce posters. Extensive experiments show our E-comIQ-M aligns more closely with expert standards and enables scalable automated assessment of e-commerce posters. All datasets, models, and evaluation tools will be released to support future research in this area.Code will be available at https://github.com/4mm7/E-comIQ-ZH.

  </details>



- **SunnyParking: Multi-Shot Trajectory Generation and Motion State Awareness for Human-like Parking**  
  Jishu Miao, Han Chen, Jiankun Zhai, Qi Liu, Tsubasa Hirakawa, Takayoshi Yamashita, Hironobu Fujiyoshi  
  _2026-02-25_ · https://arxiv.org/abs/2602.21682v1  
  <details><summary>Abstract</summary>

  Autonomous parking fundamentally differs from on-road driving due to its frequent direction changes and complex maneuvering requirements. However, existing End-to-End (E2E) planning methods often simplify the parking task into a geometric path regression problem, neglecting explicit modeling of the vehicle's kinematic state. This "dimensionality deficiency" easily leads to physically infeasible trajectories and deviates from real human driving behavior, particularly at critical gear-shift points in multi-shot parking scenarios. In this paper, we propose SunnyParking, a novel dual-branch E2E architecture that achieves motion state awareness by jointly predicting spatial trajectories and discrete motion state sequences (e.g., forward/reverse). Additionally, we introduce a Fourier feature-based representation of target parking slots to overcome the resolution limitations of traditional bird's-eye view (BEV) approaches, enabling high-precision target interactions. Experimental results demonstrate that our framework generates more robust and human-like trajectories in complex multi-shot parking scenarios, while significantly improving gear-shift point localization accuracy compared to state-of-the-art methods. We open-source a new parking dataset of the CARLA simulator, specifically designed to evaluate full prediction capabilities under complex maneuvers.

  </details>



- **Hierarchical LLM-Based Multi-Agent Framework with Prompt Optimization for Multi-Robot Task Planning**  
  Tomoya Kawabe, Rin Takano  
  _2026-02-25_ · https://arxiv.org/abs/2602.21670v1  
  <details><summary>Abstract</summary>

  Multi-robot task planning requires decomposing natural-language instructions into executable actions for heterogeneous robot teams. Conventional Planning Domain Definition Language (PDDL) planners provide rigorous guarantees but struggle to handle ambiguous or long-horizon missions, while large language models (LLMs) can interpret instructions and propose plans but may hallucinate or produce infeasible actions. We present a hierarchical multi-agent LLM-based planner with prompt optimization: an upper layer decomposes tasks and assigns them to lower-layer agents, which generate PDDL problems solved by a classical planner. When plans fail, the system applies TextGrad-inspired textual-gradient updates to optimize each agent's prompt and thereby improve planning accuracy. In addition, meta-prompts are learned and shared across agents within the same layer, enabling efficient prompt optimization in multi-agent settings. On the MAT-THOR benchmark, our planner achieves success rates of 0.95 on compound tasks, 0.84 on complex tasks, and 0.60 on vague tasks, improving over the previous state-of-the-art LaMMA-P by 2, 7, and 15 percentage points respectively. An ablation study shows that the hierarchical structure, prompt optimization, and meta-prompt sharing contribute roughly +59, +37, and +4 percentage points to the overall success rate.

  </details>



- **Space-Time Forecasting of Dynamic Scenes with Motion-aware Gaussian Grouping**  
  Junmyeong Lee, Hoseung Choi, Minsu Cho  
  _2026-02-25_ · https://arxiv.org/abs/2602.21668v1  
  <details><summary>Abstract</summary>

  Forecasting dynamic scenes remains a fundamental challenge in computer vision, as limited observations make it difficult to capture coherent object-level motion and long-term temporal evolution. We present Motion Group-aware Gaussian Forecasting (MoGaF), a framework for long-term scene extrapolation built upon the 4D Gaussian Splatting representation. MoGaF introduces motion-aware Gaussian grouping and group-wise optimization to enforce physically consistent motion across both rigid and non-rigid regions, yielding spatially coherent dynamic representations. Leveraging this structured space-time representation, a lightweight forecasting module predicts future motion, enabling realistic and temporally stable scene evolution. Experiments on synthetic and real-world datasets demonstrate that MoGaF consistently outperforms existing baselines in rendering quality, motion plausibility, and long-term forecasting stability. Our project page is available at https://slime0519.github.io/mogaf

  </details>



- **Send Less, Perceive More: Masked Quantized Point Cloud Communication for Loss-Tolerant Collaborative Perception**  
  Sheng Xu, Enshu Wang, Hongfei Xue, Jian Teng, Bingyi Liu, Yi Zhu, Pu Wang, Libing Wu, Chunming Qiao  
  _2026-02-25_ · https://arxiv.org/abs/2602.21667v1  
  <details><summary>Abstract</summary>

  Collaborative perception allows connected vehicles to overcome occlusions and limited viewpoints by sharing sensory information. However, existing approaches struggle to achieve high accuracy under strict bandwidth constraints and remain highly vulnerable to random transmission packet loss. We introduce QPoint2Comm, a quantized point-cloud communication framework that dramatically reduces bandwidth while preserving high-fidelity 3D information. Instead of transmitting intermediate features, QPoint2Comm directly communicates quantized point-cloud indices using a shared codebook, enabling efficient reconstruction with lower bandwidth than feature-based methods. To ensure robustness to possible communication packet loss, we employ a masked training strategy that simulates random packet loss, allowing the model to maintain strong performance even under severe transmission failures. In addition, a cascade attention fusion module is proposed to enhance multi-vehicle information integration. Extensive experiments on both simulated and real-world datasets demonstrate that QPoint2Comm sets a new state of the art in accuracy, communication efficiency, and resilience to packet loss.

  </details>



- **Biomechanical Comparisons Reveal Divergence of Human and Humanoid Gaits**  
  Luying Feng, Yaochu Jin, Hanze Hu, Wei Chen  
  _2026-02-25_ · https://arxiv.org/abs/2602.21666v1  
  <details><summary>Abstract</summary>

  It remains challenging to achieve human-like locomotion in legged robots due to fundamental discrepancies between biological and mechanical structures. Although imitation learning has emerged as a promising approach for generating natural robotic movements, simply replicating joint angle trajectories fails to capture the underlying principles of human motion. This study proposes a Gait Divergence Analysis Framework (GDAF), a unified biomechanical evaluation framework that systematically quantifies kinematic and kinetic discrepancies between humans and bipedal robots. We apply GDAF to systematically compare human and humanoid locomotion across 28 walking speeds. To enable reproducible analysis, we collect and release a speed-continuous humanoid locomotion dataset from a state-of-the-art humanoid controller. We further provide an open-source implementation of GDAF, including analysis, visualization, and MuJoCo-based tools, enabling quantitative, interpretable, and reproducible biomechanical analysis of humanoid locomotion. Results demonstrate that despite visually human-like motion generated by modern humanoid controllers, significant biomechanical divergence persists across speeds. Robots exhibit systematic deviations in gait symmetry, energy distribution, and joint coordination, indicating that substantial room remains for improving the biomechanical fidelity and energetic efficiency of humanoid locomotion. This work provides a quantitative benchmark for evaluating humanoid locomotion and offers data and versatile tools to support the development of more human-like and energetically efficient locomotion controllers. The data and code will be made publicly available upon acceptance of the paper.

  </details>



- **Following the Diagnostic Trace: Visual Cognition-guided Cooperative Network for Chest X-Ray Diagnosis**  
  Shaoxuan Wu, Jingkun Chen, Chong Ma, Cong Shen, Xiao Zhang, Jun Feng  
  _2026-02-25_ · https://arxiv.org/abs/2602.21657v1  
  <details><summary>Abstract</summary>

  Computer-aided diagnosis (CAD) has significantly advanced automated chest X-ray diagnosis but remains isolated from clinical workflows and lacks reliable decision support and interpretability. Human-AI collaboration seeks to enhance the reliability of diagnostic models by integrating the behaviors of controllable radiologists. However, the absence of interactive tools seamlessly embedded within diagnostic routines impedes collaboration, while the semantic gap between radiologists' decision-making patterns and model representations further limits clinical adoption. To overcome these limitations, we propose a visual cognition-guided collaborative network (VCC-Net) to achieve the cooperative diagnostic paradigm. VCC-Net centers on visual cognition (VC) and employs clinically compatible interfaces, such as eye-tracking or the mouse, to capture radiologists' visual search traces and attention patterns during diagnosis. VCC-Net employs VC as a spatial cognition guide, learning hierarchical visual search strategies to localize diagnostically key regions. A cognition-graph co-editing module subsequently integrates radiologist VC with model inference to construct a disease-aware graph. The module captures dependencies among anatomical regions and aligns model representations with VC-driven features, mitigating radiologist bias and facilitating complementary, transparent decision-making. Experiments on the public datasets SIIM-ACR, EGD-CXR, and self-constructed TB-Mouse dataset achieved classification accuracies of 88.40%, 85.05%, and 92.41%, respectively. The attention maps produced by VCC-Net exhibit strong concordance with radiologists' gaze distributions, demonstrating a mutual reinforcement of radiologist and model inference. The code is available at https://github.com/IPMI-NWU/VCC-Net.

  </details>



- **CCCaption: Dual-Reward Reinforcement Learning for Complete and Correct Image Captioning**  
  Zhijiang Tang, Linhua Wang, Jiaxin Qi, Weihao Jiang, Peng Hou, Anxiang Zeng, Jianqiang Huang  
  _2026-02-25_ · https://arxiv.org/abs/2602.21655v1  
  <details><summary>Abstract</summary>

  Image captioning remains a fundamental task for vision language understanding, yet ground-truth supervision still relies predominantly on human-annotated references. Because human annotations reflect subjective preferences and expertise, ground-truth captions are often incomplete or even incorrect, which in turn limits caption models. We argue that caption quality should be assessed by two objective aspects: completeness (does the caption cover all salient visual facts?) and correctness (are the descriptions true with respect to the image?). To this end, we introduce CCCaption: a dual-reward reinforcement learning framework with a dedicated fine-tuning corpus that explicitly optimizes these properties to generate \textbf{C}omplete and \textbf{C}orrect \textbf{Captions}. For completeness, we use diverse LVLMs to disentangle the image into a set of visual queries, and reward captions that answer more of these queries, with a dynamic query sampling strategy to improve training efficiency. For correctness, we penalize captions that contain hallucinations by validating the authenticity of sub-caption queries, which are derived from the caption decomposition. Our symmetric dual-reward optimization jointly maximizes completeness and correctness, guiding models toward captions that better satisfy these objective criteria. Extensive experiments across standard captioning benchmarks show consistent improvements, offering a principled path to training caption models beyond human-annotation imitation.

  </details>



- **Lie Flow: Video Dynamic Fields Modeling and Predicting with Lie Algebra as Geometric Physics Principle**  
  Weidong Qiao, Wangmeng Zuo, Hui Li  
  _2026-02-25_ · https://arxiv.org/abs/2602.21645v1  
  <details><summary>Abstract</summary>

  Modeling 4D scenes requires capturing both spatial structure and temporal motion, which is challenging due to the need for physically consistent representations of complex rigid and non-rigid motions. Existing approaches mainly rely on translational displacements, which struggle to represent rotations, articulated transformations, often leading to spatial inconsistency and physically implausible motion. LieFlow, a dynamic radiance representation framework that explicitly models motion within the SE(3) Lie group, enabling coherent learning of translation and rotation in a unified geometric space. The SE(3) transformation field enforces physically inspired constraints to maintain motion continuity and geometric consistency. The evaluation includes a synthetic dataset with rigid-body trajectories and two real-world datasets capturing complex motion under natural lighting and occlusions. Across all datasets, LieFlow consistently improves view-synthesis fidelity, temporal coherence, and physical realism over NeRF-based baselines. These results confirm that SE(3)-based motion modeling offers a robust and physically grounded framework for representing dynamic 4D scenes.

  </details>



- **Axial-Centric Cross-Plane Attention for 3D Medical Image Classification**  
  Doyoung Park, Jinsoo Kim, Lohendran Baskaran  
  _2026-02-25_ · https://arxiv.org/abs/2602.21636v1  
  <details><summary>Abstract</summary>

  Clinicians commonly interpret three-dimensional (3D) medical images, such as computed tomography (CT) scans, using multiple anatomical planes rather than as a single volumetric representation. In this multi-planar approach, the axial plane typically serves as the primary acquisition and diagnostic reference, while the coronal and sagittal planes provide complementary spatial information to increase diagnostic confidence. However, many existing 3D deep learning methods either process volumetric data holistically or assign equal importance to all planes, failing to reflect the axial-centric clinical interpretation workflow. To address this gap, we propose an axial-centric cross-plane attention architecture for 3D medical image classification that captures the inherent asymmetric dependencies between different anatomical planes. Our architecture incorporates MedDINOv3, a medical vision foundation model pretrained via self-supervised learning on large-scale axial CT images, as a frozen feature extractor for the axial, coronal, and sagittal planes. RICA blocks and intra-plane transformer encoders capture plane-specific positional and contextual information within each anatomical plane, while axial-centric cross-plane transformer encoders condition axial features on complementary information from auxiliary planes. Experimental results on six datasets from the MedMNIST3D benchmark demonstrate that the proposed architecture consistently outperforms existing 3D and multi-plane models in terms of accuracy and AUC. Ablation studies further confirm the importance of axial-centric query-key-value allocation and directional cross-plane fusion. These results highlight the importance of aligning architectural design with clinical interpretation workflows for robust and data-efficient 3D medical image analysis.

  </details>



- **Virtual Biopsy for Intracranial Tumors Diagnosis on MRI**  
  Xinzhe Luo, Shuai Shao, Yan Wang, Jiangtao Wang, Yutong Bai, Jianguo Zhang  
  _2026-02-25_ · https://arxiv.org/abs/2602.21613v1  
  <details><summary>Abstract</summary>

  Deep intracranial tumors situated in eloquent brain regions controlling vital functions present critical diagnostic challenges. Clinical practice has shifted toward stereotactic biopsy for pathological confirmation before treatment. Yet biopsy carries inherent risks of hemorrhage and neurological deficits and struggles with sampling bias due to tumor spatial heterogeneity, because pathological changes are typically region-selective rather than tumor-wide. Therefore, advancing non-invasive MRI-based pathology prediction is essential for holistic tumor assessment and modern clinical decision-making. The primary challenge lies in data scarcity: low tumor incidence requires long collection cycles, and annotation demands biopsy-verified pathology from neurosurgical experts. Additionally, tiny lesion volumes lacking segmentation masks cause critical features to be overwhelmed by background noise. To address these challenges, we construct the ICT-MRI dataset - the first public biopsy-verified benchmark with 249 cases across four categories. We propose a Virtual Biopsy framework comprising: MRI-Processor for standardization; Tumor-Localizer employing vision-language models for coarse-to-fine localization via weak supervision; and Adaptive-Diagnoser with a Masked Channel Attention mechanism fusing local discriminative features with global contexts. Experiments demonstrate over 90% accuracy, outperforming baselines by more than 20%.

  </details>



- **Iterative Closed-Loop Motion Synthesis for Scaling the Capabilities of Humanoid Control**  
  Weisheng Xu, Qiwei Wu, Jiaxi Zhang, Tan Jing, Yangfan Li, Yuetong Fang, Jiaqi Xiong, Kai Wu, Rong Ou, Renjing Xu  
  _2026-02-25_ · https://arxiv.org/abs/2602.21599v1  
  <details><summary>Abstract</summary>

  Physics-based humanoid control relies on training with motion datasets that have diverse data distributions. However, the fixed difficulty distribution of datasets limits the performance ceiling of the trained control policies. Additionally, the method of acquiring high-quality data through professional motion capture systems is constrained by costs, making it difficult to achieve large-scale scalability. To address these issues, we propose a closed-loop automated motion data generation and iterative framework. It can generate high-quality motion data with rich action semantics, including martial arts, dance, combat, sports, gymnastics, and more. Furthermore, our framework enables difficulty iteration of policies and data through physical metrics and objective evaluations, allowing the trained tracker to break through its original difficulty limits. On the PHC single-primitive tracker, using only approximately 1/10 of the AMASS dataset size, the average failure rate on the test set (2201 clips) is reduced by 45\% compared to the baseline. Finally, we conduct comprehensive ablation and comparative experiments to highlight the rationality and advantages of our framework.

  </details>



- **SPOC: Safety-Aware Planning Under Partial Observability And Physical Constraints**  
  Hyungmin Kim, Hobeom Jeon, Dohyung Kim, Minsu Jang, Jeahong Kim  
  _2026-02-25_ · https://arxiv.org/abs/2602.21595v1  
  <details><summary>Abstract</summary>

  Embodied Task Planning with large language models faces safety challenges in real-world environments, where partial observability and physical constraints must be respected. Existing benchmarks often overlook these critical factors, limiting their ability to evaluate both feasibility and safety. We introduce SPOC, a benchmark for safety-aware embodied task planning, which integrates strict partial observability, physical constraints, step-by-step planning, and goal-condition-based evaluation. Covering diverse household hazards such as fire, fluid, injury, object damage, and pollution, SPOC enables rigorous assessment through both state and constraint-based online metrics. Experiments with state-of-the-art LLMs reveal that current models struggle to ensure safety-aware planning, particularly under implicit constraints. Code and dataset are available at https://github.com/khm159/SPOC

  </details>



- **MultiAnimate: Pose-Guided Image Animation Made Extensible**  
  Yingcheng Hu, Haowen Gong, Chuanguang Yang, Zhulin An, Yongjun Xu, Songhua Liu  
  _2026-02-25_ · https://arxiv.org/abs/2602.21581v1  
  <details><summary>Abstract</summary>

  Pose-guided human image animation aims to synthesize realistic videos of a reference character driven by a sequence of poses. While diffusion-based methods have achieved remarkable success, most existing approaches are limited to single-character animation. We observe that naively extending these methods to multi-character scenarios often leads to identity confusion and implausible occlusions between characters. To address these challenges, in this paper, we propose an extensible multi-character image animation framework built upon modern Diffusion Transformers (DiTs) for video generation. At its core, our framework introduces two novel components-Identifier Assigner and Identifier Adapter - which collaboratively capture per-person positional cues and inter-person spatial relationships. This mask-driven scheme, along with a scalable training strategy, not only enhances flexibility but also enables generalization to scenarios with more characters than those seen during training. Remarkably, trained on only a two-character dataset, our model generalizes to multi-character animation while maintaining compatibility with single-character cases. Extensive experiments demonstrate that our approach achieves state-of-the-art performance in multi-character image animation, surpassing existing diffusion-based baselines.

  </details>



- **VasGuideNet: Vascular Topology-Guided Couinaud Liver Segmentation with Structural Contrastive Loss**  
  Chaojie Shen, Jingjun Gu, Zihao Zhao, Ruocheng Li, Cunyuan Yang, Jiajun Bu, Lei Wu  
  _2026-02-25_ · https://arxiv.org/abs/2602.21539v1  
  <details><summary>Abstract</summary>

  Accurate Couinaud liver segmentation is critical for preoperative surgical planning and tumor localization.However, existing methods primarily rely on image intensity and spatial location cues, without explicitly modeling vascular topology. As a result, they often produce indistinct boundaries near vessels and show limited generalization under anatomical variability.We propose VasGuideNet, the first Couinaud segmentation framework explicitly guided by vascular topology. Specifically, skeletonized vessels, Euclidean distance transform (EDT)--derived geometry, and k-nearest neighbor (kNN) connectivity are encoded into topology features using Graph Convolutional Networks (GCNs). These features are then injected into a 3D encoder--decoder backbone via a cross-attention fusion module. To further improve inter-class separability and anatomical consistency, we introduce a Structural Contrastive Loss (SCL) with a global memory bank.On Task08_HepaticVessel and our private LASSD dataset, VasGuideNet achieves Dice scores of 83.68% and 76.65% with RVDs of 1.68 and 7.08, respectively. It consistently outperforms representative baselines including UNETR, Swin UNETR, and G-UNETR++, delivering higher Dice/mIoU and lower RVD across datasets, demonstrating its effectiveness for anatomically consistent segmentation. Code is available at https://github.com/Qacket/VasGuideNet.git.

  </details>



- **LiLo-VLA: Compositional Long-Horizon Manipulation via Linked Object-Centric Policies**  
  Yue Yang, Shuo Cheng, Yu Fang, Homanga Bharadhwaj, Mingyu Ding, Gedas Bertasius, Daniel Szafir  
  _2026-02-25_ · https://arxiv.org/abs/2602.21531v1  
  <details><summary>Abstract</summary>

  General-purpose robots must master long-horizon manipulation, defined as tasks involving multiple kinematic structure changes (e.g., attaching or detaching objects) in unstructured environments. While Vision-Language-Action (VLA) models offer the potential to master diverse atomic skills, they struggle with the combinatorial complexity of sequencing them and are prone to cascading failures due to environmental sensitivity. To address these challenges, we propose LiLo-VLA (Linked Local VLA), a modular framework capable of zero-shot generalization to novel long-horizon tasks without ever being trained on them. Our approach decouples transport from interaction: a Reaching Module handles global motion, while an Interaction Module employs an object-centric VLA to process isolated objects of interest, ensuring robustness against irrelevant visual features and invariance to spatial configurations. Crucially, this modularity facilitates robust failure recovery through dynamic replanning and skill reuse, effectively mitigating the cascading errors common in end-to-end approaches. We introduce a 21-task simulation benchmark consisting of two challenging suites: LIBERO-Long++ and Ultra-Long. In these simulations, LiLo-VLA achieves a 69% average success rate, outperforming Pi0.5 by 41% and OpenVLA-OFT by 67%. Furthermore, real-world evaluations across 8 long-horizon tasks demonstrate an average success rate of 85%. Project page: https://yy-gx.github.io/LiLo-VLA/.

  </details>



- **AHAN: Asymmetric Hierarchical Attention Network for Identical Twin Face Verification**  
  Hoang-Nhat Nguyen  
  _2026-02-25_ · https://arxiv.org/abs/2602.21503v1  
  <details><summary>Abstract</summary>

  Identical twin face verification represents an extreme fine-grained recognition challenge where even state-of-the-art systems fail due to overwhelming genetic similarity. Current face recognition methods achieve over 99.8% accuracy on standard benchmarks but drop dramatically to 88.9% when distinguishing identical twins, exposing critical vulnerabilities in biometric security systems. The difficulty lies in learning features that capture subtle, non-genetic variations that uniquely identify individuals. We propose the Asymmetric Hierarchical Attention Network (AHAN), a novel architecture specifically designed for this challenge through multi-granularity facial analysis. AHAN introduces a Hierarchical Cross-Attention (HCA) module that performs multi-scale analysis on semantic facial regions, enabling specialized processing at optimal resolutions. We further propose a Facial Asymmetry Attention Module (FAAM) that learns unique biometric signatures by computing cross-attention between left and right facial halves, capturing subtle asymmetric patterns that differ even between twins. To ensure the network learns truly individuating features, we introduce Twin-Aware Pair-Wise Cross-Attention (TA-PWCA), a training-only regularization strategy that uses each subject's own twin as the hardest possible distractor. Extensive experiments on the ND_TWIN dataset demonstrate that AHAN achieves 92.3% twin verification accuracy, representing a 3.4% improvement over state-of-the-art methods.

  </details>



- **Unified Unsupervised and Sparsely-Supervised 3D Object Detection by Semantic Pseudo-Labeling and Prototype Learning**  
  Yushen He  
  _2026-02-25_ · https://arxiv.org/abs/2602.21484v1  
  <details><summary>Abstract</summary>

  3D object detection is essential for autonomous driving and robotic perception, yet its reliance on large-scale manually annotated data limits scalability and adaptability. To reduce annotation dependency, unsupervised and sparsely-supervised paradigms have emerged. However, they face intertwined challenges: low-quality pseudo-labels, unstable feature mining, and a lack of a unified training framework. This paper proposes SPL, a unified training framework for both Unsupervised and Sparsely-Supervised 3D Object Detection via Semantic Pseudo-labeling and prototype Learning. SPL first generates high-quality pseudo-labels by integrating image semantics, point cloud geometry, and temporal cues, producing both 3D bounding boxes for dense objects and 3D point labels for sparse ones. These pseudo-labels are not used directly but as probabilistic priors within a novel, multi-stage prototype learning strategy. This strategy stabilizes feature representation learning through memory-based initialization and momentum-based prototype updating, effectively mining features from both labeled and unlabeled data. Extensive experiments on KITTI and nuScenes datasets demonstrate that SPL significantly outperforms state-of-the-art methods in both settings. Our work provides a robust and generalizable solution for learning 3D object detectors with minimal or no manual annotations.

  </details>



- **Perceptual Quality Optimization of Image Super-Resolution**  
  Wei Zhou, Yixiao Li, Hadi Amirpour, Xiaoshuai Hao, Jiang Liu, Peng Wang, Hantao Liu  
  _2026-02-25_ · https://arxiv.org/abs/2602.21482v1  
  <details><summary>Abstract</summary>

  Single-image super-resolution (SR) has achieved remarkable progress with deep learning, yet most approaches rely on distortion-oriented losses or heuristic perceptual priors, which often lead to a trade-off between fidelity and visual quality. To address this issue, we propose an \textit{Efficient Perceptual Bi-directional Attention Network (Efficient-PBAN)} that explicitly optimizes SR towards human-preferred quality. Unlike patch-based quality models, Efficient-PBAN avoids extensive patch sampling and enables efficient image-level perception. The proposed framework is trained on our self-constructed SR quality dataset that covers a wide range of state-of-the-art SR methods with corresponding human opinion scores. Using this dataset, Efficient-PBAN learns to predict perceptual quality in a way that correlates strongly with subjective judgments. The learned metric is further integrated into SR training as a differentiable perceptual loss, enabling closed-loop alignment between reconstruction and perceptual assessment. Extensive experiments demonstrate that our approach delivers superior perceptual quality. Code is publicly available at https://github.com/Lighting-YXLI/Efficient-PBAN.

  </details>


