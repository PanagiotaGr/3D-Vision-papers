# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-02-08 07:05 UTC_

Total papers shown: **37**


---

- **SwimBird: Eliciting Switchable Reasoning Mode in Hybrid Autoregressive MLLMs**  
  Jintao Tong, Shilin Yan, Hongwei Xue, Xiaojun Tang, Kunyu Shi, Guannan Zhang, Ruixuan Li, Yixiong Zou  
  _2026-02-05_ · https://arxiv.org/abs/2602.06040v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have made remarkable progress in multimodal perception and reasoning by bridging vision and language. However, most existing MLLMs perform reasoning primarily with textual CoT, which limits their effectiveness on vision-intensive tasks. Recent approaches inject a fixed number of continuous hidden states as "visual thoughts" into the reasoning process and improve visual performance, but often at the cost of degraded text-based logical reasoning. We argue that the core limitation lies in a rigid, pre-defined reasoning pattern that cannot adaptively choose the most suitable thinking modality for different user queries. We introduce SwimBird, a reasoning-switchable MLLM that dynamically switches among three reasoning modes conditioned on the input: (1) text-only reasoning, (2) vision-only reasoning (continuous hidden states as visual thoughts), and (3) interleaved vision-text reasoning. To enable this capability, we adopt a hybrid autoregressive formulation that unifies next-token prediction for textual thoughts with next-embedding prediction for visual thoughts, and design a systematic reasoning-mode curation strategy to construct SwimBird-SFT-92K, a diverse supervised fine-tuning dataset covering all three reasoning patterns. By enabling flexible, query-adaptive mode selection, SwimBird preserves strong textual logic while substantially improving performance on vision-dense tasks. Experiments across diverse benchmarks covering textual reasoning and challenging visual understanding demonstrate that SwimBird achieves state-of-the-art results and robust gains over prior fixed-pattern multimodal reasoning methods.

  </details>



- **CommCP: Efficient Multi-Agent Coordination via LLM-Based Communication with Conformal Prediction**  
  Xiaopan Zhang, Zejin Wang, Zhixu Li, Jianpeng Yao, Jiachen Li  
  _2026-02-05_ · https://arxiv.org/abs/2602.06038v1  
  <details><summary>Abstract</summary>

  To complete assignments provided by humans in natural language, robots must interpret commands, generate and answer relevant questions for scene understanding, and manipulate target objects. Real-world deployments often require multiple heterogeneous robots with different manipulation capabilities to handle different assignments cooperatively. Beyond the need for specialized manipulation skills, effective information gathering is important in completing these assignments. To address this component of the problem, we formalize the information-gathering process in a fully cooperative setting as an underexplored multi-agent multi-task Embodied Question Answering (MM-EQA) problem, which is a novel extension of canonical Embodied Question Answering (EQA), where effective communication is crucial for coordinating efforts without redundancy. To address this problem, we propose CommCP, a novel LLM-based decentralized communication framework designed for MM-EQA. Our framework employs conformal prediction to calibrate the generated messages, thereby minimizing receiver distractions and enhancing communication reliability. To evaluate our framework, we introduce an MM-EQA benchmark featuring diverse, photo-realistic household scenarios with embodied questions. Experimental results demonstrate that CommCP significantly enhances the task success rate and exploration efficiency over baselines. The experiment videos, code, and dataset are available on our project website: https://comm-cp.github.io.

  </details>



- **GenArena: How Can We Achieve Human-Aligned Evaluation for Visual Generation Tasks?**  
  Ruihang Li, Leigang Qu, Jingxu Zhang, Dongnan Gui, Mengde Xu, Xiaosong Zhang, Han Hu, Wenjie Wang, Jiaqi Wang  
  _2026-02-05_ · https://arxiv.org/abs/2602.06013v1  
  <details><summary>Abstract</summary>

  The rapid advancement of visual generation models has outpaced traditional evaluation approaches, necessitating the adoption of Vision-Language Models as surrogate judges. In this work, we systematically investigate the reliability of the prevailing absolute pointwise scoring standard, across a wide spectrum of visual generation tasks. Our analysis reveals that this paradigm is limited due to stochastic inconsistency and poor alignment with human perception. To resolve these limitations, we introduce GenArena, a unified evaluation framework that leverages a pairwise comparison paradigm to ensure stable and human-aligned evaluation. Crucially, our experiments uncover a transformative finding that simply adopting this pairwise protocol enables off-the-shelf open-source models to outperform top-tier proprietary models. Notably, our method boosts evaluation accuracy by over 20% and achieves a Spearman correlation of 0.86 with the authoritative LMArena leaderboard, drastically surpassing the 0.36 correlation of pointwise methods. Based on GenArena, we benchmark state-of-the-art visual generation models across diverse tasks, providing the community with a rigorous and automated evaluation standard for visual generation.

  </details>



- **RISE-Video: Can Video Generators Decode Implicit World Rules?**  
  Mingxin Liu, Shuran Ma, Shibei Meng, Xiangyu Zhao, Zicheng Zhang, Shaofeng Zhang, Zhihang Zhong, Peixian Chen, Haoyu Cao, Xing Sun, et al.  
  _2026-02-05_ · https://arxiv.org/abs/2602.05986v1  
  <details><summary>Abstract</summary>

  While generative video models have achieved remarkable visual fidelity, their capacity to internalize and reason over implicit world rules remains a critical yet under-explored frontier. To bridge this gap, we present RISE-Video, a pioneering reasoning-oriented benchmark for Text-Image-to-Video (TI2V) synthesis that shifts the evaluative focus from surface-level aesthetics to deep cognitive reasoning. RISE-Video comprises 467 meticulously human-annotated samples spanning eight rigorous categories, providing a structured testbed for probing model intelligence across diverse dimensions, ranging from commonsense and spatial dynamics to specialized subject domains. Our framework introduces a multi-dimensional evaluation protocol consisting of four metrics: \textit{Reasoning Alignment}, \textit{Temporal Consistency}, \textit{Physical Rationality}, and \textit{Visual Quality}. To further support scalable evaluation, we propose an automated pipeline leveraging Large Multimodal Models (LMMs) to emulate human-centric assessment. Extensive experiments on 11 state-of-the-art TI2V models reveal pervasive deficiencies in simulating complex scenarios under implicit constraints, offering critical insights for the advancement of future world-simulating generative models.

  </details>



- **Contour Refinement using Discrete Diffusion in Low Data Regime**  
  Fei Yu Guan, Ian Keefe, Sophie Wilkinson, Daniel D. B. Perrakis, Steven Waslander  
  _2026-02-05_ · https://arxiv.org/abs/2602.05880v1  
  <details><summary>Abstract</summary>

  Boundary detection of irregular and translucent objects is an important problem with applications in medical imaging, environmental monitoring and manufacturing, where many of these applications are plagued with scarce labeled data and low in situ computational resources. While recent image segmentation studies focus on segmentation mask alignment with ground-truth, the task of boundary detection remains understudied, especially in the low data regime. In this work, we present a lightweight discrete diffusion contour refinement pipeline for robust boundary detection in the low data regime. We use a Convolutional Neural Network(CNN) architecture with self-attention layers as the core of our pipeline, and condition on a segmentation mask, iteratively denoising a sparse contour representation. We introduce multiple novel adaptations for improved low-data efficacy and inference efficiency, including using a simplified diffusion process, a customized model architecture, and minimal post processing to produce a dense, isolated contour given a dataset of size <500 training images. Our method outperforms several SOTA baselines on the medical imaging dataset KVASIR, is competitive on HAM10K and our custom wildfire dataset, Smoke, while improving inference framerate by 3.5X.

  </details>



- **NVS-HO: A Benchmark for Novel View Synthesis of Handheld Objects**  
  Musawar Ali, Manuel Carranza-García, Nicola Fioraio, Samuele Salti, Luigi Di Stefano  
  _2026-02-05_ · https://arxiv.org/abs/2602.05822v1  
  <details><summary>Abstract</summary>

  We propose NVS-HO, the first benchmark designed for novel view synthesis of handheld objects in real-world environments using only RGB inputs. Each object is recorded in two complementary RGB sequences: (1) a handheld sequence, where the object is manipulated in front of a static camera, and (2) a board sequence, where the object is fixed on a ChArUco board to provide accurate camera poses via marker detection. The goal of NVS-HO is to learn a NVS model that captures the full appearance of an object from (1), whereas (2) provides the ground-truth images used for evaluation. To establish baselines, we consider both a classical SfM pipeline and a state-of-the-art pre-trained feed-forward neural network (VGGT) as pose estimators, and train NVS models based on NeRF and Gaussian Splatting. Our experiments reveal significant performance gaps in current methods under unconstrained handheld conditions, highlighting the need for more robust approaches. NVS-HO thus offers a challenging real-world benchmark to drive progress in RGB-based novel view synthesis of handheld objects.

  </details>



- **Neuro-Inspired Visual Pattern Recognition via Biological Reservoir Computing**  
  Luca Ciampi, Ludovico Iannello, Fabrizio Tonelli, Gabriele Lagani, Angelo Di Garbo, Federico Cremisi, Giuseppe Amato  
  _2026-02-05_ · https://arxiv.org/abs/2602.05737v1  
  <details><summary>Abstract</summary>

  In this paper, we present a neuro-inspired approach to reservoir computing (RC) in which a network of in vitro cultured cortical neurons serves as the physical reservoir. Rather than relying on artificial recurrent models to approximate neural dynamics, our biological reservoir computing (BRC) system leverages the spontaneous and stimulus-evoked activity of living neural circuits as its computational substrate. A high-density multi-electrode array (HD-MEA) provides simultaneous stimulation and readout across hundreds of channels: input patterns are delivered through selected electrodes, while the remaining ones capture the resulting high-dimensional neural responses, yielding a biologically grounded feature representation. A linear readout layer (single-layer perceptron) is then trained to classify these reservoir states, enabling the living neural network to perform static visual pattern-recognition tasks within a computer-vision framework. We evaluate the system across a sequence of tasks of increasing difficulty, ranging from pointwise stimuli to oriented bars, clock-digit-like shapes, and handwritten digits from the MNIST dataset. Despite the inherent variability of biological neural responses-arising from noise, spontaneous activity, and inter-session differences-the system consistently generates high-dimensional representations that support accurate classification. These results demonstrate that in vitro cortical networks can function as effective reservoirs for static visual pattern recognition, opening new avenues for integrating living neural substrates into neuromorphic computing frameworks. More broadly, this work contributes to the effort to incorporate biological principles into machine learning and supports the goals of neuro-inspired vision by illustrating how living neural systems can inform the design of efficient and biologically grounded computational models.

  </details>



- **Adaptive Global and Fine-Grained Perceptual Fusion for MLLM Embeddings Compatible with Hard Negative Amplification**  
  Lexiang Hu, Youze Xue, Dian Li, Gang Liu, Zhouchen Lin  
  _2026-02-05_ · https://arxiv.org/abs/2602.05729v1  
  <details><summary>Abstract</summary>

  Multimodal embeddings serve as a bridge for aligning vision and language, with the two primary implementations -- CLIP-based and MLLM-based embedding models -- both limited to capturing only global semantic information. Although numerous studies have focused on fine-grained understanding, we observe that complex scenarios currently targeted by MLLM embeddings often involve a hybrid perceptual pattern of both global and fine-grained elements, thus necessitating a compatible fusion mechanism. In this paper, we propose Adaptive Global and Fine-grained perceptual Fusion for MLLM Embeddings (AGFF-Embed), a method that prompts the MLLM to generate multiple embeddings focusing on different dimensions of semantic information, which are then adaptively and smoothly aggregated. Furthermore, we adapt AGFF-Embed with the Explicit Gradient Amplification (EGA) technique to achieve in-batch hard negatives enhancement without requiring fine-grained editing of the dataset. Evaluation on the MMEB and MMVP-VLM benchmarks shows that AGFF-Embed comprehensively achieves state-of-the-art performance in both general and fine-grained understanding compared to other multimodal embedding models.

  </details>



- **Exploring the Temporal Consistency for Point-Level Weakly-Supervised Temporal Action Localization**  
  Yunchuan Ma, Laiyun Qing, Guorong Li, Yuqing Liu, Yuankai Qi, Qingming Huang  
  _2026-02-05_ · https://arxiv.org/abs/2602.05718v1  
  <details><summary>Abstract</summary>

  Point-supervised Temporal Action Localization (PTAL) adopts a lightly frame-annotated paradigm (\textit{i.e.}, labeling only a single frame per action instance) to train a model to effectively locate action instances within untrimmed videos. Most existing approaches design the task head of models with only a point-supervised snippet-level classification, without explicit modeling of understanding temporal relationships among frames of an action. However, understanding the temporal relationships of frames is crucial because it can help a model understand how an action is defined and therefore benefits localizing the full frames of an action. To this end, in this paper, we design a multi-task learning framework that fully utilizes point supervision to boost the model's temporal understanding capability for action localization. Specifically, we design three self-supervised temporal understanding tasks: (i) Action Completion, (ii) Action Order Understanding, and (iii) Action Regularity Understanding. These tasks help a model understand the temporal consistency of actions across videos. To the best of our knowledge, this is the first attempt to explicitly explore temporal consistency for point supervision action localization. Extensive experimental results on four benchmark datasets demonstrate the effectiveness of the proposed method compared to several state-of-the-art approaches.

  </details>



- **Enhancing Personality Recognition by Comparing the Predictive Power of Traits, Facets, and Nuances**  
  Amir Ansari, Jana Subirana, Bruna Silva, Sergio Escalera, David Gallardo-Pujol, Cristina Palmero  
  _2026-02-05_ · https://arxiv.org/abs/2602.05650v1  
  <details><summary>Abstract</summary>

  Personality is a complex, hierarchical construct typically assessed through item-level questionnaires aggregated into broad trait scores. Personality recognition models aim to infer personality traits from different sources of behavioral data. However, reliance on broad trait scores as ground truth, combined with limited training data, poses challenges for generalization, as similar trait scores can manifest through diverse, context dependent behaviors. In this work, we explore the predictive impact of the more granular hierarchical levels of the Big-Five Personality Model, facets and nuances, to enhance personality recognition from audiovisual interaction data. Using the UDIVA v0.5 dataset, we trained a transformer-based model including cross-modal (audiovisual) and cross-subject (dyad-aware) attention mechanisms. Results show that nuance-level models consistently outperform facet and trait-level models, reducing mean squared error by up to 74% across interaction scenarios.

  </details>



- **UniSurg: A Video-Native Foundation Model for Universal Understanding of Surgical Videos**  
  Jinlin Wu, Felix Holm, Chuxi Chen, An Wang, Yaxin Hu, Xiaofan Ye, Zelin Zang, Miao Xu, Lihua Zhou, Huai Liao, et al.  
  _2026-02-05_ · https://arxiv.org/abs/2602.05638v1  
  <details><summary>Abstract</summary>

  While foundation models have advanced surgical video analysis, current approaches rely predominantly on pixel-level reconstruction objectives that waste model capacity on low-level visual details - such as smoke, specular reflections, and fluid motion - rather than semantic structures essential for surgical understanding. We present UniSurg, a video-native foundation model that shifts the learning paradigm from pixel-level reconstruction to latent motion prediction. Built on the Video Joint Embedding Predictive Architecture (V-JEPA), UniSurg introduces three key technical innovations tailored to surgical videos: 1) motion-guided latent prediction to prioritize semantically meaningful regions, 2) spatiotemporal affinity self-distillation to enforce relational consistency, and 3) feature diversity regularization to prevent representation collapse in texture-sparse surgical scenes. To enable large-scale pretraining, we curate UniSurg-15M, the largest surgical video dataset to date, comprising 3,658 hours of video from 50 sources across 13 anatomical regions. Extensive experiments across 17 benchmarks demonstrate that UniSurg significantly outperforms state-of-the-art methods on surgical workflow recognition (+14.6% F1 on EgoSurgery, +10.3% on PitVis), action triplet recognition (39.54% mAP-IVT on CholecT50), skill assessment, polyp segmentation, and depth estimation. These results establish UniSurg as a new standard for universal, motion-oriented surgical video understanding.

  </details>



- **Unified Sensor Simulation for Autonomous Driving**  
  Nikolay Patakin, Arsenii Shirokov, Anton Konushin, Dmitry Senushkin  
  _2026-02-05_ · https://arxiv.org/abs/2602.05617v1  
  <details><summary>Abstract</summary>

  In this work, we introduce \textbf{XSIM}, a sensor simulation framework for autonomous driving. XSIM extends 3DGUT splatting with a generalized rolling-shutter modeling tailored for autonomous driving applications. Our framework provides a unified and flexible formulation for appearance and geometric sensor modeling, enabling rendering of complex sensor distortions in dynamic environments. We identify spherical cameras, such as LiDARs, as a critical edge case for existing 3DGUT splatting due to cyclic projection and time discontinuities at azimuth boundaries leading to incorrect particle projection. To address this issue, we propose a phase modeling mechanism that explicitly accounts temporal and shape discontinuities of Gaussians projected by the Unscented Transform at azimuth borders. In addition, we introduce an extended 3D Gaussian representation that incorporates two distinct opacity parameters to resolve mismatches between geometry and color distributions. As a result, our framework provides enhanced scene representations with improved geometric consistency and photorealistic appearance. We evaluate our framework extensively on multiple autonomous driving datasets, including Waymo Open Dataset, Argoverse 2, and PandaSet. Our framework consistently outperforms strong recent baselines and achieves state-of-the-art performance across all datasets. The source code is publicly available at \href{https://github.com/whesense/XSIM}{https://github.com/whesense/XSIM}.

  </details>



- **HiCrowd: Hierarchical Crowd Flow Alignment for Dense Human Environments**  
  Yufei Zhu, Shih-Min Yang, Martin Magnusson, Allan Wang  
  _2026-02-05_ · https://arxiv.org/abs/2602.05608v1  
  <details><summary>Abstract</summary>

  Navigating through dense human crowds remains a significant challenge for mobile robots. A key issue is the freezing robot problem, where the robot struggles to find safe motions and becomes stuck within the crowd. To address this, we propose HiCrowd, a hierarchical framework that integrates reinforcement learning (RL) with model predictive control (MPC). HiCrowd leverages surrounding pedestrian motion as guidance, enabling the robot to align with compatible crowd flows. A high-level RL policy generates a follow point to align the robot with a suitable pedestrian group, while a low-level MPC safely tracks this guidance with short horizon planning. The method combines long-term crowd aware decision making with safe short-term execution. We evaluate HiCrowd against reactive and learning-based baselines in offline setting (replaying recorded human trajectories) and online setting (human trajectories are updated to react to the robot in simulation). Experiments on a real-world dataset and a synthetic crowd dataset show that our method outperforms in navigation efficiency and safety, while reducing freezing behaviors. Our results suggest that leveraging human motion as guidance, rather than treating humans solely as dynamic obstacles, provides a powerful principle for safe and efficient robot navigation in crowds.

  </details>



- **CAViT -- Channel-Aware Vision Transformer for Dynamic Feature Fusion**  
  Aon Safdar, Mohamed Saadeldin  
  _2026-02-05_ · https://arxiv.org/abs/2602.05598v1  
  <details><summary>Abstract</summary>

  Vision Transformers (ViTs) have demonstrated strong performance across a range of computer vision tasks by modeling long-range spatial interactions via self-attention. However, channel-wise mixing in ViTs remains static, relying on fixed multilayer perceptrons (MLPs) that lack adaptability to input content. We introduce 'CAViT', a dual-attention architecture that replaces the static MLP with a dynamic, attention-based mechanism for feature interaction. Each Transformer block in CAViT performs spatial self-attention followed by channel-wise self-attention, allowing the model to dynamically recalibrate feature representations based on global image context. This unified and content-aware token mixing strategy enhances representational expressiveness without increasing depth or complexity. We validate CAViT across five benchmark datasets spanning both natural and medical domains, where it outperforms the standard ViT baseline by up to +3.6% in accuracy, while reducing parameter count and FLOPs by over 30%. Qualitative attention maps reveal sharper and semantically meaningful activation patterns, validating the effectiveness of our attention-driven token mixing.

  </details>



- **EgoPoseVR: Spatiotemporal Multi-Modal Reasoning for Egocentric Full-Body Pose in Virtual Reality**  
  Haojie Cheng, Shaun Jing Heng Ong, Shaoyu Cai, Aiden Tat Yang Koh, Fuxi Ouyang, Eng Tat Khoo  
  _2026-02-05_ · https://arxiv.org/abs/2602.05590v1  
  <details><summary>Abstract</summary>

  Immersive virtual reality (VR) applications demand accurate, temporally coherent full-body pose tracking. Recent head-mounted camera-based approaches show promise in egocentric pose estimation, but encounter challenges when applied to VR head-mounted displays (HMDs), including temporal instability, inaccurate lower-body estimation, and the lack of real-time performance. To address these limitations, we present EgoPoseVR, an end-to-end framework for accurate egocentric full-body pose estimation in VR that integrates headset motion cues with egocentric RGB-D observations through a dual-modality fusion pipeline. A spatiotemporal encoder extracts frame- and joint-level representations, which are fused via cross-attention to fully exploit complementary motion cues across modalities. A kinematic optimization module then imposes constraints from HMD signals, enhancing the accuracy and stability of pose estimation. To facilitate training and evaluation, we introduce a large-scale synthetic dataset of over 1.8 million temporally aligned HMD and RGB-D frames across diverse VR scenarios. Experimental results show that EgoPoseVR outperforms state-of-the-art egocentric pose estimation models. A user study in real-world scenes further shows that EgoPoseVR achieved significantly higher subjective ratings in accuracy, stability, embodiment, and intention for future use compared to baseline methods. These results show that EgoPoseVR enables robust full-body pose tracking, offering a practical solution for accurate VR embodiment without requiring additional body-worn sensors or room-scale tracking systems.

  </details>



- **LoGoSeg: Integrating Local and Global Features for Open-Vocabulary Semantic Segmentation**  
  Junyang Chen, Xiangbo Lv, Zhiqiang Kou, Xingdong Sheng, Ning Xu, Yiguo Qiao  
  _2026-02-05_ · https://arxiv.org/abs/2602.05578v1  
  <details><summary>Abstract</summary>

  Open-vocabulary semantic segmentation (OVSS) extends traditional closed-set segmentation by enabling pixel-wise annotation for both seen and unseen categories using arbitrary textual descriptions. While existing methods leverage vision-language models (VLMs) like CLIP, their reliance on image-level pretraining often results in imprecise spatial alignment, leading to mismatched segmentations in ambiguous or cluttered scenes. However, most existing approaches lack strong object priors and region-level constraints, which can lead to object hallucination or missed detections, further degrading performance. To address these challenges, we propose LoGoSeg, an efficient single-stage framework that integrates three key innovations: (i) an object existence prior that dynamically weights relevant categories through global image-text similarity, effectively reducing hallucinations; (ii) a region-aware alignment module that establishes precise region-level visual-textual correspondences; and (iii) a dual-stream fusion mechanism that optimally combines local structural information with global semantic context. Unlike prior works, LoGoSeg eliminates the need for external mask proposals, additional backbones, or extra datasets, ensuring efficiency. Extensive experiments on six benchmarks (A-847, PC-459, A-150, PC-59, PAS-20, and PAS-20b) demonstrate its competitive performance and strong generalization in open-vocabulary settings.

  </details>



- **LocateEdit-Bench: A Benchmark for Instruction-Based Editing Localization**  
  Shiyu Wu, Shuyan Li, Jing Li, Jing Liu, Yequan Wang  
  _2026-02-05_ · https://arxiv.org/abs/2602.05577v1  
  <details><summary>Abstract</summary>

  Recent advancements in image editing have enabled highly controllable and semantically-aware alteration of visual content, posing unprecedented challenges to manipulation localization. However, existing AI-generated forgery localization methods primarily focus on inpainting-based manipulations, making them ineffective against the latest instruction-based editing paradigms. To bridge this critical gap, we propose LocateEdit-Bench, a large-scale dataset comprising $231$K edited images, designed specifically to benchmark localization methods against instruction-driven image editing. Our dataset incorporates four cutting-edge editing models and covers three common edit types. We conduct a detailed analysis of the dataset and develop two multi-metric evaluation protocols to assess existing localization methods. Our work establishes a foundation to keep pace with the evolving landscape of image editing, thereby facilitating the development of effective methods for future forgery localization. Dataset will be open-sourced upon acceptance.

  </details>



- **Visual Implicit Geometry Transformer for Autonomous Driving**  
  Arsenii Shirokov, Mikhail Kuznetsov, Danila Stepochkin, Egor Evdokimov, Daniil Glazkov, Nikolay Patakin, Anton Konushin, Dmitry Senushkin  
  _2026-02-05_ · https://arxiv.org/abs/2602.05573v1  
  <details><summary>Abstract</summary>

  We introduce the Visual Implicit Geometry Transformer (ViGT), an autonomous driving geometric model that estimates continuous 3D occupancy fields from surround-view camera rigs. ViGT represents a step towards foundational geometric models for autonomous driving, prioritizing scalability, architectural simplicity, and generalization across diverse sensor configurations. Our approach achieves this through a calibration-free architecture, enabling a single model to adapt to different sensor setups. Unlike general-purpose geometric foundational models that focus on pixel-aligned predictions, ViGT estimates a continuous 3D occupancy field in a birds-eye-view (BEV) addressing domain-specific requirements. ViGT naturally infers geometry from multiple camera views into a single metric coordinate frame, providing a common representation for multiple geometric tasks. Unlike most existing occupancy models, we adopt a self-supervised training procedure that leverages synchronized image-LiDAR pairs, eliminating the need for costly manual annotations. We validate the scalability and generalizability of our approach by training our model on a mixture of five large-scale autonomous driving datasets (NuScenes, Waymo, NuPlan, ONCE, and Argoverse) and achieving state-of-the-art performance on the pointmap estimation task, with the best average rank across all evaluated baselines. We further evaluate ViGT on the Occ3D-nuScenes benchmark, where ViGT achieves comparable performance with supervised methods. The source code is publicly available at \href{https://github.com/whesense/ViGT}{https://github.com/whesense/ViGT}.

  </details>



- **IndustryShapes: An RGB-D Benchmark dataset for 6D object pose estimation of industrial assembly components and tools**  
  Panagiotis Sapoutzoglou, Orestis Vaggelis, Athina Zacharia, Evangelos Sartinas, Maria Pateraki  
  _2026-02-05_ · https://arxiv.org/abs/2602.05555v1  
  <details><summary>Abstract</summary>

  We introduce IndustryShapes, a new RGB-D benchmark dataset of industrial tools and components, designed for both instance-level and novel object 6D pose estimation approaches. The dataset provides a realistic and application-relevant testbed for benchmarking these methods in the context of industrial robotics bridging the gap between lab-based research and deployment in real-world manufacturing scenarios. Unlike many previous datasets that focus on household or consumer products or use synthetic, clean tabletop datasets, or objects captured solely in controlled lab environments, IndustryShapes introduces five new object types with challenging properties, also captured in realistic industrial assembly settings. The dataset has diverse complexity, from simple to more challenging scenes, with single and multiple objects, including scenes with multiple instances of the same object and it is organized in two parts: the classic set and the extended set. The classic set includes a total of 4,6k images and 6k annotated poses. The extended set introduces additional data modalities to support the evaluation of model-free and sequence-based approaches. To the best of our knowledge, IndustryShapes is the first dataset to offer RGB-D static onboarding sequences. We further evaluate the dataset on a representative set of state-of-the art methods for instance-based and novel object 6D pose estimation, including also object detection, segmentation, showing that there is room for improvement in this domain. The dataset page can be found in https://pose-lab.github.io/IndustryShapes.

  </details>



- **VLN-Pilot: Large Vision-Language Model as an Autonomous Indoor Drone Operator**  
  Bessie Dominguez-Dager, Sergio Suescun-Ferrandiz, Felix Escalona, Francisco Gomez-Donoso, Miguel Cazorla  
  _2026-02-05_ · https://arxiv.org/abs/2602.05552v1  
  <details><summary>Abstract</summary>

  This paper introduces VLN-Pilot, a novel framework in which a large Vision-and-Language Model (VLLM) assumes the role of a human pilot for indoor drone navigation. By leveraging the multimodal reasoning abilities of VLLMs, VLN-Pilot interprets free-form natural language instructions and grounds them in visual observations to plan and execute drone trajectories in GPS-denied indoor environments. Unlike traditional rule-based or geometric path-planning approaches, our framework integrates language-driven semantic understanding with visual perception, enabling context-aware, high-level flight behaviors with minimal task-specific engineering. VLN-Pilot supports fully autonomous instruction-following for drones by reasoning about spatial relationships, obstacle avoidance, and dynamic reactivity to unforeseen events. We validate our framework on a custom photorealistic indoor simulation benchmark and demonstrate the ability of the VLLM-driven agent to achieve high success rates on complex instruction-following tasks, including long-horizon navigation with multiple semantic targets. Experimental results highlight the promise of replacing remote drone pilots with a language-guided autonomous agent, opening avenues for scalable, human-friendly control of indoor UAVs in tasks such as inspection, search-and-rescue, and facility monitoring. Our results suggest that VLLM-based pilots may dramatically reduce operator workload while improving safety and mission flexibility in constrained indoor environments.

  </details>



- **A Comparative Study of 3D Person Detection: Sensor Modalities and Robustness in Diverse Indoor and Outdoor Environments**  
  Malaz Tamim, Andrea Matic-Flierl, Karsten Roscher  
  _2026-02-05_ · https://arxiv.org/abs/2602.05538v1  
  <details><summary>Abstract</summary>

  Accurate 3D person detection is critical for safety in applications such as robotics, industrial monitoring, and surveillance. This work presents a systematic evaluation of 3D person detection using camera-only, LiDAR-only, and camera-LiDAR fusion. While most existing research focuses on autonomous driving, we explore detection performance and robustness in diverse indoor and outdoor scenes using the JRDB dataset. We compare three representative models - BEVDepth (camera), PointPillars (LiDAR), and DAL (camera-LiDAR fusion) - and analyze their behavior under varying occlusion and distance levels. Our results show that the fusion-based approach consistently outperforms single-modality models, particularly in challenging scenarios. We further investigate robustness against sensor corruptions and misalignments, revealing that while DAL offers improved resilience, it remains sensitive to sensor misalignment and certain LiDAR-based corruptions. In contrast, the camera-based BEVDepth model showed the lowest performance and was most affected by occlusion, distance, and noise. Our findings highlight the importance of utilizing sensor fusion for enhanced 3D person detection, while also underscoring the need for ongoing research to address the vulnerabilities inherent in these systems.

  </details>



- **Generalization of Self-Supervised Vision Transformers for Protein Localization Across Microscopy Domains**  
  Ben Isselmann, Dilara Göksu, Andreas Weinmann  
  _2026-02-05_ · https://arxiv.org/abs/2602.05527v1  
  <details><summary>Abstract</summary>

  Task-specific microscopy datasets are often too small to train deep learning models that learn robust feature representations. Self-supervised learning (SSL) can mitigate this by pretraining on large unlabeled datasets, but it remains unclear how well such representations transfer across microscopy domains with different staining protocols and channel configurations. We investigate the cross-domain transferability of DINO-pretrained Vision Transformers for protein localization on the OpenCell dataset. We generate image embeddings using three DINO backbones pretrained on ImageNet-1k, the Human Protein Atlas (HPA), and OpenCell, and evaluate them by training a supervised classification head on OpenCell labels. All pretrained models transfer well, with the microscopy-specific HPA-pretrained model achieving the best performance (mean macro $F_1$-score = 0.8221 \pm 0.0062), slightly outperforming a DINO model trained directly on OpenCell (0.8057 \pm 0.0090). These results highlight the value of large-scale pretraining and indicate that domain-relevant SSL representations can generalize effectively to related but distinct microscopy datasets, enabling strong downstream performance even when task-specific labeled data are limited.

  </details>



- **Mapper-GIN: Lightweight Structural Graph Abstraction for Corrupted 3D Point Cloud Classification**  
  Jeongbin You, Donggun Kim, Sejun Park, Seungsang Oh  
  _2026-02-05_ · https://arxiv.org/abs/2602.05522v1  
  <details><summary>Abstract</summary>

  Robust 3D point cloud classification is often pursued by scaling up backbones or relying on specialized data augmentation. We instead ask whether structural abstraction alone can improve robustness, and study a simple topology-inspired decomposition based on the Mapper algorithm. We propose Mapper-GIN, a lightweight pipeline that partitions a point cloud into overlapping regions using Mapper (PCA lens, cubical cover, and followed by density-based clustering), constructs a region graph from their overlaps, and performs graph classification with a Graph Isomorphism Network. On the corruption benchmark ModelNet40-C, Mapper-GIN achieves competitive and stable accuracy under Noise and Transformation corruptions with only 0.5M parameters. In contrast to prior approaches that require heavier architectures or additional mechanisms to gain robustness, Mapper-GIN attains strong corruption robustness through simple region-level graph abstraction and GIN message passing. Overall, our results suggest that region-graph structure offers an efficient and interpretable source of robustness for 3D visual recognition.

  </details>



- **DECO: Decoupled Multimodal Diffusion Transformer for Bimanual Dexterous Manipulation with a Plugin Tactile Adapter**  
  Xukun Li, Yu Sun, Lei Zhang, Bosheng Huang, Yibo Peng, Yuan Meng, Haojun Jiang, Shaoxuan Xie, Guacai Yao, Alois Knoll, et al.  
  _2026-02-05_ · https://arxiv.org/abs/2602.05513v1  
  <details><summary>Abstract</summary>

  Overview of the Proposed DECO Framework.} DECO is a DiT-based policy that decouples multimodal conditioning. Image and action tokens interact via joint self attention, while proprioceptive states and optional conditions are injected through adaptive layer normalization. Tactile signals are injected via cross attention, while a lightweight LoRA-based adapter is used to efficiently fine-tune the pretrained policy. DECO is also accompanied by DECO-50, a bimanual dexterous manipulation dataset with tactile sensing, consisting of 4 scenarios and 28 sub-tasks, covering more than 50 hours of data, approximately 5 million frames, and 8,000 successful trajectories.

  </details>



- **XEmoGPT: An Explainable Multimodal Emotion Recognition Framework with Cue-Level Perception and Reasoning**  
  Hanwen Zhang, Yao Liu, Peiyuan Jiang, Lang Junjie, Xie Jun, Yihui He, Yajiao Deng, Siyu Du, Qiao Liu  
  _2026-02-05_ · https://arxiv.org/abs/2602.05496v1  
  <details><summary>Abstract</summary>

  Explainable Multimodal Emotion Recognition plays a crucial role in applications such as human-computer interaction and social media analytics. However, current approaches struggle with cue-level perception and reasoning due to two main challenges: 1) general-purpose modality encoders are pretrained to capture global structures and general semantics rather than fine-grained emotional cues, resulting in limited sensitivity to emotional signals; and 2) available datasets usually involve a trade-off between annotation quality and scale, which leads to insufficient supervision for emotional cues and ultimately limits cue-level reasoning. Moreover, existing evaluation metrics are inadequate for assessing cue-level reasoning performance. To address these challenges, we propose eXplainable Emotion GPT (XEmoGPT), a novel EMER framework capable of both perceiving and reasoning over emotional cues. It incorporates two specialized modules: the Video Emotional Cue Bridge (VECB) and the Audio Emotional Cue Bridge (AECB), which enhance the video and audio encoders through carefully designed tasks for fine-grained emotional cue perception. To further support cue-level reasoning, we construct a large-scale dataset, EmoCue, designed to teach XEmoGPT how to reason over multimodal emotional cues. In addition, we introduce EmoCue-360, an automated metric that extracts and matches emotional cues using semantic similarity, and release EmoCue-Eval, a benchmark of 400 expert-annotated samples covering diverse emotional scenarios. Experimental results show that XEmoGPT achieves strong performance in both emotional cue perception and reasoning.

  </details>



- **Feature points evaluation on omnidirectional vision with a photorealistic fisheye sequence -- A report on experiments done in 2014**  
  Julien Moreau, S. Ambellouis, Yassine Ruichek  
  _2026-02-05_ · https://arxiv.org/abs/2602.05487v1  
  <details><summary>Abstract</summary>

  What is this report: This is a scientific report, contributing with a detailed bibliography, a dataset which we will call now PFSeq for ''Photorealistic Fisheye Sequence'' and make available at https://doi.org/10. 57745/DYIVVU, and comprehensive experiments. This work should be considered as a draft, and has been done during my PhD thesis ''Construction of 3D models from fisheye video data-Application to the localisation in urban area'' in 2014 [Mor16]. These results have never been published. The aim was to find the best features detector and descriptor for fisheye images, in the context of selfcalibration, with cameras mounted on the top of a car and aiming at the zenith (to proceed then fisheye visual odometry and stereovision in urban scenes). We face a chicken and egg problem, because we can not take advantage of an accurate projection model for an optimal features detection and description, and we rightly need good features to perform the calibration (i.e. to compute the accurate projection model of the camera). What is not this report: It does not contribute with new features algorithm. It does not compare standard features algorithms to algorithms designed for omnidirectional images (unfortunately). It has not been peer-reviewed. Discussions have been translated and enhanced but the experiments have not been run again and the report has not been updated accordingly to the evolution of the state-of-the-art (read this as a 2014 report).

  </details>



- **SOMA-1M: A Large-Scale SAR-Optical Multi-resolution Alignment Dataset for Multi-Task Remote Sensing**  
  Peihao Wu, Yongxiang Yao, Yi Wan, Wenfei Zhang, Ruipeng Zhao, Jiayuan Li, Yongjun Zhang  
  _2026-02-05_ · https://arxiv.org/abs/2602.05480v1  
  <details><summary>Abstract</summary>

  Synthetic Aperture Radar (SAR) and optical imagery provide complementary strengths that constitute the critical foundation for transcending single-modality constraints and facilitating cross-modal collaborative processing and intelligent interpretation. However, existing benchmark datasets often suffer from limitations such as single spatial resolution, insufficient data scale, and low alignment accuracy, making them inadequate for supporting the training and generalization of multi-scale foundation models. To address these challenges, we introduce SOMA-1M (SAR-Optical Multi-resolution Alignment), a pixel-level precisely aligned dataset containing over 1.3 million pairs of georeferenced images with a specification of 512 x 512 pixels. This dataset integrates imagery from Sentinel-1, PIESAT-1, Capella Space, and Google Earth, achieving global multi-scale coverage from 0.5 m to 10 m. It encompasses 12 typical land cover categories, effectively ensuring scene diversity and complexity. To address multimodal projection deformation and massive data registration, we designed a rigorous coarse-to-fine image matching framework ensuring pixel-level alignment. Based on this dataset, we established comprehensive evaluation benchmarks for four hierarchical vision tasks, including image matching, image fusion, SAR-assisted cloud removal, and cross-modal translation, involving over 30 mainstream algorithms. Experimental results demonstrate that supervised training on SOMA-1M significantly enhances performance across all tasks. Notably, multimodal remote sensing image (MRSI) matching performance achieves current state-of-the-art (SOTA) levels. SOMA-1M serves as a foundational resource for robust multimodal algorithms and remote sensing foundation models. The dataset will be released publicly at: https://github.com/PeihaoWu/SOMA-1M.

  </details>



- **MerNav: A Highly Generalizable Memory-Execute-Review Framework for Zero-Shot Object Goal Navigation**  
  Dekang Qi, Shuang Zeng, Xinyuan Chang, Feng Xiong, Shichao Xie, Xiaolong Wu, Mu Xu  
  _2026-02-05_ · https://arxiv.org/abs/2602.05467v1  
  <details><summary>Abstract</summary>

  Visual Language Navigation (VLN) is one of the fundamental capabilities for embodied intelligence and a critical challenge that urgently needs to be addressed. However, existing methods are still unsatisfactory in terms of both success rate (SR) and generalization: Supervised Fine-Tuning (SFT) approaches typically achieve higher SR, while Training-Free (TF) approaches often generalize better, but it is difficult to obtain both simultaneously. To this end, we propose a Memory-Execute-Review framework. It consists of three parts: a hierarchical memory module for providing information support, an execute module for routine decision-making and actions, and a review module for handling abnormal situations and correcting behavior. We validated the effectiveness of this framework on the Object Goal Navigation task. Across 4 datasets, our average SR achieved absolute improvements of 7% and 5% compared to all baseline methods under TF and Zero-Shot (ZS) settings, respectively. On the most commonly used HM3D_v0.1 and the more challenging open vocabulary dataset HM3D_OVON, the SR improved by 8% and 6%, under ZS settings. Furthermore, on the MP3D and HM3D_OVON datasets, our method not only outperformed all TF methods but also surpassed all SFT methods, achieving comprehensive leadership in both SR (5% and 2%) and generalization.

  </details>



- **Towards Segmenting the Invisible: An End-to-End Registration and Segmentation Framework for Weakly Supervised Tumour Analysis**  
  Budhaditya Mukhopadhyay, Chirag Mandal, Pavan Tummala, Naghmeh Mahmoodian, Andreas Nürnberger, Soumick Chatterjee  
  _2026-02-05_ · https://arxiv.org/abs/2602.05453v1  
  <details><summary>Abstract</summary>

  Liver tumour ablation presents a significant clinical challenge: whilst tumours are clearly visible on pre-operative MRI, they are often effectively invisible on intra-operative CT due to minimal contrast between pathological and healthy tissue. This work investigates the feasibility of cross-modality weak supervision for scenarios where pathology is visible in one modality (MRI) but absent in another (CT). We present a hybrid registration-segmentation framework that combines MSCGUNet for inter-modal image registration with a UNet-based segmentation module, enabling registration-assisted pseudo-label generation for CT images. Our evaluation on the CHAOS dataset demonstrates that the pipeline can successfully register and segment healthy liver anatomy, achieving a Dice score of 0.72. However, when applied to clinical data containing tumours, performance degrades substantially (Dice score of 0.16), revealing the fundamental limitations of current registration methods when the target pathology lacks corresponding visual features in the target modality. We analyse the "domain gap" and "feature absence" problems, demonstrating that whilst spatial propagation of labels via registration is feasible for visible structures, segmenting truly invisible pathology remains an open challenge. Our findings highlight that registration-based label transfer cannot compensate for the absence of discriminative features in the target modality, providing important insights for future research in cross-modality medical image analysis. Code an weights are available at: https://github.com/BudhaTronix/Weakly-Supervised-Tumour-Detection

  </details>



- **Benchmarking Affordance Generalization with BusyBox**  
  Dean Fortier, Timothy Adamson, Tess Hellebrekers, Teresa LaScala, Kofi Ennin, Michael Murray, Andrey Kolobov, Galen Mullins  
  _2026-02-05_ · https://arxiv.org/abs/2602.05441v1  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models have been attracting the attention of researchers and practitioners thanks to their promise of generalization. Although single-task policies still offer competitive performance, VLAs are increasingly able to handle commands and environments unseen in their training set. While generalization in vision and language space is undoubtedly important for robust versatile behaviors, a key meta-skill VLAs need to possess is affordance generalization -- the ability to manipulate new objects with familiar physical features. In this work, we present BusyBox, a physical benchmark for systematic semi-automatic evaluation of VLAs' affordance generalization. BusyBox consists of 6 modules with switches, sliders, wires, buttons, a display, and a dial. The modules can be swapped and rotated to create a multitude of BusyBox variations with different visual appearances but the same set of affordances. We empirically demonstrate that generalization across BusyBox variants is highly challenging even for strong open-weights VLAs such as $π_{0.5}$ and GR00T-N1.6. To encourage the research community to evaluate their own VLAs on BusyBox and to propose new affordance generalization experiments, we have designed BusyBox to be easy to build in most robotics labs. We release the full set of CAD files for 3D-printing its parts as well as a bill of materials for (optionally) assembling its electronics. We also publish a dataset of language-annotated demonstrations that we collected using the common bimanual Mobile Aloha robot on the canonical BusyBox configuration. All of the released materials are available at https://microsoft.github.io/BusyBox.

  </details>



- **Synthetic Defect Geometries of Cast Metal Objects Modeled via 2d Voronoi Tessellations**  
  Natascha Jeziorski, Petra Gospodnetić, Claudia Redenbach  
  _2026-02-05_ · https://arxiv.org/abs/2602.05440v1  
  <details><summary>Abstract</summary>

  In industry, defect detection is crucial for quality control. Non-destructive testing (NDT) methods are preferred as they do not influence the functionality of the object while inspecting. Automated data evaluation for automated defect detection is a growing field of research. In particular, machine learning approaches show promising results. To provide training data in sufficient amount and quality, synthetic data can be used. Rule-based approaches enable synthetic data generation in a controllable environment. Therefore, a digital twin of the inspected object including synthetic defects is needed. We present parametric methods to model 3d mesh objects of various defect types that can then be added to the object geometry to obtain synthetic defective objects. The models are motivated by common defects in metal casting but can be transferred to other machining procedures that produce similar defect shapes. Synthetic data resembling the real inspection data can then be created by using a physically based Monte Carlo simulation of the respective testing method. Using our defect models, a variable and arbitrarily large synthetic data set can be generated with the possibility to include rarely occurring defects in sufficient quantity. Pixel-perfect annotation can be created in parallel. As an example, we will use visual surface inspection, but the procedure can be applied in combination with simulations for any other NDT method.

  </details>



- **M$^2$-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining**  
  Rui Lv, Juncheng Mo, Tianyi Chu, Chen Rao, Hongyi Jing, Jiajie Teng, Jiafu Chen, Shiqi Zhang, Liangzi Ding, Shuo Fang, et al.  
  _2026-02-05_ · https://arxiv.org/abs/2602.05429v1  
  <details><summary>Abstract</summary>

  Graphical User Interface (GUI) agent is pivotal to advancing intelligent human-computer interaction paradigms. Constructing powerful GUI agents necessitates the large-scale annotation of high-quality user-behavior trajectory data (i.e., intent-trajectory pairs) for training. However, manual annotation methods and current GUI agent data mining approaches typically face three critical challenges: high construction cost, poor data quality, and low data richness. To address these issues, we propose M$^2$-Miner, the first low-cost and automated mobile GUI agent data-mining framework based on Monte Carlo Tree Search (MCTS). For better data mining efficiency and quality, we present a collaborative multi-agent framework, comprising InferAgent, OrchestraAgent, and JudgeAgent for guidance, acceleration, and evaluation. To further enhance the efficiency of mining and enrich intent diversity, we design an intent recycling strategy to extract extra valuable interaction trajectories. Additionally, a progressive model-in-the-loop training strategy is introduced to improve the success rate of data mining. Extensive experiments have demonstrated that the GUI agent fine-tuned using our mined data achieves state-of-the-art performance on several commonly used mobile GUI benchmarks. Our work will be released to facilitate the community research.

  </details>



- **Disco: Densely-overlapping Cell Instance Segmentation via Adjacency-aware Collaborative Coloring**  
  Rui Sun, Yiwen Yang, Kaiyu Guo, Chen Jiang, Dongli Xu, Zhaonan Liu, Tan Pan, Limei Han, Xue Jiang, Wu Wei, et al.  
  _2026-02-05_ · https://arxiv.org/abs/2602.05420v1  
  <details><summary>Abstract</summary>

  Accurate cell instance segmentation is foundational for digital pathology analysis. Existing methods based on contour detection and distance mapping still face significant challenges in processing complex and dense cellular regions. Graph coloring-based methods provide a new paradigm for this task, yet the effectiveness of this paradigm in real-world scenarios with dense overlaps and complex topologies has not been verified. Addressing this issue, we release a large-scale dataset GBC-FS 2025, which contains highly complex and dense sub-cellular nuclear arrangements. We conduct the first systematic analysis of the chromatic properties of cell adjacency graphs across four diverse datasets and reveal an important discovery: most real-world cell graphs are non-bipartite, with a high prevalence of odd-length cycles (predominantly triangles). This makes simple 2-coloring theory insufficient for handling complex tissues, while higher-chromaticity models would cause representational redundancy and optimization difficulties. Building on this observation of complex real-world contexts, we propose Disco (Densely-overlapping Cell Instance Segmentation via Adjacency-aware COllaborative Coloring), an adjacency-aware framework based on the "divide and conquer" principle. It uniquely combines a data-driven topological labeling strategy with a constrained deep learning system to resolve complex adjacency conflicts. First, "Explicit Marking" strategy transforms the topological challenge into a learnable classification task by recursively decomposing the cell graph and isolating a "conflict set." Second, "Implicit Disambiguation" mechanism resolves ambiguities in conflict regions by enforcing feature dissimilarity between different instances, enabling the model to learn separable feature representations.

  </details>



- **TSBOW: Traffic Surveillance Benchmark for Occluded Vehicles Under Various Weather Conditions**  
  Ngoc Doan-Minh Huynh, Duong Nguyen-Ngoc Tran, Long Hoang Pham, Tai Huu-Phuong Tran, Hyung-Joon Jeon, Huy-Hung Nguyen, Duong Khac Vu, Hyung-Min Jeon, Son Hong Phan, Quoc Pham-Nam Ho, et al.  
  _2026-02-05_ · https://arxiv.org/abs/2602.05414v1  
  <details><summary>Abstract</summary>

  Global warming has intensified the frequency and severity of extreme weather events, which degrade CCTV signal and video quality while disrupting traffic flow, thereby increasing traffic accident rates. Existing datasets, often limited to light haze, rain, and snow, fail to capture extreme weather conditions. To address this gap, this study introduces the Traffic Surveillance Benchmark for Occluded vehicles under various Weather conditions (TSBOW), a comprehensive dataset designed to enhance occluded vehicle detection across diverse annual weather scenarios. Comprising over 32 hours of real-world traffic data from densely populated urban areas, TSBOW includes more than 48,000 manually annotated and 3.2 million semi-labeled frames; bounding boxes spanning eight traffic participant classes from large vehicles to micromobility devices and pedestrians. We establish an object detection benchmark for TSBOW, highlighting challenges posed by occlusions and adverse weather. With its varied road types, scales, and viewpoints, TSBOW serves as a critical resource for advancing Intelligent Transportation Systems. Our findings underscore the potential of CCTV-based traffic monitoring, pave the way for new research and applications. The TSBOW dataset is publicly available at: https://github.com/SKKUAutoLab/TSBOW.

  </details>



- **Dataset Distillation via Relative Distribution Matching and Cognitive Heritage**  
  Qianxin Xia, Jiawei Du, Yuhan Zhang, Jielei Wang, Guoming Lu  
  _2026-02-05_ · https://arxiv.org/abs/2602.05391v1  
  <details><summary>Abstract</summary>

  Dataset distillation seeks to synthesize a highly compact dataset that achieves performance comparable to the original dataset on downstream tasks. For the classification task that use pre-trained self-supervised models as backbones, previous linear gradient matching optimizes synthetic images by encouraging them to mimic the gradient updates induced by real images on the linear classifier. However, this batch-level formulation requires loading thousands of real images and applying multiple rounds of differentiable augmentations to synthetic images at each distillation step, leading to substantial computational and memory overhead. In this paper, we introduce statistical flow matching , a stable and efficient supervised learning framework that optimizes synthetic images by aligning constant statistical flows from target class centers to non-target class centers in the original data. Our approach loads raw statistics only once and performs a single augmentation pass on the synthetic data, achieving performance comparable to or better than the state-of-the-art methods with 10x lower GPU memory usage and 4x shorter runtime. Furthermore, we propose a classifier inheritance strategy that reuses the classifier trained on the original dataset for inference, requiring only an extremely lightweight linear projector and marginal storage while achieving substantial performance gains.

  </details>



- **Dolphin-v2: Universal Document Parsing via Scalable Anchor Prompting**  
  Hao Feng, Wei Shi, Ke Zhang, Xiang Fei, Lei Liao, Dingkang Yang, Yongkun Du, Xuecheng Wu, Jingqun Tang, Yang Liu, et al.  
  _2026-02-05_ · https://arxiv.org/abs/2602.05384v1  
  <details><summary>Abstract</summary>

  Document parsing has garnered widespread attention as vision-language models (VLMs) advance OCR capabilities. However, the field remains fragmented across dozens of specialized models with varying strengths, forcing users to navigate complex model selection and limiting system scalability. Moreover, existing two-stage approaches depend on axis-aligned bounding boxes for layout detection, failing to handle distorted or photographed documents effectively. To this end, we present Dolphin-v2, a two-stage document image parsing model that substantially improves upon the original Dolphin. In the first stage, Dolphin-v2 jointly performs document type classification (digital-born versus photographed) alongside layout analysis. For digital-born documents, it conducts finer-grained element detection with reading order prediction. In the second stage, we employ a hybrid parsing strategy: photographed documents are parsed holistically as complete pages to handle geometric distortions, while digital-born documents undergo element-wise parallel parsing guided by the detected layout anchors, enabling efficient content extraction. Compared with the original Dolphin, Dolphin-v2 introduces several crucial enhancements: (1) robust parsing of photographed documents via holistic page-level understanding, (2) finer-grained element detection (21 categories) with semantic attribute extraction such as author information and document metadata, and (3) code block recognition with indentation preservation, which existing systems typically lack. Comprehensive evaluations are conducted on DocPTBench, OmniDocBench, and our self-constructed RealDoc-160 benchmark. The results demonstrate substantial improvements: +14.78 points overall on the challenging OmniDocBench and 91% error reduction on photographed documents, while maintaining efficient inference through parallel processing.

  </details>



- **VRIQ: Benchmarking and Analyzing Visual-Reasoning IQ of VLMs**  
  Tina Khezresmaeilzadeh, Jike Zhong, Konstantinos Psounis  
  _2026-02-05_ · https://arxiv.org/abs/2602.05382v1  
  <details><summary>Abstract</summary>

  Recent progress in Vision Language Models (VLMs) has raised the question of whether they can reliably perform nonverbal reasoning. To this end, we introduce VRIQ (Visual Reasoning IQ), a novel benchmark designed to assess and analyze the visual reasoning ability of VLMs. We evaluate models on two sets of tasks: abstract puzzle-style and natural-image reasoning tasks. We find that on abstract puzzles, performance remains near random with an average accuracy of around 28%, while natural tasks yield better but still weak results with 45% accuracy. We also find that tool-augmented reasoning demonstrates only modest improvements. To uncover the source of this weakness, we introduce diagnostic probes targeting perception and reasoning. Our analysis demonstrates that around 56% of failures arise from perception alone, 43% from both perception and reasoning, and only a mere 1% from reasoning alone. This motivates us to design fine-grained diagnostic probe questions targeting specific perception categories (e.g., shape, count, position, 3D/depth), revealing that certain categories cause more failures than others. Our benchmark and analysis establish that current VLMs, even with visual reasoning tools, remain unreliable abstract reasoners, mostly due to perception limitations, and offer a principled basis for improving visual reasoning in multimodal systems.

  </details>


