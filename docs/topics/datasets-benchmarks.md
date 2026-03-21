# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-03-21 07:01 UTC_

Total papers shown: **50**


---

- **NavTrust: Benchmarking Trustworthiness for Embodied Navigation**  
  Huaide Jiang, Yash Chaudhary, Yuping Wang, Zehao Wang, Raghav Sharma, Manan Mehta, Yang Zhou, Lichao Sun, Zhiwen Fan, Zhengzhong Tu, et al.  
  _2026-03-19_ · https://arxiv.org/abs/2603.19229v1  
  <details><summary>Abstract</summary>

  There are two major categories of embodied navigation: Vision-Language Navigation (VLN), where agents navigate by following natural language instructions; and Object-Goal Navigation (OGN), where agents navigate to a specified target object. However, existing work primarily evaluates model performance under nominal conditions, overlooking the potential corruptions that arise in real-world settings. To address this gap, we present NavTrust, a unified benchmark that systematically corrupts input modalities, including RGB, depth, and instructions, in realistic scenarios and evaluates their impact on navigation performance. To our best knowledge, NavTrust is the first benchmark that exposes embodied navigation agents to diverse RGB-Depth corruptions and instruction variations in a unified framework. Our extensive evaluation of seven state-of-the-art approaches reveals substantial performance degradation under realistic corruptions, which highlights critical robustness gaps and provides a roadmap toward more trustworthy embodied navigation systems. Furthermore, we systematically evaluate four distinct mitigation strategies to enhance robustness against RGB-Depth and instructions corruptions. Our base models include Uni-NaVid and ETPNav. We deployed them on a real mobile robot and observed improved robustness to corruptions. The project website is: https://navtrust.github.io.

  </details>



- **EffectErase: Joint Video Object Removal and Insertion for High-Quality Effect Erasing**  
  Yang Fu, Yike Zheng, Ziyun Dai, Henghui Ding  
  _2026-03-19_ · https://arxiv.org/abs/2603.19224v1  
  <details><summary>Abstract</summary>

  Video object removal aims to eliminate dynamic target objects and their visual effects, such as deformation, shadows, and reflections, while restoring seamless backgrounds. Recent diffusion-based video inpainting and object removal methods can remove the objects but often struggle to erase these effects and to synthesize coherent backgrounds. Beyond method limitations, progress is further hampered by the lack of a comprehensive dataset that systematically captures common object effects across varied environments for training and evaluation. To address this, we introduce VOR (Video Object Removal), a large-scale dataset that provides diverse paired videos, each consisting of one video where the target object is present with its effects and a counterpart where the object and effects are absent, with corresponding object masks. VOR contains 60K high-quality video pairs from captured and synthetic sources, covers five effects types, and spans a wide range of object categories as well as complex, dynamic multi-object scenes. Building on VOR, we propose EffectErase, an effect-aware video object removal method that treats video object insertion as the inverse auxiliary task within a reciprocal learning scheme. The model includes task-aware region guidance that focuses learning on affected areas and enables flexible task switching. Then, an insertion-removal consistency objective that encourages complementary behaviors and shared localization of effect regions and structural cues. Trained on VOR, EffectErase achieves superior performance in extensive experiments, delivering high-quality video object effect erasing across diverse scenarios.

  </details>



- **DriveTok: 3D Driving Scene Tokenization for Unified Multi-View Reconstruction and Understanding**  
  Dong Zhuo, Wenzhao Zheng, Sicheng Zuo, Siming Yan, Lu Hou, Jie Zhou, Jiwen Lu  
  _2026-03-19_ · https://arxiv.org/abs/2603.19219v1  
  <details><summary>Abstract</summary>

  With the growing adoption of vision-language-action models and world models in autonomous driving systems, scalable image tokenization becomes crucial as the interface for the visual modality. However, most existing tokenizers are designed for monocular and 2D scenes, leading to inefficiency and inter-view inconsistency when applied to high-resolution multi-view driving scenes. To address this, we propose DriveTok, an efficient 3D driving scene tokenizer for unified multi-view reconstruction and understanding. DriveTok first obtains semantically rich visual features from vision foundation models and then transforms them into the scene tokens with 3D deformable cross-attention. For decoding, we employ a multi-view transformer to reconstruct multi-view features from the scene tokens and use multiple heads to obtain RGB, depth, and semantic reconstructions. We also add a 3D head directly on the scene tokens for 3D semantic occupancy prediction for better spatial awareness. With the multiple training objectives, DriveTok learns unified scene tokens that integrate semantic, geometric, and textural information for efficient multi-view tokenization. Extensive experiments on the widely used nuScenes dataset demonstrate that the scene tokens from DriveTok perform well on image reconstruction, semantic segmentation, depth prediction, and 3D occupancy prediction tasks.

  </details>



- **LVOmniBench: Pioneering Long Audio-Video Understanding Evaluation for Omnimodal LLMs**  
  Keda Tao, Yuhua Zheng, Jia Xu, Wenjie Du, Kele Shao, Hesong Wang, Xueyi Chen, Xin Jin, Junhan Zhu, Bohan Yu, et al.  
  _2026-03-19_ · https://arxiv.org/abs/2603.19217v1  
  <details><summary>Abstract</summary>

  Recent advancements in omnimodal large language models (OmniLLMs) have significantly improved the comprehension of audio and video inputs. However, current evaluations primarily focus on short audio and video clips ranging from 10 seconds to 5 minutes, failing to reflect the demands of real-world applications, where videos typically run for tens of minutes. To address this critical gap, we introduce LVOmniBench, a new benchmark designed specifically for the cross-modal comprehension of long-form audio and video. This dataset comprises high-quality videos sourced from open platforms that feature rich audio-visual dynamics. Through rigorous manual selection and annotation, LVOmniBench comprises 275 videos, ranging in duration from 10 to 90 minutes, and 1,014 question-answer (QA) pairs. LVOmniBench aims to rigorously evaluate the capabilities of OmniLLMs across domains, including long-term memory, temporal localization, fine-grained understanding, and multimodal perception. Our extensive evaluation reveals that current OmniLLMs encounter significant challenges when processing extended audio-visual inputs. Open-source models generally achieve accuracies below 35%, whereas the Gemini 3 Pro reaches a peak accuracy of approximately 65%. We anticipate that this dataset, along with our empirical findings, will stimulate further research and the development of advanced models capable of resolving complex cross-modal understanding problems within long-form audio-visual contexts.

  </details>



- **OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Robotic Manipulation**  
  Yuhang Zheng, Songen Gu, Weize Li, Yupeng Zheng, Yujie Zang, Shuai Tian, Xiang Li, Ruihai Wu, Ce Hao, Chen Gao, et al.  
  _2026-03-19_ · https://arxiv.org/abs/2603.19201v1  
  <details><summary>Abstract</summary>

  Contact-rich manipulation tasks, such as wiping and assembly, require accurate perception of contact forces, friction changes, and state transitions that cannot be reliably inferred from vision alone. Despite growing interest in visuo-tactile manipulation, progress is constrained by two persistent limitations: existing datasets are small in scale and narrow in task coverage, and current methods treat tactile signals as passive observations rather than using them to model contact dynamics or enable closed-loop control explicitly. In this paper, we present \textbf{OmniViTac}, a large-scale visuo-tactile-action dataset comprising $21{,}000+$ trajectories across $86$ tasks and $100+$ objects, organized into six physics-grounded interaction patterns. Building on this dataset, we propose \textbf{OmniVTA}, a world-model-based visuo-tactile manipulation framework that integrates four tightly coupled modules: a self-supervised tactile encoder, a two-stream visuo-tactile world model for predicting short-horizon contact evolution, a contact-aware fusion policy for action generation, and a 60Hz reflexive controller that corrects deviations between predicted and observed tactile signals in a closed loop. Real-robot experiments across all six interaction categories show that OmniVTA outperforms existing methods and generalizes well to unseen objects and geometric configurations, confirming the value of combining predictive contact modeling with high-frequency tactile feedback for contact-rich manipulation. All data, models, and code will be made publicly available on the project website at https://mrsecant.github.io/OmniVTA.

  </details>



- **Reconstruction Matters: Learning Geometry-Aligned BEV Representation through 3D Gaussian Splatting**  
  Yiren Lu, Xin Ye, Burhaneddin Yaman, Jingru Luo, Zhexiao Xiong, Liu Ren, Yu Yin  
  _2026-03-19_ · https://arxiv.org/abs/2603.19193v1  
  <details><summary>Abstract</summary>

  Bird's-Eye-View (BEV) perception serves as a cornerstone for autonomous driving, offering a unified spatial representation that fuses surrounding-view images to enable reasoning for various downstream tasks, such as semantic segmentation, 3D object detection, and motion prediction. However, most existing BEV perception frameworks adopt an end-to-end training paradigm, where image features are directly transformed into the BEV space and optimized solely through downstream task supervision. This formulation treats the entire perception process as a black box, often lacking explicit 3D geometric understanding and interpretability, leading to suboptimal performance. In this paper, we claim that an explicit 3D representation matters for accurate BEV perception, and we propose Splat2BEV, a Gaussian Splatting-assisted framework for BEV tasks. Splat2BEV aims to learn BEV feature representations that are both semantically rich and geometrically precise. We first pre-train a Gaussian generator that explicitly reconstructs 3D scenes from multi-view inputs, enabling the generation of geometry-aligned feature representations. These representations are then projected into the BEV space to serve as inputs for downstream tasks. Extensive experiments on nuScenes and argoverse dataset demonstrate that Splat2BEV achieves state-of-the-art performance and validate the effectiveness of incorporating explicit 3D reconstruction into BEV perception.

  </details>



- **Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models**  
  Aiden Swann, Lachlain McGranahan, Hugo Buurmeijer, Monroe Kennedy, Mac Schwager  
  _2026-03-19_ · https://arxiv.org/abs/2603.19183v1  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models have emerged as a promising approach for general-purpose robot manipulation. However, their generalization is inconsistent: while these models can perform impressively in some settings, fine-tuned variants often fail on novel objects, scenes, and instructions. We apply mechanistic interpretability techniques to better understand the inner workings of VLA models. To probe internal representations, we train Sparse Autoencoders (SAEs) on hidden layer activations of the VLA. SAEs learn a sparse dictionary whose features act as a compact, interpretable basis for the model's computation. We find that the large majority of extracted SAE features correspond to memorized sequences from specific training demonstrations. However, some features correspond to interpretable, general, and steerable motion primitives and semantic properties, offering a promising glimpse toward VLA generalizability. We propose a metric to categorize features according to whether they represent generalizable transferable primitives or episode-specific memorization. We validate these findings through steering experiments on the LIBERO benchmark. We show that individual SAE features causally influence robot behavior. Steering general features induces behaviors consistent with their semantic meaning and can be applied across tasks and scenes. This work provides the first mechanistic evidence that VLAs can learn generalizable features across tasks and scenes. We observe that supervised fine-tuning on small robotics datasets disproportionately amplifies memorization. In contrast, training on larger, more diverse datasets (e.g., DROID) or using knowledge insulation promotes more general features. We provide an open-source codebase and user-friendly interface for activation collection, SAE training, and feature steering. Our project page is located at http://drvla.github.io

  </details>



- **Meanings and Measurements: Multi-Agent Probabilistic Grounding for Vision-Language Navigation**  
  Swagat Padhan, Lakshya Jain, Bhavya Minesh Shah, Omkar Patil, Thao Nguyen, Nakul Gopalan  
  _2026-03-19_ · https://arxiv.org/abs/2603.19166v1  
  <details><summary>Abstract</summary>

  Robots collaborating with humans must convert natural language goals into actionable, physically grounded decisions. For example, executing a command such as "go two meters to the right of the fridge" requires grounding semantic references, spatial relations, and metric constraints within a 3D scene. While recent vision language models (VLMs) demonstrate strong semantic grounding capabilities, they are not explicitly designed to reason about metric constraints in physically defined spaces. In this work, we empirically demonstrate that state-of-the-art VLM-based grounding approaches struggle with complex metric-semantic language queries. To address this limitation, we propose MAPG (Multi-Agent Probabilistic Grounding), an agentic framework that decomposes language queries into structured subcomponents and queries a VLM to ground each component. MAPG then probabilistically composes these grounded outputs to produce metrically consistent, actionable decisions in 3D space. We evaluate MAPG on the HM-EQA benchmark and show consistent performance improvements over strong baselines. Furthermore, we introduce a new benchmark, MAPG-Bench, specifically designed to evaluate metric-semantic goal grounding, addressing a gap in existing language grounding evaluations. We also present a real-world robot demonstration showing that MAPG transfers beyond simulation when a structured scene representation is available.

  </details>



- **ADAPT: Attention Driven Adaptive Prompt Scheduling and InTerpolating Orthogonal Complements for Rare Concepts Generation**  
  Kwanyoung Lee, Hyunwoo Oh, SeungJu Cha, Sungho Koh, Dong-Jin Kim  
  _2026-03-19_ · https://arxiv.org/abs/2603.19157v1  
  <details><summary>Abstract</summary>

  Generating rare compositional concepts in text-to-image synthesis remains a challenge for diffusion models, particularly for attributes that are uncommon in the training data. While recent approaches, such as R2F, address this challenge by utilizing LLM for prompt scheduling, they suffer from inherent variance due to the randomness of language models and suboptimal guidance from iterative text embedding switching. To address these problems, we propose the ADAPT framework, a training-free framework that deterministically plans and semantically aligns prompt schedules, providing consistent guidance to enhance the composition of rare concepts. By leveraging attention scores and orthogonal components, ADAPT significantly enhances compositional generation of rare concepts in the RareBench benchmark without additional training or fine-tuning. Through comprehensive experiments, we demonstrate that ADAPT achieves superior performance in RareBench and accurately reflects the semantic information of rare attributes, providing deterministic and precise control over the generation of rare compositions without compromising visual integrity.

  </details>



- **TAU-R1: Visual Language Model for Traffic Anomaly Understanding**  
  Yuqiang Lin, Kehua Chen, Sam Lockyer, Arjun Yadav, Mingxuan Sui, Shucheng Zhang, Yan Shi, Bingzhang Wang, Yuang Zhang, Markus Zarbock, et al.  
  _2026-03-19_ · https://arxiv.org/abs/2603.19098v1  
  <details><summary>Abstract</summary>

  Traffic Anomaly Understanding (TAU) is important for traffic safety in Intelligent Transportation Systems. Recent vision-language models (VLMs) have shown strong capabilities in video understanding. However, progress on TAU remains limited due to the lack of benchmarks and task-specific methodologies. To address this limitation, we introduce Roundabout-TAU, a dataset constructed from real-world roundabout videos collected in collaboration with the City of Carmel, Indiana. The dataset contains 342 clips and is annotated with more than 2,000 question-answer pairs covering multiple aspects of traffic anomaly understanding. Building on this benchmark, we propose TAU-R1, a two-layer vision-language framework for TAU. The first layer is a lightweight anomaly classifier that performs coarse anomaly categorisation, while the second layer is a larger anomaly reasoner that generates detailed event summaries. To improve task-specific reasoning, we introduce a two-stage training strategy consisting of decomposed-QA-enhanced supervised fine-tuning followed by TAU-GRPO, a GRPO-based post-training method with TAU-specific reward functions. Experimental results show that TAU-R1 achieves strong performance on both anomaly classification and reasoning tasks while maintaining deployment efficiency. The dataset and code are available at: https://github.com/siri-rouser/TAU-R1

  </details>



- **SAVeS: Steering Safety Judgments in Vision-Language Models via Semantic Cues**  
  Carlos Hinojosa, Clemens Grange, Bernard Ghanem  
  _2026-03-19_ · https://arxiv.org/abs/2603.19092v1  
  <details><summary>Abstract</summary>

  Vision-language models (VLMs) are increasingly deployed in real-world and embodied settings where safety decisions depend on visual context. However, it remains unclear which visual evidence drives these judgments. We study whether multimodal safety behavior in VLMs can be steered by simple semantic cues. We introduce a semantic steering framework that applies controlled textual, visual, and cognitive interventions without changing the underlying scene content. To evaluate these effects, we propose SAVeS, a benchmark for situational safety under semantic cues, together with an evaluation protocol that separates behavioral refusal, grounded safety reasoning, and false refusals. Experiments across multiple VLMs and an additional state-of-the-art benchmark show that safety decisions are highly sensitive to semantic cues, indicating reliance on learned visual-linguistic associations rather than grounded visual understanding. We further demonstrate that automated steering pipelines can exploit these mechanisms, highlighting a potential vulnerability in multimodal safety systems.

  </details>



- **Multi-Modal Building Change Detection for Large-Scale Small Changes: Benchmark and Baseline**  
  Ye Wang, Wei Lu, Zhihui You, Keyan Chen, Tongfei Liu, Kaiyu Li, Hongruixuan Chen, Qingling Shu, Sibao Chen  
  _2026-03-19_ · https://arxiv.org/abs/2603.19077v1  
  <details><summary>Abstract</summary>

  Change detection in optical remote sensing imagery is susceptible to illumination fluctuations, seasonal changes, and variations in surface land-cover materials. Relying solely on RGB imagery often produces pseudo-changes and leads to semantic ambiguity in features. Incorporating near-infrared (NIR) information provides heterogeneous physical cues that are complementary to visible light, thereby enhancing the discriminability of building materials and tiny structures while improving detection accuracy. However, existing multi-modal datasets generally lack high-resolution and accurately registered bi-temporal imagery, and current methods often fail to fully exploit the inherent heterogeneity between these modalities. To address these issues, we introduce the Large-scale Small-change Multi-modal Dataset (LSMD), a bi-temporal RGB-NIR building change detection benchmark dataset targeting small changes in realistic scenarios, providing a rigorous testing platform for evaluating multi-modal change detection methods in complex environments. Based on LSMD, we further propose the Multi-modal Spectral Complementarity Network (MSCNet) to achieve effective cross-modal feature fusion. MSCNet comprises three key components: the Neighborhood Context Enhancement Module (NCEM) to strengthen local spatial details, the Cross-modal Alignment and Interaction Module (CAIM) to enable deep interaction between RGB and NIR features, and the Saliency-aware Multisource Refinement Module (SMRM) to progressively refine fused features. Extensive experiments demonstrate that MSCNet effectively leverages multi-modal information and consistently outperforms existing methods under multiple input configurations, validating its efficacy for fine-grained building change detection. The source code will be made publicly available at: https://github.com/AeroVILab-AHU/LSMD

  </details>



- **Fire as a Service: Augmenting Robot Simulators with Thermally and Visually Accurate Fire Dynamics**  
  Anton R. Wagner, Madhan Balaji Rao, Helge Wrede, Sören Pirk, Xuesu Xiao  
  _2026-03-19_ · https://arxiv.org/abs/2603.19063v1  
  <details><summary>Abstract</summary>

  Most existing robot simulators prioritize rigid-body dynamics and photorealistic rendering, but largely neglect the thermally and optically complex phenomena that characterize real-world fire environments. For robots envisioned as future firefighters, this limitation hinders both reliable capability evaluation and the generation of representative training data prior to deployment in hazardous scenarios. To address these challenges, we introduce Fire as a Service (FaaS), a novel, asynchronous co-simulation framework that augments existing robot simulators with high-fidelity and computationally efficient fire simulations. Our pipeline enables robots to experience accurate, multi-species thermodynamic heat transfer and visually consistent volumetric smoke without disrupting high-frequency rigid-body control loops. We demonstrate that our framework can be integrated with diverse robot simulators to generate physically accurate fire behavior, benchmark thermal hazards encountered by robotic platforms, and collect realistic multimodal perceptual data. Crucially, its real-time performance supports human-in-the-loop teleoperation, enabling the successful training of reactive, multimodal policies via Behavioral Cloning. By adding fire dynamics to robot simulations, FaaS provides a scalable pathway toward safer, more reliable deployment of robots in fire scenarios.

  </details>



- **SignAgent: Agentic LLMs for Linguistically-Grounded Sign Language Annotation and Dataset Curation**  
  Oliver Cory, Ozge Mercanoglu Sincan, Richard Bowden  
  _2026-03-19_ · https://arxiv.org/abs/2603.19059v1  
  <details><summary>Abstract</summary>

  This paper introduces SignAgent, a novel agentic framework that utilises Large Language Models (LLMs) for scalable, linguistically-grounded Sign Language (SL) annotation and dataset curation. Traditional computational methods for SLs often operate at the gloss level, overlooking crucial linguistic nuances, while manual linguistic annotation remains a significant bottleneck, proving too slow and expensive for the creation of large-scale, phonologically-aware datasets. SignAgent addresses these challenges through SignAgent Orchestrator, a reasoning LLM that coordinates a suite of linguistic tools, and SignGraph, a knowledge-grounded LLM that provides lexical and linguistic grounding. We evaluate our framework on two downstream annotation tasks. First, on Pseudo-gloss Annotation, where the agent performs constrained assignment, using multi-modal evidence to extract and order suitable gloss labels for signed sequences. Second, on ID Glossing, where the agent detects and refines visual clusters by reasoning over both visual similarity and phonological overlap to correctly identify and group lexical sign variants. Our results demonstrate that our agentic approach achieves strong performance for large-scale, linguistically-aware data annotation and curation.

  </details>



- **TerraScope: Pixel-Grounded Visual Reasoning for Earth Observation**  
  Yan Shu, Bin Ren, Zhitong Xiong, Xiao Xiang Zhu, Begüm Demir, Nicu Sebe, Paolo Rota  
  _2026-03-19_ · https://arxiv.org/abs/2603.19039v1  
  <details><summary>Abstract</summary>

  Vision-language models (VLMs) have shown promise in earth observation (EO), yet they struggle with tasks that require grounding complex spatial reasoning in precise pixel-level visual representations. To address this problem, we introduce TerraScope, a unified VLM that delivers pixel-grounded geospatial reasoning with two key capabilities: (1) modality-flexible reasoning: it handles single-modality inputs (optical or SAR) and adaptively fuses different modalities into the reasoning process when both are available; (2) multi-temporal reasoning: it integrates temporal sequences for change analysis across multiple time points. In addition, we curate Terra-CoT, a large-scale dataset containing 1 million samples with pixel-level masks embedded in reasoning chains across multiple sources. We also propose TerraScope-Bench, the first benchmark for pixel-grounded geospatial reasoning with six sub-tasks that evaluates both answer accuracy and mask quality to ensure authentic pixel-grounded reasoning. Experiments show that TerraScope significantly outperforms existing VLMs on pixel-grounded geospatial reasoning while providing interpretable visual evidence.

  </details>



- **SEM: Sparse Embedding Modulation for Post-Hoc Debiasing of Vision-Language Models**  
  Quentin Guimard, Federico Bartsch, Simone Caldarella, Rahaf Aljundi, Elisa Ricci, Massimiliano Mancini  
  _2026-03-19_ · https://arxiv.org/abs/2603.19028v1  
  <details><summary>Abstract</summary>

  Models that bridge vision and language, such as CLIP, are key components of multimodal AI, yet their large-scale, uncurated training data introduce severe social and spurious biases. Existing post-hoc debiasing methods often operate directly in the dense CLIP embedding space, where bias and task-relevant information are highly entangled. This entanglement limits their ability to remove bias without degrading semantic fidelity. In this work, we propose Sparse Embedding Modulation (SEM), a post-hoc, zero-shot debiasing framework that operates in a Sparse Autoencoder (SAE) latent space. By decomposing CLIP text embeddings into disentangled features, SEM identifies and modulates bias-relevant neurons while preserving query-relevant ones. This enables more precise, non-linear interventions. Across four benchmark datasets and two CLIP backbones, SEM achieves substantial fairness gains in retrieval and zero-shot classification. Our results demonstrate that sparse latent representations provide an effective foundation for post-hoc debiasing of vision-language models.

  </details>



- **CRAFT: Aligning Diffusion Models with Fine-Tuning Is Easier Than You Think**  
  Zening Sun, Zhengpeng Xie, Lichen Bai, Shitong Shao, Shuo Yang, Zeke Xie  
  _2026-03-19_ · https://arxiv.org/abs/2603.18991v1  
  <details><summary>Abstract</summary>

  Aligning Diffusion models has achieved remarkable breakthroughs in generating high-quality, human preference-aligned images. Existing techniques, such as supervised fine-tuning (SFT) and DPO-style preference optimization, have become principled tools for fine-tuning diffusion models. However, SFT relies on high-quality images that are costly to obtain, while DPO-style methods depend on large-scale preference datasets, which are often inconsistent in quality. Beyond data dependency, these methods are further constrained by computational inefficiency. To address these two challenges, we propose Composite Reward Assisted Fine-Tuning (CRAFT), a lightweight yet powerful fine-tuning paradigm that requires significantly reduced training data while maintaining computational efficiency. It first leverages a Composite Reward Filtering (CRF) technique to construct a high-quality and consistent training dataset and then perform an enhanced variant of SFT. We also theoretically prove that CRAFT actually optimizes the lower bound of group-based reinforcement learning, establishing a principled connection between SFT with selected data and reinforcement learning. Our extensive empirical results demonstrate that CRAFT with only 100 samples can easily outperform recent SOTA preference optimization methods with thousands of preference-paired samples. Moreover, CRAFT can even achieve 11-220$\times$ faster convergences than the baseline preference optimization methods, highlighting its extremely high efficiency.

  </details>



- **MERGE: Guided Vision-Language Models for Multi-Actor Event Reasoning and Grounding in Human-Robot Interaction**  
  Joerg Deigmoeller, Nakul Agarwal, Stephan Hasler, Daniel Tanneberg, Anna Belardinelli, Reza Ghoddoosian, Chao Wang, Felix Ocker, Fan Zhang, Behzad Dariush, et al.  
  _2026-03-19_ · https://arxiv.org/abs/2603.18988v1  
  <details><summary>Abstract</summary>

  We introduce MERGE, a system for situational grounding of actors, objects, and events in dynamic human-robot group interactions. Effective collaboration in such settings requires consistent situational awareness, built on persistent representations of people and objects and an episodic abstraction of events. MERGE achieves this by uniquely identifying physical instances of actors (humans or robots) and objects and structuring them into actor-action-object relations, ensuring temporal consistency across interactions. Central to MERGE is the integration of Vision-Language Models (VLMs) guided with a perception pipeline: a lightweight streaming module continuously processes visual input to detect changes and selectively invokes the VLM only when necessary. This decoupled design preserves the reasoning power and zero-shot generalization of VLMs while improving efficiency, avoiding both the high monetary cost and the latency of frame-by-frame captioning that leads to fragmented and delayed outputs. To address the absence of suitable benchmarks for multi-actor collaboration, we introduce the GROUND dataset, which offers fine-grained situational annotations of multi-person and human-robot interactions. On this dataset, our approach improves the average grounding score by a factor of 2 compared to the performance of VLM-only baselines - including GPT-4o, GPT-5 and Gemini 2.5 Flash - while also reducing run-time by a factor of 4. The code and data are available at www.github.com/HRI-EU/merge.

  </details>



- **MultihopSpatial: Multi-hop Compositional Spatial Reasoning Benchmark for Vision-Language Model**  
  Youngwan Lee, Soojin Jang, Yoorhim Cho, Seunghwan Lee, Yong-Ju Lee, Sung Ju Hwang  
  _2026-03-19_ · https://arxiv.org/abs/2603.18892v1  
  <details><summary>Abstract</summary>

  Spatial reasoning is foundational for Vision-Language Models (VLMs), particularly when deployed as Vision-Language-Action (VLA) agents in physical environments. However, existing benchmarks predominantly focus on elementary, single-hop relations, neglecting the multi-hop compositional reasoning and precise visual grounding essential for real-world scenarios. To address this, we introduce MultihopSpatial, offering three key contributions: (1) A comprehensive benchmark designed for multi-hop and compositional spatial reasoning, featuring 1- to 3-hop complex queries across diverse spatial perspectives. (2) Acc@50IoU, a complementary metric that simultaneously evaluates reasoning and visual grounding by requiring both answer selection and precise bounding box prediction - capabilities vital for robust VLA deployment. (3) MultihopSpatial-Train, a dedicated large-scale training corpus to foster spatial intelligence. Extensive evaluation of 37 state-of-the-art VLMs yields eight key insights, revealing that compositional spatial reasoning remains a formidable challenge. Finally, we demonstrate that reinforcement learning post-training on our corpus enhances both intrinsic VLM spatial reasoning and downstream embodied manipulation performance.

  </details>



- **Motion-o: Trajectory-Grounded Video Reasoning**  
  Bishoy Galoaa, Shayda Moezzi, Xiangyu Bai, Sarah Ostadabbas  
  _2026-03-19_ · https://arxiv.org/abs/2603.18856v1  
  <details><summary>Abstract</summary>

  Recent research has made substantial progress on video reasoning, with many models leveraging spatio-temporal evidence chains to strengthen their inference capabilities. At the same time, a growing set of datasets and benchmarks now provides structured annotations designed to support and evaluate such reasoning. However, little attention has been paid to reasoning about \emph{how} objects move between observations: no prior work has articulated the motion patterns by connecting successive observations, leaving trajectory understanding implicit and difficult to verify. We formalize this missing capability as Spatial-Temporal-Trajectory (STT) reasoning and introduce \textbf{Motion-o}, a motion-centric video understanding extension to visual language models that makes trajectories explicit and verifiable. To enable motion reasoning, we also introduce a trajectory-grounding dataset artifact that expands sparse keyframe supervision via augmentation to yield denser bounding box tracks and a stronger trajectory-level training signal. Finally, we introduce Motion Chain of Thought (MCoT), a structured reasoning pathway that makes object trajectories through discrete \texttt{<motion/>} tag summarizing per-object direction, speed, and scale (of velocity) change to explicitly connect grounded observations into trajectories. To train Motion-o, we design a reward function that compels the model to reason directly over visual evidence, all while requiring no architectural modifications. Empirical results demonstrate that Motion-o improves spatial-temporal grounding and trajectory prediction while remaining fully compatible with existing frameworks, establishing motion reasoning as a critical extension for evidence-based video understanding. Code is available at https://github.com/ostadabbas/Motion-o.

  </details>



- **Statistical Characteristic-Guided Denoising for Rapid High-Resolution Transmission Electron Microscopy Imaging**  
  Hesong Li, Ziqi Wu, Ruiwen Shao, Ying Fu  
  _2026-03-19_ · https://arxiv.org/abs/2603.18834v1  
  <details><summary>Abstract</summary>

  High-Resolution Transmission Electron Microscopy (HRTEM) enables atomic-scale observation of nucleation dynamics, which boosts the studies of advanced solid materials. Nonetheless, due to the millisecond-scale rapid change of nucleation, it requires short-exposure rapid imaging, leading to severe noise that obscures atomic positions. In this work, we propose a statistical characteristic-guided denoising network, which utilizes statistical characteristics to guide the denoising process in both spatial and frequency domains. In the spatial domain, we present spatial deviation-guided weighting to select appropriate convolution operations for each spatial position based on deviation characteristic. In the frequency domain, we present frequency band-guided weighting to enhance signals and suppress noise based on band characteristics. We also develop an HRTEM-specific noise calibration method and generate a dataset with disordered structures and realistic HRTEM image noises. It can ensure the denoising performance of models on real images for nucleation observation. Experiments on synthetic and real data show our method outperforms the state-of-the-art methods in HRTEM image denoising, with effectiveness in the localization downstream task. Code will be available at https://github.com/HeasonLee/SCGN.

  </details>



- **Rethinking Uncertainty Quantification and Entanglement in Image Segmentation**  
  Jakob Lønborg Christensen, Vedrana Andersen Dahl, Morten Rieger Hannemose, Anders Bjorholm Dahl, Christian F. Baumgartner  
  _2026-03-19_ · https://arxiv.org/abs/2603.18792v1  
  <details><summary>Abstract</summary>

  Uncertainty quantification (UQ) is crucial in safety-critical applications such as medical image segmentation. Total uncertainty is typically decomposed into data-related aleatoric uncertainty (AU) and model-related epistemic uncertainty (EU). Many methods exist for modeling AU (such as Probabilistic UNet, Diffusion) and EU (such as ensembles, MC Dropout), but it is unclear how they interact when combined. Additionally, recent work has revealed substantial entanglement between AU and EU, undermining the interpretability and practical usefulness of the decomposition. We present a comprehensive empirical study covering a broad range of AU-EU model combinations, propose a metric to quantify uncertainty entanglement, and evaluate both across downstream UQ tasks. For out-of-distribution detection, ensembles exhibit consistently lower entanglement and superior performance. For ambiguity modeling and calibration the best models are dataset-dependent, with softmax/SSN-based methods performing well and Probabilistic UNets being less entangled. A softmax ensemble fares remarkably well on all tasks. Finally, we analyze potential sources of uncertainty entanglement and outline directions for mitigating this effect.

  </details>



- **SEAR: Simple and Efficient Adaptation of Visual Geometric Transformers for RGB+Thermal 3D Reconstruction**  
  Vsevolod Skorokhodov, Chenghao Xu, Shuo Sun, Olga Fink, Malcolm Mielle  
  _2026-03-19_ · https://arxiv.org/abs/2603.18774v1  
  <details><summary>Abstract</summary>

  Foundational feed-forward visual geometry models enable accurate and efficient camera pose estimation and scene reconstruction by learning strong scene priors from massive RGB datasets. However, their effectiveness drops when applied to mixed sensing modalities, such as RGB-thermal (RGB-T) images. We observe that while a visual geometry grounded transformer pretrained on RGB data generalizes well to thermal-only reconstruction, it struggles to align RGB and thermal modalities when processed jointly. To address this, we propose SEAR, a simple yet efficient fine-tuning strategy that adapts a pretrained geometry transformer to multimodal RGB-T inputs. Despite being trained on a relatively small RGB-T dataset, our approach significantly outperforms state-of-the-art methods for 3D reconstruction and camera pose estimation, achieving significant improvements over all metrics (e.g., over 29\% in AUC@30) and delivering higher detail and consistency between modalities with negligible overhead in inference time compared to the original pretrained model. Notably, SEAR enables reliable multimodal pose estimation and reconstruction even under challenging conditions, such as low lighting and dense smoke. We validate our architecture through extensive ablation studies, demonstrating how the model aligns both modalities. Additionally, we introduce a new dataset featuring RGB and thermal sequences captured at different times, viewpoints, and illumination conditions, providing a robust benchmark for future work in multimodal 3D scene reconstruction. Code and models are publicly available at https://www.github.com/Schindler-EPFL-Lab/SEAR.

  </details>



- **EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation**  
  Longfei Liu, Yongjie Hou, Yang Li, Qirui Wang, Youyang Sha, Yongjun Yu, Yinzhi Wang, Peizhe Ru, Xuanlong Yu, Xi Shen  
  _2026-03-19_ · https://arxiv.org/abs/2603.18739v1  
  <details><summary>Abstract</summary>

  Deploying high-performance dense prediction models on resource-constrained edge devices remains challenging due to strict limits on computation and memory. In practice, lightweight systems for object detection, instance segmentation, and pose estimation are still dominated by CNN-based architectures such as YOLO, while compact Vision Transformers (ViTs) often struggle to achieve similarly strong accuracy efficiency tradeoff, even with large scale pretraining. We argue that this gap is largely due to insufficient task specific representation learning in small scale ViTs, rather than an inherent mismatch between ViTs and edge dense prediction. To address this issue, we introduce EdgeCrafter, a unified compact ViT framework for edge dense prediction centered on ECDet, a detection model built from a distilled compact backbone and an edge-friendly encoder decoder design. On the COCO dataset, ECDet-S achieves 51.7 AP with fewer than 10M parameters using only COCO annotations. For instance segmentation, ECInsSeg achieves performance comparable to RF-DETR while using substantially fewer parameters. For pose estimation, ECPose-X reaches 74.8 AP, significantly outperforming YOLO26Pose-X (71.6 AP) despite the latter's reliance on extensive Objects365 pretraining. These results show that compact ViTs, when paired with task-specialized distillation and edge-aware design, can be a practical and competitive option for edge dense prediction. Code is available at: https://intellindust-ai-lab.github.io/projects/EdgeCrafter/

  </details>



- **Click-to-Ask: An AI Live Streaming Assistant with Offline Copywriting and Online Interactive QA**  
  Ruizhi Yu, Keyang Zhong, Peng Liu, Qi Wu, Haoran Zhang, Yanhao Zhang, Chen Chen, Haonan Lu  
  _2026-03-19_ · https://arxiv.org/abs/2603.18649v1  
  <details><summary>Abstract</summary>

  Live streaming commerce has become a prominent form of broadcasting in the modern era. To facilitate more efficient and convenient product promotions for streamers, we present Click-to-Ask, an AI-driven assistant for live streaming commerce with complementary offline and online components. The offline module processes diverse multimodal product information, transforming complex inputs into structured product data and generating compliant promotional copywriting. During live broadcasts, the online module enables real-time responses to viewer inquiries by allowing streamers to click on questions and leveraging both the structured product information generated by the offline module and an event-level historical memory maintained in a streaming architecture. This system significantly reduces the time needed for promotional preparation, enhances content engagement, and enables prompt interaction with audience inquiries, ultimately improving the effectiveness of live streaming commerce. On our collected dataset of TikTok live stream frames, the proposed method achieves a Question Recognition Accuracy of 0.913 and a Response Quality score of 0.876, demonstrating considerable potential for practical application. The video demonstration can be viewed here: https://www.youtube.com/shorts/mWIXK-SWhiE.

  </details>



- **PhysVideo: Physically Plausible Video Generation with Cross-View Geometry Guidance**  
  Cong Wang, Hanxin Zhu, Xiao Tang, Jiayi Luo, Xin Jin, Long Chen, Fei-Yue Wang, Zhibo Chen  
  _2026-03-19_ · https://arxiv.org/abs/2603.18639v1  
  <details><summary>Abstract</summary>

  Recent progress in video generation has led to substantial improvements in visual fidelity, yet ensuring physically consistent motion remains a fundamental challenge. Intuitively, this limitation can be attributed to the fact that real-world object motion unfolds in three-dimensional space, while video observations provide only partial, view-dependent projections of such dynamics. To address these issues, we propose PhysVideo, a two-stage framework that first generates physics-aware orthogonal foreground videos and then synthesizes full videos with background. In the first stage, Phys4View leverages physics-aware attention to capture the influence of physical attributes on motion dynamics, and enhances spatio-temporal consistency by incorporating geometry-enhanced cross-view attention and temporal attention. In the second stage, VideoSyn uses the generated foreground videos as guidance and learns the interactions between foreground dynamics and background context for controllable video synthesis. To support training, we construct PhysMV, a dataset containing 40K scenes, each consisting of four orthogonal viewpoints, resulting in a total of 160K video sequences. Extensive experiments demonstrate that PhysVideo significantly improves physical realism and spatial-temporal coherence over existing video generation methods. Home page: https://anonymous.4open.science/w/Phys4D/.

  </details>



- **GEAR: Geography-knowledge Enhanced Analog Recognition Framework in Extreme Environments**  
  Zelin Liu, Bocheng Li, Yuling Zhou, Xuanting Li, Yixuan Yang, Jing Wang, Weishu Zhao, Xiaofeng Gao  
  _2026-03-19_ · https://arxiv.org/abs/2603.18626v1  
  <details><summary>Abstract</summary>

  The Mariana Trench and the Qinghai-Tibet Plateau exhibit significant similarities in geological origins and microbial metabolic functions. Given that deep-sea biological sampling faces prohibitive costs, recognizing structurally homologous terrestrial analogs of the Mariana Trench on the Qinghai-Tibet Plateau is of great significance. Yet, no existing model adequately addresses cross-domain topographic similarity retrieval, either neglecting geographical knowledge or sacrificing computational efficiency. To address these challenges, we present \underline{\textbf{G}}eography-knowledge \underline{\textbf{E}}nhanced \underline{\textbf{A}}nalog \underline{\textbf{R}}ecognition (\textbf{GEAR}) Framework, a three-stage pipeline designed to efficiently retrieve analogs from 2.5 million square kilometers of the Qinghai-Tibet Plateau: (1) Skeleton guided Screening and Clipping: Recognition of candidate valleys and initial screening based on size and linear morphological criteria. (2) Physics aware Filtering: The Topographic Waveform Comparator (TWC) and Morphological Texture Module (MTM) evaluate the waveform and texture and filter out inconsistent candidate valleys. (3) Graph based Fine Recognition: We design a \underline{\textbf{M}}orphology-integrated \underline{\textbf{S}}iamese \underline{\textbf{G}}raph \underline{\textbf{N}}etwork (\textbf{MSG-Net}) based on geomorphological metrics. Correspondingly, we release an expert-annotated topographic similarity dataset targeting tectonic collision zones. Experiments demonstrate the effectiveness of every stage. Besides, MSG-Net achieved an F1-Score 1.38 percentage points higher than the SOTA baseline. Using features extracted by MSG-Net, we discovered a significant correlation with biological data, providing evidence for future biological analysis.

  </details>



- **GenVideoLens: Where LVLMs Fall Short in AI-Generated Video Detection?**  
  Yueying Zou, Pei Pei Li, Zekun Li, Xinyu Guo, Xing Cui, Huaibo Huang, Ran He  
  _2026-03-19_ · https://arxiv.org/abs/2603.18625v1  
  <details><summary>Abstract</summary>

  In recent years, AI-generated videos have become increasingly realistic and sophisticated. Meanwhile, Large Vision-Language Models (LVLMs) have shown strong potential for detecting such content. However, existing evaluation protocols largely treat the task as a binary classification problem and rely on coarse-grained metrics such as overall accuracy, providing limited insight into where LVLMs succeed or fail. To address this limitation, we introduce GenVideoLens, a fine-grained benchmark that enables dimension-wise evaluation of LVLM capabilities in AI-generated video detection. The benchmark contains 400 highly deceptive AI-generated videos and 100 real videos, annotated by experts across 15 authenticity dimensions covering perceptual, optical, physical, and temporal cues. We evaluate eleven representative LVLMs on this benchmark. Our analysis reveals a pronounced dimensional imbalance. While LVLMs perform relatively well on perceptual cues, they struggle with optical consistency, physical interactions, and temporal-causal reasoning. Model performance also varies substantially across dimensions, with smaller open-source models sometimes outperforming stronger proprietary models on specific authenticity cues. Temporal perturbation experiments further show that current LVLMs make limited use of temporal information. Overall, GenVideoLens provides diagnostic insights into LVLM behavior, revealing key capability gaps and offering guidance for improving future AI-generated video detection systems.

  </details>



- **OpenT2M: No-frill Motion Generation with Open-source,Large-scale, High-quality Data**  
  Bin Cao, Sipeng Zheng, Hao Luo, Boyuan Li, Jing Liu, Zongqing Lu  
  _2026-03-19_ · https://arxiv.org/abs/2603.18623v1  
  <details><summary>Abstract</summary>

  Text-to-motion (T2M) generation aims to create realistic human movements from text descriptions, with promising applications in animation and robotics. Despite recent progress, current T2M models perform poorly on unseen text descriptions due to the small scale and limited diversity of existing motion datasets. To address this problem, we introduce OpenT2M, a million-level, high-quality, and open-source motion dataset containing over 2800 hours of human motion. Each sequence undergoes rigorous quality control through physical feasibility validation and multi-granularity filtering, with detailed second-wise text annotations. We also develop an automated pipeline for creating long-horizon sequences, enabling complex motion generation. Building upon OpenT2M, we introduce MonoFrill, a pretrained motion model that achieves compelling T2M results without complicated designs or technique tricks as "frills". Its core component is 2D-PRQ, a novel motion tokenizer that captures spatiotemporal dependencies by dividing the human body into biology parts. Experiments show that OpenT2M significantly improves generalization of existing T2M models, while 2D-PRQ achieves superior reconstruction and strong zero-shot performance. We expect OpenT2M and MonoFrill will advance the T2M field by addressing longstanding data quality and benchmarking challenges.

  </details>



- **Benchmarking CNN-based Models against Transformer-based Models for Abdominal Multi-Organ Segmentation on the RATIC Dataset**  
  Lukas Bayer, Sheethal Bhat, Andreas Maier  
  _2026-03-19_ · https://arxiv.org/abs/2603.18616v1  
  <details><summary>Abstract</summary>

  Accurate multi-organ segmentation in abdominal CT scans is essential for computer-aided diagnosis and treatment. While convolutional neural networks (CNNs) have long been the standard approach in medical image segmentation, transformer-based architectures have recently gained attention due to their ability to model long-range dependencies. In this study, we systematically benchmark the three hybrid transformer-based models UNETR, SwinUNETR, and UNETR++ against a strong CNN baseline, SegResNet, for volumetric multi-organ segmentation on the heterogeneous RATIC dataset. The dataset comprises 206 annotated CT scans from 23 institutions worldwide, covering five abdominal organs. All models were trained and evaluated under identical preprocessing and training conditions using the Dice Similarity Coefficient (DSC) as the primary metric. The results show that the CNN-based SegResNet achieves the highest overall performance, outperforming all hybrid transformer-based models across all organs. Among the transformer-based approaches, UNETR++ delivers the most competitive results, while UNETR demonstrates notably faster convergence with fewer training iterations. These findings suggest that, for small- to medium-sized heterogeneous datasets, well-optimized CNN architectures remain highly competitive and may outperform hybrid transformer-based designs.

  </details>



- **Cross-Modal Rationale Transfer for Explainable Humanitarian Classification on Social Media**  
  Thi Huyen Nguyen, Koustav Rudra, Wolfgang Nejdl  
  _2026-03-19_ · https://arxiv.org/abs/2603.18611v1  
  <details><summary>Abstract</summary>

  Advances in social media data dissemination enable the provision of real-time information during a crisis. The information comes from different classes, such as infrastructure damages, persons missing or stranded in the affected zone, etc. Existing methods attempted to classify text and images into various humanitarian categories, but their decision-making process remains largely opaque, which affects their deployment in real-life applications. Recent work has sought to improve transparency by extracting textual rationales from tweets to explain predicted classes. However, such explainable classification methods have mostly focused on text, rather than crisis-related images. In this paper, we propose an interpretable-by-design multimodal classification framework. Our method first learns the joint representation of text and image using a visual language transformer model and extracts text rationales. Next, it extracts the image rationales via the mapping with text rationales. Our approach demonstrates how to learn rationales in one modality from another through cross-modal rationale transfer, which saves annotation effort. Finally, tweets are classified based on extracted rationales. Experiments are conducted over CrisisMMD benchmark dataset, and results show that our proposed method boosts the classification Macro-F1 by 2-35% while extracting accurate text tokens and image patches as rationales. Human evaluation also supports the claim that our proposed method is able to retrieve better image rationale patches (12%) that help to identify humanitarian classes. Our method adapts well to new, unseen datasets in zero-shot mode, achieving an accuracy of 80%.

  </details>



- **myMNIST: Benchmark of PETNN, KAN, and Classical Deep Learning Models for Burmese Handwritten Digit Recognition**  
  Ye Kyaw Thu, Thazin Myint Oo, Thepchai Supnithi  
  _2026-03-19_ · https://arxiv.org/abs/2603.18597v1  
  <details><summary>Abstract</summary>

  We present the first systematic benchmark on myMNIST (formerly BHDD), a publicly available Burmese handwritten digit dataset important for Myanmar NLP/AI research. We evaluate eleven architectures spanning classical deep learning models (Multi-Layer Perceptron, Convolutional Neural Network, Long Short-Term Memory, Gated Recurrent Unit, Transformer), recent alternatives (FastKAN, EfficientKAN), an energy-based model (JEM), and physics-inspired PETNN variants (Sigmoid, GELU, SiLU). Using Precision, Recall, F1-Score, and Accuracy as evaluation metrics, our results show that the CNN remains a strong baseline, achieving the best overall scores (F1 = 0.9959, Accuracy = 0.9970). The PETNN (GELU) model closely follows (F1 = 0.9955, Accuracy = 0.9966), outperforming LSTM, GRU, Transformer, and KAN variants. JEM, representing energy-based modeling, performs competitively (F1 = 0.9944, Accuracy = 0.9958). KAN-based models (FastKAN, EfficientKAN) trail the top performers but provide a meaningful alternative baseline (Accuracy ~0.992). These findings (i) establish reproducible baselines for myMNIST across diverse modeling paradigms, (ii) highlight PETNN's strong performance relative to classical and Transformer-based models, and (iii) quantify the gap between energy-inspired PETNNs and a true energy-based model (JEM). We release this benchmark to facilitate future research on Myanmar digit recognition and to encourage broader evaluation of emerging architectures on regional scripts.

  </details>



- **AU Codes, Language, and Synthesis: Translating Anatomy to Text for Facial Behavior Synthesis**  
  Jiahe Wang, Cong Liang, Xuandong Huang, Yuxin Wang, Xin Yun, Yi Wu, Yanan Chang, Shangfei Wang  
  _2026-03-19_ · https://arxiv.org/abs/2603.18588v1  
  <details><summary>Abstract</summary>

  Facial behavior synthesis remains a critical yet underexplored challenge. While text-to-face models have made progress, they often rely on coarse emotion categories, which lack the nuance needed to capture the full spectrum of human nonverbal communication. Action Units (AUs) provide a more precise and anatomically grounded alternative. However, current AU-based approaches typically encode AUs as one-hot vectors, modeling compound expressions as simple linear combinations of individual AUs. This linearity becomes problematic when handling conflicting AUs--defined as those which activate the same facial muscle with opposing actions. Such cases lead to anatomically implausible artifacts and unnatural motion superpositions. To address this, we propose a novel method that represents facial behavior through natural language descriptions of AUs. This approach preserves the expressiveness of the AU framework while enabling explicit modeling of complex and conflicting AUs. It also unlocks the potential of modern text-to-image models for high-fidelity facial synthesis. Supporting this direction, we introduce BP4D-AUText, the first large-scale text-image paired dataset for complex facial behavior. It is synthesized by applying a rule-based Dynamic AU Text Processor to the BP4D and BP4D+ datasets. We further propose VQ-AUFace, a generative model that leverages facial structural priors to synthesize realistic and diverse facial behaviors from text. Extensive quantitative experiments and user studies demonstrate that our approach significantly outperforms existing methods. It excels in generating facial expressions that are anatomically plausible, behaviorally rich, and perceptually convincing, particularly under challenging conditions involving conflicting AUs.

  </details>



- **UEPS: Robust and Efficient MRI Reconstruction**  
  Xiang Zhou, Hong Shang, Zijian Zhan, Tianyu He, Jintao Meng, Dong Liang  
  _2026-03-19_ · https://arxiv.org/abs/2603.18572v1  
  <details><summary>Abstract</summary>

  Deep unrolled models (DUMs) have become the state of the art for accelerated MRI reconstruction, yet their robustness under domain shift remains a critical barrier to clinical adoption. In this work, we identify coil sensitivity map (CSM) estimation as the primary bottleneck limiting generalization. To address this, we propose UEPS, a novel DUM architecture featuring three key innovations: (i) an Unrolled Expanded (UE) design that eliminates CSM dependency by reconstructing each coil independently; (ii) progressive resolution, which leverages k-space-to-image mapping for efficient coarse-to-fine refinement; and (iii) sparse attention tailored to MRI's 1D undersampling nature. These physics-grounded designs enable simultaneous gains in robustness and computational efficiency. We construct a large-scale zero-shot transfer benchmark comprising 10 out-of-distribution test sets spanning diverse clinical shifts -- anatomy, view, contrast, vendor, field strength, and coil configurations. Extensive experiments demonstrate that UEPS consistently and substantially outperforms existing DUM, end-to-end, diffusion, and untrained methods across all OOD tests, achieving state-of-the-art robustness with low-latency inference suitable for real-time deployment.

  </details>



- **CausalVAD: De-confounding End-to-End Autonomous Driving via Causal Intervention**  
  Jiacheng Tang, Zhiyuan Zhou, Zhuolin He, Jia Zhang, Kai Zhang, Jian Pu  
  _2026-03-19_ · https://arxiv.org/abs/2603.18561v1  
  <details><summary>Abstract</summary>

  Planning-oriented end-to-end driving models show great promise, yet they fundamentally learn statistical correlations instead of true causal relationships. This vulnerability leads to causal confusion, where models exploit dataset biases as shortcuts, critically harming their reliability and safety in complex scenarios. To address this, we introduce CausalVAD, a de-confounding training framework that leverages causal intervention. At its core, we design the sparse causal intervention scheme (SCIS), a lightweight, plug-and-play module to instantiate the backdoor adjustment theory in neural networks. SCIS constructs a dictionary of prototypes representing latent driving contexts. It then uses this dictionary to intervene on the model's sparse vectorized queries. This step actively eliminates spurious associations induced by confounders, thereby eliminating spurious factors from the representations for downstream tasks. Extensive experiments on benchmarks like nuScenes show CausalVAD achieves state-of-the-art planning accuracy and safety. Furthermore, our method demonstrates superior robustness against both data bias and noisy scenarios configured to induce causal confusion.

  </details>



- **HEP Statistical Inference for UAV Fault Detection: CLs, LRT, and SBI Applied to Blade Damage**  
  Khushiyant  
  _2026-03-19_ · https://arxiv.org/abs/2603.18546v1  
  <details><summary>Abstract</summary>

  This paper transfers three statistical methods from particle physics to multirotor propeller fault detection: the likelihood ratio test (LRT) for binary detection, the CLs modified frequentist method for false alarm rate control, and sequential neural posterior estimation (SNPE) for quantitative fault characterization. Operating on spectral features tied to rotor harmonic physics, the system returns three outputs: binary detection, controlled false alarm rates, and calibrated posteriors over fault severity and motor location. On UAV-FD, a hexarotor dataset of 18 real flights with 5% and 10% blade damage, leave-one-flight-out cross-validation gives AUC 0.862 +/- 0.007 (95% CI: 0.849--0.876), outperforming CUSUM (0.708 +/- 0.010), autoencoder (0.753 +/- 0.009), and LSTM autoencoder (0.551). At 5% false alarm rate the system detects 93% of significant and 81% of subtle blade damage. On PADRE, a quadrotor platform, AUC reaches 0.986 after refitting only the generative models. SNPE gives a full posterior over fault severity (90% credible interval coverage 92--100%, MAE 0.012), so the output includes uncertainty rather than just a point estimate or fault flag. Per-flight sequential detection achieves 100% fault detection with 94% overall accuracy.

  </details>



- **SCISSR: Scribble-Conditioned Interactive Surgical Segmentation and Refinement**  
  Haonan Ping, Jian Jiang, Cheng Yuan, Qizhen Sun, Lv Wu, Yutong Ban  
  _2026-03-19_ · https://arxiv.org/abs/2603.18544v1  
  <details><summary>Abstract</summary>

  Accurate segmentation of tissues and instruments in surgical scenes is annotation-intensive due to irregular shapes, thin structures, specularities, and frequent occlusions. While SAM models support point, box, and mask prompts, points are often too sparse and boxes too coarse to localize such challenging targets. We present SCISSR, a scribble-promptable framework for interactive surgical scene segmentation. It introduces a lightweight Scribble Encoder that converts freehand scribbles into dense prompt embeddings compatible with the mask decoder, enabling iterative refinement for a target object by drawing corrective strokes on error regions. Because all added modules (the Scribble Encoder, Spatial Gated Fusion, and LoRA adapters) interact with the backbone only through its standard embedding interfaces, the framework is not tied to a single model: we build on SAM 2 in this work, yet the same components transfer to other prompt-driven segmentation architectures such as SAM 3 without structural modification. To preserve pre-trained capabilities, we train only these lightweight additions while keeping the remaining backbone frozen. Experiments on EndoVis 2018 demonstrate strong in-domain performance, while evaluation on the out-of-distribution CholecSeg8k further confirms robustness across surgical domains. SCISSR achieves 95.41% Dice on EndoVis 2018 with five interaction rounds and 96.30% Dice on CholecSeg8k with three interaction rounds, outperforming iterative point prompting on both benchmarks.

  </details>



- **NymeriaPlus: Enriching Nymeria Dataset with Additional Annotations and Data**  
  Daniel DeTone, Federica Bogo, Eric-Tuan Le, Duncan Frost, Julian Straub, Yawar Siddiqui, Yuting Ye, Jakob Engel, Richard Newcombe, Lingni Ma  
  _2026-03-19_ · https://arxiv.org/abs/2603.18496v1  
  <details><summary>Abstract</summary>

  The Nymeria Dataset, released in 2024, is a large-scale collection of in-the-wild human activities captured with multiple egocentric wearable devices that are spatially localized and temporally synchronized. It provides body-motion ground truth recorded with a motion-capture suit, device trajectories, semi-dense 3D point clouds, and in-context narrations. In this paper, we upgrade Nymeria and introduce NymeriaPlus. NymeriaPlus features: (1) improved human motion in Momentum Human Rig (MHR) and SMPL formats; (2) dense 3D and 2D bounding box annotations for indoor objects and structural elements; (3) instance-level 3D object reconstructions; and (4) additional modalities e.g., basemap recordings, audio, and wristband videos. By consolidating these complementary modalities and annotations into a single, coherent benchmark, NymeriaPlus strengthens Nymeria into a more powerful in-the-wild egocentric dataset. We expect NymeriaPlus to bridge a key gap in existing egocentric resources and to support a broader range of research, including unique explorations of multimodal learning for embodied AI.

  </details>



- **TexEditor: Structure-Preserving Text-Driven Texture Editing**  
  Bo Zhao, Yihang Liu, Chenfeng Zhang, Huan Yang, Kun Gai, Wei Ji  
  _2026-03-19_ · https://arxiv.org/abs/2603.18488v1  
  <details><summary>Abstract</summary>

  Text-guided texture editing aims to modify object appearance while preserving the underlying geometric structure. However, our empirical analysis reveals that even SOTA editing models frequently struggle to maintain structural consistency during texture editing, despite the intended changes being purely appearance-related. Motivated by this observation, we jointly enhance structure preservation from both data and training perspectives, and build TexEditor, a dedicated texture editing model based on Qwen-Image-Edit-2509. Firstly, we construct TexBlender, a high-quality SFT dataset generated with Blender, which provides strong structural priors for a cold start. Sec- ondly, we introduce StructureNFT, a RL-based approach that integrates structure-preserving losses to transfer the structural priors learned during SFT to real-world scenes. Moreover, due to the limited realism and evaluation coverage of existing benchmarks, we introduce TexBench, a general-purpose real-world benchmark for text-guided texture editing. Extensive experiments on existing Blender-based texture benchmarks and our TexBench show that TexEditor consistently outperforms strong baselines such as Nano Banana Pro. In addition, we assess TexEditor on the general purpose benchmark ImgEdit to validate its generalization. Our code and data are available at https://github.com/KlingAIResearch/TexEditor.

  </details>



- **Do Vision Language Models Understand Human Engagement in Games?**  
  Ziyi Wang, Qizan Guo, Rishitosh Singh, Xiyang Hu  
  _2026-03-19_ · https://arxiv.org/abs/2603.18480v1  
  <details><summary>Abstract</summary>

  Inferring human engagement from gameplay video is important for game design and player-experience research, yet it remains unclear whether vision--language models (VLMs) can infer such latent psychological states from visual cues alone. Using the GameVibe Few-Shot dataset across nine first-person shooter games, we evaluate three VLMs under six prompting strategies, including zero-shot prediction, theory-guided prompts grounded in Flow, GameFlow, Self-Determination Theory, and MDA, and retrieval-augmented prompting. We consider both pointwise engagement prediction and pairwise prediction of engagement change between consecutive windows. Results show that zero-shot VLM predictions are generally weak and often fail to outperform simple per-game majority-class baselines. Memory- or retrieval-augmented prompting improves pointwise prediction in some settings, whereas pairwise prediction remains consistently difficult across strategies. Theory-guided prompting alone does not reliably help and can instead reinforce surface-level shortcuts. These findings suggest a perception--understanding gap in current VLMs: although they can recognize visible gameplay cues, they still struggle to robustly infer human engagement across games.

  </details>



- **Cognitive Mismatch in Multimodal Large Language Models for Discrete Symbol Understanding**  
  Yinghui Li, Jiayi Kuang, Peng Xing, Daixian Liu, Junnan Dong, Shu-Yu Guo, Yangning Li, Qingyu Zhou, Wenhao Jiang, Hai-Tao Zheng, et al.  
  _2026-03-19_ · https://arxiv.org/abs/2603.18472v1  
  <details><summary>Abstract</summary>

  While Multimodal Large Language Models (MLLMs) have achieved remarkable success in interpreting natural scenes, their ability to process discrete symbols -- the fundamental building blocks of human cognition -- remains a critical open question. Unlike continuous visual data, symbols such as mathematical formulas, chemical structures, and linguistic characters require precise, deeper interpretation. This paper introduces a comprehensive benchmark to evaluate how top-tier MLLMs navigate these "discrete semantic spaces" across five domains: language, culture, mathematics, physics, and chemistry. Our investigation uncovers a counterintuitive phenomenon: models often fail at basic symbol recognition yet succeed in complex reasoning tasks, suggesting they rely on linguistic probability rather than true visual perception. By exposing this "cognitive mismatch", we highlight a significant gap in current AI capabilities: the struggle to truly perceive and understand the symbolic languages that underpin scientific discovery and abstract thought. This work offers a roadmap for developing more rigorous, human-aligned intelligent systems.

  </details>



- **Recolour What Matters: Region-Aware Colour Editing via Token-Level Diffusion**  
  Yuqi Yang, Dongliang Chang, Yijia Ling, Ruoyi Du, Zhanyu Ma  
  _2026-03-19_ · https://arxiv.org/abs/2603.18466v1  
  <details><summary>Abstract</summary>

  Colour is one of the most perceptually salient yet least controllable attributes in image generation. Although recent diffusion models can modify object colours from user instructions, their results often deviate from the intended hue, especially for fine-grained and local edits. Early text-driven methods rely on discrete language descriptions that cannot accurately represent continuous chromatic variations. To overcome this limitation, we propose ColourCrafter, a unified diffusion framework that transforms colour editing from global tone transfer into a structured, region-aware generation process. Unlike traditional colour driven methods, ColourCrafter performs token-level fusion of RGB colour tokens and image tokens in latent space, selectively propagating colour information to semantically relevant regions while preserving structural fidelity. A perceptual Lab-space Loss further enhances pixel-level precision by decoupling luminance and chrominance and constraining edits within masked areas. Additionally, we build ColourfulSet, a largescale dataset of high-quality image pairs with continuous and diverse colour variations. Extensive experiments demonstrate that ColourCrafter achieves state-of-the-art colour accuracy, controllability and perceptual fidelity in fine-grained colour editing. Our project is available at https://yangyuqi317.github.io/ColourCrafter.github.io/.

  </details>



- **MedQ-UNI: Toward Unified Medical Image Quality Assessment and Restoration via Vision-Language Modeling**  
  Jiyao Liu, Junzhi Ning, Wanying Qu, Lihao Liu, Chenglong Ma, Junjun He, Ningsheng Xu  
  _2026-03-19_ · https://arxiv.org/abs/2603.18465v1  
  <details><summary>Abstract</summary>

  Existing medical image restoration (Med-IR) methods are typically modality-specific or degradation-specific, failing to generalize across the heterogeneous degradations encountered in clinical practice. We argue this limitation stems from the isolation of Med-IR from medical image quality assessment (Med-IQA), as restoration models without explicit quality understanding struggle to adapt to diverse degradation types across modalities. To address these challenges, we propose MedQ-UNI, a unified vision-language model that follows an assess-then-restore paradigm, explicitly leveraging Med-IQA to guide Med-IR across arbitrary modalities and degradation types. MedQ-UNI adopts a multimodal autoregressive dual-expert architecture with shared attention: a quality assessment expert first identifies degradation issues through structured natural language descriptions, and a restoration expert then conditions on these descriptions to perform targeted image restoration. To support this paradigm, we construct a large-scale dataset of approximately 50K paired samples spanning three imaging modalities and five restoration tasks, each annotated with structured quality descriptions for joint Med-IQA and Med-IR training, along with a 2K-sample benchmark for evaluation. Extensive experiments demonstrate that a single MedQ-UNI model, without any task-specific adaptation, achieves state-of-the-art restoration performance across all tasks while generating superior descriptions, confirming that explicit quality understanding meaningfully improves restoration fidelity and interpretability.

  </details>



- **Interpretable Prostate Cancer Detection using a Small Cohort of MRI Images**  
  Vahid Monfared, Mohammad Hadi Gharib, Ali Sabri, Maryam Shahali, Farid Rashidi, Amit Mehta, Reza Rawassizadeh  
  _2026-03-19_ · https://arxiv.org/abs/2603.18460v1  
  <details><summary>Abstract</summary>

  Prostate cancer is a leading cause of mortality in men, yet interpretation of T2-weighted prostate MRI remains challenging due to subtle and heterogeneous lesions. We developed an interpretable framework for automatic cancer detection using a small dataset of 162 T2-weighted images (102 cancer, 60 normal), addressing data scarcity through transfer learning and augmentation. We performed a comprehensive comparison of Vision Transformers (ViT, Swin), CNNs (ResNet18), and classical methods (Logistic Regression, SVM, HOG+SVM). Transfer-learned ResNet18 achieved the best performance (90.9% accuracy, 95.2% sensitivity, AUC 0.905) with only 11M parameters, while Vision Transformers showed lower performance despite substantially higher complexity. Notably, HOG+SVM achieved comparable accuracy (AUC 0.917), highlighting the effectiveness of handcrafted features in small datasets. Unlike state-of-the-art approaches relying on biparametric MRI (T2+DWI) and large cohorts, our method achieves competitive performance using only T2-weighted images, reducing acquisition complexity and computational cost. In a reader study of 22 cases, five radiologists achieved a mean sensitivity of 67.5% (Fleiss Kappa = 0.524), compared to 95.2% for the AI model, suggesting potential for AI-assisted screening to reduce missed cancers and improve consistency. Code and data are publicly available.

  </details>



- **SODIUM: From Open Web Data to Queryable Databases**  
  Chuxuan Hu, Philip Li, Maxwell Yang, Daniel Kang  
  _2026-03-19_ · https://arxiv.org/abs/2603.18447v1  
  <details><summary>Abstract</summary>

  During research, domain experts often ask analytical questions whose answers require integrating data from a wide range of web sources. Thus, they must spend substantial effort searching, extracting, and organizing raw data before analysis can begin. We formalize this process as the SODIUM task, where we conceptualize open domains such as the web as latent databases that must be systematically instantiated to support downstream querying. Solving SODIUM requires (1) conducting in-depth and specialized exploration of the open web, which is further strengthened by (2) exploiting structural correlations for systematic information extraction and (3) integrating collected information into coherent, queryable database instances. To quantify the challenges in automating SODIUM, we construct SODIUM-Bench, a benchmark of 105 tasks derived from published academic papers across 6 domains, where systems are tasked with exploring the open web to collect and aggregate data from diverse sources into structured tables. Existing systems struggle with SODIUM tasks: we evaluate 6 advanced AI agents on SODIUM-Bench, with the strongest baseline achieving only 46.5% accuracy. To bridge this gap, we develop SODIUM-Agent, a multi-agent system composed of a web explorer and a cache manager. Powered by our proposed ATP-BFS algorithm and optimized through principled management of cached sources and navigation paths, SODIUM-Agent conducts deep and comprehensive web exploration and performs structurally coherent information extraction. SODIUM-Agent achieves 91.1% accuracy on SODIUM-Bench, outperforming the strongest baseline by approximately 2 times and the weakest by up to 73 times.

  </details>



- **AndroTMem: From Interaction Trajectories to Anchored Memory in Long-Horizon GUI Agents**  
  Yibo Shi, Jungang Li, Linghao Zhang, Zihao Dongfang, Biao Wu, Sicheng Tao, Yibo Yan, Chenxi Qin, Weiting Liu, Zhixin Lin, et al.  
  _2026-03-19_ · https://arxiv.org/abs/2603.18429v1  
  <details><summary>Abstract</summary>

  Long-horizon GUI agents are a key step toward real-world deployment, yet effective interaction memory under prevailing paradigms remains under-explored. Replaying full interaction sequences is redundant and amplifies noise, while summaries often erase dependency-critical information and traceability. We present AndroTMem, a diagnostic framework for anchored memory in long-horizon Android GUI agents. Its core benchmark, AndroTMem-Bench, comprises 1,069 tasks with 34,473 interaction steps (avg. 32.1 per task, max. 65). We evaluate agents with TCR (Task Complete Rate), focusing on tasks whose completion requires carrying forward critical intermediate state; AndroTMem-Bench is designed to enforce strong step-to-step causal dependencies, making sparse yet essential intermediate states decisive for downstream actions and centering interaction memory in evaluation. Across open- and closed-source GUI agents, we observe a consistent pattern: as interaction sequences grow longer, performance drops are driven mainly by within-task memory failures, not isolated perception errors or local action mistakes. Guided by this diagnosis, we propose Anchored State Memory (ASM), which represents interaction sequences as a compact set of causally linked intermediate-state anchors to enable subgoal-targeted retrieval and attribution-aware decision making. Across multiple settings and 12 evaluated GUI agents, ASM consistently outperforms full-sequence replay and summary-based baselines, improving TCR by 5%-30.16% and AMS by 4.93%-24.66%, indicating that anchored, structured memory effectively mitigates the interaction-memory bottleneck in long-horizon GUI tasks. The code, benchmark, and related resources are publicly available at [https://github.com/CVC2233/AndroTMem](https://github.com/CVC2233/AndroTMem).

  </details>



- **R&D: Balancing Reliability and Diversity in Synthetic Data Augmentation for Semantic Segmentation**  
  Huy Che, Dinh-Duy Phan, Duc-Khai Lam  
  _2026-03-19_ · https://arxiv.org/abs/2603.18427v1  
  <details><summary>Abstract</summary>

  Collecting and annotating datasets for pixel-level semantic segmentation tasks are highly labor-intensive. Data augmentation provides a viable solution by enhancing model generalization without additional real-world data collection. Traditional augmentation techniques, such as translation, scaling, and color transformations, create geometric variations but fail to generate new structures. While generative models have been employed to extend semantic information of datasets, they often struggle to maintain consistency between the original and generated images, particularly for pixel-level tasks. In this work, we propose a novel synthetic data augmentation pipeline that integrates controllable diffusion models. Our approach balances diversity and reliability data, effectively bridging the gap between synthetic and real data. We utilize class-aware prompting and visual prior blending to improve image quality further, ensuring precise alignment with segmentation labels. By evaluating benchmark datasets such as PASCAL VOC and BDD100K, we demonstrate that our method significantly enhances semantic segmentation performance, especially in data-scarce scenarios, while improving model robustness in real-world applications. Our code is available at \href{https://github.com/chequanghuy/Enhanced-Generative-Data-Augmentation-for-Semantic-Segmentation-via-Stronger-Guidance}{https://github.com/chequanghuy/Enhanced-Generative-Data-Augmentation-for-Semantic-Segmentation-via-Stronger-Guidance}.

  </details>



- **SynQ: Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning**  
  Minjun Kim, Jongjin Kim, U Kang  
  _2026-03-19_ · https://arxiv.org/abs/2603.18423v1  
  <details><summary>Abstract</summary>

  How can we accurately quantize a pre-trained model without any data? Quantization algorithms are widely used for deploying neural networks on resource-constrained edge devices. Zero-shot Quantization (ZSQ) addresses the crucial and practical scenario where training data are inaccessible for privacy or security reasons. However, three significant challenges hinder the performance of existing ZSQ methods: 1) noise in the synthetic dataset, 2) predictions based on off-target patterns, and the 3) misguidance by erroneous hard labels. In this paper, we propose SynQ (Synthesis-aware Fine-tuning for Zero-shot Quantization), a carefully designed ZSQ framework to overcome the limitations of existing methods. SynQ minimizes the noise from the generated samples by exploiting a low-pass filter. Then, SynQ trains the quantized model to improve accuracy by aligning its class activation map with the pre-trained model. Furthermore, SynQ mitigates misguidance from the pre-trained model's error by leveraging only soft labels for difficult samples. Extensive experiments show that SynQ provides the state-of-the-art accuracy, over existing ZSQ methods.

  </details>



- **Mind the Rarities: Can Rare Skin Diseases Be Reliably Diagnosed via Diagnostic Reasoning?**  
  Yang Liu, Jiyao Yang, Hongjin Zhao, Xiaoyong Li, Yanzhe Ji, Xingjian Li, Runmin Jiang, Tianyang Wang, Saeed Anwar, Dongwoo Kim, et al.  
  _2026-03-19_ · https://arxiv.org/abs/2603.18418v1  
  <details><summary>Abstract</summary>

  Large vision-language models (LVLMs) demonstrate strong performance in dermatology; however, evaluating diagnostic reasoning for rare conditions remains largely unexplored. Existing benchmarks focus on common diseases and assess only final accuracy, overlooking the clinical reasoning process, which is critical for complex cases. We address this gap by constructing DermCase, a long-context benchmark derived from peer-reviewed case reports. Our dataset contains 26,030 multi-modal image-text pairs and 6,354 clinically challenging cases, each annotated with comprehensive clinical information and step-by-step reasoning chains. To enable reliable evaluation, we establish DermLIP-based similarity metrics that achieve stronger alignment with dermatologists for assessing differential diagnosis quality. Benchmarking 22 leading LVLMs exposes significant deficiencies across diagnosis accuracy, differential diagnosis, and clinical reasoning. Fine-tuning experiments demonstrate that instruction tuning substantially improves performance while Direct Preference Optimization (DPO) yields minimal gains. Systematic error analysis further reveals critical limitations in current models' reasoning capabilities.

  </details>



- **Inst4DGS: Instance-Decomposed 4D Gaussian Splatting with Multi-Video Label Permutation Learning**  
  Yonghan Lee, Dinesh Manocha  
  _2026-03-19_ · https://arxiv.org/abs/2603.18402v1  
  <details><summary>Abstract</summary>

  We present Inst4DGS, an instance-decomposed 4D Gaussian Splatting (4DGS) approach with long-horizon per-Gaussian trajectories. While dynamic 4DGS has advanced rapidly, instance-decomposed 4DGS remains underexplored, largely due to the difficulty of associating inconsistent instance labels across independently segmented multi-view videos. We address this challenge by introducing per-video label-permutation latents that learn cross-video instance matches through a differentiable Sinkhorn layer, enabling direct multi-view supervision with consistent identity preservation. This explicit label alignment yields sharp decision boundaries and temporally stable identities without identity drift. To further improve efficiency, we propose instance-decomposed motion scaffolds that provide low-dimensional motion bases per object for long-horizon trajectory optimization. Experiments on Panoptic Studio and Neural3DV show that Inst4DGS jointly supports tracking and instance decomposition while achieving state-of-the-art rendering and segmentation quality. On the Panoptic Studio dataset, Inst4DGS improves PSNR from 26.10 to 28.36, and instance mIoU from 0.6310 to 0.9129, over the strongest baseline.

  </details>


