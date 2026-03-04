# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-03-04 07:04 UTC_

Total papers shown: **50**


---

- **LoGeR: Long-Context Geometric Reconstruction with Hybrid Memory**  
  Junyi Zhang, Charles Herrmann, Junhwa Hur, Chen Sun, Ming-Hsuan Yang, Forrester Cole, Trevor Darrell, Deqing Sun  
  _2026-03-03_ · https://arxiv.org/abs/2603.03269v1  
  <details><summary>Abstract</summary>

  Feedforward geometric foundation models achieve strong short-window reconstruction, yet scaling them to minutes-long videos is bottlenecked by quadratic attention complexity or limited effective memory in recurrent designs. We present LoGeR (Long-context Geometric Reconstruction), a novel architecture that scales dense 3D reconstruction to extremely long sequences without post-optimization. LoGeR processes video streams in chunks, leveraging strong bidirectional priors for high-fidelity intra-chunk reasoning. To manage the critical challenge of coherence across chunk boundaries, we propose a learning-based hybrid memory module. This dual-component system combines a parametric Test-Time Training (TTT) memory to anchor the global coordinate frame and prevent scale drift, alongside a non-parametric Sliding Window Attention (SWA) mechanism to preserve uncompressed context for high-precision adjacent alignment. Remarkably, this memory architecture enables LoGeR to be trained on sequences of 128 frames, and generalize up to thousands of frames during inference. Evaluated across standard benchmarks and a newly repurposed VBR dataset with sequences of up to 19k frames, LoGeR substantially outperforms prior state-of-the-art feedforward methods--reducing ATE on KITTI by over 74%--and achieves robust, globally consistent reconstruction over unprecedented horizons.

  </details>



- **UniG2U-Bench: Do Unified Models Advance Multimodal Understanding?**  
  Zimo Wen, Boxiu Li, Wanbo Zhang, Junxiang Lei, Xiaoyu Chen, Yijia Fan, Qi Zhang, Yujiang Wang, Lili Qiu, Bo Li, et al.  
  _2026-03-03_ · https://arxiv.org/abs/2603.03241v1  
  <details><summary>Abstract</summary>

  Unified multimodal models have recently demonstrated strong generative capabilities, yet whether and when generation improves understanding remains unclear. Existing benchmarks lack a systematic exploration of the specific tasks where generation facilitates understanding. To this end, we introduce UniG2U-Bench, a comprehensive benchmark categorizing generation-to-understanding (G2U) evaluation into 7 regimes and 30 subtasks, requiring varying degrees of implicit or explicit visual transformations. Extensive evaluation of over 30 models reveals three core findings: 1) Unified models generally underperform their base Vision-Language Models (VLMs), and Generate-then-Answer (GtA) inference typically degrades performance relative to direct inference. 2) Consistent enhancements emerge in spatial intelligence, visual illusions, or multi-round reasoning subtasks, where enhanced spatial and shape perception, as well as multi-step intermediate image states, prove beneficial. 3) Tasks with similar reasoning structures and models sharing architectures exhibit correlated behaviors, suggesting that generation-understanding coupling induces class-consistent inductive biases over tasks, pretraining data, and model architectures. These findings highlight the necessity for more diverse training data and novel paradigms to fully unlock the potential of unified multimodal modeling.

  </details>



- **COP-GEN: Latent Diffusion Transformer for Copernicus Earth Observation Data -- Generation Stochastic by Design**  
  Miguel Espinosa, Eva Gmelich Meijling, Valerio Marsocci, Elliot J. Crowley, Mikolaj Czerkawski  
  _2026-03-03_ · https://arxiv.org/abs/2603.03239v1  
  <details><summary>Abstract</summary>

  Earth observation applications increasingly rely on data from multiple sensors, including optical, radar, elevation, and land-cover products. Relationships between these modalities are fundamental for data integration but are inherently non-injective: identical conditioning information can correspond to multiple physically plausible observations. Thus, such conditional mappings should be parametrised as data distributions. As a result, deterministic models tend to collapse toward conditional means and fail to represent the uncertainty and variability required for tasks such as data completion and cross-sensor translation. We introduce COP-GEN, a multimodal latent diffusion transformer that models the joint distribution of heterogeneous Earth Observation modalities at their native spatial resolutions. By parameterising cross-modal mappings as conditional distributions, COP-GEN enables flexible any-to-any conditional generation, including zero-shot modality translation, spectral band infilling, and generation under partial or missing inputs, without task-specific retraining. Experiments on a large-scale global multimodal dataset show that COP-GEN generates diverse yet physically consistent realisations while maintaining strong peak fidelity across optical, radar, and elevation modalities. Qualitative and quantitative analyses demonstrate that the model captures meaningful cross-modal structure and systematically adapts its output uncertainty as conditioning information increases. These results highlight the practical importance of stochastic generative modeling for Earth observation and motivate evaluation protocols that move beyond single-reference, pointwise metrics. Website: https:// miquel-espinosa.github.io/cop-gen

  </details>



- **Conditioned Activation Transport for T2I Safety Steering**  
  Maciej Chrabąszcz, Aleksander Szymczyk, Jan Dubiński, Tomasz Trzciński, Franziska Boenisch, Adam Dziedzic  
  _2026-03-03_ · https://arxiv.org/abs/2603.03163v1  
  <details><summary>Abstract</summary>

  Despite their impressive capabilities, current Text-to-Image (T2I) models remain prone to generating unsafe and toxic content. While activation steering offers a promising inference-time intervention, we observe that linear activation steering frequently degrades image quality when applied to benign prompts. To address this trade-off, we first construct SafeSteerDataset, a contrastive dataset containing 2300 safe and unsafe prompt pairs with high cosine similarity. Leveraging this data, we propose Conditioned Activation Transport (CAT), a framework that employs a geometry-based conditioning mechanism and nonlinear transport maps. By conditioning transport maps to activate only within unsafe activation regions, we minimize interference with benign queries. We validate our approach on two state-of-the-art architectures: Z-Image and Infinity. Experiments demonstrate that CAT generalizes effectively across these backbones, significantly reducing Attack Success Rate while maintaining image fidelity compared to unsteered generations. Warning: This paper contains potentially offensive text and images.

  </details>



- **AWDiff: An a trous wavelet diffusion model for lung ultrasound image synthesis**  
  Maryam Heidari, Nantheera Anantrasirichai, Steven Walker, Rahul Bhatnagar, Alin Achim  
  _2026-03-03_ · https://arxiv.org/abs/2603.03125v1  
  <details><summary>Abstract</summary>

  Lung ultrasound (LUS) is a safe and portable imaging modality, but the scarcity of data limits the development of machine learning methods for image interpretation and disease monitoring. Existing generative augmentation methods, such as Generative Adversarial Networks (GANs) and diffusion models, often lose subtle diagnostic cues due to resolution reduction, particularly B-lines and pleural irregularities. We propose A trous Wavelet Diffusion (AWDiff), a diffusion based augmentation framework that integrates the a trous wavelet transform to preserve fine-scale structures while avoiding destructive downsampling. In addition, semantic conditioning with BioMedCLIP, a vision language foundation model trained on large scale biomedical corpora, enforces alignment with clinically meaningful labels. On a LUS dataset, AWDiff achieved lower distortion and higher perceptual quality compared to existing methods, demonstrating both structural fidelity and clinical diversity.

  </details>



- **MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection**  
  Jun Yeong Park, JunYoung Seo, Minji Kang, Yu Rang Park  
  _2026-03-03_ · https://arxiv.org/abs/2603.03101v1  
  <details><summary>Abstract</summary>

  The CLIP model's outstanding generalization has driven recent success in Zero-Shot Anomaly Detection (ZSAD) for detecting anomalies in unseen categories. The core challenge in ZSAD is to specialize the model for anomaly detection tasks while preserving CLIP's powerful generalization capability. Existing approaches attempting to solve this challenge share the fundamental limitation of a patch-agnostic design that processes all patches monolithically without regard for their unique characteristics. To address this limitation, we propose \textbf{MoECLIP}, a Mixture-of-Experts (MoE) architecture for the ZSAD task, which achieves patch-level adaptation by dynamically routing each image patch to a specialized Low-Rank Adaptation (LoRA) expert based on its unique characteristics. Furthermore, to prevent functional redundancy among the LoRA experts, we introduce (1) Frozen Orthogonal Feature Separation (FOFS), which orthogonally separates the input feature space to force experts to focus on distinct information, and (2) a simplex equiangular tight frame (ETF) loss to regulate the expert outputs to form maximally equiangular representations. Comprehensive experimental results across 14 benchmark datasets spanning industrial and medical domains demonstrate that MoECLIP outperforms existing state-of-the-art methods. The code is available at https://github.com/CoCoRessa/MoECLIP.

  </details>



- **TinyIceNet: Low-Power SAR Sea Ice Segmentation for On-Board FPGA Inference**  
  Mhd Rashed Al Koutayni, Mohamed Selim, Gerd Reis, Alain Pagani, Didier Stricker  
  _2026-03-03_ · https://arxiv.org/abs/2603.03075v1  
  <details><summary>Abstract</summary>

  Accurate sea ice mapping is essential for safe maritime navigation in polar regions, where rapidly changing ice conditions require timely and reliable information. While Sentinel-1 Synthetic Aperture Radar (SAR) provides high-resolution, all-weather observations of sea ice, conventional ground-based processing is limited by downlink bandwidth, latency, and energy costs associated with transmitting large volumes of raw data. On-board processing, enabled by dedicated inference chips integrated directly within the satellite payload, offers a transformative alternative by generating actionable sea ice products in orbit. In this context, we present TinyIceNet, a compact semantic segmentation network co-designed for on-board Stage of Development (SOD) mapping from dual-polarized Sentinel-1 SAR imagery under strict hardware and power constraints. Trained on the AI4Arctic dataset, TinyIceNet combines SAR-aware architectural simplifications with low-precision quantization to balance accuracy and efficiency. The model is synthesized using High-Level Synthesis and deployed on a Xilinx Zynq UltraScale+ FPGA platform, demonstrating near-real-time inference with significantly reduced energy consumption. Experimental results show that TinyIceNet achieves 75.216% F1 score on SOD segmentation while reducing energy consumption by 2x compared to full-precision GPU baselines, underscoring the potential of chip-level hardware-algorithm co-design for future spaceborne and edge AI systems.

  </details>



- **TikZilla: Scaling Text-to-TikZ with High-Quality Data and Reinforcement Learning**  
  Christian Greisinger, Steffen Eger  
  _2026-03-03_ · https://arxiv.org/abs/2603.03072v1  
  <details><summary>Abstract</summary>

  Large language models (LLMs) are increasingly used to assist scientists across diverse workflows. A key challenge is generating high-quality figures from textual descriptions, often represented as TikZ programs that can be rendered as scientific images. Prior research has proposed a variety of datasets and modeling approaches for this task. However, existing datasets for Text-to-TikZ are too small and noisy to capture the complexity of TikZ, causing mismatches between text and rendered figures. Moreover, prior approaches rely solely on supervised fine-tuning (SFT), which does not expose the model to the rendered semantics of the figure, often resulting in errors such as looping, irrelevant content, and incorrect spatial relations. To address these issues, we construct DaTikZ-V4, a dataset more than four times larger and substantially higher in quality than DaTikZ-V3, enriched with LLM-generated figure descriptions. Using this dataset, we train TikZilla, a family of small open-source Qwen models (3B and 8B) with a two-stage pipeline of SFT followed by reinforcement learning (RL). For RL, we leverage an image encoder trained via inverse graphics to provide semantically faithful reward signals. Extensive human evaluations with over 1,000 judgments show that TikZilla improves by 1.5-2 points over its base models on a 5-point scale, surpasses GPT-4o by 0.5 points, and matches GPT-5 in the image-based evaluation, while operating at much smaller model sizes. Code, data, and models will be made available.

  </details>



- **EduVQA: Benchmarking AI-Generated Video Quality Assessment for Education**  
  Baoliang Chen, Xinlong Bu, Lingyu Zhu, Hanwei Zhu, Xiangjie Sui  
  _2026-03-03_ · https://arxiv.org/abs/2603.03066v1  
  <details><summary>Abstract</summary>

  While AI-generated content (AIGC) models have achieved remarkable success in generating photorealistic videos, their potential to support visual, story-driven learning in education remains largely untapped. To close this gap, we present EduAIGV-1k, the first benchmark dataset and evaluation framework dedicated to assessing the quality of AI-generated videos (AIGVs) designed to teach foundational math concepts, such as numbers and geometry, to young learners. EduAIGV-1k contains 1,130 short videos produced by ten state-of-the-art text-to-video (T2V) models using 113 pedagogy-oriented prompts. Each video is accompanied by rich, fine-grained annotations along two complementary axes: (1) Perceptual quality, disentangled into spatial and temporal fidelity, and (2) Prompt alignment, labeled at the word-level and sentence-level to quantify the degree to which each mathematical concept in the prompt is accurately grounded in the generated video. These fine-grained annotations transform each video into a multi-dimensional, interpretable supervision signal, far beyond a single quality score. Leveraging this dense feedback, we introduce EduVQA for both perceptual and alignment quality assessment of AIGVs. In particular, we propose a Structured 2D Mixture-of-Experts (S2D-MoE) module, which enhances the dependency between overall quality and each sub-dimension by shared experts and dynamic 2D gating matrix. Extensive experiments show our EduVQA consistently outperforms existing VQA baselines. Both our dataset and code will be publicly available.

  </details>



- **MA-CoNav: A Master-Slave Multi-Agent Framework with Hierarchical Collaboration and Dual-Level Reflection for Long-Horizon Embodied VLN**  
  Ling Luo, Qianqian Bai  
  _2026-03-03_ · https://arxiv.org/abs/2603.03024v1  
  <details><summary>Abstract</summary>

  Vision-Language Navigation (VLN) aims to empower robots with the ability to perform long-horizon navigation in unfamiliar environments based on complex linguistic instructions. Its success critically hinges on establishing an efficient ``language-understanding -- visual-perception -- embodied-execution'' closed loop. Existing methods often suffer from perceptual distortion and decision drift in complex, long-distance tasks due to the cognitive overload of a single agent. Inspired by distributed cognition theory, this paper proposes MA-CoNav, a Multi-Agent Collaborative Navigation framework. This framework adopts a ``Master-Slave'' hierarchical agent collaboration architecture, decoupling and distributing the perception, planning, execution, and memory functions required for navigation tasks to specialized agents. Specifically, the Master Agent is responsible for global orchestration, while the Subordinate Agent group collaborates through a clear division of labor: an Observation Agent generates environment descriptions, a Planning Agent performs task decomposition and dynamic verification, an Execution Agent handles simultaneous mapping and action, and a Memory Agent manages structured experiences. Furthermore, the framework introduces a ``Local-Global'' dual-stage reflection mechanism to dynamically optimize the entire navigation pipeline. Empirical experiments were conducted using a real-world indoor dataset collected by a Limo Pro robot, with no scene-specific fine-tuning performed on the models throughout the process. The results demonstrate that MA-CoNav comprehensively outperforms existing mainstream VLN methods across multiple metrics.

  </details>



- **The Dresden Dataset for 4D Reconstruction of Non-Rigid Abdominal Surgical Scenes**  
  Reuben Docea, Rayan Younis, Yonghao Long, Maxime Fleury, Jinjing Xu, Chenyang Li, André Schulze, Ann Wierick, Johannes Bender, Micha Pfeiffer, et al.  
  _2026-03-03_ · https://arxiv.org/abs/2603.02985v1  
  <details><summary>Abstract</summary>

  The D4D Dataset provides paired endoscopic video and high-quality structured-light geometry for evaluating 3D reconstruction of deforming abdominal soft tissue in realistic surgical conditions. Data were acquired from six porcine cadaver sessions using a da Vinci Xi stereo endoscope and a Zivid structured-light camera, registered via optical tracking and manually curated iterative alignment methods. Three sequence types - whole deformations, incremental deformations, and moved-camera clips - probe algorithm robustness to non-rigid motion, deformation magnitude, and out-of-view updates. Each clip provides rectified stereo images, per-frame instrument masks, stereo depth, start/end structured-light point clouds, curated camera poses and camera intrinsics. In postprocessing, ICP and semi-automatic registration techniques are used to register data, and instrument masks are created. The dataset enables quantitative geometric evaluation in both visible and occluded regions, alongside photometric view-synthesis baselines. Comprising over 300,000 frames and 369 point clouds across 98 curated recordings, this resource can serve as a comprehensive benchmark for developing and evaluating non-rigid SLAM, 4D reconstruction, and depth estimation methods.

  </details>



- **Spatial Autoregressive Modeling of DINOv3 Embeddings for Unsupervised Anomaly Detection**  
  Ertunc Erdil, Nico Schulthess, Guney Tombak, Ender Konukoglu  
  _2026-03-03_ · https://arxiv.org/abs/2603.02974v1  
  <details><summary>Abstract</summary>

  DINO models provide rich patch-level representations that have recently enabled strong performance in unsupervised anomaly detection (UAD). Most existing methods extract patch embeddings from ``normal'' images and model them independently, ignoring spatial and neighborhood relationships between patches. This implicitly assumes that self-attention and positional encodings sufficiently encode contextual information within each patch embedding. In addition, the normative distribution is often modeled as memory banks or prototype-based representations, which require storing large numbers of features and performing costly comparisons at inference time, leading to substantial memory and computational overhead. In this work, we address both limitations by proposing a simple and efficient framework that explicitly models spatial and contextual dependencies between patch embeddings using a 2D autoregressive (AR) model. Instead of storing embeddings or clustering prototypes, our approach learns a compact parametric model of the normative distribution via an AR convolutional neural network (CNN). At test time, anomaly detection reduces to a single forward pass through the network and enables fast and memory-efficient inference. We evaluate our method on the BMAD benchmark, which comprises three medical imaging datasets, and compare it against existing work including recent DINO-based methods. Experimental results demonstrate that explicitly modeling spatial dependencies achieves competitive anomaly detection performance while substantially reducing inference time and memory requirements. Code is available at the project page: https://eerdil.github.io/spatial-ar-dinov3-uad/.

  </details>



- **TagaVLM: Topology-Aware Global Action Reasoning for Vision-Language Navigation**  
  Jiaxing Liu, Zexi Zhang, Xiaoyan Li, Boyue Wang, Yongli Hu, Baocai Yin  
  _2026-03-03_ · https://arxiv.org/abs/2603.02972v1  
  <details><summary>Abstract</summary>

  Vision-Language Navigation (VLN) presents a unique challenge for Large Vision-Language Models (VLMs) due to their inherent architectural mismatch: VLMs are primarily pretrained on static, disembodied vision-language tasks, which fundamentally clash with the dynamic, embodied, and spatially-structured nature of navigation. Existing large-model-based methods often resort to converting rich visual and spatial information into text, forcing models to implicitly infer complex visual-topological relationships or limiting their global action capabilities. To bridge this gap, we propose TagaVLM (Topology-Aware Global Action reasoning), an end-to-end framework that explicitly injects topological structures into the VLM backbone. To introduce topological edge information, Spatial Topology Aware Residual Attention (STAR-Att) directly integrates it into the VLM's self-attention mechanism, enabling intrinsic spatial reasoning while preserving pretrained knowledge. To enhance topological node information, an Interleaved Navigation Prompt strengthens node-level visual-text alignment. Finally, with the embedded topological graph, the model is capable of global action reasoning, allowing for robust path correction. On the R2R benchmark, TagaVLM achieves state-of-the-art performance among large-model-based methods, with a Success Rate (SR) of 51.09% and SPL of 47.18 in unseen environments, outperforming prior work by 3.39% in SR and 9.08 in SPL. This demonstrates that, for embodied spatial reasoning, targeted enhancements on smaller open-source VLMs can be more effective than brute-force model scaling. The code will be released upon publication.Project page: https://apex-bjut.github.io/Taga-VLM

  </details>



- **Semi-Supervised Few-Shot Adaptation of Vision-Language Models**  
  Julio Silva-Rodríguez, Ender Konukoglu  
  _2026-03-03_ · https://arxiv.org/abs/2603.02959v1  
  <details><summary>Abstract</summary>

  Vision-language models (VLMs) pre-trained on large, heterogeneous data sources are becoming increasingly popular, providing rich multi-modal embeddings that enable efficient transfer to new tasks. A particularly relevant application is few-shot adaptation, where only a handful of annotated examples are available to adapt the model through multi-modal linear probes. In medical imaging, specialized VLMs have shown promising performance in zero- and few-shot image classification, which is valuable for mitigating the high cost of expert annotations. However, challenges remain in extremely low-shot regimes: the inherent class imbalances in medical tasks often lead to underrepresented categories, penalizing overall model performance. To address this limitation, we propose leveraging unlabeled data by introducing an efficient semi-supervised solver that propagates text-informed pseudo-labels during few-shot adaptation. The proposed method enables lower-budget annotation pipelines for adapting VLMs, reducing labeling effort by >50% in low-shot regimes.

  </details>



- **Leveraging Label Proportion Prior for Class-Imbalanced Semi-Supervised Learning**  
  Kohki Akiba, Shinnosuke Matsuo, Shota Harada, Ryoma Bise  
  _2026-03-03_ · https://arxiv.org/abs/2603.02957v1  
  <details><summary>Abstract</summary>

  Semi-supervised learning (SSL) often suffers under class imbalance, where pseudo-labeling amplifies majority bias and suppresses minority performance. We address this issue with a lightweight framework that, to our knowledge, is the first to introduce Proportion Loss from learning from label proportions (LLP) into SSL as a regularization term. Proportion Loss aligns model predictions with the global class distribution, mitigating bias across both majority and minority classes. To further stabilize training, we formulate a stochastic variant that accounts for fluctuations in mini-batch composition. Experiments on the Long-tailed CIFAR-10 benchmark show that integrating Proportion Loss into FixMatch and ReMixMatch consistently improves performance over the baselines across imbalance severities and label ratios, and achieves competitive or superior results compared to existing CISSL methods, particularly under scarce-label conditions.

  </details>



- **CGL: Advancing Continual GUI Learning via Reinforcement Fine-Tuning**  
  Zhenquan Yao, Zitong Huang, Yihan Zeng, Jianhua Han, Hang Xu, Chun-Mei Feng, Jianwei Ma, Wangmeng Zuo  
  _2026-03-03_ · https://arxiv.org/abs/2603.02951v1  
  <details><summary>Abstract</summary>

  Graphical User Interface (GUI) Agents, benefiting from recent advances in multimodal large language models (MLLM), have achieved significant development. However, due to the frequent updates of GUI applications, adapting to new tasks without forgetting old tasks in GUI continual learning remains an open problem. In this work, we reveal that while Supervised Fine-Tuning (SFT) facilitates fast adaptation, it often triggers knowledge overwriting, whereas Reinforcement Learning (RL) demonstrates an inherent resilience that shields prior interaction logic from erasure. Based on this insight, we propose a \textbf{C}ontinual \textbf{G}UI \textbf{L}earning (CGL) framework that dynamically balances adaptation efficiency and skill retention by enhancing the synergy between SFT and RL. Specifically, we introduce an SFT proportion adjustment mechanism guided by policy entropy to dynamically control the weight allocation between the SFT and RL training phases. To resolve explicit gradient interference, we further develop a specialized gradient surgery strategy. By projecting exploratory SFT gradients onto GRPO-based anchor gradients, our method explicitly clips the components of SFT gradients that conflict with GRPO. On top of that, we establish an AndroidControl-CL benchmark, which divides GUI applications into distinct task groups to effectively simulate and evaluate the performance of continual GUI learning. Experimental results demonstrate the effectiveness of our proposed CGL framework across continual learning scenarios. The benchmark, code, and model will be made publicly available.

  </details>



- **Self-supervised Domain Adaptation for Visual 3D Pose Estimation of Nano-drone Racing Gates by Enforcing Geometric Consistency**  
  Nicholas Carlotti, Michele Antonazzi, Elia Cereda, Mirko Nava, Nicola Basilico, Daniele Palossi, Alessandro Giusti  
  _2026-03-03_ · https://arxiv.org/abs/2603.02936v1  
  <details><summary>Abstract</summary>

  We consider the task of visually estimating the relative pose of a drone racing gate in front of a nano-quadrotor, using a convolutional neural network pre-trained on simulated data to regress the gate's pose. Due to the sim-to-real gap, the pre-trained model underperforms in the real world and must be adapted to the target domain. We propose an unsupervised domain adaptation (UDA) approach using only real image sequences collected by the drone flying an arbitrary trajectory in front of a gate; sequences are annotated in a self-supervised fashion with the drone's odometry as measured by its onboard sensors. On this dataset, a state consistency loss enforces that two images acquired at different times yield pose predictions that are consistent with the drone's odometry. Results indicate that our approach outperforms other SoA UDA approaches, has a low mean absolute error in position (x=26, y=28, z=10 cm) and orientation ($ψ$=13${^{\circ}}$), an improvement of 40% in position and 37% in orientation over a baseline. The approach's effectiveness is appreciable with as few as 10 minutes of real-world flight data and yields models with an inference time of 30.4ms (33 fps) when deployed aboard the Crazyflie 2.1 Brushless nano-drone.

  </details>



- **TRACE: Task-Adaptive Reasoning and Representation Learning for Universal Multimodal Retrieval**  
  Xiangzhao Hao, Shijie Wang, Tianyu Yang, Tianyue Wang, Haiyun Guo, JinQiao Wang  
  _2026-03-03_ · https://arxiv.org/abs/2603.02929v1  
  <details><summary>Abstract</summary>

  Universal Multimodal Retrieval requires unified embedding models capable of interpreting diverse user intents, ranging from simple keywords to complex compositional instructions. While Multimodal Large Language Models (MLLMs) possess strong reasoning capabilities, prevailing adaptations confine them to static encoders, underutilizing their generative potential. This encoder-only paradigm struggles with complex intents that demand logical deduction rather than superficial pattern matching. To address this, we introduce TRACE (Task-adaptive Reasoning And Compressing Embeddings). TRACE unifies generative reasoning with discriminative representation learning. It first generates a structured Chain-of-Thought (CoT) to explicitly reason about the query, and subsequently compresses this reasoning trace into a compact embedding via a dedicated token. To train this framework, we construct M-BEIR-CoT, a large-scale dataset featuring a difficulty-aware routing strategy. Experiments on the M-BEIR benchmark establish TRACE as the new state-of-the-art. Crucially, TRACE demonstrates a learned implicit routing behavior. It autonomously activates reasoning for complex queries while bypassing it for simpler ones, achieving an optimal balance between retrieval accuracy and inference throughput. Furthermore, by internalizing the deductive process, TRACE exhibits remarkable zero-shot transferability to unseen domains and novel constraints.

  </details>



- **ProGIC: Progressive and Lightweight Generative Image Compression with Residual Vector Quantization**  
  Hao Cao, Chengbin Liang, Wenqi Guo, Zhijin Qin, Jungong Han  
  _2026-03-03_ · https://arxiv.org/abs/2603.02897v1  
  <details><summary>Abstract</summary>

  Recent advances in generative image compression (GIC) have delivered remarkable improvements in perceptual quality. However, many GICs rely on large-scale and rigid models, which severely constrain their utility for flexible transmission and practical deployment in low-bitrate scenarios. To address these issues, we propose Progressive Generative Image Compression (ProGIC), a compact codec built on residual vector quantization (RVQ). In RVQ, a sequence of vector quantizers encodes the residuals stage by stage, each with its own codebook. The resulting codewords sum to a coarse-to-fine reconstruction and a progressive bitstream, enabling previews from partial data. We pair this with a lightweight backbone based on depthwise-separable convolutions and small attention blocks, enabling practical deployment on both GPUs and CPU-only devices. Experimental results show that ProGIC attains comparable compression performance compared with previous methods. It achieves bitrate savings of up to 57.57% on DISTS and 58.83% on LPIPS compared to MS-ILLM on the Kodak dataset. Beyond perceptual quality, ProGIC enables progressive transmission for flexibility, and also delivers over 10 times faster encoding and decoding compared with MS-ILLM on GPUs for efficiency.

  </details>



- **3D-DRES: Detailed 3D Referring Expression Segmentation**  
  Qi Chen, Changli Wu, Jiayi Ji, Yiwei Ma, Liujuan Cao  
  _2026-03-03_ · https://arxiv.org/abs/2603.02896v1  
  <details><summary>Abstract</summary>

  Current 3D visual grounding tasks only process sentence level detection or segmentation, which critically fails to leverage the rich compositional contextual reasonings within natural language expressions. To address this challenge, we introduce Detailed 3D Referring Expression Segmentation (3D-DRES), a new task that provides a phrase to 3D instance mapping, aiming at enhancing fine-grained 3D vision language understanding. To support 3D-DRES, we present DetailRefer, a new dataset comprising 54,432 descriptions spanning 11,054 distinct objects. Unlike previous datasets, DetailRefer implements a pioneering phrase-instance annotation paradigm where each referenced noun phrase is explicitly mapped to its corresponding 3D elements. Additionally, we introduce DetailBase, a purposefully streamlined yet effective baseline architecture that supports dual-mode segmentation at both sentence and phrase levels. Our experimental results demonstrate that models trained on DetailRefer not only excel at phrase-level segmentation but also show surprising improvements on traditional 3D-RES benchmarks.

  </details>



- **Nodes Are Early, Edges Are Late: Probing Diagram Representations in Large Vision-Language Models**  
  Haruto Yoshida, Keito Kudo, Yoichi Aoki, Ryota Tanaka, Itsumi Saito, Keisuke Sakaguchi, Kentaro Inui  
  _2026-03-03_ · https://arxiv.org/abs/2603.02865v1  
  <details><summary>Abstract</summary>

  Large vision-language models (LVLMs) demonstrate strong performance on diagram understanding benchmarks, yet they still struggle with understanding relationships between elements, particularly those represented by nodes and directed edges (e.g., arrows and lines). To investigate the underlying causes of this limitation, we probe the internal representation of LVLMs using a carefully constructed synthetic diagram dataset based on directed graphs. Our probing experiments reveal that edge information is not linearly separable in the vision encoder and becomes linearly encoded only in the text tokens in the language model. In contrast, node information and global structural features are already linearly encoded in individual hidden states of the vision encoder. These findings suggest that the stage at which linearly separable representations are formed varies depending on the type of visual information. In particular, the delayed emergence of edge representations may help explain why LVLMs struggle with relational understanding, such as interpreting edge directions, which require more abstract, compositionally integrated processes.

  </details>



- **CoFL: Continuous Flow Fields for Language-Conditioned Navigation**  
  Haokun Liu, Zhaoqi Ma, Yicheng Chen, Masaki Kitagawa, Wentao Zhang, Jinjie Li, Moju Zhao  
  _2026-03-03_ · https://arxiv.org/abs/2603.02854v1  
  <details><summary>Abstract</summary>

  Language-conditioned navigation pipelines often rely on brittle modular components or costly action-sequence generation. To address these limitations, we present CoFL, an end-to-end policy that directly maps a bird's-eye view (BEV) observation and a language instruction to a continuous flow field for navigation. Instead of predicting discrete action tokens or sampling action chunks via iterative denoising, CoFL outputs instantaneous velocities that can be queried at arbitrary 2D projected locations. Trajectories are obtained by numerical integration of the predicted field, producing smooth motion that remains reactive under closed-loop execution. To enable large-scale training, we build a dataset of over 500k BEV image-instruction pairs, each procedurally annotated with a flow field and a trajectory derived from BEV semantic maps built on Matterport3D and ScanNet. By training on a mixed distribution, CoFL significantly outperforms modular Vision-Language Model (VLM)-based planners and generative policy baselines on strictly unseen scenes. Finally, we deploy CoFL zero-shot in real-world experiments with overhead BEV observations across multiple layouts, maintaining reliable closed-loop control and a high success rate.

  </details>



- **Scale-invariant Gaussian derivative residual networks**  
  Andrzej Perzanowski, Tony Lindeberg  
  _2026-03-03_ · https://arxiv.org/abs/2603.02843v1  
  <details><summary>Abstract</summary>

  Generalisation across image scales remains a fundamental challenge for deep networks, which often fail to handle images at scales not seen during training (the out-of-distribution problem). In this paper, we present provably scale-invariant Gaussian derivative residual networks (GaussDerResNets), constructed out of scale-covariant Gaussian derivative residual blocks coupled in cascade, aimed at addressing this problem. By adding residual skip connections to the previous notion of Gaussian derivative layers, deeper networks with substantially increased accuracy can be constructed, while preserving very good scale generalisation properties at the higher level of accuracy. Explicit proofs are provided regarding the underlying scale-covariant and scale-invariant properties in arbitrary dimensions. To analyse the ability of GaussDerResNets to generalise to new scales, we apply them on the new rescaled version of the STL-10 dataset, where training is done at a single fixed scale and evaluation is performed on multiple copies of the test set, each rescaled to a single distinct spatial scale, with scale factors extending over a range of 4. We also conduct similar systematic experiments on the rescaled versions of Fashion-MNIST and CIFAR-10 datasets. Experimentally, we demonstrate that the GaussDerResNets have strong scale generalisation and scale selection properties on all the three rescaled datasets. In our ablation studies, we investigate different architectural variants of GaussDerResNets, demonstrating that basing the architecture on depthwise-separable convolutions allows for decreasing both the number of parameters and the amount of computations, with reasonably maintained accuracy and scale generalisation.

  </details>



- **R3GW: Relightable 3D Gaussians for Outdoor Scenes in the Wild**  
  Margherita Lea Corona, Wieland Morgenstern, Peter Eisert, Anna Hilsmann  
  _2026-03-03_ · https://arxiv.org/abs/2603.02801v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has established itself as a leading technique for 3D reconstruction and novel view synthesis of static scenes, achieving outstanding rendering quality and fast training. However, the method does not explicitly model the scene illumination, making it unsuitable for relighting tasks. Furthermore, 3DGS struggles to reconstruct scenes captured in the wild by unconstrained photo collections featuring changing lighting conditions. In this paper, we present R3GW, a novel method that learns a relightable 3DGS representation of an outdoor scene captured in the wild. Our approach separates the scene into a relightable foreground and a non-reflective background (the sky), using two distinct sets of Gaussians. R3GW models view-dependent lighting effects in the foreground reflections by combining Physically Based Rendering with the 3DGS scene representation in a varying illumination setting. We evaluate our method quantitatively and qualitatively on the NeRF-OSR dataset, offering state-of-the-art performance and enhanced support for physically-based relighting of unconstrained scenes. Our method synthesizes photorealistic novel views under arbitrary illumination conditions. Additionally, our representation of the sky mitigates depth reconstruction artifacts, improving rendering quality at the sky-foreground boundary

  </details>



- **VSearcher: Long-Horizon Multimodal Search Agent via Reinforcement Learning**  
  Ruiyang Zhang, Qianguo Sun, Chao Song, Yiyan Qi, Zhedong Zheng  
  _2026-03-03_ · https://arxiv.org/abs/2603.02795v1  
  <details><summary>Abstract</summary>

  Large models are increasingly becoming autonomous agents that interact with real-world environments and use external tools to augment their static capabilities. However, most recent progress has focused on text-only large language models, which are limited to a single modality and therefore have narrower application scenarios. On the other hand, multimodal large models, while offering stronger perceptual capabilities, remain limited to static knowledge and lack the ability to access and leverage up-to-date web information. In this paper, we propose VSearcher, turning static multimodal model into multimodal search agent capable of long-horizon, multi-turn tool use in real-world web environments, including text search, image search, and web browsing, via reinforcement learning. Specifically, we introduce Iterative Injection Data Synthesis pipeline to generate large-scale, complex multimodal QA questions, which are further filtered with comprehensive metrics to ensure high quality and sufficient difficulty. We then adopt an SFT-then-RL training pipeline to turn base multimodal models to agent capable of multi-turn tool calling in real-world web environments. Besides, we propose a multimodal search benchmark MM-SearchExam dedicated to evaluating search capabilities of multimodal search agents, which proves highly challenging for recent proprietary models. Extensive evaluations across multiple multimodal search benchmarks reveal effectiveness of our method. VSearcher achieves superior performance compared to recent multimodal search agents and even surpasses several proprietary models on multimodal web search tasks.

  </details>



- **Designing UNICORN: a Unified Benchmark for Imaging in Computational Pathology, Radiology, and Natural Language**  
  Michelle Stegeman, Lena Philipp, Fennie van der Graaf, Marina D'Amato, Clément Grisi, Luc Builtjes, Joeran S. Bosma, Judith Lefkes, Rianne A. Weber, James A. Meakin, et al.  
  _2026-03-03_ · https://arxiv.org/abs/2603.02790v1  
  <details><summary>Abstract</summary>

  Medical foundation models show promise to learn broadly generalizable features from large, diverse datasets. This could be the base for reliable cross-modality generalization and rapid adaptation to new, task-specific goals, with only a few task-specific examples. Yet, evidence for this is limited by the lack of public, standardized, and reproducible evaluation frameworks, as existing public benchmarks are often fragmented across task-, organ-, or modality-specific settings, limiting assessment of cross-task generalization. We introduce UNICORN, a public benchmark designed to systematically evaluate medical foundation models under a unified protocol. To isolate representation quality, we built the benchmark on a novel two-step framework that decouples model inference from task-specific evaluation based on standardized few-shot adaptation. As a central design choice, we constructed indirectly accessible sequestered test sets derived from clinically relevant cohorts, along with standardized evaluation code and a submission interface on an open benchmarking platform. Performance is aggregated into a single UNICORN Score, a new metric that we introduce to support direct comparison of foundation models across diverse medical domains, modalities, and task types. The UNICORN test dataset includes data from more than 2,400 patients, including over 3,700 vision cases and over 2,400 clinical reports collected from 17 institutions across eight countries. The benchmark spans eight anatomical regions and four imaging modalities. Both task-specific and aggregated leaderboards enable accessible, standardized, and reproducible evaluation. By standardizing multi-task, multi-modality assessment, UNICORN establishes a foundation for reproducible benchmarking of medical foundation models. Data, baseline methods, and the evaluation platform are publicly available via unicorn.grand-challenge.org.

  </details>



- **Seeing Clearly without Training: Mitigating Hallucinations in Multimodal LLMs for Remote Sensing**  
  Yi Liu, Jing Zhang, Di Wang, Xiaoyu Tian, Haonan Guo, Bo Du  
  _2026-03-03_ · https://arxiv.org/abs/2603.02754v1  
  <details><summary>Abstract</summary>

  Multimodal large language models (MLLMs) suffer from pronounced hallucinations in remote sensing visual question-answering (RS-VQA), primarily caused by visual grounding failures in large-scale scenes or misinterpretation of fine-grained small targets. To systematically analyze these issues, we introduce RSHBench, a protocol-based benchmark for fine-grained diagnosis of factual and logical hallucinations. To mitigate grounding-induced factual hallucinations, we further propose Relative Attention-Driven Actively Reasoning (RADAR), a training-free inference method that leverages intrinsic attention in MLLMs to guide progressive localization and fine-grained local reasoning at test time. Extensive experiments across diverse MLLMs demonstrate that RADAR consistently improves RS-VQA performance and reduces both factual and logical hallucinations. Code and data will be publicly available at: https://github.com/MiliLab/RADAR

  </details>



- **CoShadow: Multi-Object Shadow Generation for Image Compositing via Diffusion Model**  
  Waqas Ahmed, Dean Diepeveen, Ferdous Sohel  
  _2026-03-03_ · https://arxiv.org/abs/2603.02743v1  
  <details><summary>Abstract</summary>

  Realistic shadow generation is crucial for achieving seamless image compositing, yet existing methods primarily focus on single-object insertion and often fail to generalize when multiple foreground objects are composited into a background scene. In practice, however, modern compositing pipelines and real-world applications often insert multiple objects simultaneously, necessitating shadows that are jointly consistent in terms of geometry, attachment, and location. In this paper, we address the under-explored problem of multi-object shadow generation, aiming to synthesize physically plausible shadows for multiple inserted objects. Our approach exploits the multimodal capabilities of a pre-trained text-to-image diffusion model. An image pathway injects dense, multi-scale features to provide fine-grained spatial guidance, while a text-based pathway encodes per-object shadow bounding boxes as learned positional tokens and fuses them via cross-attention. An attention-alignment loss further grounds these tokens to their corresponding shadow regions. To support this task, we augment the DESOBAv2 dataset by constructing composite scenes with multiple inserted objects and automatically derive prompts combining object category and shadow positioning information. Experimental results demonstrate that our method achieves state-of-the-art performance in both single and multi-object shadow generation settings.

  </details>



- **Robust Tightly-Coupled Filter-Based Monocular Visual-Inertial State Estimation and Graph-Based Evaluation for Autonomous Drone Racing**  
  Maulana Bisyir Azhari, Donghun Han, SungJun Park, David Hyunchul Shim  
  _2026-03-03_ · https://arxiv.org/abs/2603.02742v1  
  <details><summary>Abstract</summary>

  Autonomous drone racing (ADR) demands state estimation that is simultaneously computationally efficient and resilient to the perceptual degradation experienced during extreme velocity and maneuvers. Traditional frameworks typically rely on conventional visual-inertial pipelines with loosely-coupled gate-based Perspective-n-Points (PnP) corrections that suffer from a rigid requirement for four visible features and information loss in intermediate steps. Furthermore, the absence of GNSS and Motion Capture systems in uninstrumented, competitive racing environments makes the objective evaluation of such systems remarkably difficult. To address these limitations, we propose ADR-VINS, a robust, monocular visual-inertial state estimation framework based on an Error-State Kalman Filter (ESKF) tailored for autonomous drone racing. Our approach integrates direct pixel reprojection errors from gate corners features as innovation terms within the filter. By bypassing intermediate PnP solvers, ADR-VINS maintains valid state updates with as few as two visible corners and utilizes robust reweighting instead of RANSAC-based schemes to handle outliers, enhancing computational efficiency. Furthermore, we introduce ADR-FGO, an offline Factor-Graph Optimization framework to generate high-fidelity reference trajectories that facilitate post-flight performance evaluation and analysis on uninstrumented, GNSS-denied environments. The proposed system is validated using TII-RATM dataset, where ADR-VINS achieves an average RMS translation error of 0.134 m, while ADR-FGO yields 0.060 m as a smoothing-based reference. Finally, ADR-VINS was successfully deployed in the A2RL Drone Championship Season 2, maintaining stable and robust estimation despite noisy detections during high-agility flight at top speeds of 20.9 m/s. We further utilize ADR-FGO for post-flight evaluation in uninstrumented racing environments.

  </details>



- **ShareVerse: Multi-Agent Consistent Video Generation for Shared World Modeling**  
  Jiayi Zhu, Jianing Zhang, Yiying Yang, Wei Cheng, Xiaoyun Yuan  
  _2026-03-03_ · https://arxiv.org/abs/2603.02697v1  
  <details><summary>Abstract</summary>

  This paper presents ShareVerse, a video generation framework enabling multi-agent shared world modeling, addressing the gap in existing works that lack support for unified shared world construction with multi-agent interaction. ShareVerse leverages the generation capability of large video models and integrates three key innovations: 1) A dataset for large-scale multi-agent interactive world modeling is built on the CARLA simulation platform, featuring diverse scenes, weather conditions, and interactive trajectories with paired multi-view videos (front/ rear/ left/ right views per agent) and camera data. 2) We propose a spatial concatenation strategy for four-view videos of independent agents to model a broader environment and to ensure internal multi-view geometric consistency. 3) We integrate cross-agent attention blocks into the pretrained video model, which enable interactive transmission of spatial-temporal information across agents, guaranteeing shared world consistency in overlapping regions and reasonable generation in non-overlapping regions. ShareVerse, which supports 49-frame large-scale video generation, accurately perceives the position of dynamic agents and achieves consistent shared world modeling.

  </details>



- **Retrieval-Augmented Robots via Retrieve-Reason-Act**  
  Izat Temiraliev, Diji Yang, Yi Zhang  
  _2026-03-03_ · https://arxiv.org/abs/2603.02688v1  
  <details><summary>Abstract</summary>

  To achieve general-purpose utility, we argue that robots must evolve from passive executors into active Information Retrieval users. In strictly zero-shot settings where no prior demonstrations exist, robots face a critical information gap, such as the exact sequence required to assemble a complex furniture kit, that cannot be satisfied by internal parametric knowledge (common sense) or past internal memory. While recent robotic works attempt to use search before action, they primarily focus on retrieving past kinematic trajectories (analogous to searching internal memory) or text-based safety rules (searching for constraints). These approaches fail to address the core information need of active task construction: acquiring unseen procedural knowledge from external, unstructured documentation. In this paper, we define the paradigm as Retrieval-Augmented Robotics (RAR), empowering the robot with the information-seeking capability that bridges the gap between visual documentation and physical actuation. We formulate the task execution as an iterative Retrieve-Reason-Act loop: the robot or embodied agent actively retrieves relevant visual procedural manuals from an unstructured corpus, grounds the abstract 2D diagrams to 3D physical parts via cross-modal alignment, and synthesizes executable plans. We validate this paradigm on a challenging long-horizon assembly benchmark. Our experiments demonstrate that grounding robotic planning in retrieved visual documents significantly outperforms baselines relying on zero-shot reasoning or few-shot example retrieval. This work establishes the basis of RAR, extending the scope of Information Retrieval from answering user queries to driving embodied physical actions.

  </details>



- **VisionCreator: A Native Visual-Generation Agentic Model with Understanding, Thinking, Planning and Creation**  
  Jinxiang Lai, Zexin Lu, Jiajun He, Rongwei Quan, Wenzhe Zhao, Qinyu Yang, Qi Chen, Qin Lin, Chuyue Li, Tao Gao, et al.  
  _2026-03-03_ · https://arxiv.org/abs/2603.02681v1  
  <details><summary>Abstract</summary>

  Visual content creation tasks demand a nuanced understanding of design conventions and creative workflows-capabilities challenging for general models, while workflow-based agents lack specialized knowledge for autonomous creative planning. To overcome these challenges, we propose VisionCreator, a native visual-generation agentic model that unifies Understanding, Thinking, Planning, and Creation (UTPC) capabilities within an end-to-end learnable framework. Our work introduces four key contributions: (i) VisGenData-4k and its construction methodology using metacognition-based VisionAgent to generate high-quality creation trajectories with explicit UTPC structures; (ii) The VisionCreator agentic model, optimized through Progressive Specialization Training (PST) and Virtual Reinforcement Learning (VRL) within a high-fidelity simulated environment, enabling stable and efficient acquisition of UTPC capabilities for complex creation tasks; (iii) VisGenBench, a comprehensive benchmark featuring 1.2k test samples across diverse scenarios for standardized evaluation of multi-step visual creation capabilities; (iv) Remarkably, our VisionCreator-8B/32B models demonstrate superior performance over larger closed-source models across multiple evaluation dimensions. Overall, this work provides a foundation for future research in visual-generation agentic systems.

  </details>



- **IMR-LLM: Industrial Multi-Robot Task Planning and Program Generation using Large Language Models**  
  Xiangyu Su, Juzhan Xu, Oliver van Kaick, Kai Xu, Ruizhen Hu  
  _2026-03-03_ · https://arxiv.org/abs/2603.02669v1  
  <details><summary>Abstract</summary>

  In modern industrial production, multiple robots often collaborate to complete complex manufacturing tasks. Large language models (LLMs), with their strong reasoning capabilities, have shown potential in coordinating robots for simple household and manipulation tasks. However, in industrial scenarios, stricter sequential constraints and more complex dependencies within tasks present new challenges for LLMs. To address this, we propose IMR-LLM, a novel LLM-driven Industrial Multi-Robot task planning and program generation framework. Specifically, we utilize LLMs to assist in constructing disjunctive graphs and employ deterministic solving methods to obtain a feasible and efficient high-level task plan. Based on this, we use a process tree to guide LLMs to generate executable low-level programs. Additionally, we create IMR-Bench, a challenging benchmark that encompasses multi-robot industrial tasks across three levels of complexity. Experimental results indicate that our method significantly surpasses existing methods across all evaluation metrics.

  </details>



- **OmniFashion: Towards Generalist Fashion Intelligence via Multi-Task Vision-Language Learning**  
  Zhengwei Yang, Andi Long, Hao Li, Zechao Hu, Kui Jiang, Zheng Wang  
  _2026-03-03_ · https://arxiv.org/abs/2603.02658v1  
  <details><summary>Abstract</summary>

  Fashion intelligence spans multiple tasks, i.e., retrieval, recommendation, recognition, and dialogue, yet remains hindered by fragmented supervision and incomplete fashion annotations. These limitations jointly restrict the formation of consistent visual-semantic structures, preventing recent vision-language models (VLMs) from serving as a generalist fashion brain that unifies understanding and reasoning across tasks. Therefore, we construct FashionX, a million-scale dataset that exhaustively annotates visible fashion items within an outfit and organizes attributes from global to part-level. Built upon this foundation, we propose OmniFashion, a unified vision-language framework that bridges diverse fashion tasks under a unified fashion dialogue paradigm, enabling both multi-task reasoning and interactive dialogue. Experiments on multi-subtasks and retrieval benchmarks show that OmniFashion achieves strong task-level accuracy and cross-task generalization, highlighting its offering of a scalable path toward universal, dialogue-oriented fashion intelligence.

  </details>



- **SEP-YOLO: Fourier-Domain Feature Representation for Transparent Object Instance Segmentation**  
  Fengming Zhang, Tao Yan, Jianchao Huang  
  _2026-03-03_ · https://arxiv.org/abs/2603.02648v1  
  <details><summary>Abstract</summary>

  Transparent object instance segmentation presents significant challenges in computer vision, due to the inherent properties of transparent objects, including boundary blur, low contrast, and high dependence on background context. Existing methods often fail as they depend on strong appearance cues and clear boundaries. To address these limitations, we propose SEP-YOLO, a novel framework that integrates a dual-domain collaborative mechanism for transparent object instance segmentation. Our method incorporates a Frequency Domain Detail Enhancement Module, which separates and enhances weak highfrequency boundary components via learnable complex weights. We further design a multi-scale spatial refinement stream, which consists of a Content-Aware Alignment Neck and a Multi-scale Gated Refinement Block, to ensure precise feature alignment and boundary localization in deep semantic features. We also provide high-quality instance-level annotations for the Trans10K dataset, filling the critical data gap in transparent object instance segmentation. Extensive experiments on the Trans10K and GVD datasets show that SEP-YOLO achieves state-of-the-art (SOTA) performance.

  </details>



- **Uni-Skill: Building Self-Evolving Skill Repository for Generalizable Robotic Manipulation**  
  Senwei Xie, Yuntian Zhang, Ruiping Wang, Xilin Chen  
  _2026-03-03_ · https://arxiv.org/abs/2603.02623v1  
  <details><summary>Abstract</summary>

  While skill-centric approaches leverage foundation models to enhance generalization in compositional tasks, they often rely on fixed skill libraries, limiting adaptability to new tasks without manual intervention. To address this, we propose Uni-Skill, a Unified Skill-centric framework that supports skill-aware planning and facilitates automatic skill evolution. Unlike prior methods that restrict planning to predefined skills, Uni-Skill requests for new skill implementations when existing ones are insufficient, ensuring adaptable planning with self-augmented skill library. To support automatic implementation of diverse skills requested by the planning module, we construct SkillFolder, a VerbNet-inspired repository derived from large-scale unstructured robotic videos. SkillFolder introduces a hierarchical skill taxonomy that captures diverse skill descriptions at multiple levels of abstraction. By populating this taxonomy with large-scale, automatically annotated demonstrations, Uni-Skill shifts the paradigm of skill acquisition from inefficient manual annotation to efficient offline structural retrieval. Retrieved examples provide semantic supervision over behavior patterns and fine-grained references for spatial trajectories, enabling few-shot skill inference without deployment-time demonstrations. Comprehensive experiments in both simulation and real-world settings verify the state-of-the-art performance of Uni-Skill over existing VLM-based skill-centric approaches, highlighting its advanced reasoning capabilities and strong zero-shot generalization across a wide range of novel tasks.

  </details>



- **Direct Reward Fine-Tuning on Poses for Single Image to 3D Human in the Wild**  
  Seunguk Do, Minwoo Huh, Joonghyuk Shin, Jaesik Park  
  _2026-03-03_ · https://arxiv.org/abs/2603.02619v1  
  <details><summary>Abstract</summary>

  Single-view 3D human reconstruction has achieved remarkable progress through the adoption of multi-view diffusion models, yet the recovered 3D humans often exhibit unnatural poses. This phenomenon becomes pronounced when reconstructing 3D humans with dynamic or challenging poses, which we attribute to the limited scale of available 3D human datasets with diverse poses. To address this limitation, we introduce DrPose, Direct Reward fine-tuning algorithm on Poses, which enables post-training of a multi-view diffusion model on diverse poses without requiring expensive 3D human assets. DrPose trains a model using only human poses paired with single-view images, employing a direct reward fine-tuning to maximize PoseScore, which is our proposed differentiable reward that quantifies consistency between a generated multi-view latent image and a ground-truth human pose. This optimization is conducted on DrPose15K, a novel dataset that was constructed from an existing human motion dataset and a pose-conditioned video generative model. Constructed from abundant human pose sequence data, DrPose15K exhibits a broader pose distribution compared to existing 3D human datasets. We validate our approach through evaluation on conventional benchmark datasets, in-the-wild images, and a newly constructed benchmark, with a particular focus on assessing performance on challenging human poses. Our results demonstrate consistent qualitative and quantitative improvements across all benchmarks. Project page: https://seunguk-do.github.io/drpose.

  </details>



- **Mind the Way You Select Negative Texts: Pursuing the Distance Consistency in OOD Detection with VLMs**  
  Zhikang Xu, Qianqian Xu, Zitai Wang, Cong Hua, Sicong Li, Zhiyong Yang, Qingming Huang  
  _2026-03-03_ · https://arxiv.org/abs/2603.02618v1  
  <details><summary>Abstract</summary>

  Out-of-distribution (OOD) detection seeks to identify samples from unknown classes, a critical capability for deploying machine learning models in open-world scenarios. Recent research has demonstrated that Vision-Language Models (VLMs) can effectively leverage their multi-modal representations for OOD detection. However, current methods often incorporate intra-modal distance during OOD detection, such as comparing negative texts with ID labels or comparing test images with image proxies. This design paradigm creates an inherent inconsistency against the inter-modal distance that CLIP-like VLMs are optimized for, potentially leading to suboptimal performance. To address this limitation, we propose InterNeg, a simple yet effective framework that systematically utilizes consistent inter-modal distance enhancement from textual and visual perspectives. From the textual perspective, we devise an inter-modal criterion for selecting negative texts. From the visual perspective, we dynamically identify high-confidence OOD images and invert them into the textual space, generating extra negative text embeddings guided by inter-modal distance. Extensive experiments across multiple benchmarks demonstrate the superiority of our approach. Notably, our InterNeg achieves state-of-the-art performance compared to existing works, with a 3.47\% reduction in FPR95 on the large-scale ImageNet benchmark and a 5.50\% improvement in AUROC on the challenging Near-OOD benchmark.

  </details>



- **Real-Time Generative Policy via Langevin-Guided Flow Matching for Autonomous Driving**  
  Tianze Zhu, Yinuo Wang, Wenjun Zou, Tianyi Zhang, Likun Wang, Letian Tao, Feihong Zhang, Yao Lyu, Shengbo Eben Li  
  _2026-03-03_ · https://arxiv.org/abs/2603.02613v1  
  <details><summary>Abstract</summary>

  Reinforcement learning (RL) is a fundamental methodology in autonomous driving systems, where generative policies exhibit considerable potential by leveraging their ability to model complex distributions to enhance exploration. However, their inherent high inference latency severely impedes their deployment in real-time decision-making and control. To address this issue, we propose diffusion actor-critic with entropy regulator via flow matching (DACER-F) by introducing flow matching into online RL, enabling the generation of competitive actions in a single inference step. By leveraging Langevin dynamics and gradients of the Q-function, DACER-F dynamically optimizes actions from experience replay toward a target distribution that balances high Q-value information with exploratory behavior. The flow policy is then trained to efficiently learn a mapping from a simple prior distribution to this dynamic target. In complex multi-lane and intersection simulations, DACER-F outperforms baselines diffusion actor-critic with entropy regulator (DACER) and distributional soft actor-critic (DSAC), while maintaining an ultra-low inference latency. DACER-F further demonstrates its scalability on standard RL benchmark DeepMind Control Suite (DMC), achieving a score of 775.8 in the humanoid-stand task and surpassing prior methods. Collectively, these results establish DACER-F as a high-performance and computationally efficient RL algorithm.

  </details>



- **Synthetic-Child: An AIGC-Based Synthetic Data Pipeline for Privacy-Preserving Child Posture Estimation**  
  Taowen Zeng  
  _2026-03-03_ · https://arxiv.org/abs/2603.02598v1  
  <details><summary>Abstract</summary>

  Accurate child posture estimation is critical for AI-powered study companion devices, yet collecting large-scale annotated datasets of children is both expensive and ethically prohibitive due to privacy concerns. We present Synthetic-Child, an AIGC-based synthetic data pipeline that produces photorealistic child posture training images with ground-truth-projected keypoint annotations, requiring zero real child photographs. The pipeline comprises four stages: (1) a programmable 3D child body model (SMPL-X) in Blender generates diverse desk-study poses with IK-constrained anatomical plausibility and automatic COCO-format ground-truth export; (2) a custom PoseInjectorNode feeds 3D-derived skeletons into a dual ControlNet (pose + depth) conditioned on FLUX-1 Dev, synthesizing 12,000 photorealistic images across 10 posture categories with low annotation drift; (3) ViTPose-based confidence filtering and targeted augmentation remove generation failures and improve robustness; (4) RTMPose-M (13.6M params) is fine-tuned on the synthetic data and paired with geometric feature engineering and a lightweight MLP for posture classification, then quantized to INT8 for real-time edge deployment. On a real-child test set (n~300), the FP16 model achieves 71.2 AP -- a +12.5 AP improvement over the COCO-pretrained adult-data baseline at identical model capacity. After INT8 quantization the model retains 70.4 AP while running at 22 FPS on a 0.8-TOPS Rockchip RK3568 NPU. In a single-subject controlled comparison with a commercial posture corrector, our system achieves substantially higher recognition rates across most tested categories and responds ~1.8x faster on average. These results demonstrate that carefully designed AIGC pipelines can substantially reduce dependence on real child imagery while achieving deployment-ready accuracy, with potential applications to other privacy-sensitive domains.

  </details>



- **Maximizing Generalization: The Effect of Different Augmentation Techniques on Lightweight Vision Transformer for Bengali Character Classification**  
  Rafi Hassan Chowdhury, Naimul Haque, Kaniz Fatiha  
  _2026-03-03_ · https://arxiv.org/abs/2603.02591v1  
  <details><summary>Abstract</summary>

  Deep learning models have proven to be highly effective in computer vision, with deep convolutional neural networks achieving impressive results across various computer vision tasks. However, these models rely heavily on large datasets to avoid overfitting. When a model learns features with either low or high variance, it can lead to underfitting or overfitting on the training data. Unfortunately, large-scale datasets may not be available in many domains, particularly for resource-limited languages such as Bengali. In this experiment, a series of tests were conducted in the field of image data augmentation as an approach to addressing the limited data problem for Bengali handwritten characters. The study also provides an in-depth analysis of the performance of different augmentation techniques. Data augmentation refers to a set of techniques applied to data to increase its size and diversity, making it more suitable for training deep learning models. The image augmentation techniques evaluated in this study include CLAHE, Random Rotation, Random Affine, Color Jitter, and their combinations. The study further explores the use of augmentation methods with a lightweight model such as EfficientViT. Among the different augmentation strategies, the combination of Random Affine and Color Jitter produced the best accuracy on the Ekush [1] and AIBangla [2] datasets, achieving accuracies of 97.48% and 97.57%, respectively. This combination outperformed all other individual and combined augmentation techniques. Overall, this analysis presents a thorough examination of the impact of image data augmentation in resource-scarce languages, particularly in the context of Bengali handwritten character recognition using lightweight models.

  </details>



- **Neural Electromagnetic Fields for High-Resolution Material Parameter Reconstruction**  
  Zhe Chen, Peilin Zheng, Wenshuo Chen, Xiucheng Wang, Yutao Yue, Nan Cheng  
  _2026-03-03_ · https://arxiv.org/abs/2603.02582v1  
  <details><summary>Abstract</summary>

  Creating functional Digital Twins, simulatable 3D replicas of the real world, is a central challenge in computer vision. Current methods like NeRF produce visually rich but functionally incomplete twins. The key barrier is the lack of underlying material properties (e.g., permittivity, conductivity). Acquiring this information for every point in a scene via non-contact, non-invasive sensing is a primary goal, but it demands solving a notoriously ill-posed physical inversion problem. Standard remote signals, like images and radio frequencies (RF), deeply entangle the unknown geometry, ambient field, and target materials. We introduce NEMF, a novel framework for dense, non-invasive physical inversion designed to build functional digital twins. Our key insight is a systematic disentanglement strategy. NEMF leverages high-fidelity geometry from images as a powerful anchor, which first enables the resolution of the ambient field. By constraining both geometry and field using only non-invasive data, the original ill-posed problem transforms into a well-posed, physics-supervised learning task. This transformation unlocks our core inversion module: a decoder. Guided by ambient RF signals and a differentiable layer incorporating physical reflection models, it learns to explicitly output a continuous, spatially-varying field of the scene's underlying material parameters. We validate our framework on high-fidelity synthetic datasets. Experiments show our non-invasive inversion reconstructs these material maps with high accuracy, and the resulting functional twin enables high-fidelity physical simulation. This advance moves beyond passive visual replicas, enabling the creation of truly functional and simulatable models of the physical world.

  </details>



- **CAWM-Mamba: A unified model for infrared-visible image fusion and compound adverse weather restoration**  
  Huichun Liu, Xiaosong Li, Zhuangfan Huang, Tao Ye, Yang Liu, Haishu Tan  
  _2026-03-03_ · https://arxiv.org/abs/2603.02560v1  
  <details><summary>Abstract</summary>

  Multimodal Image Fusion (MMIF) integrates complementary information from various modalities to produce clearer and more informative fused images. MMIF under adverse weather is particularly crucial in autonomous driving and UAV monitoring applications. However, existing adverse weather fusion methods generally only tackle single types of degradation such as haze, rain, or snow, and fail when multiple degradations coexist (e.g., haze+rain, rain+snow). To address this challenge, we propose Compound Adverse Weather Mamba (CAWM-Mamba), the first end-to-end framework that jointly performs image fusion and compound weather restoration with unified shared weights. Our network contains three key components: (1) a Weather-Aware Preprocess Module (WAPM) to enhance degraded visible features and extracts global weather embeddings; (2) a Cross-modal Feature Interaction Module (CFIM) to facilitate the alignment of heterogeneous modalities and exchange of complementary features across modalities; and (3) a Wavelet Space State Block (WSSB) that leverages wavelet-domain decomposition to decouple multi-frequency degradations. WSSB includes Freq-SSM, a module that models anisotropic high-frequency degradation without redundancy, and a unified degradation representation mechanism to further improve generalization across complex compound weather conditions. Extensive experiments on the AWMM-100K benchmark and three standard fusion datasets demonstrate that CAWM-Mamba consistently outperforms state-of-the-art methods in both compound and single-weather scenarios. In addition, our fusion results excel in downstream tasks covering semantic segmentation and object detection, confirming the practical value in real-world adverse weather perception. The source code will be available at https://github.com/Feecuin/CAWM-Mamba.

  </details>



- **CAPT: Confusion-Aware Prompt Tuning for Reducing Vision-Language Misalignment**  
  Maoyuan Shao, Yutong Gao, Xinyang Huang, Chuang Zhu, Lijuan Sun, Guoshun Nan  
  _2026-03-03_ · https://arxiv.org/abs/2603.02557v1  
  <details><summary>Abstract</summary>

  Vision-language models like CLIP have achieved remarkable progress in cross-modal representation learning, yet suffer from systematic misclassifications among visually and semantically similar categories. We observe that such confusion patterns are not random but persistently occur between specific category pairs, revealing the model's intrinsic bias and limited fine-grained discriminative ability. To address this, we propose CAPT, a Confusion-Aware Prompt Tuning framework that enables models to learn from their own misalignment. Specifically, we construct a Confusion Bank to explicitly model stable confusion relationships across categories and misclassified samples. On this basis, we introduce a Semantic Confusion Miner (SEM) to capture global inter-class confusion through semantic difference and commonality prompts, and a Sample Confusion Miner (SAM) to retrieve representative misclassified instances from the bank and capture sample-level cues through a Diff-Manner Adapter that integrates global and local contexts. To further unify confusion information across different granularities, a Multi-Granularity Difference Expert (MGDE) module is designed to jointly leverage semantic- and sample-level experts for more robust confusion-aware reasoning. Extensive experiments on 11 benchmark datasets demonstrate that our method significantly reduces confusion-induced errors while enhancing the discriminability and generalization of both base and novel classes, successfully resolving 50.72 percent of confusable sample pairs. Code will be released at https://github.com/greatest-gourmet/CAPT.

  </details>



- **Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs**  
  Zhiyu Pan, Yizheng Wu, Jiashen Hua, Junyi Feng, Shaotian Yan, Bing Deng, Zhiguo Cao, Jieping Ye  
  _2026-03-03_ · https://arxiv.org/abs/2603.02556v1  
  <details><summary>Abstract</summary>

  Reasoning has emerged as a key capability of large language models. In linguistic tasks, this capability can be enhanced by self-improving techniques that refine reasoning paths for subsequent finetuning. However, extending these language-based self-improving approaches to vision language models (VLMs) presents a unique challenge:~visual hallucinations in reasoning paths cannot be effectively verified or rectified. Our solution starts with a key observation about visual contrast: when presented with a contrastive VQA pair, i.e., two visually similar images with synonymous questions, VLMs identify relevant visual cues more precisely. Motivated by this observation, we propose Visual Contrastive Self-Taught Reasoner (VC-STaR), a novel self-improving framework that leverages visual contrast to mitigate hallucinations in model-generated rationales. We collect a diverse suite of VQA datasets, curate contrastive pairs according to multi-modal similarity, and generate rationales using VC-STaR. Consequently, we obtain a new visual reasoning dataset, VisCoR-55K, which is then used to boost the reasoning capability of various VLMs through supervised finetuning. Extensive experiments show that VC-STaR not only outperforms existing self-improving approaches but also surpasses models finetuned on the SoTA visual reasoning datasets, demonstrating that the inherent contrastive ability of VLMs can bootstrap their own visual reasoning. Project at: https://github.com/zhiyupan42/VC-STaR.

  </details>



- **SemGS: Feed-Forward Semantic 3D Gaussian Splatting from Sparse Views for Generalizable Scene Understanding**  
  Sheng Ye, Zhen-Hui Dong, Ruoyu Fan, Tian Lv, Yong-Jin Liu  
  _2026-03-03_ · https://arxiv.org/abs/2603.02548v1  
  <details><summary>Abstract</summary>

  Semantic understanding of 3D scenes is essential for robots to operate effectively and safely in complex environments. Existing methods for semantic scene reconstruction and semantic-aware novel view synthesis often rely on dense multi-view inputs and require scene-specific optimization, limiting their practicality and scalability in real-world applications. To address these challenges, we propose SemGS, a feed-forward framework for reconstructing generalizable semantic fields from sparse image inputs. SemGS uses a dual-branch architecture to extract color and semantic features, where the two branches share shallow CNN layers, allowing semantic reasoning to leverage textural and structural cues in color appearance. We also incorporate a camera-aware attention mechanism into the feature extractor to explicitly model geometric relationships between camera viewpoints. The extracted features are decoded into dual-Gaussians that share geometric consistency while preserving branch-specific attributes, and further rasterized to synthesize semantic maps under novel viewpoints. Additionally, we introduce a regional smoothness loss to enhance semantic coherence. Experiments show that SemGS achieves state-of-the-art performance on benchmark datasets, while providing rapid inference and strong generalization capabilities across diverse synthetic and real-world scenarios.

  </details>



- **On Discriminative vs. Generative classifiers: Rethinking MLLMs for Action Understanding**  
  Zhanzhong Pang, Dibyadip Chatterjee, Fadime Sener, Angela Yao  
  _2026-03-03_ · https://arxiv.org/abs/2603.02546v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have advanced open-world action understanding and can be adapted as generative classifiers for closed-set settings by autoregressively generating action labels as text. However, this approach is inefficient, and shared subwords across action labels introduce semantic overlap, leading to ambiguity in generation. In contrast, discriminative classifiers learn task-specific representations with clear decision boundaries, enabling efficient one-step classification without autoregressive decoding. We first compare generative and discriminative classifiers with MLLMs for closed-set action understanding, revealing the superior accuracy and efficiency of the latter. To bridge the performance gap, we design strategies that elevate generative classifiers toward performance comparable with discriminative ones. Furthermore, we show that generative modeling can complement discriminative classifiers, leading to better performance while preserving efficiency. To this end, we propose Generation-Assisted Discriminative~(GAD) classifier for closed-set action understanding. GAD operates only during fine-tuning, preserving full compatibility with MLLM pretraining. Extensive experiments on temporal action understanding benchmarks demonstrate that GAD improves both accuracy and efficiency over generative methods, achieving state-of-the-art results on four tasks across five datasets, including an average 2.5% accuracy gain and 3x faster inference on our largest COIN benchmark.

  </details>



- **ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person Detection**  
  Deokyun Kim, Jeongjun Lee, Jungwon Choi, Jonggeon Park, Giyoung Lee, Yookyung Kim, Myungseok Ki, Juho Lee, Jihun Cha  
  _2026-03-03_ · https://arxiv.org/abs/2603.02541v1  
  <details><summary>Abstract</summary>

  Detecting missing persons in forest environments remains a challenge, as dense canopy cover often conceals individuals from detection in top-down or oblique aerial imagery typically captured by Unmanned Aerial Vehicles (UAVs). While UAVs are effective for covering large, inaccessible areas, their aerial perspectives often miss critical visual cues beneath the forest canopy. This limitation underscores the need for under-canopy perspectives better suited for detecting missing persons in such environments. To address this gap, we introduce ForestPersons, a novel large-scale dataset specifically designed for under-canopy person detection. ForestPersons contains 96,482 images and 204,078 annotations collected under diverse environmental and temporal conditions. Each annotation includes a bounding box, pose, and visibility label for occlusion-aware analysis. ForestPersons provides ground-level and low-altitude perspectives that closely reflect the visual conditions encountered by Micro Aerial Vehicles (MAVs) during forest Search and Rescue (SAR) missions. Our baseline evaluations reveal that standard object detection models, trained on prior large-scale object detection datasets or SAR-oriented datasets, show limited performance on ForestPersons. This indicates that prior benchmarks are not well aligned with the challenges of missing person detection under the forest canopy. We offer this benchmark to support advanced person detection capabilities in real-world SAR scenarios. The dataset is publicly available at https://huggingface.co/datasets/etri/ForestPersons.

  </details>



- **LLM-MLFFN: Multi-Level Autonomous Driving Behavior Feature Fusion via Large Language Model**  
  Xiangyu Li, Tianyi Wang, Xi Cheng, Rakesh Chowdary Machineni, Zhaomiao Guo, Sikai Chen, Junfeng Jiao, Christian Claudel  
  _2026-03-03_ · https://arxiv.org/abs/2603.02528v1  
  <details><summary>Abstract</summary>

  Accurate classification of autonomous vehicle (AV) driving behaviors is critical for safety validation, performance diagnosis, and traffic integration analysis. However, existing approaches primarily rely on numerical time-series modeling and often lack semantic abstraction, limiting interpretability and robustness in complex traffic environments. This paper presents LLM-MLFFN, a novel large language model (LLM)-enhanced multi-level feature fusion network designed to address the complexities of multi-dimensional driving data. The proposed LLM-MLFFN framework integrates priors from largescale pre-trained models and employs a multi-level approach to enhance classification accuracy. LLM-MLFFN comprises three core components: (1) a multi-level feature extraction module that extracts statistical, behavioral, and dynamic features to capture the quantitative aspects of driving behaviors; (2) a semantic description module that leverages LLMs to transform raw data into high-level semantic features; and (3) a dual-channel multi-level feature fusion network that combines numerical and semantic features using weighted attention mechanisms to improve robustness and prediction accuracy. Evaluation on the Waymo open trajectory dataset demonstrates the superior performance of the proposed LLM-MLFFN, achieving a classification accuracy of over 94%, surpassing existing machine learning models. Ablation studies further validate the critical contributions of multi-level fusion, feature extraction strategies, and LLM-derived semantic reasoning. These results suggest that integrating structured feature modeling with language-driven semantic abstraction provides a principled and interpretable pathway for robust autonomous driving behavior classification.

  </details>



- **Beyond Anatomy: Explainable ASD Classification from rs-fMRI via Functional Parcellation and Graph Attention Networks**  
  Syeda Hareem Madani, Noureen Bibi, Adam Rafiq Jeraj, Sumra Khan, Anas Zafar, Rizwan Qureshi  
  _2026-03-03_ · https://arxiv.org/abs/2603.02518v1  
  <details><summary>Abstract</summary>

  Anatomical brain parcellations dominate rs-fMRI-based Autism Spectrum Disorder (ASD) classification, yet their rigid boundaries may fail to capture the idiosyncratic connectivity patterns that characterise ASD. We present a graph-based deep learning framework comparing anatomical (AAL, 116 ROIs) and functionally-derived (MSDL, 39 ROIs) parcellation strategies on the ABIDE I dataset. Our FSL preprocessing pipeline handles multi-site heterogeneity across 400 balanced subjects, with site-stratified 70/15/15 splits to prevent data leakage. Gaussian noise augmentation within training folds expands samples from 280 to 1,680. A three phase pipeline progresses from a baseline GCN with AAL (73.3% accuracy, AUC=0.74), to an optimised GCN with MSDL (84.0%, AUC=0.84), to a Graph Attention Network ensemble achieving 95.0% accuracy (AUC=0.98), outperforming all recent GNN-based benchmarks on ABIDE I. The 10.7-point gain from atlas substitution alone demonstrates that functional parcellation is the most impactful modelling decision. Gradient-based saliency and GNNExplainer analyses converge on the Posterior Cingulate Cortex and Precuneus as core Default Mode Network hubs, validating that model decisions reflect ASD neuropathology rather than acquisition artefacts. All code and datasets will be publicly released upon acceptance.

  </details>


