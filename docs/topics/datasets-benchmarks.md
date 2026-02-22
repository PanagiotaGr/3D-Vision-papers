# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-02-22 07:04 UTC_

Total papers shown: **14**


---

- **When Vision Overrides Language: Evaluating and Mitigating Counterfactual Failures in VLAs**  
  Yu Fang, Yuchun Feng, Dong Jing, Jiaqi Liu, Yue Yang, Zhenyu Wei, Daniel Szafir, Mingyu Ding  
  _2026-02-19_ · https://arxiv.org/abs/2602.17659v1  
  <details><summary>Abstract</summary>

  Vision-Language-Action models (VLAs) promise to ground language instructions in robot control, yet in practice often fail to faithfully follow language. When presented with instructions that lack strong scene-specific supervision, VLAs suffer from counterfactual failures: they act based on vision shortcuts induced by dataset biases, repeatedly executing well-learned behaviors and selecting objects frequently seen during training regardless of language intent. To systematically study it, we introduce LIBERO-CF, the first counterfactual benchmark for VLAs that evaluates language following capability by assigning alternative instructions under visually plausible LIBERO layouts. Our evaluation reveals that counterfactual failures are prevalent yet underexplored across state-of-the-art VLAs. We propose Counterfactual Action Guidance (CAG), a simple yet effective dual-branch inference scheme that explicitly regularizes language conditioning in VLAs. CAG combines a standard VLA policy with a language-unconditioned Vision-Action (VA) module, enabling counterfactual comparison during action selection. This design reduces reliance on visual shortcuts, improves robustness on under-observed tasks, and requires neither additional demonstrations nor modifications to existing architectures or pretrained models. Extensive experiments demonstrate its plug-and-play integration across diverse VLAs and consistent improvements. For example, on LIBERO-CF, CAG improves $π_{0.5}$ by 9.7% in language following accuracy and 3.6% in task success on under-observed tasks using a training-free strategy, with further gains of 15.5% and 8.5%, respectively, when paired with a VA model. In real-world evaluations, CAG reduces counterfactual failures of 9.4% and improves task success by 17.2% on average.

  </details>



- **IntRec: Intent-based Retrieval with Contrastive Refinement**  
  Pourya Shamsolmoali, Masoumeh Zareapoor, Eric Granger, Yue Lu  
  _2026-02-19_ · https://arxiv.org/abs/2602.17639v1  
  <details><summary>Abstract</summary>

  Retrieving user-specified objects from complex scenes remains a challenging task, especially when queries are ambiguous or involve multiple similar objects. Existing open-vocabulary detectors operate in a one-shot manner, lacking the ability to refine predictions based on user feedback. To address this, we propose IntRec, an interactive object retrieval framework that refines predictions based on user feedback. At its core is an Intent State (IS) that maintains dual memory sets for positive anchors (confirmed cues) and negative constraints (rejected hypotheses). A contrastive alignment function ranks candidate objects by maximizing similarity to positive cues while penalizing rejected ones, enabling fine-grained disambiguation in cluttered scenes. Our interactive framework provides substantial improvements in retrieval accuracy without additional supervision. On LVIS, IntRec achieves 35.4 AP, outperforming OVMR, CoDet, and CAKE by +2.3, +3.7, and +0.5, respectively. On the challenging LVIS-Ambiguous benchmark, it improves performance by +7.9 AP over its one-shot baseline after a single corrective feedback, with less than 30 ms of added latency per interaction.

  </details>



- **CORAL: Correspondence Alignment for Improved Virtual Try-On**  
  Jiyoung Kim, Youngjin Shin, Siyoon Jin, Dahyun Chung, Jisu Nam, Tongmin Kim, Jongjae Park, Hyeonwoo Kang, Seungryong Kim  
  _2026-02-19_ · https://arxiv.org/abs/2602.17636v1  
  <details><summary>Abstract</summary>

  Existing methods for Virtual Try-On (VTON) often struggle to preserve fine garment details, especially in unpaired settings where accurate person-garment correspondence is required. These methods do not explicitly enforce person-garment alignment and fail to explain how correspondence emerges within Diffusion Transformers (DiTs). In this paper, we first analyze full 3D attention in DiT-based architecture and reveal that the person-garment correspondence critically depends on precise person-garment query-key matching within the full 3D attention. Building on this insight, we then introduce CORrespondence ALignment (CORAL), a DiT-based framework that explicitly aligns query-key matching with robust external correspondences. CORAL integrates two complementary components: a correspondence distillation loss that aligns reliable matches with person-garment attention, and an entropy minimization loss that sharpens the attention distribution. We further propose a VLM-based evaluation protocol to better reflect human preference. CORAL consistently improves over the baseline, enhancing both global shape transfer and local detail preservation. Extensive ablations validate our design choices.

  </details>



- **Adapting Actively on the Fly: Relevance-Guided Online Meta-Learning with Latent Concepts for Geospatial Discovery**  
  Jowaria Khan, Anindya Sarkar, Yevgeniy Vorobeychik, Elizabeth Bondi-Kelly  
  _2026-02-19_ · https://arxiv.org/abs/2602.17605v1  
  <details><summary>Abstract</summary>

  In many real-world settings, such as environmental monitoring, disaster response, or public health, with costly and difficult data collection and dynamic environments, strategically sampling from unobserved regions is essential for efficiently uncovering hidden targets under tight resource constraints. Yet, sparse and biased geospatial ground truth limits the applicability of existing learning-based methods, such as reinforcement learning. To address this, we propose a unified geospatial discovery framework that integrates active learning, online meta-learning, and concept-guided reasoning. Our approach introduces two key innovations built on a shared notion of *concept relevance*, which captures how domain-specific factors influence target presence: a *concept-weighted uncertainty sampling strategy*, where uncertainty is modulated by learned relevance based on readily-available domain-specific concepts (e.g., land cover, source proximity); and a *relevance-aware meta-batch formation strategy* that promotes semantic diversity during online-meta updates, improving generalization in dynamic environments. Our experiments include testing on a real-world dataset of cancer-causing PFAS (Per- and polyfluoroalkyl substances) contamination, showcasing our method's reliability at uncovering targets with limited data and a varying environment.

  </details>



- **Art2Mus: Artwork-to-Music Generation via Visual Conditioning and Large-Scale Cross-Modal Alignment**  
  Ivan Rinaldi, Matteo Mendula, Nicola Fanelli, Florence Levé, Matteo Testi, Giovanna Castellano, Gennaro Vessio  
  _2026-02-19_ · https://arxiv.org/abs/2602.17599v1  
  <details><summary>Abstract</summary>

  Music generation has advanced markedly through multimodal deep learning, enabling models to synthesize audio from text and, more recently, from images. However, existing image-conditioned systems suffer from two fundamental limitations: (i) they are typically trained on natural photographs, limiting their ability to capture the richer semantic, stylistic, and cultural content of artworks; and (ii) most rely on an image-to-text conversion stage, using language as a semantic shortcut that simplifies conditioning but prevents direct visual-to-audio learning. Motivated by these gaps, we introduce ArtSound, a large-scale multimodal dataset of 105,884 artwork-music pairs enriched with dual-modality captions, obtained by extending ArtGraph and the Free Music Archive. We further propose ArtToMus, the first framework explicitly designed for direct artwork-to-music generation, which maps digitized artworks to music without image-to-text translation or language-based semantic supervision. The framework projects visual embeddings into the conditioning space of a latent diffusion model, enabling music synthesis guided solely by visual information. Experimental results show that ArtToMus generates musically coherent and stylistically consistent outputs that reflect salient visual cues of the source artworks. While absolute alignment scores remain lower than those of text-conditioned systems-as expected given the substantially increased difficulty of removing linguistic supervision-ArtToMus achieves competitive perceptual quality and meaningful cross-modal correspondence. This work establishes direct visual-to-music generation as a distinct and challenging research direction, and provides resources that support applications in multimedia art, cultural heritage, and AI-assisted creative practice. Code and dataset will be publicly released upon acceptance.

  </details>



- **Conditional Flow Matching for Continuous Anomaly Detection in Autonomous Driving on a Manifold-Aware Spectral Space**  
  Antonio Guillen-Perez  
  _2026-02-19_ · https://arxiv.org/abs/2602.17586v1  
  <details><summary>Abstract</summary>

  Safety validation for Level 4 autonomous vehicles (AVs) is currently bottlenecked by the inability to scale the detection of rare, high-risk long-tail scenarios using traditional rule-based heuristics. We present Deep-Flow, an unsupervised framework for safety-critical anomaly detection that utilizes Optimal Transport Conditional Flow Matching (OT-CFM) to characterize the continuous probability density of expert human driving behavior. Unlike standard generative approaches that operate in unstable, high-dimensional coordinate spaces, Deep-Flow constrains the generative process to a low-rank spectral manifold via a Principal Component Analysis (PCA) bottleneck. This ensures kinematic smoothness by design and enables the computation of the exact Jacobian trace for numerically stable, deterministic log-likelihood estimation. To resolve multi-modal ambiguity at complex junctions, we utilize an Early Fusion Transformer encoder with lane-aware goal conditioning, featuring a direct skip-connection to the flow head to maintain intent-integrity throughout the network. We introduce a kinematic complexity weighting scheme that prioritizes high-energy maneuvers (quantified via path tortuosity and jerk) during the simulation-free training process. Evaluated on the Waymo Open Motion Dataset (WOMD), our framework achieves an AUC-ROC of 0.766 against a heuristic golden set of safety-critical events. More significantly, our analysis reveals a fundamental distinction between kinematic danger and semantic non-compliance. Deep-Flow identifies a critical predictability gap by surfacing out-of-distribution behaviors, such as lane-boundary violations and non-normative junction maneuvers, that traditional safety filters overlook. This work provides a mathematically rigorous foundation for defining statistical safety gates, enabling objective, data-driven validation for the safe deployment of autonomous fleets.

  </details>



- **FR-GESTURE: An RGBD Dataset For Gesture-based Human-Robot Interaction In First Responder Operations**  
  Konstantinos Foteinos, Georgios Angelidis, Aggelos Psiris, Vasileios Argyriou, Panagiotis Sarigiannidis, Georgios Th. Papadopoulos  
  _2026-02-19_ · https://arxiv.org/abs/2602.17573v1  
  <details><summary>Abstract</summary>

  The ever increasing intensity and number of disasters make even more difficult the work of First Responders (FRs). Artificial intelligence and robotics solutions could facilitate their operations, compensating these difficulties. To this end, we propose a dataset for gesture-based UGV control by FRs, introducing a set of 12 commands, drawing inspiration from existing gestures used by FRs and tactical hand signals and refined after incorporating feedback from experienced FRs. Then we proceed with the data collection itself, resulting in 3312 RGBD pairs captured from 2 viewpoints and 7 distances. To the best of our knowledge, this is the first dataset especially intended for gesture-based UGV guidance by FRs. Finally we define evaluation protocols for our RGBD dataset, termed FR-GESTURE, and we perform baseline experiments, which are put forward for improvement. We have made data publicly available to promote future research on the domain: https://doi.org/10.5281/zenodo.18131333.

  </details>



- **RetouchIQ: MLLM Agents for Instruction-Based Image Retouching with Generalist Reward**  
  Qiucheng Wu, Jing Shi, Simon Jenni, Kushal Kafle, Tianyu Wang, Shiyu Chang, Handong Zhao  
  _2026-02-19_ · https://arxiv.org/abs/2602.17558v1  
  <details><summary>Abstract</summary>

  Recent advances in multimodal large language models (MLLMs) have shown great potential for extending vision-language reasoning to professional tool-based image editing, enabling intuitive and creative editing. A promising direction is to use reinforcement learning (RL) to enable MLLMs to reason about and execute optimal tool-use plans within professional image-editing software. However, training remains challenging due to the lack of reliable, verifiable reward signals that can reflect the inherently subjective nature of creative editing. In this work, we introduce RetouchIQ, a framework that performs instruction-based executable image editing through MLLM agents guided by a generalist reward model. RetouchIQ interprets user-specified editing intentions and generates corresponding, executable image adjustments, bridging high-level aesthetic goals with precise parameter control. To move beyond conventional, rule-based rewards that compute similarity against a fixed reference image using handcrafted metrics, we propose a generalist reward model, an RL fine-tuned MLLM that evaluates edited results through a set of generated metrics on a case-by-case basis. Then, the reward model provides scalar feedback through multimodal reasoning, enabling reinforcement learning with high-quality, instruction-consistent gradients. We curate an extended dataset with 190k instruction-reasoning pairs and establish a new benchmark for instruction-based image editing. Experiments show that RetouchIQ substantially improves both semantic consistency and perceptual quality over previous MLLM-based and diffusion-based editing systems. Our findings demonstrate the potential of generalist reward-driven MLLM agents as flexible, explainable, and executable assistants for professional image editing.

  </details>



- **Tracing Copied Pixels and Regularizing Patch Affinity in Copy Detection**  
  Yichen Lu, Siwei Nie, Minlong Lu, Xudong Yang, Xiaobo Zhang, Peng Zhang  
  _2026-02-19_ · https://arxiv.org/abs/2602.17484v1  
  <details><summary>Abstract</summary>

  Image Copy Detection (ICD) aims to identify manipulated content between image pairs through robust feature representation learning. While self-supervised learning (SSL) has advanced ICD systems, existing view-level contrastive methods struggle with sophisticated edits due to insufficient fine-grained correspondence learning. We address this limitation by exploiting the inherent geometric traceability in edited content through two key innovations. First, we propose PixTrace - a pixel coordinate tracking module that maintains explicit spatial mappings across editing transformations. Second, we introduce CopyNCE, a geometrically-guided contrastive loss that regularizes patch affinity using overlap ratios derived from PixTrace's verified mappings. Our method bridges pixel-level traceability with patch-level similarity learning, suppressing supervision noise in SSL training. Extensive experiments demonstrate not only state-of-the-art performance (88.7% uAP / 83.9% RP90 for matcher, 72.6% uAP / 68.4% RP90 for descriptor on DISC21 dataset) but also better interpretability over existing methods.

  </details>



- **QuPAINT: Physics-Aware Instruction Tuning Approach to Quantum Material Discovery**  
  Xuan-Bac Nguyen, Hoang-Quan Nguyen, Sankalp Pandey, Tim Faltermeier, Nicholas Borys, Hugh Churchill, Khoa Luu  
  _2026-02-19_ · https://arxiv.org/abs/2602.17478v1  
  <details><summary>Abstract</summary>

  Characterizing two-dimensional quantum materials from optical microscopy images is challenging due to the subtle layer-dependent contrast, limited labeled data, and significant variation across laboratories and imaging setups. Existing vision models struggle in this domain since they lack physical priors and cannot generalize to new materials or hardware conditions. This work presents a new physics-aware multimodal framework that addresses these limitations from both the data and model perspectives. We first present Synthia, a physics-based synthetic data generator that simulates realistic optical responses of quantum material flakes under thin-film interference. Synthia produces diverse and high-quality samples, helping reduce the dependence on expert manual annotation. We introduce QMat-Instruct, the first large-scale instruction dataset for quantum materials, comprising multimodal, physics-informed question-answer pairs designed to teach Multimodal Large Language Models (MLLMs) to understand the appearance and thickness of flakes. Then, we propose Physics-Aware Instruction Tuning (QuPAINT), a multimodal architecture that incorporates a Physics-Informed Attention module to fuse visual embeddings with optical priors, enabling more robust and discriminative flake representations. Finally, we establish QF-Bench, a comprehensive benchmark spanning multiple materials, substrates, and imaging settings, offering standardized protocols for fair and reproducible evaluation.

  </details>



- **Attachment Anchors: A Novel Framework for Laparoscopic Grasping Point Prediction in Colorectal Surgery**  
  Dennis N. Schneider, Lars Wagner, Daniel Rueckert, Dirk Wilhelm  
  _2026-02-19_ · https://arxiv.org/abs/2602.17310v1  
  <details><summary>Abstract</summary>

  Accurate grasping point prediction is a key challenge for autonomous tissue manipulation in minimally invasive surgery, particularly in complex and variable procedures such as colorectal interventions. Due to their complexity and prolonged duration, colorectal procedures have been underrepresented in current research. At the same time, they pose a particularly interesting learning environment due to repetitive tissue manipulation, making them a promising entry point for autonomous, machine learning-driven support. Therefore, in this work, we introduce attachment anchors, a structured representation that encodes the local geometric and mechanical relationships between tissue and its anatomical attachments in colorectal surgery. This representation reduces uncertainty in grasping point prediction by normalizing surgical scenes into a consistent local reference frame. We demonstrate that attachment anchors can be predicted from laparoscopic images and incorporated into a grasping framework based on machine learning. Experiments on a dataset of 90 colorectal surgeries demonstrate that attachment anchors improve grasping point prediction compared to image-only baselines. There are particularly strong gains in out-of-distribution settings, including unseen procedures and operating surgeons. These results suggest that attachment anchors are an effective intermediate representation for learning-based tissue manipulation in colorectal surgery.

  </details>



- **Physics Encoded Spatial and Temporal Generative Adversarial Network for Tropical Cyclone Image Super-resolution**  
  Ruoyi Zhang, Jiawei Yuan, Lujia Ye, Runling Yu, Liling Zhao  
  _2026-02-19_ · https://arxiv.org/abs/2602.17277v1  
  <details><summary>Abstract</summary>

  High-resolution satellite imagery is indispensable for tracking the genesis, intensification, and trajectory of tropical cyclones (TCs). However, existing deep learning-based super-resolution (SR) methods often treat satellite image sequences as generic videos, neglecting the underlying atmospheric physical laws governing cloud motion. To address this, we propose a Physics Encoded Spatial and Temporal Generative Adversarial Network (PESTGAN) for TC image super-resolution. Specifically, we design a disentangled generator architecture incorporating a PhyCell module, which approximates the vorticity equation via constrained convolutions and encodes the resulting approximate physical dynamics as implicit latent representations to separate physical dynamics from visual textures. Furthermore, a dual-discriminator framework is introduced, employing a temporal discriminator to enforce motion consistency alongside spatial realism. Experiments on the Digital Typhoon dataset for 4$\times$ upscaling demonstrate that PESTGAN establishes a better performance in structural fidelity and perceptual quality. While maintaining competitive pixel-wise accuracy compared to existing approaches, our method significantly excels in reconstructing meteorologically plausible cloud structures with superior physical fidelity.

  </details>



- **EA-Swin: An Embedding-Agnostic Swin Transformer for AI-Generated Video Detection**  
  Hung Mai, Loi Dinh, Duc Hai Nguyen, Dat Do, Luong Doan, Khanh Nguyen Quoc, Huan Vu, Phong Ho, Naeem Ul Islam, Tuan Do  
  _2026-02-19_ · https://arxiv.org/abs/2602.17260v1  
  <details><summary>Abstract</summary>

  Recent advances in foundation video generators such as Sora2, Veo3, and other commercial systems have produced highly realistic synthetic videos, exposing the limitations of existing detection methods that rely on shallow embedding trajectories, image-based adaptation, or computationally heavy MLLMs. We propose EA-Swin, an Embedding-Agnostic Swin Transformer that models spatiotemporal dependencies directly on pretrained video embeddings via a factorized windowed attention design, making it compatible with generic ViT-style patch-based encoders. Alongside the model, we construct the EA-Video dataset, a benchmark dataset comprising 130K videos that integrates newly collected samples with curated existing datasets, covering diverse commercial and open-source generators and including unseen-generator splits for rigorous cross-distribution evaluation. Extensive experiments show that EA-Swin achieves 0.97-0.99 accuracy across major generators, outperforming prior SoTA methods (typically 0.8-0.9) by a margin of 5-20%, while maintaining strong generalization to unseen distributions, establishing a scalable and robust solution for modern AI-generated video detection.

  </details>



- **FRAPPE: Infusing World Modeling into Generalist Policies via Multiple Future Representation Alignment**  
  Han Zhao, Jingbo Wang, Wenxuan Song, Shuai Chen, Yang Liu, Yan Wang, Haoang Li, Donglin Wang  
  _2026-02-19_ · https://arxiv.org/abs/2602.17259v1  
  <details><summary>Abstract</summary>

  Enabling VLA models to predict environmental dynamics, known as world modeling, has been recognized as essential for improving robotic reasoning and generalization. However, current approaches face two main issues: 1. The training objective forces models to over-emphasize pixel-level reconstruction, which constrains semantic learning and generalization 2. Reliance on predicted future observations during inference often leads to error accumulation. To address these challenges, we introduce Future Representation Alignment via Parallel Progressive Expansion (FRAPPE). Our method adopts a two-stage fine-tuning strategy: In the mid-training phase, the model learns to predict the latent representations of future observations; In the post-training phase, we expand the computational workload in parallel and align the representation simultaneously with multiple different visual foundation models. By significantly improving fine-tuning efficiency and reducing dependence on action-annotated data, FRAPPE provides a scalable and data-efficient pathway to enhance world-awareness in generalist robotic policies. Experiments on the RoboTwin benchmark and real-world tasks demonstrate that FRAPPE outperforms state-of-the-art approaches and shows strong generalization in long-horizon and unseen scenarios.

  </details>


