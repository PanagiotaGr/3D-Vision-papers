# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-04-08 07:51 UTC_

Total papers shown: **50**


---

- **The Character Error Vector: Decomposable errors for page-level OCR evaluation**  
  Jonathan Bourne, Mwiza Simbeye, Joseph Nockels  
  _2026-04-07_ · https://arxiv.org/abs/2604.06160v1  
  <details><summary>Abstract</summary>

  The Character Error Rate (CER) is a key metric for evaluating the quality of Optical Character Recognition (OCR). However, this metric assumes that text has been perfectly parsed, which is often not the case. Under page-parsing errors, CER becomes undefined, limiting its use as a metric and making evaluating page-level OCR challenging, particularly when using data that do not share a labelling schema. We introduce the Character Error Vector (CEV), a bag-of-characters evaluator for OCR. The CEV can be decomposed into parsing and OCR, and interaction error components. This decomposability allows practitioners to focus on the part of the Document Understanding pipeline that will have the greatest impact on overall text extraction quality. The CEV can be implemented using a variety of methods, of which we demonstrate SpACER (Spatially Aware Character Error Rate) and a Character distribution method using the Jensen-Shannon Distance. We validate the CEV's performance against other metrics: first, the relationship with CER; then, parse quality; and finally, as a direct measure of page-level OCR quality. The validation process shows that the CEV is a valuable bridge between parsing metrics and local metrics like CER. We analyse a dataset of archival newspapers made of degraded images with complex layouts and find that state-of-the-art end-to-end models are outperformed by more traditional pipeline approaches. Whilst the CEV requires character-level positioning for optimal triage, thresholding on easily available values can predict the main error source with an F1 of 0.91. We provide the CEV as part of a Python library to support Document understanding research.

  </details>



- **MMEmb-R1: Reasoning-Enhanced Multimodal Embedding with Pair-Aware Selection and Adaptive Control**  
  Yuchi Wang, Haiyang Yu, Weikang Bian, Jiefeng Long, Xiao Liang, Chao Feng, Hongsheng Li  
  _2026-04-07_ · https://arxiv.org/abs/2604.06156v1  
  <details><summary>Abstract</summary>

  MLLMs have been successfully applied to multimodal embedding tasks, yet their generative reasoning capabilities remain underutilized. Directly incorporating chain-of-thought reasoning into embedding learning introduces two fundamental challenges. First, structural misalignment between instance-level reasoning and pairwise contrastive supervision may lead to shortcut behavior, where the model merely learns the superficial format of reasoning. Second, reasoning is not universally beneficial for embedding tasks. Enforcing reasoning for all inputs may introduce unnecessary computation and latency, and can even obscure salient semantic signals for simple cases. To address these issues, we propose MMEmb-R1, an adaptive reasoning-based multimodal embedding framework. We formulate reasoning as a latent variable and introduce pair-aware reasoning selection that employs counterfactual intervention to identify reasoning paths beneficial for query-target alignment. Furthermore, we adopt reinforcement learning to selectively invoke reasoning only when necessary. Experiments on the MMEB-V2 benchmark demonstrate that our model achieves a score of 71.2 with only 4B parameters, establishing a new state-of-the-art while significantly reducing reasoning overhead and inference latency.

  </details>



- **Lightweight Multimodal Adaptation of Vision Language Models for Species Recognition and Habitat Context Interpretation in Drone Thermal Imagery**  
  Hao Chen, Fang Qiu, Fangchao Dong, Defei Yang, Eve Bohnett, Li An  
  _2026-04-07_ · https://arxiv.org/abs/2604.06124v1  
  <details><summary>Abstract</summary>

  This study proposes a lightweight multimodal adaptation framework to bridge the representation gap between RGB-pretrained VLMs and thermal infrared imagery, and demonstrates its practical utility using a real drone-collected dataset. A thermal dataset was developed from drone-collected imagery and was used to fine-tune VLMs through multimodal projector alignment, enabling the transfer of information from RGB-based visual representations to thermal radiometric inputs. Three representative models, including InternVL3-8B-Instruct, Qwen2.5-VL-7B-Instruct, and Qwen3-VL-8B-Instruct, were benchmarked under both closed-set and open-set prompting conditions for species recognition and instance enumeration. Among the tested models, Qwen3-VL-8B-Instruct with open-set prompting achieved the best overall performance, with F1 scores of 0.935 for deer, 0.915 for rhino, and 0.968 for elephant, and within-1 enumeration accuracies of 0.779, 0.982, and 1.000, respectively. In addition, combining thermal imagery with simultaneously collected RGB imagery enabled the model to generate habitat-context information, including land-cover characteristics, key landscape features, and visible human disturbance. Overall, the findings demonstrate that lightweight projector-based adaptation provides an effective and practical route for transferring RGB-pretrained VLMs to thermal drone imagery, expanding their utility from object-level recognition to habitat-context interpretation in ecological monitoring.

  </details>



- **Extending ZACH-ViT to Robust Medical Imaging: Corruption and Adversarial Stress Testing in Low-Data Regimes**  
  Athanasios Angelakis, Marta Gomez-Barrero  
  _2026-04-07_ · https://arxiv.org/abs/2604.06099v1  
  <details><summary>Abstract</summary>

  The recently introduced ZACH-ViT (Zero-token Adaptive Compact Hierarchical Vision Transformer) formalized a compact permutation-invariant Vision Transformer for medical imaging and argued that architectural alignment with spatial structure can matter more than universal benchmark dominance. Its design was motivated by the observation that positional embeddings and a dedicated class token encode fixed spatial assumptions that may be suboptimal when spatial organization is weakly informative, locally distributed, or variable across biomedical images. The foundational study established a regime-dependent clean performance profile across MedMNIST, but did not examine robustness in detail. In this work, we present the first robustness-focused extension of ZACH-ViT by evaluating its behavior under common image corruptions and adversarial perturbations in the same low-data setting. We compare ZACH-ViT with three scratch-trained compact baselines, ABMIL, Minimal-ViT, and TransMIL, on seven MedMNIST datasets using 50 samples per class, fixed hyperparameters, and five random seeds. Across the benchmark, ZACH-ViT achieves the best overall mean rank on clean data (1.57) and under common corruptions (1.57), indicating a favorable balance between baseline predictive performance and robustness to realistic image degradation. Under adversarial stress, all models deteriorate substantially; nevertheless, ZACH-ViT remains competitive, ranking first under FGSM (2.00) and second under PGD (2.29), where ABMIL performs best overall. These results extend the original ZACH-ViT narrative: the advantages of compact permutation-invariant transformers are not limited to clean evaluation, but can persist under realistic perturbation stress in low-data medical imaging, while adversarial robustness remains an open challenge for all evaluated models.

  </details>



- **Scientific Graphics Program Synthesis via Dual Self-Consistency Reinforcement Learning**  
  Juekai Lin, Yun Zhu, Honglin Lin, Sijing Li, Tianwei Lin, Zheng Liu, Xiaoyang Wang, Wenqiao Zhang, Lijun Wu  
  _2026-04-07_ · https://arxiv.org/abs/2604.06079v1  
  <details><summary>Abstract</summary>

  Graphics Program Synthesis is pivotal for interpreting and editing visual data, effectively facilitating the reverse-engineering of static visuals into editable TikZ code. While TikZ is the de facto standard for scientific schematics due to its programmatic flexibility, its requirement for rigorous spatial precision presents a significant challenge for Multimodal Large Language Models. Progress is currently stifled by two primary gaps: (1) Data Quality Gap: existing image-TikZ corpora often lack strict executability and reliable visual alignment; (2) Evaluation Gap: a lack of benchmarks for both structural and visual fidelity. To address these, we present a closed-loop framework featuring: SciTikZ-230K, a large-scale, high-quality dataset from our Execution-Centric Data Engine covering 11 diverse scientific disciplines; SciTikZ-Bench, a multifaceted benchmark spanning from basic geometric constructs to intricate hierarchical schematics to evaluate both visual fidelity and structural logic. To further broaden the scope of visual-code optimization methodology, we introduce a novel Dual Self-Consistency Reinforcement Learning optimization paradigm, which utilizes Round-Trip Verification to penalize degenerate code and boost overall self-consistency. Empowered by these, our trained model SciTikZer-8B achieves state-of-the-art performance, consistently outperforming proprietary giants like Gemini-2.5-Pro and massive models like Qwen3-VL-235B-A22B-Instruct.

  </details>



- **OmniCamera: A Unified Framework for Multi-task Video Generation with Arbitrary Camera Control**  
  Yukun Wang, Ruihuang Li, Jiale Tao, Shiyuan Yang, Liyi Chen, Zhantao Yang, Handz, Yulan Guo, Shuai Shao, Qinglin Lu  
  _2026-04-07_ · https://arxiv.org/abs/2604.06010v1  
  <details><summary>Abstract</summary>

  Video fundamentally intertwines two crucial axes: the dynamic content of a scene and the camera motion through which it is observed. However, existing generation models often entangle these factors, limiting independent control. In this work, we introduce OmniCamera, a unified framework designed to explicitly disentangle and command these two dimensions. This compositional approach enables flexible video generation by allowing arbitrary pairings of camera and content conditions, unlocking unprecedented creative control. To overcome the fundamental challenges of modality conflict and data scarcity inherent in such a system, we present two key innovations. First, we construct OmniCAM, a novel hybrid dataset combining curated real-world videos with synthetic data that provides diverse paired examples for robust multi-task learning. Second, we propose a Dual-level Curriculum Co-Training strategy that mitigates modality interference and synergistically learns from diverse data sources. This strategy operates on two levels: first, it progressively introduces control modalities by difficulties (condition-level), and second, trains for precise control on synthetic data before adapting to real data for photorealism (data-level). As a result, OmniCamera achieves state-of-the-art performance, enabling flexible control for complex camera movements while maintaining superior visual quality.

  </details>



- **Mixture-of-Modality-Experts with Holistic Token Learning for Fine-Grained Multimodal Visual Analytics in Driver Action Recognition**  
  Tianyi Liu, Yiming Li, Wenqian Wang, Jiaojiao Wang, Chen Cai, Yi Wang, Kim-Hui Yap  
  _2026-04-07_ · https://arxiv.org/abs/2604.05947v1  
  <details><summary>Abstract</summary>

  Robust multimodal visual analytics remains challenging when heterogeneous modalities provide complementary but input-dependent evidence for decision-making.Existing multimodal learning methods mainly rely on fixed fusion modules or predefined cross-modal interactions, which are often insufficient to adapt to changing modality reliability and to capture fine-grained action cues. To address this issue, we propose a Mixture-of-Modality-Experts (MoME) framework with a Holistic Token Learning (HTL) strategy. MoME enables adaptive collaboration among modality-specific experts, while HTL improves both intra-expert refinement and inter-expert knowledge transfer through class tokens and spatio-temporal tokens. In this way, our method forms a knowledge-centric multimodal learning framework that improves expert specialization while reducing ambiguity in multimodal fusion.We validate the proposed framework on driver action recognition as a representative multimodal understanding taskThe experimental results on the public benchmark show that the proposed MoME framework and the HTL strategy jointly outperform representative single-modal and multimodal baselines. Additional ablation, validation, and visualization results further verify that the proposed HTL strategy improves subtle multimodal understanding and offers better interpretability.

  </details>



- **Leveraging Image Editing Foundation Models for Data-Efficient CT Metal Artifact Reduction**  
  Ahmet Rasim Emirdagi, Süleyman Aslan, Mısra Yavuz, Görkay Aydemir, Yunus Bilge Kurt, Nasrin Rahimi, Burak Can Biner, M. Akın Yılmaz  
  _2026-04-07_ · https://arxiv.org/abs/2604.05934v1  
  <details><summary>Abstract</summary>

  Metal artifacts from high-attenuation implants severely degrade CT image quality, obscuring critical anatomical structures and posing a challenge for standard deep learning methods that require extensive paired training data. We propose a paradigm shift: reframing artifact reduction as an in-context reasoning task by adapting a general-purpose vision-language diffusion foundation model via parameter-efficient Low-Rank Adaptation (LoRA). By leveraging rich visual priors, our approach achieves effective artifact suppression with only 16 to 128 paired training examples reducing data requirements by two orders of magnitude. Crucially, we demonstrate that domain adaptation is essential for hallucination mitigation; without it, foundation models interpret streak artifacts as erroneous natural objects (e.g., waffles or petri dishes). To ground the restoration, we propose a multi-reference conditioning strategy where clean anatomical exemplars from unrelated subjects are provided alongside the corrupted input, enabling the model to exploit category-specific context to infer uncorrupted anatomy. Extensive evaluation on the AAPM CT-MAR benchmark demonstrates that our method achieves state-of-the-art performance on perceptual and radiological-feature metrics . This work establishes that foundation models, when appropriately adapted, offer a scalable alternative for interpretable, data-efficient medical image reconstruction. Code is available at https://github.com/ahmetemirdagi/CT-EditMAR.

  </details>



- **Saliency-Guided Representation with Consistency Policy Learning for Visual Unsupervised Reinforcement Learning**  
  Jingbo Sun, Qichao Zhang, Songjun Tu, Xing Fang, Yupeng Zheng, Haoran Li, Ke Chen, Dongbin Zhao  
  _2026-04-07_ · https://arxiv.org/abs/2604.05931v1  
  <details><summary>Abstract</summary>

  Zero-shot unsupervised reinforcement learning (URL) offers a promising direction for building generalist agents capable of generalizing to unseen tasks without additional supervision. Among existing approaches, successor representations (SR) have emerged as a prominent paradigm due to their effectiveness in structured, low-dimensional settings. However, SR methods struggle to scale to high-dimensional visual environments. Through empirical analysis, we identify two key limitations of SR in visual URL: (1) SR objectives often lead to suboptimal representations that attend to dynamics-irrelevant regions, resulting in inaccurate successor measures and degraded task generalization; and (2) these flawed representations hinder SR policies from modeling multi-modal skill-conditioned action distributions and ensuring skill controllability. To address these limitations, we propose Saliency-Guided Representation with Consistency Policy Learning (SRCP), a novel framework that improves zero-shot generalization of SR methods in visual URL. SRCP decouples representation learning from successor training by introducing a saliency-guided dynamics task to capture dynamics-relevant representations, thereby improving successor measure and task generalization. Moreover, it integrates a fast-sampling consistency policy with URL-specific classifier-free guidance and tailored training objectives to improve skill-conditioned policy modeling and controllability. Extensive experiments on 16 tasks across 4 datasets from the ExORL benchmark demonstrate that SRCP achieves state-of-the-art zero-shot generalization in visual URL and is compatible with various SR methods.

  </details>



- **AICA-Bench: Holistically Examining the Capabilities of VLMs in Affective Image Content Analysis**  
  Dong She, Xianrong Yao, Liqun Chen, Jinghe Yu, Yang Gao, Zhanpeng Jin  
  _2026-04-07_ · https://arxiv.org/abs/2604.05900v1  
  <details><summary>Abstract</summary>

  Vision-Language Models (VLMs) have demonstrated strong capabilities in perception, yet holistic Affective Image Content Analysis (AICA), which integrates perception, reasoning, and generation into a unified framework, remains underexplored. To address this gap, we introduce AICA-Bench, a comprehensive benchmark with three core tasks: Emotion Understanding (EU), Emotion Reasoning (ER), and Emotion-Guided Content Generation (EGCG). We evaluate 23 VLMs and identify two major limitations: weak intensity calibration and shallow open-ended descriptions. To address these issues, we propose Grounded Affective Tree (GAT) Prompting, a training-free framework that combines visual scaffolding with hierarchical reasoning. Experiments show that GAT reduces intensity errors and improves descriptive depth, providing a strong baseline for future research on affective multimodal understanding and generation.

  </details>



- **Physics-Aware Video Instance Removal Benchmark**  
  Zirui Li, Xinghao Chen, Lingyu Jiang, Dengzhe Hou, Fangzhou Lin, Kazunori Yamada, Xiangbo Gao, Zhengzhong Tu  
  _2026-04-07_ · https://arxiv.org/abs/2604.05898v1  
  <details><summary>Abstract</summary>

  Video Instance Removal (VIR) requires removing target objects while maintaining background integrity and physical consistency, such as specular reflections and illumination interactions. Despite advancements in text-guided editing, current benchmarks primarily assess visual plausibility, often overlooking the physical causalities, such as lingering shadows, triggered by object removal. We introduce the Physics-Aware Video Instance Removal (PVIR) benchmark, featuring 95 high-quality videos annotated with instance-accurate masks and removal prompts. PVIR is partitioned into Simple and Hard subsets, the latter explicitly targeting complex physical interactions. We evaluate four representative methods, PISCO-Removal, UniVideo, DiffuEraser, and CoCoCo, using a decoupled human evaluation protocol across three dimensions to isolate semantic, visual, and spatial failures: instruction following, rendering quality, and edit exclusivity. Our results show that PISCO-Removal and UniVideo achieve state-of-the-art performance, while DiffuEraser frequently introduces blurring artifacts and CoCoCo struggles significantly with instruction following. The persistent performance drop on the Hard subset highlights the ongoing challenge of recovering complex physical side effects.

  </details>



- **Neural Network Pruning via QUBO Optimization**  
  Osama Orabi, Artur Zagitov, Hadi Salloum, Viktor A. Lobachev, Kasymkhan Khubiev, Yaroslav Kholodov  
  _2026-04-07_ · https://arxiv.org/abs/2604.05856v1  
  <details><summary>Abstract</summary>

  Neural network pruning can be formulated as a combinatorial optimization problem, yet most existing approaches rely on greedy heuristics that ignore complex interactions between filters. Formal optimization methods such as Quadratic Unconstrained Binary Optimization (QUBO) provide a principled alternative but have so far underperformed due to oversimplified objective formulations based on metrics like the L1-norm. In this work, we propose a unified Hybrid QUBO framework that bridges heuristic importance estimation with global combinatorial optimization. Our formulation integrates gradient-aware sensitivity metrics - specifically first-order Taylor and second-order Fisher information - into the linear term, while utilizing data-driven activation similarity in the quadratic term. This allows the QUBO objective to jointly capture individual filter relevance and inter-filter functional redundancy. We further introduce a dynamic capacity-driven search to strictly enforce target sparsity without distorting the optimization landscape. Finally, we employ a two-stage pipeline featuring a Tensor-Train (TT) Refinement stage - a gradient-free optimizer that fine-tunes the QUBO-derived solution directly against the true evaluation metric. Experiments on the SIDD image denoising dataset demonstrate that the proposed Hybrid QUBO significantly outperforms both greedy Taylor pruning and traditional L1-based QUBO, with TT Refinement providing further consistent gains at appropriate combinatorial scales. This highlights the potential of hybrid combinatorial formulations for robust, scalable, and interpretable neural network compression.

  </details>



- **BiCoord: A Bimanual Manipulation Benchmark towards Long-Horizon Spatial-Temporal Coordination**  
  Xingyu Peng, Chen Gao, Liankai Jin, Annan Li, Si Liu  
  _2026-04-07_ · https://arxiv.org/abs/2604.05831v1  
  <details><summary>Abstract</summary>

  Bimanual manipulation, i.e., the coordinated use of two robotic arms to complete tasks, is essential for achieving human-level dexterity in robotics. Recent simulation benchmarks, e.g., RoboTwin and RLBench2, have advanced data-driven learning for bimanual manipulation. However, existing tasks are short-horizon and only loosely coordinated, failing to capture the spatial-temporal coupling inherent in real-world bimanual behaviors. To address this gap, we introduce BiCoord, a benchmark for long-horizon and tightly coordinated bimanual manipulation. Specifically, BiCoord comprises diverse tasks that require continuous inter-arm dependency and dynamic role exchange across multiple sub-goals. Also, we propose a suite of quantitative metrics that evaluate coordination from temporal, spatial, and spatial-temporal perspectives, enabling systematic measurement of bimanual cooperation. Experimental results show that representative manipulation policies, e.g., DP, RDT, Pi0, and OpenVLA-OFT, struggle with long-duration and highly coupled tasks, revealing fundamental challenges in achieving long-horizon and tight coordination tasks. We hope BiCoord can serve as a foundation for studying long-horizon cooperative manipulation and inspire future research on coordination-aware robotic learning. All datasets, codes and supplements could be found at https://buaa-colalab.github.io/BiCoord/.

  </details>



- **BodhiPromptShield: Pre-Inference Prompt Mediation for Suppressing Privacy Propagation in LLM/VLM Agents**  
  Bo Ma, Jinsong Wu, Weiqi Yan  
  _2026-04-07_ · https://arxiv.org/abs/2604.05793v1  
  <details><summary>Abstract</summary>

  In LLM/VLM agents, prompt privacy risk propagates beyond a single model call because raw user content can flow into retrieval queries, memory writes, tool calls, and logs. Existing de-identification pipelines address document boundaries but not this cross-stage propagation. We propose BodhiPromptShield, a policy-aware framework that detects sensitive spans, routes them via typed placeholders, semantic abstraction, or secure symbolic mapping, and delays restoration to authorized boundaries. Relative to enterprise redaction, this adds explicit propagation-aware mediation and restoration timing as a security variable. Under controlled evaluation on the Controlled Prompt-Privacy Benchmark (CPPB), stage-wise propagation suppresses from 10.7\% to 7.1\% across retrieval, memory, and tool stages; PER reaches 9.3\% with 0.94 AC and 0.92 TSR, outperforming generic de-identification. These are controlled systems results on CPPB rather than formal privacy guarantees or public-benchmark transfer claims. The project repository is available at https://github.com/mabo1215/BodhiPromptShield.git.

  </details>



- **Sparse Gain Radio Map Reconstruction With Geometry Priors and Uncertainty-Guided Measurement Selection**  
  Zhihan Zeng, Ning Wei, Muhammad Baqer Mollah, Kaihe Wang, Phee Lep Yeoh, Fei Xu, Yue Xiu, Zhongpei Zhang  
  _2026-04-07_ · https://arxiv.org/abs/2604.05788v1  
  <details><summary>Abstract</summary>

  Radio maps are important for environment-aware wireless communication, network planning, and radio resource optimization. However, dense radio map construction remains challenging when only a limited number of measurements are available, especially in complex urban environments with strong blockages, irregular geometry, and restricted sensing accessibility. Existing methods have explored interpolation, low-rank cartography, deep completion, and channel knowledge map (CKM) construction, but many of these methods insufficiently exploit explicit geometric priors or overlook the value of predictive uncertainty for subsequent sensing. In this paper, we study sparse gain radio map reconstruction from a geometry-aware and active sensing perspective. We first construct \textbf{UrbanRT-RM}, a controllable ray-tracing benchmark with diverse urban layouts, multiple base-station deployments, and multiple sparse sampling modes. We then propose \textbf{GeoUQ-GFNet}, a lightweight network that jointly predicts a dense gain radio map and a spatial uncertainty map from sparse measurements and structured scene priors. The predicted uncertainty is further used to guide active measurement selection under limited sensing budgets. Extensive experiments show that our proposed GeoUQ-GFNet method achieves strong and consistent reconstruction performance across different scenes and transmitter placements generated using UrbanRT-RM. Moreover, uncertainty-guided querying provides more effective reconstruction improvement than non-adaptive sampling under the same additional measurement budget. These results demonstrate the effectiveness of combining geometry-aware learning, uncertainty estimation, and benchmark-driven evaluation for sparse radio map reconstruction in complex urban environments.

  </details>



- **Beyond the Beep: Scalable Collision Anticipation and Real-Time Explainability with BADAS-2.0**  
  Roni Goldshmidt, Hamish Scott, Lorenzo Niccolini, Hernan Matzner  
  _2026-04-07_ · https://arxiv.org/abs/2604.05767v1  
  <details><summary>Abstract</summary>

  We present BADAS-2.0, the second generation of our collision anticipation system, building on BADAS-1.0 [7], which showed that fine-tuning V-JEPA2 [1] on large-scale ego-centric dashcam data outperforms both academic baselines and production ADAS systems. BADAS-2.0 advances the state of the art along three axes. (i) Long-tail benchmark and accuracy: We introduce a 10-group long-tail benchmark targeting rare and safety-critical scenarios. To construct it, BADAS-1.0 is used as an active oracle to score millions of unlabeled drives and surface high-risk candidates for annotation. Combined with Nexar's Atlas platform [13] for targeted data collection, this expands the dataset from 40k to 178,500 labeled videos (~2M clips), yielding consistent gains across all subgroups, with the largest improvements on the hardest long-tail cases. (ii) Knowledge distillation to edge: Domain-specific self-supervised pre-training on 2.25M unlabeled driving videos enables distillation into compact models, BADAS-2.0-Flash (86M) and BADAS-2.0-Flash-Lite (22M), achieving 7-12x speedup with near-parity accuracy, enabling real-time edge deployment. (iii) Explainability: BADAS-2.0 produces real-time object-centric attention heatmaps that localize the evidence behind predictions. BADAS-Reason [17] extends this with a vision-language model that consumes the last frame and heatmap to generate driver actions and structured textual reasoning. Inference code and evaluation benchmarks are publicly available.

  </details>



- **Physics-Informed Neural Optimal Control for Precision Immobilization Technique in Emergency Scenarios**  
  Yangye Jiang, Jiachen Wang, Daofei Li  
  _2026-04-07_ · https://arxiv.org/abs/2604.05758v1  
  <details><summary>Abstract</summary>

  Precision Immobilization Technique (PIT) is a potentially effective intervention maneuver for emergency out-of-control vehicle, but its automation is challenged by highly nonlinear collision dynamics, strict safety constraints, and real-time computation requirements. This work presents a PIT-oriented neural optimal-control framework built around PicoPINN (Planning-Informed Compact Physics-Informed Neural Network), a compact physics-informed surrogate obtained through knowledge distillation, hierarchical parameter clustering, and relation-matrix-based parameter reconstruction. A hierarchical neural-OCP (Optimal Control Problem) architecture is then developed, in which an upper virtual decision layer generates PIT decision packages under scenario constraints and a lower coupled-MPC (Model Predictive Control) layer executes interaction-aware control. To evaluate the framework, we construct a PIT Scenario Dataset and conduct surrogate-model comparison, planning-structure ablation, and multi-fidelity assessment from simulation to scaled by-wire vehicle tests. In simulation, adding the upper planning layer improves PIT success rate from 63.8% to 76.7%, and PicoPINN reduces the original PINN parameter count from 8965 to 812 and achieves the smallest average heading error among the learned surrogates (0.112 rad). Scaled vehicle experiments are further used as evidence of control feasibility, with 3 of 4 low-speed controllable-contact PIT trials achieving successful yaw reversal.

  </details>



- **ASSR-Net: Anisotropic Structure-Aware and Spectrally Recalibrated Network for Hyperspectral Image Fusion**  
  Qiya Song, Hongzhi Zhou, Lishan Tan, Renwei Dian, Shutao Li  
  _2026-04-07_ · https://arxiv.org/abs/2604.05742v1  
  <details><summary>Abstract</summary>

  Hyperspectral image fusion aims to reconstruct high-spatial-resolution hyperspectral images (HR-HSI) by integrating complementary information from multi-source inputs. Despite recent progress, existing methods still face two critical challenges: (1) inadequate reconstruction of anisotropic spatial structures, resulting in blurred details and compromised spatial quality; and (2) spectral distortion during fusion, which hinders fine-grained spectral representation. To address these issues, we propose \textbf{ASSR-Net}: an Anisotropic Structure-Aware and Spectrally Recalibrated Network for Hyperspectral Image Fusion. ASSR-Net adopts a two-stage fusion strategy comprising anisotropic structure-aware spatial enhancement (ASSE) and hierarchical prior-guided spectral calibration (HPSC). In the first stage, a directional perception fusion module adaptively captures structural features along multiple orientations, effectively reconstructing anisotropic spatial patterns. In the second stage, a spectral recalibration module leverages the original low-resolution HSI as a spectral prior to explicitly correct spectral deviations in the fused results, thereby enhancing spectral fidelity. Extensive experiments on various benchmark datasets demonstrate that ASSR-Net consistently outperforms state-of-the-art methods, achieving superior spatial detail preservation and spectral consistency.

  </details>



- **FoleyDesigner: Immersive Stereo Foley Generation with Precise Spatio-Temporal Alignment for Film Clips**  
  Mengtian Li, Kunyan Dai, Yi Ding, Ruobing Ni, Ying Zhang, Wenwu Wang, Zhifeng Xie  
  _2026-04-07_ · https://arxiv.org/abs/2604.05731v1  
  <details><summary>Abstract</summary>

  Foley art plays a pivotal role in enhancing immersive auditory experiences in film, yet manual creation of spatio-temporally aligned audio remains labor-intensive. We propose FoleyDesigner, a novel framework inspired by professional Foley workflows, integrating film clip analysis, spatio-temporally controllable Foley generation, and professional audio mixing capabilities. FoleyDesigner employs a multi-agent architecture for precise spatio-temporal analysis. It achieves spatio-temporal alignment through latent diffusion models trained on spatio-temporal cues extracted from video frames, combined with large language model (LLM)-driven hybrid mechanisms that emulate post-production practices in film industry. To address the lack of high-quality stereo audio datasets in film, we introduce FilmStereo, the first professional stereo audio dataset containing spatial metadata, precise timestamps, and semantic annotations for eight common Foley categories. For applications, the framework supports interactive user control while maintaining seamless integration with professional pipelines, including 5.1-channel Dolby Atmos systems compliant with ITU-R BS.775 standards, thereby offering extensive creative flexibility. Extensive experiments demonstrate that our method achieves superior spatio-temporal alignment compared to existing baselines, with seamless compatibility with professional film production standards. The project page is available at https://gekiii996.github.io/FoleyDesigner/ .

  </details>



- **MPM: Mutual Pair Merging for Efficient Vision Transformers**  
  Simon Ravé, Pejman Rasti, David Rousseau  
  _2026-04-07_ · https://arxiv.org/abs/2604.05718v1  
  <details><summary>Abstract</summary>

  Decreasing sequence length is a common way to accelerate transformers, but prior token reduction work often targets classification and reports proxy metrics rather than end-to-end latency. For semantic segmentation, token reduction is further constrained by the need to reconstruct dense, pixel-aligned features, and on modern accelerators the overhead of computing merge maps can erase expected gains. We propose Mutual Pair Merging (MPM), a training-free token aggregation module that forms mutual nearest-neighbor pairs in cosine space, averages each pair, and records a merge map enabling a gather-based reconstruction before the decoder so that existing segmentation heads can be used unchanged. MPM introduces no learned parameters and no continuous compression knob (no keep-rate or threshold). The speed-accuracy trade-off is set by a discrete insertion schedule. We benchmark end-to-end latency on an NVIDIA H100 GPU (with and without FlashAttention-2) and a Raspberry Pi 5 across standard segmentation datasets. On ADE20K, MPM reduces per-image latency by up to 60% for ViT-Tiny on Raspberry Pi 5, and increases throughput by up to 20% on H100 with FlashAttention-2 while keeping the mIoU drop below 3%. These results suggest that simple, reconstruction-aware, training-free token merging can translate into practical wall-clock gains for segmentation when overhead is explicitly accounted for.

  </details>



- **PanopticQuery: Unified Query-Time Reasoning for 4D Scenes**  
  Ruilin Tang, Yang Zhou, Zhong Ye, Wenxi Liu, Yan Huang, Shengfeng He  
  _2026-04-07_ · https://arxiv.org/abs/2604.05638v1  
  <details><summary>Abstract</summary>

  Understanding dynamic 4D environments through natural language queries requires not only accurate scene reconstruction but also robust semantic grounding across space, time, and viewpoints. While recent methods using neural representations have advanced 4D reconstruction, they remain limited in contextual reasoning, especially for complex semantics such as interactions, temporal actions, and spatial relations. A key challenge lies in transforming noisy, view-dependent predictions into globally consistent 4D interpretations. We introduce PanopticQuery, a framework for unified query-time reasoning in 4D scenes. Our approach builds on 4D Gaussian Splatting for high-fidelity dynamic reconstruction and introduces a multi-view semantic consensus mechanism that grounds natural language queries by aggregating 2D semantic predictions across multiple views and time frames. This process filters inconsistent outputs, enforces geometric consistency, and lifts 2D semantics into structured 4D groundings via neural field optimization. To support evaluation, we present Panoptic-L4D, a new benchmark for language-based querying in dynamic scenes. Experiments demonstrate that PanopticQuery sets a new state of the art on complex language queries, effectively handling attributes, actions, spatial relationships, and multi-object interactions. A video demonstration is available in the supplementary materials.

  </details>



- **Towards Athlete Fatigue Assessment from Association Football Videos**  
  Xavier Bou, Nathan Correger, Alexandre Cloots, Cédric Gavage, Silvio Giancola, Cédric Schwartz, François Delvaux, Rudi Cloots, Marc Van Droogenbroeck, Anthony Cioppa  
  _2026-04-07_ · https://arxiv.org/abs/2604.05636v1  
  <details><summary>Abstract</summary>

  Fatigue monitoring is central in association football due to its links with injury risk and tactical performance. However, objective fatigue-related indicators are commonly derived from subjective self-reported metrics, biomarkers derived from laboratory tests, or, more recently, intrusive sensors such as heart monitors or GPS tracking data. This paper studies whether monocular broadcast videos can provide spatio-temporal signals of sufficient quality to support fatigue-oriented analysis. Building on state-of-the-art Game State Reconstruction methods, we extract player trajectories in pitch coordinates and propose a novel kinematics processing algorithm to obtain temporally consistent speed and acceleration estimates from reconstructed tracks. We then construct acceleration--speed (A-S) profiles from these signals and analyze their behavior as fatigue-related performance indicators. We evaluate the full pipeline on the public SoccerNet-GSR benchmark, considering both 30-second clips and a complete 45-minute half to examine short-term reliability and longer-term temporal consistency. Our results indicate that monocular GSR can recover kinematic patterns that are compatible with A-S profiling while also revealing sensitivity to trajectory noise, calibration errors, and temporal discontinuities inherent to broadcast footage. These findings support monocular broadcast video as a low-cost basis for fatigue analysis and delineate the methodological challenges for future research.

  </details>



- **A Unified Foundation Model for All-in-One Multi-Modal Remote Sensing Image Restoration and Fusion with Language Prompting**  
  Yongchuan Cui, Peng Liu  
  _2026-04-07_ · https://arxiv.org/abs/2604.05629v1  
  <details><summary>Abstract</summary>

  Remote sensing imagery suffers from clouds, haze, noise, resolution limits, and sensor heterogeneity. Existing restoration and fusion approaches train separate models per degradation type. In this work, we present Language-conditioned Large-scale Remote Sensing restoration model (LLaRS), the first unified foundation model for multi-modal and multi-task remote sensing low-level vision. LLaRS employs Sinkhorn-Knopp optimal transport to align heterogeneous bands into semantically matched slots, routes features through three complementary mixture-of-experts layers (convolutional experts for spatial patterns, channel-mixing experts for spectral fidelity, and attention experts with low-rank adapters for global context), and stabilizes joint training via step-level dynamic weight adjustment. To train LLaRS, we construct LLaRS1M, a million-scale multi-task dataset spanning eleven restoration and enhancement tasks, integrating real paired observations and controlled synthetic degradations with diverse natural language prompts. Experiments show LLaRS consistently outperforms seven competitive models, and parameter-efficient finetuning experiments demonstrate strong transfer capability and adaptation efficiency on unseen data. Repo: https://github.com/yc-cui/LLaRS

  </details>



- **DetailVerifyBench: A Benchmark for Dense Hallucination Localization in Long Image Captions**  
  Xinran Wang, Yuxuan Zhang, Xiao Zhang, Haolong Yan, Muxi Diao, Songyu Xu, Zhonghao Yan, Hongbing Li, Kongming Liang, Zhanyu Ma  
  _2026-04-07_ · https://arxiv.org/abs/2604.05623v1  
  <details><summary>Abstract</summary>

  Accurately detecting and localizing hallucinations is a critical task for ensuring high reliability of image captions. In the era of Multimodal Large Language Models (MLLMs), captions have evolved from brief sentences into comprehensive narratives, often spanning hundreds of words. This shift exponentially increases the challenge: models must now pinpoint specific erroneous spans or words within extensive contexts, rather than merely flag response-level inconsistencies. However, existing benchmarks lack the fine granularity and domain diversity required to evaluate this capability. To bridge this gap, we introduce DetailVerifyBench, a rigorous benchmark comprising 1,000 high-quality images across five distinct domains. With an average caption length of over 200 words and dense, token-level annotations of multiple hallucination types, it stands as the most challenging benchmark for precise hallucination localization in the field of long image captioning to date. Our benchmark is available at https://zyx-hhnkh.github.io/DetailVerifyBench/.

  </details>



- **Evaluation of Randomization through Style Transfer for Enhanced Domain Generalization**  
  Dustin Eisenhardt, Timothy Schaumlöffel, Alperen Kantarci, Gemma Roig  
  _2026-04-07_ · https://arxiv.org/abs/2604.05616v1  
  <details><summary>Abstract</summary>

  Deep learning models for computer vision often suffer from poor generalization when deployed in real-world settings, especially when trained on synthetic data due to the well-known Sim2Real gap. Despite the growing popularity of style transfer as a data augmentation strategy for domain generalization, the literature contains unresolved contradictions regarding three key design axes: the diversity of the style pool, the role of texture complexity, and the choice of style source. We present a systematic empirical study that isolates and evaluates each of these factors for driving scene understanding, resolving inconsistencies in prior work. Our findings show that (i) expanding the style pool yields larger gains than repeated augmentation with few styles, (ii) texture complexity has no significant effect when the pool is sufficiently large, and (iii) diverse artistic styles outperform domain-aligned alternatives. Guided by these insights, we derive StyleMixDG (Style-Mixing for Domain Generalization), a lightweight, model-agnostic augmentation recipe that requires no architectural modifications or additional losses. Evaluated on the GTAV $\rightarrow$ {BDD100k, Cityscapes, Mapillary Vistas} benchmark, StyleMixDG demonstrates consistent improvements over strong baselines, confirming that the empirically identified design principles translate into practical gains. The code will be released on GitHub.

  </details>



- **Grounding Hierarchical Vision-Language-Action Models Through Explicit Language-Action Alignment**  
  Theodor Wulff, Federico Tavella, Rahul Singh Maharjan, Manith Adikari, Angelo Cangelosi  
  _2026-04-07_ · https://arxiv.org/abs/2604.05614v1  
  <details><summary>Abstract</summary>

  Achieving robot transparency is a critical step toward effective human-robot collaboration. To be transparent, a robot's natural language communication must be consistent with its actions and explicitly grounded in the task and environment. Existing hierarchical Vision-Language-Action (VLA) models can generate language (e.g., through chain-of-thought) and low-level actions. However, current work does not consider explicit alignment between these modalities during training. To address this crucial gap, we propose a novel training framework that explicitly grounds hierarchical VLA sub-task descriptions with respect to the visual observation and action space. Our framework uses a contrastive model to assess the alignment between generated language and corresponding action trajectories. This contrastive model enables direct ranking of different language-trajectory pairs based on their alignment, allowing us to refine the grounding of our hierarchical VLA through offline preference learning. We apply our framework to the LanguageTable dataset, a benchmark dataset of human language-annotated trajectories, and provide critical insights into multimodal grounding representations, all while establishing a strong baseline that achieves performance comparable to fully supervised fine-tuning and minimizing the need for costly data annotations.

  </details>



- **BPC-Net: Annotation-Free Skin Lesion Segmentation via Boundary Probability Calibration**  
  Yujie Yao, Yuhaohang He, Junjie Huang, Zhou Liu, Jiangzhao Li, Yan Qiao, Wen Xiao, Yunsen Liang, Xiaofan Li  
  _2026-04-07_ · https://arxiv.org/abs/2604.05594v1  
  <details><summary>Abstract</summary>

  Annotation-free skin lesion segmentation is attractive for low-resource dermoscopic deployment. However, its performance remains constrained by three coupled challenges: noisy pseudo-label supervision, unstable transfer under limited target-domain data, and boundary probability under-confidence. Most existing annotation-free methods primarily focus on pseudo-label denoising. In contrast, the effect of compressed boundary probabilities on final mask quality has received less explicit attention, although it directly affects contour completeness and cannot be adequately corrected by global threshold adjustment alone. To address this issue, we propose BPC-Net, a boundary probability calibration framework for annotation-free skin lesion segmentation. The core of the framework is Gaussian Probability Smoothing (GPS), which performs localized probability-space calibration before thresholding to recover under-confident lesion boundaries without inducing indiscriminate foreground expansion. To support this calibration under noisy pseudo-supervision and cross-domain transfer, we further incorporate two auxiliary designs: a feature-decoupled decoder that separately handles context suppression, detail recovery, and boundary refinement, and an interaction-branch adaptation strategy that updates only the pseudo-label interaction branch while preserving the deployed image-only segmentation path. Under a strictly annotation-free protocol, no manual masks are used during training or target-domain adaptation, and validation labels, when available, are used only for final operating-point selection. Experiments on ISIC-2017, ISIC-2018, and PH2 show that the proposed framework achieves state-of-the-art performance among published unsupervised methods, reaching a macro-average Dice coefficient and Jaccard index of 85.80\% and 76.97\%, respectively, while approaching supervised reference performance on PH2.

  </details>



- **WRF4CIR: Weight-Regularized Fine-Tuning Network for Composed Image Retrieval**  
  Yizhuo Xu, Chaojian Yu, Yuanjie Shao, Tongliang Liu, Qinmu Peng, Xinge You  
  _2026-04-07_ · https://arxiv.org/abs/2604.05583v1  
  <details><summary>Abstract</summary>

  Composed Image Retrieval (CIR) task aims to retrieve target images based on reference images and modification texts. Current CIR methods primarily rely on fine-tuning vision-language pre-trained models. However, we find that these approaches commonly suffer from severe overfitting, posing challenges for CIR with limited triplet data. To better understand this issue, we present a systematic study of overfitting in VLP-based CIR, revealing a significant and previously overlooked generalization gap across different models and datasets. Motivated by these findings, we introduce WRF4CIR, a Weight-Regularized Fine-tuning network for CIR. Specifically, during the fine-tuning process, we apply adversarial perturbations to the model weights for regularization, where these perturbations are generated in the opposite direction of gradient descent. Intuitively, WRF4CIR increases the difficulty of fitting the training data, which helps mitigate overfitting in CIR under limited triplet supervision. Extensive experiments on benchmark datasets demonstrate that WRF4CIR significantly narrows the generalization gap and achieves substantial improvements over existing methods.

  </details>



- **Prior-guided Fusion of Multimodal Features for Change Detection from Optical-SAR Images**  
  Xuanguang Liu, Lei Ding, Yujie Li, Chenguang Dai, Zhenchao Zhang, Mengmeng Li, Ziyi Yang, Yifan Sun, Yongqi Sun, Hanyun Wang  
  _2026-04-07_ · https://arxiv.org/abs/2604.05527v1  
  <details><summary>Abstract</summary>

  Multimodal change detection (MMCD) identifies changed areas in multimodal remote sensing (RS) data, demonstrating significant application value in land use monitoring, disaster assessment, and urban sustainable development. However, literature MMCD approaches exhibit limitations in cross-modal interaction and exploiting modality-specific characteristics. This leads to insufficient modeling of fine-grained change information, thus hindering the precise detection of semantic changes in multimodal data. To address the above problems, we propose STSF-Net, a framework designed for MMCD between optical and SAR images. STSF-Net jointly models modality-specific and spatio-temporal common features to enhance change representations. Specifically, modality-specific features are exploited to capture genuine semantic change signals, while spatio-temporal common features are embedded to suppress pseudo-changes caused by differences in imaging mechanisms. Furthermore, we introduce an optical and SAR feature fusion strategy that adaptively adjusts feature importance based on semantic priors obtained from pre-trained foundational models, enabling semantic-guided adaptive fusion of multi-modal information. In addition, we introduce the Delta-SN6 dataset, the first openly-accessible multiclass MMCD benchmark consisting of very-high-resolution (VHR) fully polarimetric SAR and optical images. Experimental results on Delta-SN6, BRIGHT, and Wuhan-Het datasets demonstrate that our method outperforms the state-of-the-art (SOTA) by 3.21%, 1.08%, and 1.32% in mIoU, respectively. The associated code and Delta-SN6 dataset will be released at: https://github.com/liuxuanguang/STSF-Net.

  </details>



- **Benchmarking Vision-Language Models under Contradictory Virtual Content Attacks in Augmented Reality**  
  Yanming Xiu, Zhengayuan Jiang, Neil Zhenqiang Gong, Maria Gorlatova  
  _2026-04-07_ · https://arxiv.org/abs/2604.05510v1  
  <details><summary>Abstract</summary>

  Augmented reality (AR) has rapidly expanded over the past decade. As AR becomes increasingly integrated into daily life, its security and reliability emerge as critical challenges. Among various threats, contradictory virtual content attacks, where malicious or inconsistent virtual elements are introduced into the user's view, pose a unique risk by misleading users, creating semantic confusion, or delivering harmful information. In this work, we systematically model such attacks and present ContrAR, a novel benchmark for evaluating the robustness of vision-language models (VLMs) against virtual content manipulation and contradiction in AR. ContrAR contains 312 real-world AR videos validated by 10 human participants. We further benchmark 11 VLMs, including both commercial and open-source models. Experimental results reveal that while current VLMs exhibit reasonable understanding of contradictory virtual content, room still remains for improvement in detecting and reasoning about adversarial content manipulations in AR environments. Moreover, balancing detection accuracy and latency remains challenging.

  </details>



- **JailWAM: Jailbreaking World Action Models in Robot Control**  
  Hanqing Liu, Songping Wang, Jiahuan Long, Jiacheng Hou, Jialiang Sun, Chao Li, Yang Yang, Wei Peng, Xu Liu, Tingsong Jiang, et al.  
  _2026-04-07_ · https://arxiv.org/abs/2604.05498v1  
  <details><summary>Abstract</summary>

  The World Action Model (WAM) can jointly predict future world states and actions, exhibiting stronger physical manipulation capabilities compared with traditional models. Such powerful physical interaction ability is a double-edged sword: if safety is ignored, it will directly threaten personal safety, property security and environmental safety. However, existing research pays extremely limited attention to the critical security gap: the vulnerability of WAM to jailbreak attacks. To fill this gap, we define the Three-Level Safety Classification Framework to systematically quantify the safety of robotic arm motions. Furthermore, we propose JailWAM, the first dedicated jailbreak attack and evaluation framework for WAM, which consists of three core components: (1) Visual-Trajectory Mapping, which unifies heterogeneous action spaces into visual trajectory representations and enables cross-architectural unified evaluation; (2) Risk Discriminator, which serves as a high-recall screening tool that optimizes the efficiency-accuracy trade-off when identifying destructive behaviors in visual trajectories; (3) Dual-Path Verification Strategy, which first conducts rapid coarse screening via a single-image-based video-action generation module, and then performs efficient and comprehensive verification through full closed-loop physical simulation. In addition, we construct JailWAM-Bench, a benchmark for comprehensively evaluating the safety alignment performance of WAM under jailbreak attacks. Experiments in RoboTwin simulation environment demonstrate that the proposed framework efficiently exposes physical vulnerabilities, achieving an 84.2% attack success rate on the state-of-the-art LingBot-VA. Meanwhile, robust defense mechanisms can be constructed based on JailWAM, providing an effective technical solution for designing safe and reliable robot control systems.

  </details>



- **Unifying VLM-Guided Flow Matching and Spectral Anomaly Detection for Interpretable Veterinary Diagnosis**  
  Pu Wang, Zhixuan Mao, Jialu Li, Zhuoran Zheng, Dianjie Lu, Youshan Zhang  
  _2026-04-07_ · https://arxiv.org/abs/2604.05482v1  
  <details><summary>Abstract</summary>

  Automatic diagnosis of canine pneumothorax is challenged by data scarcity and the need for trustworthy models. To address this, we first introduce a public, pixel-level annotated dataset to facilitate research. We then propose a novel diagnostic paradigm that reframes the task as a synergistic process of signal localization and spectral detection. For localization, our method employs a Vision-Language Model (VLM) to guide an iterative Flow Matching process, which progressively refines segmentation masks to achieve superior boundary accuracy. For detection, the segmented mask is used to isolate features from the suspected lesion. We then apply Random Matrix Theory (RMT), a departure from traditional classifiers, to analyze these features. This approach models healthy tissue as predictable random noise and identifies pneumothorax by detecting statistically significant outlier eigenvalues that represent a non-random pathological signal. The high-fidelity localization from Flow Matching is crucial for purifying the signal, thus maximizing the sensitivity of our RMT detector. This synergy of generative segmentation and first-principles statistical analysis yields a highly accurate and interpretable diagnostic system (source code is available at: https://github.com/Pu-Wang-alt/Canine-pneumothorax).

  </details>



- **A Synthetic Eye Movement Dataset for Script Reading Detection: Real Trajectory Replay on a 3D Simulator**  
  Kidus Zewde, Yuchen Zhou, Dennis Ng, Neo Tiangratanakul, Tommy Duong, Ankit Raj, Yuxin Zhang, Xingyu Shen, Simiao Ren  
  _2026-04-07_ · https://arxiv.org/abs/2604.05475v1  
  <details><summary>Abstract</summary>

  Large vision-language models have achieved remarkable capabilities by training on massive internet-scale data, yet a fundamental asymmetry persists: while LLMs can leverage self-supervised pretraining on abundant text and image data, the same is not true for many behavioral modalities. Video-based behavioral data -- gestures, eye movements, social signals -- remains scarce, expensive to annotate, and privacy-sensitive. A promising alternative is simulation: replace real data collection with controlled synthetic generation to produce automatically labeled data at scale. We introduce infrastructure for this paradigm applied to eye movement, a behavioral signal with applications across vision-language modeling, virtual reality, robotics, accessibility systems, and cognitive science. We present a pipeline for generating synthetic labeled eye movement video by extracting real human iris trajectories from reference videos and replaying them on a 3D eye movement simulator via headless browser automation. Applying this to the task of script-reading detection during video interviews, we release final_dataset_v1: 144 sessions (72 reading, 72 conversation) totaling 12 hours of synthetic eye movement video at 25fps. Evaluation shows that generated trajectories preserve the temporal dynamics of the source data (KS D < 0.14 across all metrics). A matched frame-by-frame comparison reveals that the 3D simulator exhibits bounded sensitivity at reading-scale movements, attributable to the absence of coupled head movement -- a finding that informs future simulator design. The pipeline, dataset, and evaluation tools are released to support downstream behavioral classifier development at the intersection of behavioral modeling and vision-language systems.

  </details>



- **Learning What Matters: Dynamic Dimension Selection and Aggregation for Interpretable Vision-Language Reward Modeling**  
  Qiyuan Chen, Hongsen Huang, Jiahe Chen, Qian Shao, Jintai Chen, Hongxia Xu, Renjie Hua, Chuan Ren, Jian Wu  
  _2026-04-07_ · https://arxiv.org/abs/2604.05445v1  
  <details><summary>Abstract</summary>

  Vision-language reward modeling faces a dilemma: generative approaches are interpretable but slow, while discriminative ones are efficient but act as opaque "black boxes." To bridge this gap, we propose VL-MDR (Vision-Language Multi-Dimensional Reward), a framework that dynamically decomposes evaluation into granular, interpretable dimensions. Instead of outputting a monolithic scalar, VL-MDR employs a visual-aware gating mechanism to identify relevant dimensions and adaptively weight them (e.g., Hallucination, Reasoning) for each specific input. To support this, we curate a dataset of 321k vision-language preference pairs annotated across 21 fine-grained dimensions. Extensive experiments show that VL-MDR consistently outperforms existing open-source reward models on benchmarks like VL-RewardBench. Furthermore, we show that VL-MDR-constructed preference pairs effectively enable DPO alignment to mitigate visual hallucinations and improve reliability, providing a scalable solution for VLM alignment.

  </details>



- **Pre-Execution Safety Gate & Task Safety Contracts for LLM-Controlled Robot Systems**  
  Ike Obi, Vishnunandan L. N. Venkatesh, Weizheng Wang, Ruiqi Wang, Dayoon Suh, Temitope I. Amosa, Wonse Jo, Byung-Cheol Min  
  _2026-04-07_ · https://arxiv.org/abs/2604.05427v1  
  <details><summary>Abstract</summary>

  Large Language Models (LLMs) are increasingly used to convert task commands into robot-executable code, however this pipeline lacks validation gates to detect unsafe and defective commands before they are translated into robot code. Furthermore, even commands that appear safe at the outset can produce unsafe state transitions during execution in the absence of continuous constraint monitoring. In this research, we introduce SafeGate, a neurosymbolic safety architecture that prevents unsafe natural language task commands from reaching robot execution. Drawing from ISO 13482 safety standard, SafeGate extracts structured safety-relevant properties from natural language commands and applies a deterministic decision gate to authorize or reject execution. In addition, we introduce Task Safety Contracts, which decomposes commands that pass through the gate into invariants, guards, and abort conditions to prevent unsafe state transitions during execution. We further incorporate Z3 SMT solving to enforce constraint checking derived from the Task Safety Contracts. We evaluate SafeGate against existing LLM-based robot safety frameworks and baseline LLMs across 230 benchmark tasks, 30 AI2-THOR simulation scenarios, and real-world robot experiments. Results show that SafeGate significantly reduces the acceptance of defective commands while maintaining a high acceptance of benign tasks, demonstrating the importance of pre-execution safety gates for LLM-controlled robot systems

  </details>



- **VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG**  
  Honghao Fu, Miao Xu, Yiwei Wang, Dailing Zhang, Liu Jun, Yujun Cai  
  _2026-04-07_ · https://arxiv.org/abs/2604.05418v1  
  <details><summary>Abstract</summary>

  Scaling multimodal large language models (MLLMs) to long videos is constrained by limited context windows. While retrieval-augmented generation (RAG) is a promising remedy by organizing query-relevant visual evidence into a compact context, most existing methods (i) flatten videos into independent segments, breaking their inherent spatio-temporal structure, and (ii) depend on explicit semantic matching, which can miss cues that are implicitly relevant to the query's intent. To overcome these limitations, we propose VideoStir, a structured and intent-aware long-video RAG framework. It firstly structures a video as a spatio-temporal graph at clip level, and then performs multi-hop retrieval to aggregate evidence across distant yet contextually related events. Furthermore, it introduces an MLLM-backed intent-relevance scorer that retrieves frames based on their alignment with the query's reasoning intent. To support this capability, we curate IR-600K, a large-scale dataset tailored for learning frame-query intent alignment. Experiments show that VideoStir is competitive with state-of-the-art baselines without relying on auxiliary information, highlighting the promise of shifting long-video RAG from flattened semantic matching to structured, intent-aware reasoning. Codes and checkpoints are available at Github.

  </details>



- **Learning to Synergize Semantic and Geometric Priors for Limited-Data Wheat Disease Segmentation**  
  Shijie Wang, Zijian Wang, Yadan Luo, Scott Chapman, Xin Yu, Zi Huang  
  _2026-04-07_ · https://arxiv.org/abs/2604.05415v1  
  <details><summary>Abstract</summary>

  Wheat disease segmentation is fundamental to precision agriculture but faces severe challenges from significant intra-class temporal variations across growth stages. Such substantial appearance shifts make collecting a representative dataset for training from scratch both labor-intensive and impractical. To address this, we propose SGPer, a Semantic-Geometric Prior Synergization framework that treats wheat disease segmentation under limited data as a coupled task of disease-specific semantic perception and disease boundary localization. Our core insight is that pretrained DINOv2 provides robust category-aware semantic priors to handle appearance shifts, which can be converted into coarse spatial prompts to guide SAM for the precise localization of disease boundaries. Specifically, SGPer designs disease-sensitive adapters with multiple disease-friendly filters and inserts them into both DINOv2 and SAM to align their pretrained representations with disease-specific characteristics. To operationalize this synergy, SGPer transforms DINOv2-derived features into dense, category-specific point prompts to ensure comprehensive spatial coverage of all disease regions. To subsequently eliminate prompt redundancy and ensure highly accurate mask generation, it dynamically filters these dense candidates by cross-referencing SAM's iterative mask confidence with the category-specific semantic consistency derived from DINOv2. Ultimately, SGPer distills a highly informative set of prompts to activate SAM's geometric priors, achieving precise and robust segmentation that remains strictly invariant to temporal appearance changes. Extensive evaluations demonstrate that SGPer consistently achieves state-of-the-art performance on wheat disease and organ segmentation benchmarks, especially in data-constrained scenarios.

  </details>



- **Weather-Conditioned Branch Routing for Robust LiDAR-Radar 3D Object Detection**  
  Hongsheng Li, Lingfeng Zhang, Zexian Yang, Liang Li, Rong Yin, Xiaoshuai Hao, Wenbo Ding  
  _2026-04-07_ · https://arxiv.org/abs/2604.05405v1  
  <details><summary>Abstract</summary>

  Robust 3D object detection in adverse weather is highly challenging due to the varying reliability of different sensors. While existing LiDAR-4D radar fusion methods improve robustness, they predominantly rely on fixed or weakly adaptive pipelines, failing to dy-namically adjust modality preferences as environmental conditions change. To bridge this gap, we reformulate multi-modal perception as a weather-conditioned branch routing problem. Instead of computing a single fused output, our framework explicitly maintains three parallel 3D feature streams: a pure LiDAR branch, a pure 4D radar branch, and a condition-gated fusion branch. Guided by a condition token extracted from visual and semantic prompts, a lightweight router dynamically predicts sample-specific weights to softly aggregate these representations. Furthermore, to prevent branch collapse, we introduce a weather-supervised learning strategy with auxiliary classification and diversity regularization to enforce distinct, condition-dependent routing behaviors. Extensive experiments on the K-Radar benchmark demonstrate that our method achieves state-of-the-art performance. Furthermore, it provides explicit and highly interpretable insights into modality preferences, transparently revealing how adaptive routing robustly shifts reliance between LiDAR and 4D radar across diverse adverse-weather scenarios. The source code with be released.

  </details>



- **Beyond Semantic Search: Towards Referential Anchoring in Composed Image Retrieval**  
  Yuxin Yang, Yinan Zhou, Yuxin Chen, Ziqi Zhang, Zongyang Ma, Chunfeng Yuan, Bing Li, Jun Gao, Weiming Hu  
  _2026-04-07_ · https://arxiv.org/abs/2604.05393v1  
  <details><summary>Abstract</summary>

  Composed Image Retrieval (CIR) has demonstrated significant potential by enabling flexible multimodal queries that combine a reference image and modification text. However, CIR inherently prioritizes semantic matching, struggling to reliably retrieve a user-specified instance across contexts. In practice, emphasizing concrete instance fidelity over broad semantics is often more consequential. In this work, we propose Object-Anchored Composed Image Retrieval (OACIR), a novel fine-grained retrieval task that mandates strict instance-level consistency. To advance research on this task, we construct OACIRR (OACIR on Real-world images), the first large-scale, multi-domain benchmark comprising over 160K quadruples and four challenging candidate galleries enriched with hard-negative instance distractors. Each quadruple augments the compositional query with a bounding box that visually anchors the object in the reference image, providing a precise and flexible way to ensure instance preservation. To address the OACIR task, we propose AdaFocal, a framework featuring a Context-Aware Attention Modulator that adaptively intensifies attention within the specified instance region, dynamically balancing focus between the anchored instance and the broader compositional context. Extensive experiments demonstrate that AdaFocal substantially outperforms existing compositional retrieval models, particularly in maintaining instance-level fidelity, thereby establishing a robust baseline for this challenging task while opening new directions for more flexible, instance-aware retrieval systems.

  </details>



- **LUMOS: Universal Semi-Supervised OCT Retinal Layer Segmentation with Hierarchical Reliable Mutual Learning**  
  Yizhou Fang, Jian Zhong, Li Lin, Xiaoying Tang  
  _2026-04-07_ · https://arxiv.org/abs/2604.05388v1  
  <details><summary>Abstract</summary>

  Optical Coherence Tomography (OCT) layer segmentation faces challenges due to annotation scarcity and heterogeneous label granularities across datasets. While semi-supervised learning helps alleviate label scarcity, existing methods typically assume a fixed granularity, failing to fully exploit cross-granularity supervision. This paper presents LUMOS, a semi-supervised universal OCT retinal layer segmentation framework based on a Dual-Decoder Network with a Hierarchical Prompting Strategy (DDN-HPS) and Reliable Progressive Multi-granularity Learning (RPML). DDN-HPS combines a dual-branch architecture with a multi-granularity prompting strategy to effectively suppress pseudo-label noise propagation. Meanwhile, RPML introduces region-level reliability weighing and a progressive training approach that guides the model from easier to more difficult tasks, ensuring the reliable selection of cross-granularity consistency targets, thereby achieving stable cross-granularity alignment. Experiments on six OCT datasets demonstrate that LUMOS largely outperforms existing methods and exhibits exceptional cross-domain and cross-granularity generalization capability.

  </details>



- **UAVReason: A Unified, Large-Scale Benchmark for Multimodal Aerial Scene Reasoning and Generation**  
  Jintao Sun, Hu Zhang, Donglin Di, Gangyi Ding, Zhedong Zheng  
  _2026-04-07_ · https://arxiv.org/abs/2604.05377v1  
  <details><summary>Abstract</summary>

  Vision-Language models (VLMs) have demonstrated remarkable capability in ground-view visual understanding but often fracture when deployed on high-altitude Unmanned Aerial Vehicles (UAVs). The failure largely stems from a pronounced domain shift, characterized by tiny and densely packed objects, repetitive textures, and ambiguous top-down orientations. These factors severely disrupt semantic grounding and hinder both spatial reasoning and controllable generation. To bridge this critical gap, we introduce UAVReason, the first unified large-scale multi-modal benchmark dedicated to nadir-view UAV scenarios, derived from a high-fidelity UAV simulation platform. In contrast to existing UAV benchmarks, which are largely siloed and focus on single tasks like object detection or segmentation, UAVReason uniquely consolidates over 273K Visual Question Answering (VQA) pairs, including 23.6K single frames with detailed captions, 68.2K 2-frame temporal sequences, and 188.8K cross-modal generation samples. The benchmark probes 22 diverse reasoning types across spatial and temporal axes while simultaneously evaluating high-fidelity generation across RGB, depth, and segmentation modalities. We further establish a strong, unified baseline model via multi-task learning. Extensive experiments validate the efficacy of our unified approach across diverse metrics, such as EM/F1 for VQA, mIoU for segmentation, and CLIP Score for generation. These results indicate limitations of general-domain vision-language models and show that unified multi-task learning substantially improves UAV-native performance. All data, code, and evaluation tools will be publicly released to advance UAV multimodal research.

  </details>



- **CI-ICM: Channel Importance-driven Learned Image Coding for Machines**  
  Yun Zhang, Junle Liu, Huan Zhang, Zhaoqing Pan, Gangyi Jiang, Weisi Lin  
  _2026-04-07_ · https://arxiv.org/abs/2604.05347v1  
  <details><summary>Abstract</summary>

  Traditional human vision-centric image compression methods are suboptimal for machine vision centric compression due to different visual properties and feature characteristics. To address this problem, we propose a Channel Importance-driven learned Image Coding for Machines (CI-ICM), aiming to maximize the performance of machine vision tasks at a given bitrate constraint. First, we propose a Channel Importance Generation (CIG) module to quantify channel importance in machine vision and develop a channel order loss to rank channels in descending order. Second, to properly allocate bitrate among feature channels, we propose a Feature Channel Grouping and Scaling (FCGS) module that non-uniformly groups the feature channels based on their importance and adjusts the dynamic range of each group. Based on FCGS, we further propose a Channel Importance-based Context (CI-CTX) module to allocate bits among feature groups and to preserve higher fidelity in critical channels. Third, to adapt to multiple machine tasks, we propose a Task-Specific Channel Adaptation (TSCA) module to adaptively enhance features for multiple downstream machine tasks. Experimental results on the COCO2017 dataset show that the proposed CI-ICM achieves BD-mAP@50:95 gains of 16.25$\%$ in object detection and 13.72$\%$ in instance segmentation over the established baseline codec. Ablation studies validate the effectiveness of each contribution, and computation complexity analysis reveals the practicability of the CI-ICM. This work establishes feature channel optimization for machine vision-centric compression, bridging the gap between image coding and machine perception.

  </details>



- **From Measurement to Mitigation: Quantifying and Reducing Identity Leakage in Image Representation Encoders with Linear Subspace Removal**  
  Daniel George, Charles Yeh, Daniel Lee, Yifei Zhang  
  _2026-04-07_ · https://arxiv.org/abs/2604.05296v1  
  <details><summary>Abstract</summary>

  Frozen visual embeddings (e.g., CLIP, DINOv2/v3, SSCD) power retrieval and integrity systems, yet their use on face-containing data is constrained by unmeasured identity leakage and a lack of deployable mitigations. We take an attacker-aware view and contribute: (i) a benchmark of visual embeddings that reports open-set verification at low false-accept rates, a calibrated diffusion-based template inversion check, and face-context attribution with equal-area perturbations; and (ii) propose a one-shot linear projector that removes an estimated identity subspace while preserving the complementary space needed for utility, which for brevity we denote as the identity sanitization projection ISP. Across CelebA-20 and VGGFace2, we show that these encoders are robust under open-set linear probes, with CLIP exhibiting relatively higher leakage than DINOv2/v3 and SSCD, robust to template inversion, and are context-dominant. In addition, we show that ISP drives linear access to near-chance while retaining high non-biometric utility, and transfers across datasets with minor degradation. Our results establish the first attacker-calibrated facial privacy audit of non-FR encoders and demonstrate that linear subspace removal achieves strong privacy guarantees while preserving utility for visual search and retrieval.

  </details>



- **Toward Unified Fine-Grained Vehicle Classification and Automatic License Plate Recognition**  
  Gabriel E. Lima, Valfride Nascimento, Eduardo Santos, Eduil Nascimento, Rayson Laroca, David Menotti  
  _2026-04-07_ · https://arxiv.org/abs/2604.05271v1  
  <details><summary>Abstract</summary>

  Extracting vehicle information from surveillance images is essential for intelligent transportation systems, enabling applications such as traffic monitoring and criminal investigations. While Automatic License Plate Recognition (ALPR) is widely used, Fine-Grained Vehicle Classification (FGVC) offers a complementary approach by identifying vehicles based on attributes such as color, make, model, and type. Although there have been advances in this field, existing studies often assume well-controlled conditions, explore limited attributes, and overlook FGVC integration with ALPR. To address these gaps, we introduce UFPR-VeSV, a dataset comprising 24,945 images of 16,297 unique vehicles with annotations for 13 colors, 26 makes, 136 models, and 14 types. Collected from the Military Police of Paraná (Brazil) surveillance system, the dataset captures diverse real-world conditions, including partial occlusions, nighttime infrared imaging, and varying lighting. All FGVC annotations were validated using license plate information, with text and corner annotations also being provided. A qualitative and quantitative comparison with established datasets confirmed the challenging nature of our dataset. A benchmark using five deep learning models further validated this, revealing specific challenges such as handling multicolored vehicles, infrared images, and distinguishing between vehicle models that share a common platform. Additionally, we apply two optical character recognition models to license plate recognition and explore the joint use of FGVC and ALPR. The results highlight the potential of integrating these complementary tasks for real-world applications. The UFPR-VeSV dataset is publicly available at: https://github.com/Lima001/UFPR-VeSV-Dataset.

  </details>



- **Active Measurement of Two-Point Correlations**  
  Max Hamilton, Daniel Sheldon, Subhransu Maji  
  _2026-04-06_ · https://arxiv.org/abs/2604.05227v1  
  <details><summary>Abstract</summary>

  Two-point correlation functions (2PCF) are widely used to characterize how points cluster in space. In this work, we study the problem of measuring the 2PCF over a large set of points, restricted to a subset satisfying a property of interest. An example comes from astronomy, where scientists measure the 2PCF of star clusters, which make up only a tiny subset of possible sources within a galaxy. This task typically requires careful labeling of sources to construct catalogs, which is time-consuming. We present a human-in-the-loop framework for efficient estimation of 2PCF of target sources. By leveraging a pre-trained classifier to guide sampling, our approach adaptively selects the most informative points for human annotation. After each annotation, it produces unbiased estimates of pair counts across multiple distance bins simultaneously. Compared to simple Monte Carlo approaches, our method achieves substantially lower variance while significantly reducing annotation effort. We introduce a novel unbiased estimator, sampling strategy, and confidence interval construction that together enable scalable and statistically grounded measurement of two-point correlations in astronomy datasets.

  </details>



- **RoboPlayground: Democratizing Robotic Evaluation through Structured Physical Domains**  
  Yi Ru Wang, Carter Ung, Evan Gubarev, Christopher Tan, Siddhartha Srinivasa, Dieter Fox  
  _2026-04-06_ · https://arxiv.org/abs/2604.05226v1  
  <details><summary>Abstract</summary>

  Evaluation of robotic manipulation systems has largely relied on fixed benchmarks authored by a small number of experts, where task instances, constraints, and success criteria are predefined and difficult to extend. This paradigm limits who can shape evaluation and obscures how policies respond to user-authored variations in task intent, constraints, and notions of success. We argue that evaluating modern manipulation policies requires reframing evaluation as a language-driven process over structured physical domains. We present RoboPlayground, a framework that enables users to author executable manipulation tasks using natural language within a structured physical domain. Natural language instructions are compiled into reproducible task specifications with explicit asset definitions, initialization distributions, and success predicates. Each instruction defines a structured family of related tasks, enabling controlled semantic and behavioral variation while preserving executability and comparability. We instantiate RoboPlayground in a structured block manipulation domain and evaluate it along three axes. A user study shows that the language-driven interface is easier to use and imposes lower cognitive workload than programming-based and code-assist baselines. Evaluating learned policies on language-defined task families reveals generalization failures that are not apparent under fixed benchmark evaluations. Finally, we show that task diversity scales with contributor diversity rather than task count alone, enabling evaluation spaces to grow continuously through crowd-authored contributions. Project Page: https://roboplayground.github.io

  </details>



- **Hierarchical Mesh Transformers with Topology-Guided Pretraining for Morphometric Analysis of Brain Structures**  
  Yujian Xiong, Mohammad Farazi, Yanxi Chen, Wenhui Zhu, Xuanzhao Dong, Natasha Lepore, Yi Su, Raza Mushtaq, Stephen Foldes, Andrew Yang, et al.  
  _2026-04-06_ · https://arxiv.org/abs/2604.05215v1  
  <details><summary>Abstract</summary>

  Representation learning on large-scale unstructured volumetric and surface meshes poses significant challenges in neuroimaging, especially when models must incorporate diverse vertex-level morphometric descriptors, such as cortical thickness, curvature, sulcal depth, and myelin content, which carry subtle disease-related signals. Current approaches either ignore these clinically informative features or support only a single mesh topology, restricting their use across imaging pipelines. We introduce a hierarchical transformer framework designed for heterogeneous mesh analysis that operates on spatially adaptive tree partitions constructed from simplicial complexes of arbitrary order. This design accommodates both volumetric and surface discretizations within a single architecture, enabling efficient multi-scale attention without topology-specific modifications. A feature projection module maps variable-length per-vertex clinical descriptors into the spatial hierarchy, separating geometric structure from feature dimensionality and allowing seamless integration of different neuroimaging feature sets. Self-supervised pretraining via masked reconstruction of both coordinates and morphometric channels on large unlabeled cohorts yields a transferable encoder backbone applicable to diverse downstream tasks and mesh modalities. We validate our approach on Alzheimer's disease classification and amyloid burden prediction using volumetric brain meshes from ADNI, as well as focal cortical dysplasia detection on cortical surface meshes from the MELD dataset, achieving state-of-the-art results across all benchmarks.

  </details>



- **Integration of Object Detection and Small VLMs for Construction Safety Hazard Identification**  
  Muhammad Adil, Mehmood Ahmed, Muhammad Aqib, Vicente A. Gonzalez, Gaang Lee, Qipei Mei  
  _2026-04-06_ · https://arxiv.org/abs/2604.05210v1  
  <details><summary>Abstract</summary>

  Accurate and timely identification of construction hazards around workers is essential for preventing workplace accidents. While large vision-language models (VLMs) demonstrate strong contextual reasoning capabilities, their high computational requirements limit their applicability in near real-time construction hazard detection. In contrast, small vision-language models (sVLMs) with fewer than 4 billion parameters offer improved efficiency but often suffer from reduced accuracy and hallucination when analyzing complex construction scenes. To address this trade-off, this study proposes a detection-guided sVLM framework that integrates object detection with multimodal reasoning for contextual hazard identification. The framework first employs a YOLOv11n detector to localize workers and construction machinery within the scene. The detected entities are then embedded into structured prompts to guide the reasoning process of sVLMs, enabling spatially grounded hazard assessment. Within this framework, six sVLMs (Gemma-3 4B, Qwen-3-VL 2B/4B, InternVL-3 1B/2B, and SmolVLM-2B) were evaluated in zero-shot settings on a curated dataset of construction site images with hazard annotations and explanatory rationales. The proposed approach consistently improved hazard detection performance across all models. The best-performing model, Gemma-3 4B, achieved an F1-score of 50.6%, compared to 34.5% in the baseline configuration. Explanation quality also improved significantly, with BERTScore F1 increasing from 0.61 to 0.82. Despite incorporating object detection, the framework introduces minimal overhead, adding only 2.5 ms per image during inference. These results demonstrate that integrating lightweight object detection with small VLM reasoning provides an effective and efficient solution for context-aware construction safety hazard detection.

  </details>



- **MIRAGE: Benchmarking and Aligning Multi-Instance Image Editing**  
  Ziqian Liu, Stephan Alaniz  
  _2026-04-06_ · https://arxiv.org/abs/2604.05180v1  
  <details><summary>Abstract</summary>

  Instruction-guided image editing has seen remarkable progress with models like FLUX.2 and Qwen-Image-Edit, yet they still struggle with complex scenarios with multiple similar instances each requiring individual edits. We observe that state-of-the-art models suffer from severe over-editing and spatial misalignment when faced with multiple identical instances and composite instructions. To this end, we introduce a comprehensive benchmark specifically designed to evaluate fine-grained consistency in multi-instance and multi-instruction settings. To address the failures of existing methods observed in our benchmark, we propose Multi-Instance Regional Alignment via Guided Editing (MIRAGE), a training-free framework that enables precise, localized editing. By leveraging a vision-language model to parse complex instructions into regional subsets, MIRAGE employs a multi-branch parallel denoising strategy. This approach injects latent representations of target regions into the global representation space while maintaining background integrity through a reference trajectory. Extensive evaluations on MIRA-Bench and RefEdit-Bench demonstrate that our framework significantly outperforms existing methods in achieving precise instance-level modifications while preserving background consistency. Our benchmark and code are available at https://github.com/ZiqianLiu666/MIRAGE.

  </details>



- **Watch Before You Answer: Learning from Visually Grounded Post-Training**  
  Yuxuan Zhang, EunJeong Hwang, Huaisong Zhang, Penghui Du, Yiming Jia, Dongfu Jiang, Xuan He, Shenhui Zhang, Ping Nie, Peter West, et al.  
  _2026-04-06_ · https://arxiv.org/abs/2604.05117v1  
  <details><summary>Abstract</summary>

  It is critical for vision-language models (VLMs) to comprehensively understand visual, temporal, and textual cues. However, despite rapid progress in multimodal modeling, video understanding performance still lags behind text-based reasoning. In this work, we find that progress is even worse than previously assumed: commonly reported long video understanding benchmarks contain 40-60% of questions that can be answered using text cues alone. Furthermore, we find that these issues are also pervasive in widely used post-training datasets, potentially undercutting the ability of post-training to improve VLM video understanding performance. Guided by this observation, we introduce VidGround as a simple yet effective solution: using only the actual visually grounded questions without any linguistic biases for post-training. When used in tandem with RL-based post-training algorithms, this simple technique improves performance by up to 6.2 points relative to using the full dataset, while using only 69.1% of the original post-training data. Moreover, we show that data curation with a simple post-training algorithm outperforms several more complex post-training techniques, highlighting that data quality is a major bottleneck for improving video understanding in VLMs. These results underscore the importance of curating post-training data and evaluation benchmarks that truly require visual grounding to advance the development of more capable VLMs. Project page: http://vidground.etuagi.com.

  </details>


