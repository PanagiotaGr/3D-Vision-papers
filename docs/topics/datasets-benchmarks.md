# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-02-03 07:08 UTC_

Total papers shown: **50**


---

- **Multi-head automated segmentation by incorporating detection head into the contextual layer neural network**  
  Edwin Kys, Febian Febian  
  _2026-02-02_ · https://arxiv.org/abs/2602.02471v1  
  <details><summary>Abstract</summary>

  Deep learning based auto segmentation is increasingly used in radiotherapy, but conventional models often produce anatomically implausible false positives, or hallucinations, in slices lacking target structures. We propose a gated multi-head Transformer architecture based on Swin U-Net, augmented with inter-slice context integration and a parallel detection head, which jointly performs slice-level structure detection via a multi-layer perceptron and pixel-level segmentation through a context-enhanced stream. Detection outputs gate the segmentation predictions to suppress false positives in anatomically invalid slices, and training uses slice-wise Tversky loss to address class imbalance. Experiments on the Prostate-Anatomical-Edge-Cases dataset from The Cancer Imaging Archive demonstrate that the gated model substantially outperforms a non-gated segmentation-only baseline, achieving a mean Dice loss of $0.013 \pm 0.036$ versus $0.732 \pm 0.314$, with detection probabilities strongly correlated with anatomical presence, effectively eliminating spurious segmentations. In contrast, the non-gated model exhibited higher variability and persistent false positives across all slices. These results indicate that detection-based gating enhances robustness and anatomical plausibility in automated segmentation applications, reducing hallucinated predictions without compromising segmentation quality in valid slices, and offers a promising approach for improving the reliability of clinical radiotherapy auto-contouring workflows.

  </details>



- **RANKVIDEO: Reasoning Reranking for Text-to-Video Retrieval**  
  Tyler Skow, Alexander Martin, Benjamin Van Durme, Rama Chellappa, Reno Kriz  
  _2026-02-02_ · https://arxiv.org/abs/2602.02444v1  
  <details><summary>Abstract</summary>

  Reranking is a critical component of modern retrieval systems, which typically pair an efficient first-stage retriever with a more expressive model to refine results. While large reasoning models have driven rapid progress in text-centric reranking, reasoning-based reranking for video retrieval remains underexplored. To address this gap, we introduce RANKVIDEO, a reasoning-based reranker for video retrieval that explicitly reasons over query-video pairs using video content to assess relevance. RANKVIDEO is trained using a two-stage curriculum consisting of perception-grounded supervised fine-tuning followed by reranking training that combines pointwise, pairwise, and teacher confidence distillation objectives, and is supported by a data synthesis pipeline for constructing reasoning-intensive query-video pairs. Experiments on the large-scale MultiVENT 2.0 benchmark demonstrate that RANKVIDEO consistently improves retrieval performance within a two-stage framework, yielding an average improvement of 31% on nDCG@10 and outperforming text-only and vision-language reranking alternatives, while more efficient.

  </details>



- **UniReason 1.0: A Unified Reasoning Framework for World Knowledge Aligned Image Generation and Editing**  
  Dianyi Wang, Chaofan Ma, Feng Han, Size Wu, Wei Song, Yibin Wang, Zhixiong Zhang, Tianhang Wang, Siyuan Wang, Zhongyu Wei, et al.  
  _2026-02-02_ · https://arxiv.org/abs/2602.02437v1  
  <details><summary>Abstract</summary>

  Unified multimodal models often struggle with complex synthesis tasks that demand deep reasoning, and typically treat text-to-image generation and image editing as isolated capabilities rather than interconnected reasoning steps. To address this, we propose UniReason, a unified framework that harmonizes these two tasks through a dual reasoning paradigm. We formulate generation as world knowledge-enhanced planning to inject implicit constraints, and leverage editing capabilities for fine-grained visual refinement to further correct visual errors via self-reflection. This approach unifies generation and editing within a shared representation, mirroring the human cognitive process of planning followed by refinement. We support this framework by systematically constructing a large-scale reasoning-centric dataset (~300k samples) covering five major knowledge domains (e.g., cultural commonsense, physics, etc.) for planning, alongside an agent-generated corpus for visual self-correction. Extensive experiments demonstrate that UniReason achieves advanced performance on reasoning-intensive benchmarks such as WISE, KrisBench and UniREditBench, while maintaining superior general synthesis capabilities.

  </details>



- **SelvaMask: Segmenting Trees in Tropical Forests and Beyond**  
  Simon-Olivier Duguay, Hugo Baudchon, Etienne Laliberté, Helene Muller-Landau, Gonzalo Rivas-Torres, Arthur Ouaknine  
  _2026-02-02_ · https://arxiv.org/abs/2602.02426v1  
  <details><summary>Abstract</summary>

  Tropical forests harbor most of the planet's tree biodiversity and are critical to global ecological balance. Canopy trees in particular play a disproportionate role in carbon storage and functioning of these ecosystems. Studying canopy trees at scale requires accurate delineation of individual tree crowns, typically performed using high-resolution aerial imagery. Despite advances in transformer-based models for individual tree crown segmentation, performance remains low in most forests, especially tropical ones. To this end, we introduce SelvaMask, a new tropical dataset containing over 8,800 manually delineated tree crowns across three Neotropical forest sites in Panama, Brazil, and Ecuador. SelvaMask features comprehensive annotations, including an inter-annotator agreement evaluation, capturing the dense structure of tropical forests and highlighting the difficulty of the task. Leveraging this benchmark, we propose a modular detection-segmentation pipeline that adapts vision foundation models (VFMs), using domain-specific detection-prompter. Our approach reaches state-of-the-art performance, outperforming both zero-shot generalist models and fully supervised end-to-end methods in dense tropical forests. We validate these gains on external tropical and temperate datasets, demonstrating that SelvaMask serves as both a challenging benchmark and a key enabler for generalized forest monitoring. Our code and dataset will be released publicly.

  </details>



- **Infinite-World: Scaling Interactive World Models to 1000-Frame Horizons via Pose-Free Hierarchical Memory**  
  Ruiqi Wu, Xuanhua He, Meng Cheng, Tianyu Yang, Yong Zhang, Zhuoliang Kang, Xunliang Cai, Xiaoming Wei, Chunle Guo, Chongyi Li, et al.  
  _2026-02-02_ · https://arxiv.org/abs/2602.02393v1  
  <details><summary>Abstract</summary>

  We propose Infinite-World, a robust interactive world model capable of maintaining coherent visual memory over 1000+ frames in complex real-world environments. While existing world models can be efficiently optimized on synthetic data with perfect ground-truth, they lack an effective training paradigm for real-world videos due to noisy pose estimations and the scarcity of viewpoint revisits. To bridge this gap, we first introduce a Hierarchical Pose-free Memory Compressor (HPMC) that recursively distills historical latents into a fixed-budget representation. By jointly optimizing the compressor with the generative backbone, HPMC enables the model to autonomously anchor generations in the distant past with bounded computational cost, eliminating the need for explicit geometric priors. Second, we propose an Uncertainty-aware Action Labeling module that discretizes continuous motion into a tri-state logic. This strategy maximizes the utilization of raw video data while shielding the deterministic action space from being corrupted by noisy trajectories, ensuring robust action-response learning. Furthermore, guided by insights from a pilot toy study, we employ a Revisit-Dense Finetuning Strategy using a compact, 30-minute dataset to efficiently activate the model's long-range loop-closure capabilities. Extensive experiments, including objective metrics and user studies, demonstrate that Infinite-World achieves superior performance in visual quality, action controllability, and spatial consistency.

  </details>



- **Enhancing Indoor Occupancy Prediction via Sparse Query-Based Multi-Level Consistent Knowledge Distillation**  
  Xiang Li, Yupeng Zheng, Pengfei Li, Yilun Chen, Ya-Qin Zhang, Wenchao Ding  
  _2026-02-02_ · https://arxiv.org/abs/2602.02318v1  
  <details><summary>Abstract</summary>

  Occupancy prediction provides critical geometric and semantic understanding for robotics but faces efficiency-accuracy trade-offs. Current dense methods suffer computational waste on empty voxels, while sparse query-based approaches lack robustness in diverse and complex indoor scenes. In this paper, we propose DiScene, a novel sparse query-based framework that leverages multi-level distillation to achieve efficient and robust occupancy prediction. In particular, our method incorporates two key innovations: (1) a Multi-level Consistent Knowledge Distillation strategy, which transfers hierarchical representations from large teacher models to lightweight students through coordinated alignment across four levels, including encoder-level feature alignment, query-level feature matching, prior-level spatial guidance, and anchor-level high-confidence knowledge transfer and (2) a Teacher-Guided Initialization policy, employing optimized parameter warm-up to accelerate model convergence. Validated on the Occ-Scannet benchmark, DiScene achieves 23.2 FPS without depth priors while outperforming our baseline method, OPUS, by 36.1% and even better than the depth-enhanced version, OPUS†. With depth integration, DiScene† attains new SOTA performance, surpassing EmbodiedOcc by 3.7% with 1.62$\times$ faster inference speed. Furthermore, experiments on the Occ3D-nuScenes benchmark and in-the-wild scenarios demonstrate the versatility of our approach in various environments. Code and models can be accessed at https://github.com/getterupper/DiScene.

  </details>



- **MIRROR: Manifold Ideal Reference ReconstructOR for Generalizable AI-Generated Image Detection**  
  Ruiqi Liu, Manni Cui, Ziheng Qin, Zhiyuan Yan, Ruoxin Chen, Yi Han, Zhiheng Li, Junkai Chen, ZhiJin Chen, Kaiqing Lin, et al.  
  _2026-02-02_ · https://arxiv.org/abs/2602.02222v1  
  <details><summary>Abstract</summary>

  High-fidelity generative models have narrowed the perceptual gap between synthetic and real images, posing serious threats to media security. Most existing AI-generated image (AIGI) detectors rely on artifact-based classification and struggle to generalize to evolving generative traces. In contrast, human judgment relies on stable real-world regularities, with deviations from the human cognitive manifold serving as a more generalizable signal of forgery. Motivated by this insight, we reformulate AIGI detection as a Reference-Comparison problem that verifies consistency with the real-image manifold rather than fitting specific forgery cues. We propose MIRROR (Manifold Ideal Reference ReconstructOR), a framework that explicitly encodes reality priors using a learnable discrete memory bank. MIRROR projects an input into a manifold-consistent ideal reference via sparse linear combination, and uses the resulting residuals as robust detection signals. To evaluate whether detectors reach the "superhuman crossover" required to replace human experts, we introduce the Human-AIGI benchmark, featuring a psychophysically curated human-imperceptible subset. Across 14 benchmarks, MIRROR consistently outperforms prior methods, achieving gains of 2.1% on six standard benchmarks and 8.1% on seven in-the-wild benchmarks. On Human-AIGI, MIRROR reaches 89.6% accuracy across 27 generators, surpassing both lay users and visual experts, and further approaching the human perceptual limit as pretrained backbones scale. The code is publicly available at: https://github.com/349793927/MIRROR

  </details>



- **LangMap: A Hierarchical Benchmark for Open-Vocabulary Goal Navigation**  
  Bo Miao, Weijia Liu, Jun Luo, Lachlan Shinnick, Jian Liu, Thomas Hamilton-Smith, Yuhe Yang, Zijie Wu, Vanja Videnovic, Feras Dayoub, et al.  
  _2026-02-02_ · https://arxiv.org/abs/2602.02220v1  
  <details><summary>Abstract</summary>

  The relationships between objects and language are fundamental to meaningful communication between humans and AI, and to practically useful embodied intelligence. We introduce HieraNav, a multi-granularity, open-vocabulary goal navigation task where agents interpret natural language instructions to reach targets at four semantic levels: scene, room, region, and instance. To this end, we present Language as a Map (LangMap), a large-scale benchmark built on real-world 3D indoor scans with comprehensive human-verified annotations and tasks spanning these levels. LangMap provides region labels, discriminative region descriptions, discriminative instance descriptions covering 414 object categories, and over 18K navigation tasks. Each target features both concise and detailed descriptions, enabling evaluation across different instruction styles. LangMap achieves superior annotation quality, outperforming GOAT-Bench by 23.8% in discriminative accuracy using four times fewer words. Comprehensive evaluations of zero-shot and supervised models on LangMap reveal that richer context and memory improve success, while long-tailed, small, context-dependent, and distant goals, as well as multi-goal completion, remain challenging. HieraNav and LangMap establish a rigorous testbed for advancing language-driven embodied navigation. Project: https://bo-miao.github.io/LangMap

  </details>



- **Learning Topology-Aware Implicit Field for Unified Pulmonary Tree Modeling with Incomplete Topological Supervision**  
  Ziqiao Weng, Jiancheng Yang, Kangxian Xie, Bo Zhou, Weidong Cai  
  _2026-02-02_ · https://arxiv.org/abs/2602.02186v1  
  <details><summary>Abstract</summary>

  Pulmonary trees extracted from CT images frequently exhibit topological incompleteness, such as missing or disconnected branches, which substantially degrades downstream anatomical analysis and limits the applicability of existing pulmonary tree modeling pipelines. Current approaches typically rely on dense volumetric processing or explicit graph reasoning, leading to limited efficiency and reduced robustness under realistic structural corruption. We propose TopoField, a topology-aware implicit modeling framework that treats topology repair as a first-class modeling problem and enables unified multi-task inference for pulmonary tree analysis. TopoField represents pulmonary anatomy using sparse surface and skeleton point clouds and learns a continuous implicit field that supports topology repair without relying on complete or explicit disconnection annotations, by training on synthetically introduced structural disruptions over \textit{already} incomplete trees. Building upon the repaired implicit representation, anatomical labeling and lung segment reconstruction are jointly inferred through task-specific implicit functions within a single forward pass.Extensive experiments on the Lung3D+ dataset demonstrate that TopoField consistently improves topological completeness and achieves accurate anatomical labeling and lung segment reconstruction under challenging incomplete scenarios. Owing to its implicit formulation, TopoField attains high computational efficiency, completing all tasks in just over one second per case, highlighting its practicality for large-scale and time-sensitive clinical applications. Code and data will be available at https://github.com/HINTLab/TopoField.

  </details>



- **Vision-DeepResearch Benchmark: Rethinking Visual and Textual Search for Multimodal Large Language Models**  
  Yu Zeng, Wenxuan Huang, Zhen Fang, Shuang Chen, Yufan Shen, Yishuo Cai, Xiaoman Wang, Zhenfei Yin, Lin Chen, Zehui Chen, et al.  
  _2026-02-02_ · https://arxiv.org/abs/2602.02185v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have advanced VQA and now support Vision-DeepResearch systems that use search engines for complex visual-textual fact-finding. However, evaluating these visual and textual search abilities is still difficult, and existing benchmarks have two major limitations. First, existing benchmarks are not visual search-centric: answers that should require visual search are often leaked through cross-textual cues in the text questions or can be inferred from the prior world knowledge in current MLLMs. Second, overly idealized evaluation scenario: On the image-search side, the required information can often be obtained via near-exact matching against the full image, while the text-search side is overly direct and insufficiently challenging. To address these issues, we construct the Vision-DeepResearch benchmark (VDR-Bench) comprising 2,000 VQA instances. All questions are created via a careful, multi-stage curation pipeline and rigorous expert review, designed to assess the behavior of Vision-DeepResearch systems under realistic real-world conditions. Moreover, to address the insufficient visual retrieval capabilities of current MLLMs, we propose a simple multi-round cropped-search workflow. This strategy is shown to effectively improve model performance in realistic visual retrieval scenarios. Overall, our results provide practical guidance for the design of future multimodal deep-research systems. The code will be released in https://github.com/Osilly/Vision-DeepResearch.

  </details>



- **Lung Nodule Image Synthesis Driven by Two-Stage Generative Adversarial Networks**  
  Lu Cao, Xiquan He, Junying Zeng, Chaoyun Mai, Min Luo  
  _2026-02-02_ · https://arxiv.org/abs/2602.02171v1  
  <details><summary>Abstract</summary>

  The limited sample size and insufficient diversity of lung nodule CT datasets severely restrict the performance and generalization ability of detection models. Existing methods generate images with insufficient diversity and controllability, suffering from issues such as monotonous texture features and distorted anatomical structures. Therefore, we propose a two-stage generative adversarial network (TSGAN) to enhance the diversity and spatial controllability of synthetic data by decoupling the morphological structure and texture features of lung nodules. In the first stage, StyleGAN is used to generate semantic segmentation mask images, encoding lung nodules and tissue backgrounds to control the anatomical structure of lung nodule images; The second stage uses the DL-Pix2Pix model to translate the mask map into CT images, employing local importance attention to capture local features, while utilizing dynamic weight multi-head window attention to enhance the modeling capability of lung nodule texture and background. Compared to the original dataset, the accuracy improved by 4.6% and mAP by 4% on the LUNA16 dataset. Experimental results demonstrate that TSGAN can enhance the quality of synthetic images and the performance of detection models.

  </details>



- **Reg4Pru: Regularisation Through Random Token Routing for Token Pruning**  
  Julian Wyatt, Ronald Clark, Irina Voiculescu  
  _2026-02-02_ · https://arxiv.org/abs/2602.02163v1  
  <details><summary>Abstract</summary>

  Transformers are widely adopted in modern vision models due to their strong ability to scale with dataset size and generalisability. However, this comes with a major drawback: computation scales quadratically to the total number of tokens. Numerous methods have been proposed to mitigate this. For example, we consider token pruning with reactivating tokens from preserved representations, but the increased computational efficiency of this method results in decreased stability from the preserved representations, leading to poorer dense prediction performance at deeper layers. In this work, we introduce Reg4Pru, a training regularisation technique that mitigates token-pruning performance loss for segmentation. We compare our models on the FIVES blood vessel segmentation dataset and find that Reg4Pru improves average precision by an absolute 46% compared to the same model trained without routing. This increase is observed using a configuration that achieves a 29% relative speedup in wall-clock time compared to the non-pruned baseline. These findings indicate that Reg4Pru is a valuable regulariser for token reduction strategies.

  </details>



- **LoopViT: Scaling Visual ARC with Looped Transformers**  
  Wen-Jie Shu, Xuerui Qiu, Rui-Jie Zhu, Harold Haodong Chen, Yexin Liu, Harry Yang  
  _2026-02-02_ · https://arxiv.org/abs/2602.02156v1  
  <details><summary>Abstract</summary>

  Recent advances in visual reasoning have leveraged vision transformers to tackle the ARC-AGI benchmark. However, we argue that the feed-forward architecture, where computational depth is strictly bound to parameter size, falls short of capturing the iterative, algorithmic nature of human induction. In this work, we propose a recursive architecture called Loop-ViT, which decouples reasoning depth from model capacity through weight-tied recurrence. Loop-ViT iterates a weight-tied Hybrid Block, combining local convolutions and global attention, to form a latent chain of thought. Crucially, we introduce a parameter-free Dynamic Exit mechanism based on predictive entropy: the model halts inference when its internal state ``crystallizes" into a low-uncertainty attractor. Empirical results on the ARC-AGI-1 benchmark validate this perspective: our 18M model achieves 65.8% accuracy, outperforming massive 73M-parameter ensembles. These findings demonstrate that adaptive iterative computation offers a far more efficient scaling axis for visual reasoning than simply increasing network width. The code is available at https://github.com/WenjieShu/LoopViT.

  </details>



- **Eliminating Registration Bias in Synthetic CT Generation: A Physics-Based Simulation Framework**  
  Lukas Zimmermann, Michael Rauter, Maximilian Schmid, Dietmar Georg, Barbara Knäusl  
  _2026-02-02_ · https://arxiv.org/abs/2602.02130v1  
  <details><summary>Abstract</summary>

  Supervised synthetic CT generation from CBCT requires registered training pairs, yet perfect registration between separately acquired scans remains unattainable. This registration bias propagates into trained models and corrupts standard evaluation metrics. This may suggest that superior benchmark performance indicates better reproduction of registration artifacts rather than anatomical fidelity. We propose physics-based CBCT simulation to provide geometrically aligned training pairs by construction, combined with evaluation using geometric alignment metrics against input CBCT rather than biased ground truth. On two independent pelvic datasets, models trained on synthetic data achieved superior geometric alignment (Normalized Mutual Information: 0.31 vs 0.22) despite lower conventional intensity scores. Intensity metrics showed inverted correlations with clinical assessment for deformably registered data, while Normalized Mutual Information consistently predicted observer preference across registration methodologies (rho = 0.31, p < 0.001). Clinical observers preferred synthetic-trained outputs in 87% of cases, demonstrating that geometric fidelity, not intensity agreement with biased ground truth, aligns with clinical requirements.

  </details>



- **Toxicity Assessment in Preclinical Histopathology via Class-Aware Mahalanobis Distance for Known and Novel Anomalies**  
  Olga Graf, Dhrupal Patel, Peter Groß, Charlotte Lempp, Matthias Hein, Fabian Heinemann  
  _2026-02-02_ · https://arxiv.org/abs/2602.02124v1  
  <details><summary>Abstract</summary>

  Drug-induced toxicity remains a leading cause of failure in preclinical development and early clinical trials. Detecting adverse effects at an early stage is critical to reduce attrition and accelerate the development of safe medicines. Histopathological evaluation remains the gold standard for toxicity assessment, but it relies heavily on expert pathologists, creating a bottleneck for large-scale screening. To address this challenge, we introduce an AI-based anomaly detection framework for histopathological whole-slide images (WSIs) in rodent livers from toxicology studies. The system identifies healthy tissue and known pathologies (anomalies) for which training data is available. In addition, it can detect rare pathologies without training data as out-of-distribution (OOD) findings. We generate a novel dataset of pixelwise annotations of healthy tissue and known pathologies and use this data to fine-tune a pre-trained Vision Transformer (DINOv2) via Low-Rank Adaptation (LoRA) in order to do tissue segmentation. Finally, we extract features for OOD detection using the Mahalanobis distance. To better account for class-dependent variability in histological data, we propose the use of class-specific thresholds. We optimize the thresholds using the mean of the false negative and false positive rates, resulting in only 0.16\% of pathological tissue classified as healthy and 0.35\% of healthy tissue classified as pathological. Applied to mouse liver WSIs with known toxicological findings, the framework accurately detects anomalies, including rare OOD morphologies. This work demonstrates the potential of AI-driven histopathology to support preclinical workflows, reduce late-stage failures, and improve efficiency in drug development.

  </details>



- **Enhancing Diffusion-Based Quantitatively Controllable Image Generation via Matrix-Form EDM and Adaptive Vicinal Training**  
  Xin Ding, Yun Chen, Sen Zhang, Kao Zhang, Nenglun Chen, Peibei Cao, Yongwei Wang, Fei Wu  
  _2026-02-02_ · https://arxiv.org/abs/2602.02114v1  
  <details><summary>Abstract</summary>

  Continuous Conditional Diffusion Model (CCDM) is a diffusion-based framework designed to generate high-quality images conditioned on continuous regression labels. Although CCDM has demonstrated clear advantages over prior approaches across a range of datasets, it still exhibits notable limitations and has recently been surpassed by a GAN-based method, namely CcGAN-AVAR. These limitations mainly arise from its reliance on an outdated diffusion framework and its low sampling efficiency due to long sampling trajectories. To address these issues, we propose an improved CCDM framework, termed iCCDM, which incorporates the more advanced \textit{Elucidated Diffusion Model} (EDM) framework with substantial modifications to improve both generation quality and sampling efficiency. Specifically, iCCDM introduces a novel matrix-form EDM formulation together with an adaptive vicinal training strategy. Extensive experiments on four benchmark datasets, spanning image resolutions from $64\times64$ to $256\times256$, demonstrate that iCCDM consistently outperforms existing methods, including state-of-the-art large-scale text-to-image diffusion models (e.g., Stable Diffusion 3, FLUX.1, and Qwen-Image), achieving higher generation quality while significantly reducing sampling cost.

  </details>



- **Multi-View Stenosis Classification Leveraging Transformer-Based Multiple-Instance Learning Using Real-World Clinical Data**  
  Nikola Cenikj, Özgün Turgut, Alexander Müller, Alexander Steger, Jan Kehrer, Marcus Brugger, Daniel Rueckert, Eimo Martens, Philip Müller  
  _2026-02-02_ · https://arxiv.org/abs/2602.02067v1  
  <details><summary>Abstract</summary>

  Coronary artery stenosis is a leading cause of cardiovascular disease, diagnosed by analyzing the coronary arteries from multiple angiography views. Although numerous deep-learning models have been proposed for stenosis detection from a single angiography view, their performance heavily relies on expensive view-level annotations, which are often not readily available in hospital systems. Moreover, these models fail to capture the temporal dynamics and dependencies among multiple views, which are crucial for clinical diagnosis. To address this, we propose SegmentMIL, a transformer-based multi-view multiple-instance learning framework for patient-level stenosis classification. Trained on a real-world clinical dataset, using patient-level supervision and without any view-level annotations, SegmentMIL jointly predicts the presence of stenosis and localizes the affected anatomical region, distinguishing between the right and left coronary arteries and their respective segments. SegmentMIL obtains high performance on internal and external evaluations and outperforms both view-level models and classical MIL baselines, underscoring its potential as a clinically viable and scalable solution for coronary stenosis diagnosis. Our code is available at https://github.com/NikolaCenic/mil-stenosis.

  </details>



- **Auto-Comp: An Automated Pipeline for Scalable Compositional Probing of Contrastive Vision-Language Models**  
  Cristian Sbrolli, Matteo Matteucci, Toshihiko Yamasaki  
  _2026-02-02_ · https://arxiv.org/abs/2602.02043v1  
  <details><summary>Abstract</summary>

  Modern Vision-Language Models (VLMs) exhibit a critical flaw in compositional reasoning, often confusing "a red cube and a blue sphere" with "a blue cube and a red sphere". Disentangling the visual and linguistic roots of these failures is a fundamental challenge for robust evaluation. To enable fine-grained, controllable analysis, we introduce Auto-Comp, a fully automated and synthetic pipeline for generating scalable benchmarks. Its controllable nature is key to dissecting and isolating different reasoning skills. Auto-Comp generates paired images from Minimal (e.g., "a monitor to the left of a bicycle on a white background") and LLM-generated Contextual captions (e.g., "In a brightly lit photography studio, a monitor is positioned to the left of a bicycle"), allowing a controlled A/B test to disentangle core binding ability from visio-linguistic complexity. Our evaluation of 20 VLMs on novel benchmarks for color binding and spatial relations reveals universal compositional failures in both CLIP and SigLIP model families. Crucially, our novel "Confusion Benchmark" reveals a deeper flaw beyond simple attribute swaps: models are highly susceptible to low-entropy distractors (e.g., repeated objects or colors), demonstrating their compositional failures extend beyond known bag-of-words limitations. we uncover a surprising trade-off: visio-linguistic context, which provides global scene cues, aids spatial reasoning but simultaneously hinders local attribute binding by introducing visual clutter. We release the Auto-Comp pipeline to facilitate future benchmark creation, alongside all our generated benchmarks (https://huggingface.co/AutoComp).

  </details>



- **One Size, Many Fits: Aligning Diverse Group-Wise Click Preferences in Large-Scale Advertising Image Generation**  
  Shuo Lu, Haohan Wang, Wei Feng, Weizhen Wang, Shen Zhang, Yaoyu Li, Ao Ma, Zheng Zhang, Jingjing Lv, Junjie Shen, et al.  
  _2026-02-02_ · https://arxiv.org/abs/2602.02033v1  
  <details><summary>Abstract</summary>

  Advertising image generation has increasingly focused on online metrics like Click-Through Rate (CTR), yet existing approaches adopt a ``one-size-fits-all" strategy that optimizes for overall CTR while neglecting preference diversity among user groups. This leads to suboptimal performance for specific groups, limiting targeted marketing effectiveness. To bridge this gap, we present \textit{One Size, Many Fits} (OSMF), a unified framework that aligns diverse group-wise click preferences in large-scale advertising image generation. OSMF begins with product-aware adaptive grouping, which dynamically organizes users based on their attributes and product characteristics, representing each group with rich collective preference features. Building on these groups, preference-conditioned image generation employs a Group-aware Multimodal Large Language Model (G-MLLM) to generate tailored images for each group. The G-MLLM is pre-trained to simultaneously comprehend group features and generate advertising images. Subsequently, we fine-tune the G-MLLM using our proposed Group-DPO for group-wise preference alignment, which effectively enhances each group's CTR on the generated images. To further advance this field, we introduce the Grouped Advertising Image Preference Dataset (GAIP), the first large-scale public dataset of group-wise image preferences, including around 600K groups built from 40M users. Extensive experiments demonstrate that our framework achieves the state-of-the-art performance in both offline and online settings. Our code and datasets will be released at https://github.com/JD-GenX/OSMF.

  </details>



- **Beyond Open Vocabulary: Multimodal Prompting for Object Detection in Remote Sensing Images**  
  Shuai Yang, Ziyue Huang, Jiaxin Chen, Qingjie Liu, Yunhong Wang  
  _2026-02-02_ · https://arxiv.org/abs/2602.01954v1  
  <details><summary>Abstract</summary>

  Open-vocabulary object detection in remote sensing commonly relies on text-only prompting to specify target categories, implicitly assuming that inference-time category queries can be reliably grounded through pretraining-induced text-visual alignment. In practice, this assumption often breaks down in remote sensing scenarios due to task- and application-specific category semantics, resulting in unstable category specification under open-vocabulary settings. To address this limitation, we propose RS-MPOD, a multimodal open-vocabulary detection framework that reformulates category specification beyond text-only prompting by incorporating instance-grounded visual prompts, textual prompts, and their multimodal integration. RS-MPOD introduces a visual prompt encoder to extract appearance-based category cues from exemplar instances, enabling text-free category specification, and a multimodal fusion module to integrate visual and textual information when both modalities are available. Extensive experiments on standard, cross-dataset, and fine-grained remote sensing benchmarks show that visual prompting yields more reliable category specification under semantic ambiguity and distribution shifts, while multimodal prompting provides a flexible alternative that remains competitive when textual semantics are well aligned.

  </details>



- **Enabling Progressive Whole-slide Image Analysis with Multi-scale Pyramidal Network**  
  Shuyang Wu, Yifu Qiu, Ines P. Nearchou, Sandrine Prost, Jonathan A Fallowfield, Hakan Bilen, Timothy J Kendall  
  _2026-02-02_ · https://arxiv.org/abs/2602.01951v1  
  <details><summary>Abstract</summary>

  Multiple-instance Learning (MIL) is commonly used to undertake computational pathology (CPath) tasks, and the use of multi-scale patches allows diverse features across scales to be learned. Previous studies using multi-scale features in clinical applications rely on multiple inputs across magnifications with late feature fusion, which does not retain the link between features across scales while the inputs are dependent on arbitrary, manufacturer-defined magnifications, being inflexible and computationally expensive. In this paper, we propose the Multi-scale Pyramidal Network (MSPN), which is plug-and-play over attention-based MIL that introduces progressive multi-scale analysis on WSI. Our MSPN consists of (1) grid-based remapping that uses high magnification features to derive coarse features and (2) the coarse guidance network (CGN) that learns coarse contexts. We benchmark MSPN as an add-on module to 4 attention-based frameworks using 4 clinically relevant tasks across 3 types of foundation model, as well as the pre-trained MIL framework. We show that MSPN consistently improves MIL across the compared configurations and tasks, while being lightweight and easy-to-use.

  </details>



- **Boundary-Constrained Diffusion Models for Floorplan Generation: Balancing Realism and Diversity**  
  Leonardo Stoppani, Davide Bacciu, Shahab Mokarizadeh  
  _2026-02-02_ · https://arxiv.org/abs/2602.01949v1  
  <details><summary>Abstract</summary>

  Diffusion models have become widely popular for automated floorplan generation, producing highly realistic layouts conditioned on user-defined constraints. However, optimizing for perceptual metrics such as the Fréchet Inception Distance (FID) causes limited design diversity. To address this, we propose the Diversity Score (DS), a metric that quantifies layout diversity under fixed constraints. Moreover, to improve geometric consistency, we introduce a Boundary Cross-Attention (BCA) module that enables conditioning on building boundaries. Our experiments show that BCA significantly improves boundary adherence, while prolonged training drives diversity collapse undiagnosed by FID, revealing a critical trade-off between realism and diversity. Out-Of-Distribution evaluations further demonstrate the models' reliance on dataset priors, emphasizing the need for generative systems that explicitly balance fidelity, diversity, and generalization in architectural design tasks.

  </details>



- **Towards Exploratory and Focused Manipulation with Bimanual Active Perception: A New Problem, Benchmark and Strategy**  
  Yuxin He, Ruihao Zhang, Tianao Shen, Cheng Liu, Qiang Nie  
  _2026-02-02_ · https://arxiv.org/abs/2602.01939v1  
  <details><summary>Abstract</summary>

  Recently, active vision has reemerged as an important concept for manipulation, since visual occlusion occurs more frequently when main cameras are mounted on the robot heads. We reflect on the visual occlusion issue and identify its essence as the absence of information useful for task completion. Inspired by this, we come up with the more fundamental problem of Exploratory and Focused Manipulation (EFM). The proposed problem is about actively collecting information to complete challenging manipulation tasks that require exploration or focus. As an initial attempt to address this problem, we establish the EFM-10 benchmark that consists of 4 categories of tasks that align with our definition (10 tasks in total). We further come up with a Bimanual Active Perception (BAP) strategy, which leverages one arm to provide active vision and another arm to provide force sensing while manipulating. Based on this idea, we collect a dataset named BAPData for the tasks in EFM-10. With the dataset, we successfully verify the effectiveness of the BAP strategy in an imitation learning manner. We hope that the EFM-10 benchmark along with the BAP strategy can become a cornerstone that facilitates future research towards this direction. Project website: EFManipulation.github.io.

  </details>



- **DSXFormer: Dual-Pooling Spectral Squeeze-Expansion and Dynamic Context Attention Transformer for Hyperspectral Image Classification**  
  Farhan Ullah, Irfan Ullah, Khalil Khan, Giovanni Pau, JaKeoung Koo  
  _2026-02-02_ · https://arxiv.org/abs/2602.01906v1  
  <details><summary>Abstract</summary>

  Hyperspectral image classification (HSIC) is a challenging task due to high spectral dimensionality, complex spectral-spatial correlations, and limited labeled training samples. Although transformer-based models have shown strong potential for HSIC, existing approaches often struggle to achieve sufficient spectral discriminability while maintaining computational efficiency. To address these limitations, we propose a novel DSXFormer, a novel dual-pooling spectral squeeze-expansion transformer with Dynamic Context Attention for HSIC. The proposed DSXFormer introduces a Dual-Pooling Spectral Squeeze-Expansion (DSX) block, which exploits complementary global average and max pooling to adaptively recalibrate spectral feature channels, thereby enhancing spectral discriminability and inter-band dependency modeling. In addition, DSXFormer incorporates a Dynamic Context Attention (DCA) mechanism within a window-based transformer architecture to dynamically capture local spectral-spatial relationships while significantly reducing computational overhead. The joint integration of spectral dual-pooling squeeze-expansion and DCA enables DSXFormer to achieve an effective balance between spectral emphasis and spatial contextual representation. Furthermore, patch extraction, embedding, and patch merging strategies are employed to facilitate efficient multi-scale feature learning. Extensive experiments conducted on four widely used hyperspectral benchmark datasets, including Salinas (SA), Indian Pines (IP), Pavia University (PU), and Kennedy Space Center (KSC), demonstrate that DSXFormer consistently outperforms state-of-the-art methods, achieving classification accuracies of 99.95%, 98.91%, 99.85%, and 98.52%, respectively.

  </details>



- **BTGenBot-2: Efficient Behavior Tree Generation with Small Language Models**  
  Riccardo Andrea Izzo, Gianluca Bardaro, Matteo Matteucci  
  _2026-02-02_ · https://arxiv.org/abs/2602.01870v1  
  <details><summary>Abstract</summary>

  Recent advances in robot learning increasingly rely on LLM-based task planning, leveraging their ability to bridge natural language with executable actions. While prior works showcased great performances, the widespread adoption of these models in robotics has been challenging as 1) existing methods are often closed-source or computationally intensive, neglecting the actual deployment on real-world physical systems, and 2) there is no universally accepted, plug-and-play representation for robotic task generation. Addressing these challenges, we propose BTGenBot-2, a 1B-parameter open-source small language model that directly converts natural language task descriptions and a list of robot action primitives into executable behavior trees in XML. Unlike prior approaches, BTGenBot-2 enables zero-shot BT generation, error recovery at inference and runtime, while remaining lightweight enough for resource-constrained robots. We further introduce the first standardized benchmark for LLM-based BT generation, covering 52 navigation and manipulation tasks in NVIDIA Isaac Sim. Extensive evaluations demonstrate that BTGenBot-2 consistently outperforms GPT-5, Claude Opus 4.1, and larger open-source models across both functional and non-functional metrics, achieving average success rates of 90.38% in zero-shot and 98.07% in one-shot, while delivering up to 16x faster inference compared to the previous BTGenBot.

  </details>



- **Fact or Fake? Assessing the Role of Deepfake Detectors in Multimodal Misinformation Detection**  
  A S M Sharifuzzaman Sagar, Mohammed Bennamoun, Farid Boussaid, Naeha Sharif, Lian Xu, Shaaban Sahmoud, Ali Kishk  
  _2026-02-02_ · https://arxiv.org/abs/2602.01854v1  
  <details><summary>Abstract</summary>

  In multimodal misinformation, deception usually arises not just from pixel-level manipulations in an image, but from the semantic and contextual claim jointly expressed by the image-text pair. Yet most deepfake detectors, engineered to detect pixel-level forgeries, do not account for claim-level meaning, despite their growing integration in automated fact-checking (AFC) pipelines. This raises a central scientific and practical question: Do pixel-level detectors contribute useful signal for verifying image-text claims, or do they instead introduce misleading authenticity priors that undermine evidence-based reasoning? We provide the first systematic analysis of deepfake detectors in the context of multimodal misinformation detection. Using two complementary benchmarks, MMFakeBench and DGM4, we evaluate: (1) state-of-the-art image-only deepfake detectors, (2) an evidence-driven fact-checking system that performs tool-guided retrieval via Monte Carlo Tree Search (MCTS) and engages in deliberative inference through Multi-Agent Debate (MAD), and (3) a hybrid fact-checking system that injects detector outputs as auxiliary evidence. Results across both benchmark datasets show that deepfake detectors offer limited standalone value, achieving F1 scores in the range of 0.26-0.53 on MMFakeBench and 0.33-0.49 on DGM4, and that incorporating their predictions into fact-checking pipelines consistently reduces performance by 0.04-0.08 F1 due to non-causal authenticity assumptions. In contrast, the evidence-centric fact-checking system achieves the highest performance, reaching F1 scores of approximately 0.81 on MMFakeBench and 0.55 on DGM4. Overall, our findings demonstrate that multimodal claim verification is driven primarily by semantic understanding and external evidence, and that pixel-level artifact signals do not reliably enhance reasoning over real-world image-text misinformation.

  </details>



- **How Well Do Models Follow Visual Instructions? VIBE: A Systematic Benchmark for Visual Instruction-Driven Image Editing**  
  Huanyu Zhang, Xuehai Bai, Chengzu Li, Chen Liang, Haochen Tian, Haodong Li, Ruichuan An, Yifan Zhang, Anna Korhonen, Zhang Zhang, et al.  
  _2026-02-02_ · https://arxiv.org/abs/2602.01851v1  
  <details><summary>Abstract</summary>

  Recent generative models have achieved remarkable progress in image editing. However, existing systems and benchmarks remain largely text-guided. In contrast, human communication is inherently multimodal, where visual instructions such as sketches efficiently convey spatial and structural intent. To address this gap, we introduce VIBE, the Visual Instruction Benchmark for Image Editing with a three-level interaction hierarchy that captures deictic grounding, morphological manipulation, and causal reasoning. Across these levels, we curate high-quality and diverse test cases that reflect progressively increasing complexity in visual instruction following. We further propose a robust LMM-as-a-judge evaluation framework with task-specific metrics to enable scalable and fine-grained assessment. Through a comprehensive evaluation of 17 representative open-source and proprietary image editing models, we find that proprietary models exhibit early-stage visual instruction-following capabilities and consistently outperform open-source models. However, performance degrades markedly with increasing task difficulty even for the strongest systems, highlighting promising directions for future research.

  </details>



- **WS-IMUBench: Can Weakly Supervised Methods from Audio, Image, and Video Be Adapted for IMU-based Temporal Action Localization?**  
  Pei Li, Jiaxi Yin, Lei Ouyang, Shihan Pan, Ge Wang, Han Ding, Fei Wang  
  _2026-02-02_ · https://arxiv.org/abs/2602.01850v1  
  <details><summary>Abstract</summary>

  IMU-based Human Activity Recognition (HAR) has enabled a wide range of ubiquitous computing applications, yet its dominant clip classification paradigm cannot capture the rich temporal structure of real-world behaviors. This motivates a shift toward IMU Temporal Action Localization (IMU-TAL), which predicts both action categories and their start/end times in continuous streams. However, current progress is strongly bottlenecked by the need for dense, frame-level boundary annotations, which are costly and difficult to scale. To address this bottleneck, we introduce WS-IMUBench, a systematic benchmark study of weakly supervised IMU-TAL (WS-IMU-TAL) under only sequence-level labels. Rather than proposing a new localization algorithm, we evaluate how well established weakly supervised localization paradigms from audio, image, and video transfer to IMU-TAL under only sequence-level labels. We benchmark seven representative weakly supervised methods on seven public IMU datasets, resulting in over 3,540 model training runs and 7,080 inference evaluations. Guided by three research questions on transferability, effectiveness, and insights, our findings show that (i) transfer is modality-dependent, with temporal-domain methods generally more stable than image-derived proposal-based approaches; (ii) weak supervision can be competitive on favorable datasets (e.g., with longer actions and higher-dimensional sensing); and (iii) dominant failure modes arise from short actions, temporal ambiguity, and proposal quality. Finally, we outline concrete directions for advancing WS-IMU-TAL (e.g., IMU-specific proposal generation, boundary-aware objectives, and stronger temporal reasoning). Beyond individual results, WS-IMUBench establishes a reproducible benchmarking template, datasets, protocols, and analyses, to accelerate community-wide progress toward scalable WS-IMU-TAL.

  </details>



- **Efficient Cross-Country Data Acquisition Strategy for ADAS via Street-View Imagery**  
  Yin Wu, Daniel Slieter, Carl Esselborn, Ahmed Abouelazm, Tsung Yuan Tseng, J. Marius Zöllner  
  _2026-02-02_ · https://arxiv.org/abs/2602.01836v1  
  <details><summary>Abstract</summary>

  Deploying ADAS and ADS across countries remains challenging due to differences in legislation, traffic infrastructure, and visual conventions, which introduce domain shifts that degrade perception performance. Traditional cross-country data collection relies on extensive on-road driving, making it costly and inefficient to identify representative locations. To address this, we propose a street-view-guided data acquisition strategy that leverages publicly available imagery to identify places of interest (POI). Two POI scoring methods are introduced: a KNN-based feature distance approach using a vision foundation model, and a visual-attribution approach using a vision-language model. To enable repeatable evaluation, we adopt a collect-detect protocol and construct a co-located dataset by pairing the Zenseact Open Dataset with Mapillary street-view images. Experiments on traffic sign detection, a task particularly sensitive to cross-country variations in sign appearance, show that our approach achieves performance comparable to random sampling while using only half of the target-domain data. We further provide cost estimations for full-country analysis, demonstrating that large-scale street-view processing remains economically feasible. These results highlight the potential of street-view-guided data acquisition for efficient and cost-effective cross-country model adaptation.

  </details>



- **Seeing Is Believing? A Benchmark for Multimodal Large Language Models on Visual Illusions and Anomalies**  
  Wenjin Hou, Wei Liu, Han Hu, Xiaoxiao Sun, Serena Yeung-Levy, Hehe Fan  
  _2026-02-02_ · https://arxiv.org/abs/2602.01816v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have shown remarkable proficiency on general-purpose vision-language benchmarks, reaching or even exceeding human-level performance. However, these evaluations typically rely on standard in-distribution data, leaving the robustness of MLLMs largely unexamined when faced with scenarios that defy common-sense priors. To address this gap, we introduce VIA-Bench, a challenging benchmark designed to probe model performance on visual illusions and anomalies. It includes six core categories: color illusions, motion illusions, gestalt illusions, geometric and spatial illusions, general visual illusions, and visual anomalies. Through careful human-in-the-loop review, we construct over 1K high-quality question-answer pairs that require nuanced visual reasoning. Extensive evaluation of over 20 state-of-the-art MLLMs, including proprietary, open-source, and reasoning-enhanced models, uncovers significant vulnerabilities. Notably, we find that Chain-of-Thought (CoT) reasoning offers negligible robustness, often yielding ``brittle mirages'' where the model's logic collapses under illusory stimuli. Our findings reveal a fundamental divergence between machine and human perception, suggesting that resolving such perceptual bottlenecks is critical for the advancement of artificial general intelligence. The benchmark data and code will be released.

  </details>



- **LDRNet: Large Deformation Registration Model for Chest CT Registration**  
  Cheng Wang, Qiyu Gao, Fandong Zhang, Shu Zhang, Yizhou Yu  
  _2026-02-02_ · https://arxiv.org/abs/2602.01812v1  
  <details><summary>Abstract</summary>

  Most of the deep learning based medical image registration algorithms focus on brain image registration tasks.Compared with brain registration, the chest CT registration has larger deformation, more complex background and region over-lap. In this paper, we propose a fast unsupervised deep learning method, LDRNet, for large deformation image registration of chest CT images. We first predict a coarse resolution registration field, then refine it from coarse to fine. We propose two innovative technical components: 1) a refine block that is used to refine the registration field in different resolutions, 2) a rigid block that is used to learn transformation matrix from high-level features. We train and evaluate our model on the private dataset and public dataset SegTHOR. We compare our performance with state-of-the-art traditional registration methods as well as deep learning registration models VoxelMorph, RCN, and LapIRN. The results demonstrate that our model achieves state-of-the-art performance for large deformation images registration and is much faster.

  </details>



- **From Knowing to Doing Precisely: A General Self-Correction and Termination Framework for VLA models**  
  Wentao Zhang, Aolan Sun, Wentao Mo, Xiaoyang Qu, Yuxin Zheng, Jianzong Wang  
  _2026-02-02_ · https://arxiv.org/abs/2602.01811v1  
  <details><summary>Abstract</summary>

  While vision-language-action (VLA) models for embodied agents integrate perception, reasoning, and control, they remain constrained by two critical weaknesses: first, during grasping tasks, the action tokens generated by the language model often exhibit subtle spatial deviations from the target object, resulting in grasp failures; second, they lack the ability to reliably recognize task completion, which leads to redundant actions and frequent timeout errors. To address these challenges and enhance robustness, we propose a lightweight, training-free framework, VLA-SCT. This framework operates as a self-correcting control loop, combining data-driven action refinement with conditional logic for termination. Consequently, compared to baseline approaches, our method achieves consistent improvements across all datasets in the LIBERO benchmark, significantly increasing the success rate of fine manipulation tasks and ensuring accurate task completion, thereby promoting the deployment of more reliable VLA agents in complex, unstructured environments.

  </details>



- **GDPR-Compliant Person Recognition in Industrial Environments Using MEMS-LiDAR and Hybrid Data**  
  Dennis Basile, Dennis Sprute, Helene Dörksen, Holger Flatt  
  _2026-02-02_ · https://arxiv.org/abs/2602.01764v1  
  <details><summary>Abstract</summary>

  The reliable detection of unauthorized individuals in safety-critical industrial indoor spaces is crucial to avoid plant shutdowns, property damage, and personal hazards. Conventional vision-based methods that use deep-learning approaches for person recognition provide image information but are sensitive to lighting and visibility conditions and often violate privacy regulations, such as the General Data Protection Regulation (GDPR) in the European Union. Typically, detection systems based on deep learning require annotated data for training. Collecting and annotating such data, however, is highly time-consuming and due to manual treatments not necessarily error free. Therefore, this paper presents a privacy-compliant approach based on Micro-Electro-Mechanical Systems LiDAR (MEMS-LiDAR), which exclusively captures anonymized 3D point clouds and avoids personal identification features. To compensate for the large amount of time required to record real LiDAR data and for post-processing and annotation, real recordings are augmented with synthetically generated scenes from the CARLA simulation framework. The results demonstrate that the hybrid data improves the average precision by 44 percentage points compared to a model trained exclusively with real data while reducing the manual annotation effort by 50 %. Thus, the proposed approach provides a scalable, cost-efficient alternative to purely real-data-based methods and systematically shows how synthetic LiDAR data can combine high performance in person detection with GDPR compliance in an industrial environment.

  </details>



- **Mind-Brush: Integrating Agentic Cognitive Search and Reasoning into Image Generation**  
  Jun He, Junyan Ye, Zilong Huang, Dongzhi Jiang, Chenjue Zhang, Leqi Zhu, Renrui Zhang, Xiang Zhang, Weijia Li  
  _2026-02-02_ · https://arxiv.org/abs/2602.01756v1  
  <details><summary>Abstract</summary>

  While text-to-image generation has achieved unprecedented fidelity, the vast majority of existing models function fundamentally as static text-to-pixel decoders. Consequently, they often fail to grasp implicit user intentions. Although emerging unified understanding-generation models have improved intent comprehension, they still struggle to accomplish tasks involving complex knowledge reasoning within a single model. Moreover, constrained by static internal priors, these models remain unable to adapt to the evolving dynamics of the real world. To bridge these gaps, we introduce Mind-Brush, a unified agentic framework that transforms generation into a dynamic, knowledge-driven workflow. Simulating a human-like 'think-research-create' paradigm, Mind-Brush actively retrieves multimodal evidence to ground out-of-distribution concepts and employs reasoning tools to resolve implicit visual constraints. To rigorously evaluate these capabilities, we propose Mind-Bench, a comprehensive benchmark comprising 500 distinct samples spanning real-time news, emerging concepts, and domains such as mathematical and Geo-Reasoning. Extensive experiments demonstrate that Mind-Brush significantly enhances the capabilities of unified models, realizing a zero-to-one capability leap for the Qwen-Image baseline on Mind-Bench, while achieving superior results on established benchmarks like WISE and RISE.

  </details>



- **Tail-Aware Post-Training Quantization for 3D Geometry Models**  
  Sicheng Pan, Chen Tang, Shuzhao Xie, Ke Yang, Weixiang Zhang, Jiawei Li, Bin Chen, Shu-Tao Xia, Zhi Wang  
  _2026-02-02_ · https://arxiv.org/abs/2602.01741v1  
  <details><summary>Abstract</summary>

  The burgeoning complexity and scale of 3D geometry models pose significant challenges for deployment on resource-constrained platforms. While Post-Training Quantization (PTQ) enables efficient inference without retraining, conventional methods, primarily optimized for 2D Vision Transformers, fail to transfer effectively to 3D models due to intricate feature distributions and prohibitive calibration overhead. To address these challenges, we propose TAPTQ, a Tail-Aware Post-Training Quantization pipeline specifically engineered for 3D geometric learning. Our contribution is threefold: (1) To overcome the data-scale bottleneck in 3D datasets, we develop a progressive coarse-to-fine calibration construction strategy that constructs a highly compact subset to achieve both statistical purity and geometric representativeness. (2) We reformulate the quantization interval search as an optimization problem and introduce a ternary-search-based solver, reducing the computational complexity from $\mathcal{O}(N)$ to $\mathcal{O}(\log N)$ for accelerated deployment. (3) To mitigate quantization error accumulation, we propose TRE-Guided Module-wise Compensation, which utilizes a Tail Relative Error (TRE) metric to adaptively identify and rectify distortions in modules sensitive to long-tailed activation outliers. Extensive experiments on the VGGT and Pi3 benchmarks demonstrate that TAPTQ consistently outperforms state-of-the-art PTQ methods in accuracy while significantly reducing calibration time. The code will be released soon.

  </details>



- **DenVisCoM: Dense Vision Correspondence Mamba for Efficient and Real-time Optical Flow and Stereo Estimation**  
  Tushar Anand, Maheswar Bora, Antitza Dantcheva, Abhijit Das  
  _2026-02-02_ · https://arxiv.org/abs/2602.01724v1  
  <details><summary>Abstract</summary>

  In this work, we propose a novel Mamba block DenVisCoM, as well as a novel hybrid architecture specifically tailored for accurate and real-time estimation of optical flow and disparity estimation. Given that such multi-view geometry and motion tasks are fundamentally related, we propose a unified architecture to tackle them jointly. Specifically, the proposed hybrid architecture is based on DenVisCoM and a Transformer-based attention block that efficiently addresses real-time inference, memory footprint, and accuracy at the same time for joint estimation of motion and 3D dense perception tasks. We extensively analyze the benchmark trade-off of accuracy and real-time processing on a large number of datasets. Our experimental results and related analysis suggest that our proposed model can accurately estimate optical flow and disparity estimation in real time. All models and associated code are available at https://github.com/vimstereo/DenVisCoM.

  </details>



- **Physics Informed Generative AI Enabling Labour Free Segmentation For Microscopy Analysis**  
  Salma Zahran, Zhou Ao, Zhengyang Zhang, Chen Chi, Chenchen Yuan, Yanming Wang  
  _2026-02-02_ · https://arxiv.org/abs/2602.01710v1  
  <details><summary>Abstract</summary>

  Semantic segmentation of microscopy images is a critical task for high-throughput materials characterisation, yet its automation is severely constrained by the prohibitive cost, subjectivity, and scarcity of expert-annotated data. While physics-based simulations offer a scalable alternative to manual labelling, models trained on such data historically fail to generalise due to a significant domain gap, lacking the complex textures, noise patterns, and imaging artefacts inherent to experimental data. This paper introduces a novel framework for labour-free segmentation that successfully bridges this simulation-to-reality gap. Our pipeline leverages phase-field simulations to generate an abundant source of microstructural morphologies with perfect, intrinsically-derived ground-truth masks. We then employ a Cycle-Consistent Generative Adversarial Network (CycleGAN) for unpaired image-to-image translation, transforming the clean simulations into a large-scale dataset of high-fidelity, realistic SEM images. A U-Net model, trained exclusively on this synthetic data, demonstrated remarkable generalisation when deployed on unseen experimental images, achieving a mean Boundary F1-Score of 0.90 and an Intersection over Union (IOU) of 0.88. Comprehensive validation using t-SNE feature-space projection and Shannon entropy analysis confirms that our synthetic images are statistically and featurally indistinguishable from the real data manifold. By completely decoupling model training from manual annotation, our generative framework transforms a data-scarce problem into one of data abundance, providing a robust and fully automated solution to accelerate materials discovery and analysis.

  </details>



- **Cross-Modal Alignment and Fusion for RGB-D Transmission-Line Defect Detection**  
  Jiaming Cui, Shuai Zhou, Wenqiang Li, Ruifeng Qin, Feng Shen  
  _2026-02-02_ · https://arxiv.org/abs/2602.01696v1  
  <details><summary>Abstract</summary>

  Transmission line defect detection remains challenging for automated UAV inspection due to the dominance of small-scale defects, complex backgrounds, and illumination variations. Existing RGB-based detectors, despite recent progress, struggle to distinguish geometrically subtle defects from visually similar background structures under limited chromatic contrast. This paper proposes CMAFNet, a Cross-Modal Alignment and Fusion Network that integrates RGB appearance and depth geometry through a principled purify-then-fuse paradigm. CMAFNet consists of a Semantic Recomposition Module that performs dictionary-based feature purification via a learned codebook to suppress modality-specific noise while preserving defect-discriminative information, and a Contextual Semantic Integration Framework that captures global spatial dependencies using partial-channel attention to enhance structural semantic reasoning. Position-wise normalization within the purification stage enforces explicit reconstruction-driven cross-modal alignment, ensuring statistical compatibility between heterogeneous features prior to fusion. Extensive experiments on the TLRGBD benchmark, where 94.5% of instances are small objects, demonstrate that CMAFNet achieves 32.2% mAP@50 and 12.5% APs, outperforming the strongest baseline by 9.8 and 4.0 percentage points, respectively. A lightweight variant reaches 24.8% mAP50 at 228 FPS with only 4.9M parameters, surpassing all YOLO-based detectors while matching transformer-based methods at substantially lower computational cost.

  </details>



- **GSR: Learning Structured Reasoning for Embodied Manipulation**  
  Kewei Hu, Michael Zhang, Wei Ying, Tianhao Liu, Guoqiang Hao, Zimeng Li, Wanchan Yu, Jiajian Jing, Fangwen Chen, Hanwen Kang  
  _2026-02-02_ · https://arxiv.org/abs/2602.01693v1  
  <details><summary>Abstract</summary>

  Despite rapid progress, embodied agents still struggle with long-horizon manipulation that requires maintaining spatial consistency, causal dependencies, and goal constraints. A key limitation of existing approaches is that task reasoning is implicitly embedded in high-dimensional latent representations, making it challenging to separate task structure from perceptual variability. We introduce Grounded Scene-graph Reasoning (GSR), a structured reasoning paradigm that explicitly models world-state evolution as transitions over semantically grounded scene graphs. By reasoning step-wise over object states and spatial relations, rather than directly mapping perception to actions, GSR enables explicit reasoning about action preconditions, consequences, and goal satisfaction in a physically grounded space. To support learning such reasoning, we construct Manip-Cognition-1.6M, a large-scale dataset that jointly supervises world understanding, action planning, and goal interpretation. Extensive evaluations across RLBench, LIBERO, GSR-benchmark, and real-world robotic tasks show that GSR significantly improves zero-shot generalization and long-horizon task completion over prompting-based baselines. These results highlight explicit world-state representations as a key inductive bias for scalable embodied reasoning.

  </details>



- **Towards Autonomous Instrument Tray Assembly for Sterile Processing Applications**  
  Raghavasimhan Sankaranarayanan, Paul Stuart, Nicholas Ahn, Arno Sungarian, Yash Chitalia  
  _2026-02-02_ · https://arxiv.org/abs/2602.01679v1  
  <details><summary>Abstract</summary>

  The Sterile Processing and Distribution (SPD) department is responsible for cleaning, disinfecting, inspecting, and assembling surgical instruments between surgeries. Manual inspection and preparation of instrument trays is a time-consuming, error-prone task, often prone to contamination and instrument breakage. In this work, we present a fully automated robotic system that sorts and structurally packs surgical instruments into sterile trays, focusing on automation of the SPD assembly stage. A custom dataset comprising 31 surgical instruments and 6,975 annotated images was collected to train a hybrid perception pipeline using YOLO12 for detection and a cascaded ResNet-based model for fine-grained classification. The system integrates a calibrated vision module, a 6-DOF Staubli TX2-60L robotic arm with a custom dual electromagnetic gripper, and a rule-based packing algorithm that reduces instrument collisions during transport. The packing framework uses 3D printed dividers and holders to physically isolate instruments, reducing collision and friction during transport. Experimental evaluations show high perception accuracy and statistically significant reduction in tool-to-tool collisions compared to human-assembled trays. This work serves as the scalable first step toward automating SPD workflows, improving safety, and consistency of surgical preparation while reducing SPD processing times.

  </details>



- **Real-Time Loop Closure Detection in Visual SLAM via NetVLAD and Faiss**  
  Enguang Fan  
  _2026-02-02_ · https://arxiv.org/abs/2602.01673v1  
  <details><summary>Abstract</summary>

  Loop closure detection (LCD) is a core component of simultaneous localization and mapping (SLAM): it identifies revisited places and enables pose-graph constraints that correct accumulated drift. Classic bag-of-words approaches such as DBoW are efficient but often degrade under appearance change and perceptual aliasing. In parallel, deep learning-based visual place recognition (VPR) descriptors (e.g., NetVLAD and Transformer-based models) offer stronger robustness, but their computational cost is often viewed as a barrier to real-time SLAM. In this paper, we empirically evaluate NetVLAD as an LCD module and compare it against DBoW on the KITTI dataset. We introduce a Fine-Grained Top-K precision-recall curve that better reflects LCD settings where a query may have zero or multiple valid matches. With Faiss-accelerated nearestneighbor search, NetVLAD achieves real-time query speed while improving accuracy and robustness over DBoW, making it a practical drop-in alternative for LCD in SLAM.

  </details>



- **Moonworks Lunara Aesthetic II: An Image Variation Dataset**  
  Yan Wang, Partho Hassan, Samiha Sadeka, Nada Soliman, M M Sayeef Abdullah, Sabit Hassan  
  _2026-02-02_ · https://arxiv.org/abs/2602.01666v1  
  <details><summary>Abstract</summary>

  We introduce Lunara Aesthetic II, a publicly released, ethically sourced image dataset designed to support controlled evaluation and learning of contextual consistency in modern image generation and editing systems. The dataset comprises 2,854 anchor-linked variation pairs derived from original art and photographs created by Moonworks. Each variation pair applies contextual transformations, such as illumination, weather, viewpoint, scene composition, color tone, or mood; while preserving a stable underlying identity. Lunara Aesthetic II operationalizes identity-preserving contextual variation as a supervision signal while also retaining Lunara's signature high aesthetic scores. Results show high identity stability, strong target attribute realization, and a robust aesthetic profile that exceeds large-scale web datasets. Released under the Apache 2.0 license, Lunara Aesthetic II is intended for benchmarking, fine-tuning, and analysis of contextual generalization, identity preservation, and edit robustness in image generation and image-to-image systems with interpretable, relational supervision. The dataset is publicly available at: https://huggingface.co/datasets/moonworks/lunara-aesthetic-image-variations.

  </details>



- **AgenticLab: A Real-World Robot Agent Platform that Can See, Think, and Act**  
  Pengyuan Guo, Zhonghao Mai, Zhengtong Xu, Kaidi Zhang, Heng Zhang, Zichen Miao, Arash Ajoudani, Zachary Kingston, Qiang Qiu, Yu She  
  _2026-02-02_ · https://arxiv.org/abs/2602.01662v1  
  <details><summary>Abstract</summary>

  Recent advances in large vision-language models (VLMs) have demonstrated generalizable open-vocabulary perception and reasoning, yet their real-robot manipulation capability remains unclear for long-horizon, closed-loop execution in unstructured, in-the-wild environments. Prior VLM-based manipulation pipelines are difficult to compare across different research groups' setups, and many evaluations rely on simulation, privileged state, or specially designed setups. We present AgenticLab, a model-agnostic robot agent platform and benchmark for open-world manipulation. AgenticLab provides a closed-loop agent pipeline for perception, task decomposition, online verification, and replanning. Using AgenticLab, we benchmark state-of-the-art VLM-based agents on real-robot tasks in unstructured environments. Our benchmark reveals several failure modes that offline vision-language tests (e.g., VQA and static image understanding) fail to capture, including breakdowns in multi-step grounding consistency, object grounding under occlusion and scene changes, and insufficient spatial reasoning for reliable manipulation. We will release the full hardware and software stack to support reproducible evaluation and accelerate research on general-purpose robot agents.

  </details>



- **Federated Vision Transformer with Adaptive Focal Loss for Medical Image Classification**  
  Xinyuan Zhao, Yihang Wu, Ahmad Chaddad, Tareef Daqqaq, Reem Kateb  
  _2026-02-02_ · https://arxiv.org/abs/2602.01633v1  
  <details><summary>Abstract</summary>

  While deep learning models like Vision Transformer (ViT) have achieved significant advances, they typically require large datasets. With data privacy regulations, access to many original datasets is restricted, especially medical images. Federated learning (FL) addresses this challenge by enabling global model aggregation without data exchange. However, the heterogeneity of the data and the class imbalance that exist in local clients pose challenges for the generalization of the model. This study proposes a FL framework leveraging a dynamic adaptive focal loss (DAFL) and a client-aware aggregation strategy for local training. Specifically, we design a dynamic class imbalance coefficient that adjusts based on each client's sample distribution and class data distribution, ensuring minority classes receive sufficient attention and preventing sparse data from being ignored. To address client heterogeneity, a weighted aggregation strategy is adopted, which adapts to data size and characteristics to better capture inter-client variations. The classification results on three public datasets (ISIC, Ocular Disease and RSNA-ICH) show that the proposed framework outperforms DenseNet121, ResNet50, ViT-S/16, ViT-L/32, FedCLIP, Swin Transformer, CoAtNet, and MixNet in most cases, with accuracy improvements ranging from 0.98\% to 41.69\%. Ablation studies on the imbalanced ISIC dataset validate the effectiveness of the proposed loss function and aggregation strategy compared to traditional loss functions and other FL approaches. The codes can be found at: https://github.com/AIPMLab/ViT-FLDAF.

  </details>



- **PISCES: Annotation-free Text-to-Video Post-Training via Optimal Transport-Aligned Rewards**  
  Minh-Quan Le, Gaurav Mittal, Cheng Zhao, David Gu, Dimitris Samaras, Mei Chen  
  _2026-02-02_ · https://arxiv.org/abs/2602.01624v1  
  <details><summary>Abstract</summary>

  Text-to-video (T2V) generation aims to synthesize videos with high visual quality and temporal consistency that are semantically aligned with input text. Reward-based post-training has emerged as a promising direction to improve the quality and semantic alignment of generated videos. However, recent methods either rely on large-scale human preference annotations or operate on misaligned embeddings from pre-trained vision-language models, leading to limited scalability or suboptimal supervision. We present $\texttt{PISCES}$, an annotation-free post-training algorithm that addresses these limitations via a novel Dual Optimal Transport (OT)-aligned Rewards module. To align reward signals with human judgment, $\texttt{PISCES}$ uses OT to bridge text and video embeddings at both distributional and discrete token levels, enabling reward supervision to fulfill two objectives: (i) a Distributional OT-aligned Quality Reward that captures overall visual quality and temporal coherence; and (ii) a Discrete Token-level OT-aligned Semantic Reward that enforces semantic, spatio-temporal correspondence between text and video tokens. To our knowledge, $\texttt{PISCES}$ is the first to improve annotation-free reward supervision in generative post-training through the lens of OT. Experiments on both short- and long-video generation show that $\texttt{PISCES}$ outperforms both annotation-based and annotation-free methods on VBench across Quality and Semantic scores, with human preference studies further validating its effectiveness. We show that the Dual OT-aligned Rewards module is compatible with multiple optimization paradigms, including direct backpropagation and reinforcement learning fine-tuning.

  </details>



- **UV-M3TL: A Unified and Versatile Multimodal Multi-Task Learning Framework for Assistive Driving Perception**  
  Wenzhuo Liu, Qiannan Guo, Zhen Wang, Wenshuo Wang, Lei Yang, Yicheng Qiao, Lening Wang, Zhiwei Li, Chen Lv, Shanghang Zhang, et al.  
  _2026-02-02_ · https://arxiv.org/abs/2602.01594v1  
  <details><summary>Abstract</summary>

  Advanced Driver Assistance Systems (ADAS) need to understand human driver behavior while perceiving their navigation context, but jointly learning these heterogeneous tasks would cause inter-task negative transfer and impair system performance. Here, we propose a Unified and Versatile Multimodal Multi-Task Learning (UV-M3TL) framework to simultaneously recognize driver behavior, driver emotion, vehicle behavior, and traffic context, while mitigating inter-task negative transfer. Our framework incorporates two core components: dual-branch spatial channel multimodal embedding (DB-SCME) and adaptive feature-decoupled multi-task loss (AFD-Loss). DB-SCME enhances cross-task knowledge transfer while mitigating task conflicts by employing a dual-branch structure to explicitly model salient task-shared and task-specific features. AFD-Loss improves the stability of joint optimization while guiding the model to learn diverse multi-task representations by introducing an adaptive weighting mechanism based on learning dynamics and feature decoupling constraints. We evaluate our method on the AIDE dataset, and the experimental results demonstrate that UV-M3TL achieves state-of-the-art performance across all four tasks. To further prove the versatility, we evaluate UV-M3TL on additional public multi-task perception benchmarks (BDD100K, CityScapes, NYUD-v2, and PASCAL-Context), where it consistently delivers strong performance across diverse task combinations, attaining state-of-the-art results on most tasks.

  </details>



- **HandMCM: Multi-modal Point Cloud-based Correspondence State Space Model for 3D Hand Pose Estimation**  
  Wencan Cheng, Gim Hee Lee  
  _2026-02-02_ · https://arxiv.org/abs/2602.01586v1  
  <details><summary>Abstract</summary>

  3D hand pose estimation that involves accurate estimation of 3D human hand keypoint locations is crucial for many human-computer interaction applications such as augmented reality. However, this task poses significant challenges due to self-occlusion of the hands and occlusions caused by interactions with objects. In this paper, we propose HandMCM to address these challenges. Our HandMCM is a novel method based on the powerful state space model (Mamba). By incorporating modules for local information injection/filtering and correspondence modeling, the proposed correspondence Mamba effectively learns the highly dynamic kinematic topology of keypoints across various occlusion scenarios. Moreover, by integrating multi-modal image features, we enhance the robustness and representational capacity of the input, leading to more accurate hand pose estimation. Empirical evaluations on three benchmark datasets demonstrate that our model significantly outperforms current state-of-the-art methods, particularly in challenging scenarios involving severe occlusions. These results highlight the potential of our approach to advance the accuracy and reliability of 3D hand pose estimation in practical applications.

  </details>



- **Multimodal UNcommonsense: From Odd to Ordinary and Ordinary to Odd**  
  Yejin Son, Saejin Kim, Dongjun Min, Younjae Yu  
  _2026-02-02_ · https://arxiv.org/abs/2602.01561v1  
  <details><summary>Abstract</summary>

  Commonsense reasoning in multimodal contexts remains a foundational challenge in artificial intelligence. We introduce Multimodal UNcommonsense(MUN), a benchmark designed to evaluate models' ability to handle scenarios that deviate from typical visual or contextual expectations. MUN pairs visual scenes with surprising or unlikely outcomes described in natural language, prompting models to either rationalize seemingly odd images using everyday logic or uncover unexpected interpretations in ordinary scenes. To support this task, we propose a retrieval-based in-context learning (R-ICL) framework that transfers reasoning capabilities from larger models to smaller ones without additional training. Leveraging a novel Multimodal Ensemble Retriever (MER), our method identifies semantically relevant exemplars even when image and text pairs are deliberately discordant. Experiments show an average improvement of 8.3% over baseline ICL methods, highlighting the effectiveness of R-ICL in low-frequency, atypical settings. MUN opens new directions for evaluating and improving visual-language models' robustness and adaptability in real-world, culturally diverse, and non-prototypical scenarios.

  </details>



- **Combined Flicker-banding and Moire Removal for Screen-Captured Images**  
  Libo Zhu, Zihan Zhou, Zhiyi Zhou, Yiyang Qu, Weihang Zhang, Keyu Shi, Yifan Fu, Yulun Zhang  
  _2026-02-02_ · https://arxiv.org/abs/2602.01559v1  
  <details><summary>Abstract</summary>

  Capturing display screens with mobile devices has become increasingly common, yet the resulting images often suffer from severe degradations caused by the coexistence of moiré patterns and flicker-banding, leading to significant visual quality degradation. Due to the strong coupling of these two artifacts in real imaging processes, existing methods designed for single degradations fail to generalize to such compound scenarios. In this paper, we present the first systematic study on joint removal of moiré patterns and flicker-banding in screen-captured images, and propose a unified restoration framework, named CLEAR. To support this task, we construct a large-scale dataset containing both moiré patterns and flicker-banding, and introduce an ISP-based flicker simulation pipeline to stabilize model training and expand the degradation distribution. Furthermore, we design a frequency-domain decomposition and re-composition module together with a trajectory alignment loss to enhance the modeling of compound artifacts. Extensive experiments demonstrate that the proposed method consistently. outperforms existing image restoration approaches across multiple evaluation metrics, validating its effectiveness in complex real-world scenarios.

  </details>



- **Toward Cognitive Supersensing in Multimodal Large Language Model**  
  Boyi Li, Yifan Shen, Yuanzhe Liu, Yifan Xu, Jiateng Liu, Xinzhuo Li, Zhengyuan Li, Jingyuan Zhu, Yunhan Zhong, Fangzhou Lan, et al.  
  _2026-02-02_ · https://arxiv.org/abs/2602.01541v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have achieved remarkable success in open-vocabulary perceptual tasks, yet their ability to solve complex cognitive problems remains limited, especially when visual details are abstract and require visual memory. Current approaches primarily scale Chain-of-Thought (CoT) reasoning in the text space, even when language alone is insufficient for clear and structured reasoning, and largely neglect visual reasoning mechanisms analogous to the human visuospatial sketchpad and visual imagery. To mitigate this deficiency, we introduce Cognitive Supersensing, a novel training paradigm that endows MLLMs with human-like visual imagery capabilities by integrating a Latent Visual Imagery Prediction (LVIP) head that jointly learns sequences of visual cognitive latent embeddings and aligns them with the answer, thereby forming vision-based internal reasoning chains. We further introduce a reinforcement learning stage that optimizes text reasoning paths based on this grounded visual latent. To evaluate the cognitive capabilities of MLLMs, we present CogSense-Bench, a comprehensive visual question answering (VQA) benchmark assessing five cognitive dimensions. Extensive experiments demonstrate that MLLMs trained with Cognitive Supersensing significantly outperform state-of-the-art baselines on CogSense-Bench and exhibit superior generalization on out-of-domain mathematics and science VQA benchmarks, suggesting that internal visual imagery is potentially key to bridging the gap between perceptual recognition and cognitive understanding. We will open-source the CogSense-Bench and our model weights.

  </details>


