# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-02-01 07:04 UTC_

Total papers shown: **33**


---

- **UEval: A Benchmark for Unified Multimodal Generation**  
  Bo Li, Yida Yin, Wenhao Chai, Xingyu Fu, Zhuang Liu  
  _2026-01-29_ · https://arxiv.org/abs/2601.22155v1  
  <details><summary>Abstract</summary>

  We introduce UEval, a benchmark to evaluate unified models, i.e., models capable of generating both images and text. UEval comprises 1,000 expert-curated questions that require both images and text in the model output, sourced from 8 real-world tasks. Our curated questions cover a wide range of reasoning types, from step-by-step guides to textbook explanations. Evaluating open-ended multimodal generation is non-trivial, as simple LLM-as-a-judge methods can miss the subtleties. Different from previous works that rely on multimodal Large Language Models (MLLMs) to rate image quality or text accuracy, we design a rubric-based scoring system in UEval. For each question, reference images and text answers are provided to a MLLM to generate an initial rubric, consisting of multiple evaluation criteria, and human experts then refine and validate these rubrics. In total, UEval contains 10,417 validated rubric criteria, enabling scalable and fine-grained automatic scoring. UEval is challenging for current unified models: GPT-5-Thinking scores only 66.4 out of 100, while the best open-source model reaches merely 49.1. We observe that reasoning models often outperform non-reasoning ones, and transferring reasoning traces from a reasoning model to a non-reasoning model significantly narrows the gap. This suggests that reasoning may be important for tasks requiring complex multimodal understanding and generation.

  </details>



- **DynamicVLA: A Vision-Language-Action Model for Dynamic Object Manipulation**  
  Haozhe Xie, Beichen Wen, Jiarui Zheng, Zhaoxi Chen, Fangzhou Hong, Haiwen Diao, Ziwei Liu  
  _2026-01-29_ · https://arxiv.org/abs/2601.22153v1  
  <details><summary>Abstract</summary>

  Manipulating dynamic objects remains an open challenge for Vision-Language-Action (VLA) models, which, despite strong generalization in static manipulation, struggle in dynamic scenarios requiring rapid perception, temporal anticipation, and continuous control. We present DynamicVLA, a framework for dynamic object manipulation that integrates temporal reasoning and closed-loop adaptation through three key designs: 1) a compact 0.4B VLA using a convolutional vision encoder for spatially efficient, structurally faithful encoding, enabling fast multimodal inference; 2) Continuous Inference, enabling overlapping reasoning and execution for lower latency and timely adaptation to object motion; and 3) Latent-aware Action Streaming, which bridges the perception-execution gap by enforcing temporally aligned action execution. To fill the missing foundation of dynamic manipulation data, we introduce the Dynamic Object Manipulation (DOM) benchmark, built from scratch with an auto data collection pipeline that efficiently gathers 200K synthetic episodes across 2.8K scenes and 206 objects, and enables fast collection of 2K real-world episodes without teleoperation. Extensive evaluations demonstrate remarkable improvements in response speed, perception, and generalization, positioning DynamicVLA as a unified framework for general dynamic object manipulation across embodiments.

  </details>



- **PI-Light: Physics-Inspired Diffusion for Full-Image Relighting**  
  Zhexin Liang, Zhaoxi Chen, Yongwei Chen, Tianyi Wei, Tengfei Wang, Xingang Pan  
  _2026-01-29_ · https://arxiv.org/abs/2601.22135v1  
  <details><summary>Abstract</summary>

  Full-image relighting remains a challenging problem due to the difficulty of collecting large-scale structured paired data, the difficulty of maintaining physical plausibility, and the limited generalizability imposed by data-driven priors. Existing attempts to bridge the synthetic-to-real gap for full-scene relighting remain suboptimal. To tackle these challenges, we introduce Physics-Inspired diffusion for full-image reLight ($π$-Light, or PI-Light), a two-stage framework that leverages physics-inspired diffusion models. Our design incorporates (i) batch-aware attention, which improves the consistency of intrinsic predictions across a collection of images, (ii) a physics-guided neural rendering module that enforces physically plausible light transport, (iii) physics-inspired losses that regularize training dynamics toward a physically meaningful landscape, thereby enhancing generalizability to real-world image editing, and (iv) a carefully curated dataset of diverse objects and scenes captured under controlled lighting conditions. Together, these components enable efficient finetuning of pretrained diffusion models while also providing a solid benchmark for downstream evaluation. Experiments demonstrate that $π$-Light synthesizes specular highlights and diffuse reflections across a wide variety of materials, achieving superior generalization to real-world scenes compared with prior approaches.

  </details>



- **ReactEMG Stroke: Healthy-to-Stroke Few-shot Adaptation for sEMG-Based Intent Detection**  
  Runsheng Wang, Katelyn Lee, Xinyue Zhu, Lauren Winterbottom, Dawn M. Nilsen, Joel Stein, Matei Ciocarlie  
  _2026-01-29_ · https://arxiv.org/abs/2601.22090v1  
  <details><summary>Abstract</summary>

  Surface electromyography (sEMG) is a promising control signal for assist-as-needed hand rehabilitation after stroke, but detecting intent from paretic muscles often requires lengthy, subject-specific calibration and remains brittle to variability. We propose a healthy-to-stroke adaptation pipeline that initializes an intent detector from a model pretrained on large-scale able-bodied sEMG, then fine-tunes it for each stroke participant using only a small amount of subject-specific data. Using a newly collected dataset from three individuals with chronic stroke, we compare adaptation strategies (head-only tuning, parameter-efficient LoRA adapters, and full end-to-end fine-tuning) and evaluate on held-out test sets that include realistic distribution shifts such as within-session drift, posture changes, and armband repositioning. Across conditions, healthy-pretrained adaptation consistently improves stroke intent detection relative to both zero-shot transfer and stroke-only training under the same data budget; the best adaptation methods improve average transition accuracy from 0.42 to 0.61 and raw accuracy from 0.69 to 0.78. These results suggest that transferring a reusable healthy-domain EMG representation can reduce calibration burden while improving robustness for real-time post-stroke intent detection.

  </details>



- **Unsupervised Decomposition and Recombination with Discriminator-Driven Diffusion Models**  
  Archer Wang, Emile Anand, Yilun Du, Marin Soljačić  
  _2026-01-29_ · https://arxiv.org/abs/2601.22057v1  
  <details><summary>Abstract</summary>

  Decomposing complex data into factorized representations can reveal reusable components and enable synthesizing new samples via component recombination. We investigate this in the context of diffusion-based models that learn factorized latent spaces without factor-level supervision. In images, factors can capture background, illumination, and object attributes; in robotic videos, they can capture reusable motion components. To improve both latent factor discovery and quality of compositional generation, we introduce an adversarial training signal via a discriminator trained to distinguish between single-source samples and those generated by recombining factors across sources. By optimizing the generator to fool this discriminator, we encourage physical and semantic consistency in the resulting recombinations. Our method outperforms implementations of prior baselines on CelebA-HQ, Virtual KITTI, CLEVR, and Falcor3D, achieving lower FID scores and better disentanglement as measured by MIG and MCC. Furthermore, we demonstrate a novel application to robotic video trajectories: by recombining learned action components, we generate diverse sequences that significantly increase state-space coverage for exploration on the LIBERO benchmark.

  </details>



- **Urban Neural Surface Reconstruction from Constrained Sparse Aerial Imagery with 3D SAR Fusion**  
  Da Li, Chen Yao, Tong Mao, Jiacheng Bao, Houjun Sun  
  _2026-01-29_ · https://arxiv.org/abs/2601.22045v1  
  <details><summary>Abstract</summary>

  Neural surface reconstruction (NSR) has recently shown strong potential for urban 3D reconstruction from multi-view aerial imagery. However, existing NSR methods often suffer from geometric ambiguity and instability, particularly under sparse-view conditions. This issue is critical in large-scale urban remote sensing, where aerial image acquisition is limited by flight paths, terrain, and cost. To address this challenge, we present the first urban NSR framework that fuses 3D synthetic aperture radar (SAR) point clouds with aerial imagery for high-fidelity reconstruction under constrained, sparse-view settings. 3D SAR can efficiently capture large-scale geometry even from a single side-looking flight path, providing robust priors that complement photometric cues from images. Our framework integrates radar-derived spatial constraints into an SDF-based NSR backbone, guiding structure-aware ray selection and adaptive sampling for stable and efficient optimization. We also construct the first benchmark dataset with co-registered 3D SAR point clouds and aerial imagery, facilitating systematic evaluation of cross-modal 3D reconstruction. Extensive experiments show that incorporating 3D SAR markedly enhances reconstruction accuracy, completeness, and robustness compared with single-modality baselines under highly sparse and oblique-view conditions, highlighting a viable route toward scalable high-fidelity urban reconstruction with advanced airborne and spaceborne optical-SAR sensing.

  </details>



- **MoE-ACT: Improving Surgical Imitation Learning Policies through Supervised Mixture-of-Experts**  
  Lorenzo Mazza, Ariel Rodriguez, Rayan Younis, Martin Lelis, Ortrun Hellig, Chenpan Li, Sebastian Bodenstedt, Martin Wagner, Stefanie Speidel  
  _2026-01-29_ · https://arxiv.org/abs/2601.21971v1  
  <details><summary>Abstract</summary>

  Imitation learning has achieved remarkable success in robotic manipulation, yet its application to surgical robotics remains challenging due to data scarcity, constrained workspaces, and the need for an exceptional level of safety and predictability. We present a supervised Mixture-of-Experts (MoE) architecture designed for phase-structured surgical manipulation tasks, which can be added on top of any autonomous policy. Unlike prior surgical robot learning approaches that rely on multi-camera setups or thousands of demonstrations, we show that a lightweight action decoder policy like Action Chunking Transformer (ACT) can learn complex, long-horizon manipulation from less than 150 demonstrations using solely stereo endoscopic images, when equipped with our architecture. We evaluate our approach on the collaborative surgical task of bowel grasping and retraction, where a robot assistant interprets visual cues from a human surgeon, executes targeted grasping on deformable tissue, and performs sustained retraction. We benchmark our method against state-of-the-art Vision-Language-Action (VLA) models and the standard ACT baseline. Our results show that generalist VLAs fail to acquire the task entirely, even under standard in-distribution conditions. Furthermore, while standard ACT achieves moderate success in-distribution, adopting a supervised MoE architecture significantly boosts its performance, yielding higher success rates in-distribution and demonstrating superior robustness in out-of-distribution scenarios, including novel grasp locations, reduced illumination, and partial occlusions. Notably, it generalizes to unseen testing viewpoints and also transfers zero-shot to ex vivo porcine tissue without additional training, offering a promising pathway toward in vivo deployment. To support this, we present qualitative preliminary results of policy roll-outs during in vivo porcine surgery.

  </details>



- **PaddleOCR-VL-1.5: Towards a Multi-Task 0.9B VLM for Robust In-the-Wild Document Parsing**  
  Cheng Cui, Ting Sun, Suyin Liang, Tingquan Gao, Zelun Zhang, Jiaxuan Liu, Xueqing Wang, Changda Zhou, Hongen Liu, Manhui Lin, et al.  
  _2026-01-29_ · https://arxiv.org/abs/2601.21957v1  
  <details><summary>Abstract</summary>

  We introduce PaddleOCR-VL-1.5, an upgraded model achieving a new state-of-the-art (SOTA) accuracy of 94.5% on OmniDocBench v1.5. To rigorously evaluate robustness against real-world physical distortions, including scanning, skew, warping, screen-photography, and illumination, we propose the Real5-OmniDocBench benchmark. Experimental results demonstrate that this enhanced model attains SOTA performance on the newly curated benchmark. Furthermore, we extend the model's capabilities by incorporating seal recognition and text spotting tasks, while remaining a 0.9B ultra-compact VLM with high efficiency. Code: https://github.com/PaddlePaddle/PaddleOCR

  </details>



- **BookNet: Book Image Rectification via Cross-Page Attention Network**  
  Shaokai Liu, Hao Feng, Bozhi Luan, Min Hou, Jiajun Deng, Wengang Zhou  
  _2026-01-29_ · https://arxiv.org/abs/2601.21938v1  
  <details><summary>Abstract</summary>

  Book image rectification presents unique challenges in document image processing due to complex geometric distortions from binding constraints, where left and right pages exhibit distinctly asymmetric curvature patterns. However, existing single-page document image rectification methods fail to capture the coupled geometric relationships between adjacent pages in books. In this work, we introduce BookNet, the first end-to-end deep learning framework specifically designed for dual-page book image rectification. BookNet adopts a dual-branch architecture with cross-page attention mechanisms, enabling it to estimate warping flows for both individual pages and the complete book spread, explicitly modeling how left and right pages influence each other. Moreover, to address the absence of specialized datasets, we present Book3D, a large-scale synthetic dataset for training, and Book100, a comprehensive real-world benchmark for evaluation. Extensive experiments demonstrate that BookNet outperforms existing state-of-the-art methods on book image rectification. Code and dataset will be made publicly available.

  </details>



- **VideoAesBench: Benchmarking the Video Aesthetics Perception Capabilities of Large Multimodal Models**  
  Yunhao Li, Sijing Wu, Zhilin Gao, Zicheng Zhang, Qi Jia, Huiyu Duan, Xiongkuo Min, Guangtao Zhai  
  _2026-01-29_ · https://arxiv.org/abs/2601.21915v1  
  <details><summary>Abstract</summary>

  Large multimodal models (LMMs) have demonstrated outstanding capabilities in various visual perception tasks, which has in turn made the evaluation of LMMs significant. However, the capability of video aesthetic quality assessment, which is a fundamental ability for human, remains underexplored for LMMs. To address this, we introduce VideoAesBench, a comprehensive benchmark for evaluating LMMs' understanding of video aesthetic quality. VideoAesBench has several significant characteristics: (1) Diverse content including 1,804 videos from multiple video sources including user-generated (UGC), AI-generated (AIGC), compressed, robotic-generated (RGC), and game videos. (2) Multiple question formats containing traditional single-choice questions, multi-choice questions, True or False questions, and a novel open-ended questions for video aesthetics description. (3) Holistic video aesthetics dimensions including visual form related questions from 5 aspects, visual style related questions from 4 aspects, and visual affectiveness questions from 3 aspects. Based on VideoAesBench, we benchmark 23 open-source and commercial large multimodal models. Our findings show that current LMMs only contain basic video aesthetics perception ability, their performance remains incomplete and imprecise. We hope our VideoAesBench can be served as a strong testbed and offer insights for explainable video aesthetics assessment.

  </details>



- **Beyond Global Alignment: Fine-Grained Motion-Language Retrieval via Pyramidal Shapley-Taylor Learning**  
  Hanmo Chen, Guangtao Lyu, Chenghao Xu, Jiexi Yan, Xu Yang, Cheng Deng  
  _2026-01-29_ · https://arxiv.org/abs/2601.21904v1  
  <details><summary>Abstract</summary>

  As a foundational task in human-centric cross-modal intelligence, motion-language retrieval aims to bridge the semantic gap between natural language and human motion, enabling intuitive motion analysis, yet existing approaches predominantly focus on aligning entire motion sequences with global textual representations. This global-centric paradigm overlooks fine-grained interactions between local motion segments and individual body joints and text tokens, inevitably leading to suboptimal retrieval performance. To address this limitation, we draw inspiration from the pyramidal process of human motion perception (from joint dynamics to segment coherence, and finally to holistic comprehension) and propose a novel Pyramidal Shapley-Taylor (PST) learning framework for fine-grained motion-language retrieval. Specifically, the framework decomposes human motion into temporal segments and spatial body joints, and learns cross-modal correspondences through progressive joint-wise and segment-wise alignment in a pyramidal fashion, effectively capturing both local semantic details and hierarchical structural relationships. Extensive experiments on multiple public benchmark datasets demonstrate that our approach significantly outperforms state-of-the-art methods, achieving precise alignment between motion segments and body joints and their corresponding text tokens. The code of this work will be released upon acceptance.

  </details>



- **GAZELOAD A Multimodal Eye-Tracking Dataset for Mental Workload in Industrial Human-Robot Collaboration**  
  Bsher Karbouj, Baha Eddin Gaaloul, Jorg Kruger  
  _2026-01-29_ · https://arxiv.org/abs/2601.21829v1  
  <details><summary>Abstract</summary>

  This article describes GAZELOAD, a multimodal dataset for mental workload estimation in industrial human-robot collaboration. The data were collected in a laboratory assembly testbed where 26 participants interacted with two collaborative robots (UR5 and Franka Emika Panda) while wearing Meta ARIA smart glasses. The dataset time-synchronizes eye-tracking signals (pupil diameter, fixations, saccades, eye gaze, gaze transition entropy, fixation dispersion index) with environmental real-time and continuous measurements (illuminance) and task and robot context (bench, task block, induced faults), under controlled manipulations of task difficulty and ambient conditions. For each participant and workload-graded task block, we provide CSV files with ocular metrics aggregated into 250 ms windows, environmental logs, and self-reported mental workload ratings on a 1-10 Likert scale, organized in participant-specific folders alongside documentation. These data can be used to develop and benchmark algorithms for mental workload estimation, feature extraction, and temporal modeling in realistic industrial HRC scenarios, and to investigate the influence of environmental factors such as lighting on eye-based workload markers.

  </details>



- **MMFineReason: Closing the Multimodal Reasoning Gap via Open Data-Centric Methods**  
  Honglin Lin, Zheng Liu, Yun Zhu, Chonghan Qin, Juekai Lin, Xiaoran Shang, Conghui He, Wentao Zhang, Lijun Wu  
  _2026-01-29_ · https://arxiv.org/abs/2601.21821v1  
  <details><summary>Abstract</summary>

  Recent advances in Vision Language Models (VLMs) have driven significant progress in visual reasoning. However, open-source VLMs still lag behind proprietary systems, largely due to the lack of high-quality reasoning data. Existing datasets offer limited coverage of challenging domains such as STEM diagrams and visual puzzles, and lack consistent, long-form Chain-of-Thought (CoT) annotations essential for eliciting strong reasoning capabilities. To bridge this gap, we introduce MMFineReason, a large-scale multimodal reasoning dataset comprising 1.8M samples and 5.1B solution tokens, featuring high-quality reasoning annotations distilled from Qwen3-VL-235B-A22B-Thinking. The dataset is established via a systematic three-stage pipeline: (1) large-scale data collection and standardization, (2) CoT rationale generation, and (3) comprehensive selection based on reasoning quality and difficulty awareness. The resulting dataset spans STEM problems, visual puzzles, games, and complex diagrams, with each sample annotated with visually grounded reasoning traces. We fine-tune Qwen3-VL-Instruct on MMFineReason to develop MMFineReason-2B/4B/8B versions. Our models establish new state-of-the-art results for their size class. Notably, MMFineReason-4B succesfully surpasses Qwen3-VL-8B-Thinking, and MMFineReason-8B even outperforms Qwen3-VL-30B-A3B-Thinking while approaching Qwen3-VL-32B-Thinking, demonstrating remarkable parameter efficiency. Crucially, we uncover a "less is more" phenomenon via our difficulty-aware filtering strategy: a subset of just 7\% (123K samples) achieves performance comparable to the full dataset. Notably, we reveal a synergistic effect where reasoning-oriented data composition simultaneously boosts general capabilities.

  </details>



- **Synthetic-to-Real Domain Bridging for Single-View 3D Reconstruction of Ships for Maritime Monitoring**  
  Borja Carrillo-Perez, Felix Sattler, Angel Bueno Rodriguez, Maurice Stephan, Sarah Barnes  
  _2026-01-29_ · https://arxiv.org/abs/2601.21786v1  
  <details><summary>Abstract</summary>

  Three-dimensional (3D) reconstruction of ships is an important part of maritime monitoring, allowing improved visualization, inspection, and decision-making in real-world monitoring environments. However, most state-ofthe-art 3D reconstruction methods require multi-view supervision, annotated 3D ground truth, or are computationally intensive, making them impractical for real-time maritime deployment. In this work, we present an efficient pipeline for single-view 3D reconstruction of real ships by training entirely on synthetic data and requiring only a single view at inference. Our approach uses the Splatter Image network, which represents objects as sparse sets of 3D Gaussians for rapid and accurate reconstruction from single images. The model is first fine-tuned on synthetic ShapeNet vessels and further refined with a diverse custom dataset of 3D ships, bridging the domain gap between synthetic and real-world imagery. We integrate a state-of-the-art segmentation module based on YOLOv8 and custom preprocessing to ensure compatibility with the reconstruction network. Postprocessing steps include real-world scaling, centering, and orientation alignment, followed by georeferenced placement on an interactive web map using AIS metadata and homography-based mapping. Quantitative evaluation on synthetic validation data demonstrates strong reconstruction fidelity, while qualitative results on real maritime images from the ShipSG dataset confirm the potential for transfer to operational maritime settings. The final system provides interactive 3D inspection of real ships without requiring real-world 3D annotations. This pipeline provides an efficient, scalable solution for maritime monitoring and highlights a path toward real-time 3D ship visualization in practical applications. Interactive demo: https://dlr-mi.github.io/ship3d-demo/.

  </details>



- **DreamActor-M2: Universal Character Image Animation via Spatiotemporal In-Context Learning**  
  Mingshuang Luo, Shuang Liang, Zhengkun Rong, Yuxuan Luo, Tianshu Hu, Ruibing Hou, Hong Chang, Yong Li, Yuan Zhang, Mingyuan Gao  
  _2026-01-29_ · https://arxiv.org/abs/2601.21716v1  
  <details><summary>Abstract</summary>

  Character image animation aims to synthesize high-fidelity videos by transferring motion from a driving sequence to a static reference image. Despite recent advancements, existing methods suffer from two fundamental challenges: (1) suboptimal motion injection strategies that lead to a trade-off between identity preservation and motion consistency, manifesting as a "see-saw", and (2) an over-reliance on explicit pose priors (e.g., skeletons), which inadequately capture intricate dynamics and hinder generalization to arbitrary, non-humanoid characters. To address these challenges, we present DreamActor-M2, a universal animation framework that reimagines motion conditioning as an in-context learning problem. Our approach follows a two-stage paradigm. First, we bridge the input modality gap by fusing reference appearance and motion cues into a unified latent space, enabling the model to jointly reason about spatial identity and temporal dynamics by leveraging the generative prior of foundational models. Second, we introduce a self-bootstrapped data synthesis pipeline that curates pseudo cross-identity training pairs, facilitating a seamless transition from pose-dependent control to direct, end-to-end RGB-driven animation. This strategy significantly enhances generalization across diverse characters and motion scenarios. To facilitate comprehensive evaluation, we further introduce AW Bench, a versatile benchmark encompassing a wide spectrum of characters types and motion scenarios. Extensive experiments demonstrate that DreamActor-M2 achieves state-of-the-art performance, delivering superior visual fidelity and robust cross-domain generalization. Project Page: https://grisoon.github.io/DreamActor-M2/

  </details>



- **Disentangling perception and reasoning for improving data efficiency in learning cloth manipulation without demonstrations**  
  Donatien Delehelle, Fei Chen, Darwin Caldwell  
  _2026-01-29_ · https://arxiv.org/abs/2601.21713v1  
  <details><summary>Abstract</summary>

  Cloth manipulation is a ubiquitous task in everyday life, but it remains an open challenge for robotics. The difficulties in developing cloth manipulation policies are attributed to the high-dimensional state space, complex dynamics, and high propensity to self-occlusion exhibited by fabrics. As analytical methods have not been able to provide robust and general manipulation policies, reinforcement learning (RL) is considered a promising approach to these problems. However, to address the large state space and complex dynamics, data-based methods usually rely on large models and long training times. The resulting computational cost significantly hampers the development and adoption of these methods. Additionally, due to the challenge of robust state estimation, garment manipulation policies often adopt an end-to-end learning approach with workspace images as input. While this approach enables a conceptually straightforward sim-to-real transfer via real-world fine-tuning, it also incurs a significant computational cost by training agents on a highly lossy representation of the environment state. This paper questions this common design choice by exploring an efficient and modular approach to RL for cloth manipulation. We show that, through careful design choices, model size and training time can be significantly reduced when learning in simulation. Furthermore, we demonstrate how the resulting simulation-trained model can be transferred to the real world. We evaluate our approach on the SoftGym benchmark and achieve significant performance improvements over available baselines on our task, while using a substantially smaller model.

  </details>



- **ChartE$^{3}$: A Comprehensive Benchmark for End-to-End Chart Editing**  
  Shuo Li, Jiajun Sun, Zhekai Wang, Xiaoran Fan, Hui Li, Dingwen Yang, Zhiheng Xi, Yijun Wang, Zifei Shan, Tao Gui, et al.  
  _2026-01-29_ · https://arxiv.org/abs/2601.21694v1  
  <details><summary>Abstract</summary>

  Charts are a fundamental visualization format for structured data analysis. Enabling end-to-end chart editing according to user intent is of great practical value, yet remains challenging due to the need for both fine-grained control and global structural consistency. Most existing approaches adopt pipeline-based designs, where natural language or code serves as an intermediate representation, limiting their ability to faithfully execute complex edits. We introduce ChartE$^{3}$, an End-to-End Chart Editing benchmark that directly evaluates models without relying on intermediate natural language programs or code-level supervision. ChartE$^{3}$ focuses on two complementary editing dimensions: local editing, which involves fine-grained appearance changes such as font or color adjustments, and global editing, which requires holistic, data-centric transformations including data filtering and trend line addition. ChartE$^{3}$ contains over 1,200 high-quality samples constructed via a well-designed data pipeline with human curation. Each sample is provided as a triplet of a chart image, its underlying code, and a multimodal editing instruction, enabling evaluation from both objective and subjective perspectives. Extensive benchmarking of state-of-the-art multimodal large language models reveals substantial performance gaps, particularly on global editing tasks, highlighting critical limitations in current end-to-end chart editing capabilities.

  </details>



- **SONIC-O1: A Real-World Benchmark for Evaluating Multimodal Large Language Models on Audio-Video Understanding**  
  Ahmed Y. Radwan, Christos Emmanouilidis, Hina Tabassum, Deval Pandya, Shaina Raza  
  _2026-01-29_ · https://arxiv.org/abs/2601.21666v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) are a major focus of recent AI research. However, most prior work focuses on static image understanding, while their ability to process sequential audio-video data remains underexplored. This gap highlights the need for a high-quality benchmark to systematically evaluate MLLM performance in a real-world setting. We introduce SONIC-O1, a comprehensive, fully human-verified benchmark spanning 13 real-world conversational domains with 4,958 annotations and demographic metadata. SONIC-O1 evaluates MLLMs on key tasks, including open-ended summarization, multiple-choice question (MCQ) answering, and temporal localization with supporting rationales (reasoning). Experiments on closed- and open-source models reveal limitations. While the performance gap in MCQ accuracy between two model families is relatively small, we observe a substantial 22.6% performance difference in temporal localization between the best performing closed-source and open-source models. Performance further degrades across demographic groups, indicating persistent disparities in model behavior. Overall, SONIC-O1 provides an open evaluation suite for temporally grounded and socially robust multimodal understanding. We release SONIC-O1 for reproducibility and research: Project page: https://vectorinstitute.github.io/sonic-o1/ Dataset: https://huggingface.co/datasets/vector-institute/sonic-o1 Github: https://github.com/vectorinstitute/sonic-o1 Leaderboard: https://huggingface.co/spaces/vector-institute/sonic-o1-leaderboard

  </details>



- **Few-Shot Domain Adaptation with Temporal References and Static Priors for Glacier Calving Front Delineation**  
  Marcel Dreier, Nora Gourmelon, Dakota Pyles, Thorsten Seehaus, Matthias H. Braun, Andreas Maier, Vincent Christlein  
  _2026-01-29_ · https://arxiv.org/abs/2601.21663v1  
  <details><summary>Abstract</summary>

  During benchmarking, the state-of-the-art model for glacier calving front delineation achieves near-human performance. However, when applied in a real-world setting at a novel study site, its delineation accuracy is insufficient for calving front products intended for further scientific analyses. This site represents an out-of-distribution domain for a model trained solely on the benchmark dataset. By employing a few-shot domain adaptation strategy, incorporating spatial static prior knowledge, and including summer reference images in the input time series, the delineation error is reduced from 1131.6 m to 68.7 m without any architectural modifications. These methodological advancements establish a framework for applying deep learning-based calving front segmentation to novel study sites, enabling calving front monitoring on a global scale.

  </details>



- **CAF-Mamba: Mamba-Based Cross-Modal Adaptive Attention Fusion for Multimodal Depression Detection**  
  Bowen Zhou, Marc-André Fiedler, Ayoub Al-Hamadi  
  _2026-01-29_ · https://arxiv.org/abs/2601.21648v1  
  <details><summary>Abstract</summary>

  Depression is a prevalent mental health disorder that severely impairs daily functioning and quality of life. While recent deep learning approaches for depression detection have shown promise, most rely on limited feature types, overlook explicit cross-modal interactions, and employ simple concatenation or static weighting for fusion. To overcome these limitations, we propose CAF-Mamba, a novel Mamba-based cross-modal adaptive attention fusion framework. CAF-Mamba not only captures cross-modal interactions explicitly and implicitly, but also dynamically adjusts modality contributions through a modality-wise attention mechanism, enabling more effective multimodal fusion. Experiments on two in-the-wild benchmark datasets, LMVD and D-Vlog, demonstrate that CAF-Mamba consistently outperforms existing methods and achieves state-of-the-art performance.

  </details>



- **A Tilted Seesaw: Revisiting Autoencoder Trade-off for Controllable Diffusion**  
  Pu Cao, Yiyang Ma, Feng Zhou, Xuedan Yin, Qing Song, Lu Yang  
  _2026-01-29_ · https://arxiv.org/abs/2601.21633v1  
  <details><summary>Abstract</summary>

  In latent diffusion models, the autoencoder (AE) is typically expected to balance two capabilities: faithful reconstruction and a generation-friendly latent space (e.g., low gFID). In recent ImageNet-scale AE studies, we observe a systematic bias toward generative metrics in handling this trade-off: reconstruction metrics are increasingly under-reported, and ablation-based AE selection often favors the best-gFID configuration even when reconstruction fidelity degrades. We theoretically analyze why this gFID-dominant preference can appear unproblematic for ImageNet generation, yet becomes risky when scaling to controllable diffusion: AEs can induce condition drift, which limits achievable condition alignment. Meanwhile, we find that reconstruction fidelity, especially instance-level measures, better indicates controllability. We empirically validate the impact of tilted autoencoder evaluation on controllability by studying several recent ImageNet AEs. Using a multi-dimensional condition-drift evaluation protocol reflecting controllable generation tasks, we find that gFID is only weakly predictive of condition preservation, whereas reconstruction-oriented metrics are substantially more aligned. ControlNet experiments further confirm that controllability tracks condition preservation rather than gFID. Overall, our results expose a gap between ImageNet-centric AE evaluation and the requirements of scalable controllable diffusion, offering practical guidance for more reliable benchmarking and model selection.

  </details>



- **PathReasoner-R1: Instilling Structured Reasoning into Pathology Vision-Language Model via Knowledge-Guided Policy Optimization**  
  Songhan Jiang, Fengchun Liu, Ziyue Wang, Linghan Cai, Yongbing Zhang  
  _2026-01-29_ · https://arxiv.org/abs/2601.21617v1  
  <details><summary>Abstract</summary>

  Vision-Language Models (VLMs) are advancing computational pathology with superior visual understanding capabilities. However, current systems often reduce diagnosis to directly output conclusions without verifiable evidence-linked reasoning, which severely limits clinical trust and hinders expert error rectification. To address these barriers, we construct PathReasoner, the first large-scale dataset of whole-slide image (WSI) reasoning. Unlike previous work reliant on unverified distillation, we develop a rigorous knowledge-guided generation pipeline. By leveraging medical knowledge graphs, we explicitly align structured pathological findings and clinical reasoning with diagnoses, generating over 20K high-quality instructional samples. Based on the database, we propose PathReasoner-R1, which synergizes trajectory-masked supervised fine-tuning with reasoning-oriented reinforcement learning to instill structured chain-of-thought capabilities. To ensure medical rigor, we engineer a knowledge-aware multi-granular reward function incorporating an Entity Reward mechanism strictly aligned with knowledge graphs. This effectively guides the model to optimize for logical consistency rather than mere outcome matching, thereby enhancing robustness. Extensive experiments demonstrate that PathReasoner-R1 achieves state-of-the-art performance on both PathReasoner and public benchmarks across various image scales, equipping pathology models with transparent, clinically grounded reasoning capabilities. Dataset and code are available at https://github.com/cyclexfy/PathReasoner-R1.

  </details>



- **AIR-VLA: Vision-Language-Action Systems for Aerial Manipulation**  
  Jianli Sun, Bin Tian, Qiyao Zhang, Chengxiang Li, Zihan Song, Zhiyong Cui, Yisheng Lv, Yonglin Tian  
  _2026-01-29_ · https://arxiv.org/abs/2601.21602v1  
  <details><summary>Abstract</summary>

  While Vision-Language-Action (VLA) models have achieved remarkable success in ground-based embodied intelligence, their application to Aerial Manipulation Systems (AMS) remains a largely unexplored frontier. The inherent characteristics of AMS, including floating-base dynamics, strong coupling between the UAV and the manipulator, and the multi-step, long-horizon nature of operational tasks, pose severe challenges to existing VLA paradigms designed for static or 2D mobile bases. To bridge this gap, we propose AIR-VLA, the first VLA benchmark specifically tailored for aerial manipulation. We construct a physics-based simulation environment and release a high-quality multimodal dataset comprising 3000 manually teleoperated demonstrations, covering base manipulation, object & spatial understanding, semantic reasoning, and long-horizon planning. Leveraging this platform, we systematically evaluate mainstream VLA models and state-of-the-art VLM models. Our experiments not only validate the feasibility of transferring VLA paradigms to aerial systems but also, through multi-dimensional metrics tailored to aerial tasks, reveal the capabilities and boundaries of current models regarding UAV mobility, manipulator control, and high-level planning. AIR-VLA establishes a standardized testbed and data foundation for future research in general-purpose aerial robotics. The resource of AIR-VLA will be available at https://anonymous.4open.science/r/AIR-VLA-dataset-B5CC/.

  </details>



- **EmboCoach-Bench: Benchmarking AI Agents on Developing Embodied Robots**  
  Zixing Lei, Genjia Liu, Yuanshuo Zhang, Qipeng Liu, Chuan Wen, Shanghang Zhang, Wenzhao Lian, Siheng Chen  
  _2026-01-29_ · https://arxiv.org/abs/2601.21570v1  
  <details><summary>Abstract</summary>

  The field of Embodied AI is witnessing a rapid evolution toward general-purpose robotic systems, fueled by high-fidelity simulation and large-scale data collection. However, this scaling capability remains severely bottlenecked by a reliance on labor-intensive manual oversight from intricate reward shaping to hyperparameter tuning across heterogeneous backends. Inspired by LLMs' success in software automation and science discovery, we introduce \textsc{EmboCoach-Bench}, a benchmark evaluating the capacity of LLM agents to autonomously engineer embodied policies. Spanning 32 expert-curated RL and IL tasks, our framework posits executable code as the universal interface. We move beyond static generation to assess a dynamic closed-loop workflow, where agents leverage environment feedback to iteratively draft, debug, and optimize solutions, spanning improvements from physics-informed reward design to policy architectures such as diffusion policies. Extensive evaluations yield three critical insights: (1) autonomous agents can qualitatively surpass human-engineered baselines by 26.5\% in average success rate; (2) agentic workflow with environment feedback effectively strengthens policy development and substantially narrows the performance gap between open-source and proprietary models; and (3) agents exhibit self-correction capabilities for pathological engineering cases, successfully resurrecting task performance from near-total failures through iterative simulation-in-the-loop debugging. Ultimately, this work establishes a foundation for self-evolving embodied intelligence, accelerating the paradigm shift from labor-intensive manual tuning to scalable, autonomous engineering in embodied AI field.

  </details>



- **HERS: Hidden-Pattern Expert Learning for Risk-Specific Vehicle Damage Adaptation in Diffusion Models**  
  Teerapong Panboonyuen  
  _2026-01-29_ · https://arxiv.org/abs/2601.21517v1  
  <details><summary>Abstract</summary>

  Recent advances in text-to-image (T2I) diffusion models have enabled increasingly realistic synthesis of vehicle damage, raising concerns about their reliability in automated insurance workflows. The ability to generate crash-like imagery challenges the boundary between authentic and synthetic data, introducing new risks of misuse in fraud or claim manipulation. To address these issues, we propose HERS (Hidden-Pattern Expert Learning for Risk-Specific Damage Adaptation), a framework designed to improve fidelity, controllability, and domain alignment of diffusion-generated damage images. HERS fine-tunes a base diffusion model via domain-specific expert adaptation without requiring manual annotation. Using self-supervised image-text pairs automatically generated by a large language model and T2I pipeline, HERS models each damage category, such as dents, scratches, broken lights, or cracked paint, as a separate expert. These experts are later integrated into a unified multi-damage model that balances specialization with generalization. We evaluate HERS across four diffusion backbones and observe consistent improvements: plus 5.5 percent in text faithfulness and plus 2.3 percent in human preference ratings compared to baselines. Beyond image fidelity, we discuss implications for fraud detection, auditability, and safe deployment of generative models in high-stakes domains. Our findings highlight both the opportunities and risks of domain-specific diffusion, underscoring the importance of trustworthy generation in safety-critical applications such as auto insurance.

  </details>



- **Don't double it: Efficient Agent Prediction in Occlusions**  
  Anna Rothenhäusler, Markus Mazzola, Andreas Look, Raghu Rajan, Joschka Bödecker  
  _2026-01-29_ · https://arxiv.org/abs/2601.21504v1  
  <details><summary>Abstract</summary>

  Occluded traffic agents pose a significant challenge for autonomous vehicles, as hidden pedestrians or vehicles can appear unexpectedly, yet this problem remains understudied. Existing learning-based methods, while capable of inferring the presence of hidden agents, often produce redundant occupancy predictions where a single agent is identified multiple times. This issue complicates downstream planning and increases computational load. To address this, we introduce MatchInformer, a novel transformer-based approach that builds on the state-of-the-art SceneInformer architecture. Our method improves upon prior work by integrating Hungarian Matching, a state-of-the-art object matching algorithm from object detection, into the training process to enforce a one-to-one correspondence between predictions and ground truth, thereby reducing redundancy. We further refine trajectory forecasts by decoupling an agent's heading from its motion, a strategy that improves the accuracy and interpretability of predicted paths. To better handle class imbalances, we propose using the Matthews Correlation Coefficient (MCC) to evaluate occupancy predictions. By considering all entries in the confusion matrix, MCC provides a robust measure even in sparse or imbalanced scenarios. Experiments on the Waymo Open Motion Dataset demonstrate that our approach improves reasoning about occluded regions and produces more accurate trajectory forecasts than prior methods.

  </details>



- **Hypernetwork-Based Adaptive Aggregation for Multimodal Multiple-Instance Learning in Predicting Coronary Calcium Debulking**  
  Kaito Shiku, Ichika Seo, Tetsuya Matoba, Rissei Hino, Yasuhiro Nakano, Ryoma Bise  
  _2026-01-29_ · https://arxiv.org/abs/2601.21479v1  
  <details><summary>Abstract</summary>

  In this paper, we present the first attempt to estimate the necessity of debulking coronary artery calcifications from computed tomography (CT) images. We formulate this task as a Multiple-instance Learning (MIL) problem. The difficulty of this task lies in that physicians adjust their focus and decision criteria for device usage according to tabular data representing each patient's condition. To address this issue, we propose a hypernetwork-based adaptive aggregation transformer (HyperAdAgFormer), which adaptively modifies the feature aggregation strategy for each patient based on tabular data through a hypernetwork. The experiments using the clinical dataset demonstrated the effectiveness of HyperAdAgFormer. The code is publicly available at https://github.com/Shiku-Kaito/HyperAdAgFormer.

  </details>



- **Mining Forgery Traces from Reconstruction Error: A Weakly Supervised Framework for Multimodal Deepfake Temporal Localization**  
  Midou Guo, Qilin Yin, Wei Lu, Xiangyang Luo, Rui Yang  
  _2026-01-29_ · https://arxiv.org/abs/2601.21458v1  
  <details><summary>Abstract</summary>

  Modern deepfakes have evolved into localized and intermittent manipulations that require fine-grained temporal localization. The prohibitive cost of frame-level annotation makes weakly supervised methods a practical necessity, which rely only on video-level labels. To this end, we propose Reconstruction-based Temporal Deepfake Localization (RT-DeepLoc), a weakly supervised temporal forgery localization framework that identifies forgeries via reconstruction errors. Our framework uses a Masked Autoencoder (MAE) trained exclusively on authentic data to learn its intrinsic spatiotemporal patterns; this allows the model to produce significant reconstruction discrepancies for forged segments, effectively providing the missing fine-grained cues for localization. To robustly leverage these indicators, we introduce a novel Asymmetric Intra-video Contrastive Loss (AICL). By focusing on the compactness of authentic features guided by these reconstruction cues, AICL establishes a stable decision boundary that enhances local discrimination while preserving generalization to unseen forgeries. Extensive experiments on large-scale datasets, including LAV-DF, demonstrate that RT-DeepLoc achieves state-of-the-art performance in weakly-supervised temporal forgery localization.

  </details>



- **4D-CAAL: 4D Radar-Camera Calibration and Auto-Labeling for Autonomous Driving**  
  Shanliang Yao, Zhuoxiao Li, Runwei Guan, Kebin Cao, Meng Xia, Fuping Hu, Sen Xu, Yong Yue, Xiaohui Zhu, Weiping Ding, et al.  
  _2026-01-29_ · https://arxiv.org/abs/2601.21454v1  
  <details><summary>Abstract</summary>

  4D radar has emerged as a critical sensor for autonomous driving, primarily due to its enhanced capabilities in elevation measurement and higher resolution compared to traditional 3D radar. Effective integration of 4D radar with cameras requires accurate extrinsic calibration, and the development of radar-based perception algorithms demands large-scale annotated datasets. However, existing calibration methods often employ separate targets optimized for either visual or radar modalities, complicating correspondence establishment. Furthermore, manually labeling sparse radar data is labor-intensive and unreliable. To address these challenges, we propose 4D-CAAL, a unified framework for 4D radar-camera calibration and auto-labeling. Our approach introduces a novel dual-purpose calibration target design, integrating a checkerboard pattern on the front surface for camera detection and a corner reflector at the center of the back surface for radar detection. We develop a robust correspondence matching algorithm that aligns the checkerboard center with the strongest radar reflection point, enabling accurate extrinsic calibration. Subsequently, we present an auto-labeling pipeline that leverages the calibrated sensor relationship to transfer annotations from camera-based segmentations to radar point clouds through geometric projection and multi-feature optimization. Extensive experiments demonstrate that our method achieves high calibration accuracy while significantly reducing manual annotation effort, thereby accelerating the development of robust multi-modal perception systems for autonomous driving.

  </details>



- **MultiModal Fine-tuning with Synthetic Captions**  
  Shohei Enomoto, Shin'ya Yamaguchi  
  _2026-01-29_ · https://arxiv.org/abs/2601.21426v1  
  <details><summary>Abstract</summary>

  In this paper, we address a fundamental gap between pre-training and fine-tuning of deep neural networks: while pre-training has shifted from unimodal to multimodal learning with enhanced visual understanding, fine-tuning predominantly remains unimodal, limiting the benefits of rich pre-trained representations. To bridge this gap, we propose a novel approach that transforms unimodal datasets into multimodal ones using Multimodal Large Language Models (MLLMs) to generate synthetic image captions for fine-tuning models with a multimodal objective. Our method employs carefully designed prompts incorporating class labels and domain context to produce high-quality captions tailored for classification tasks. Furthermore, we introduce a supervised contrastive loss function that explicitly encourages clustering of same-class representations during fine-tuning, along with a new inference technique that leverages class-averaged text embeddings from multiple synthetic captions per image. Extensive experiments across 13 image classification benchmarks demonstrate that our approach outperforms baseline methods, with particularly significant improvements in few-shot learning scenarios. Our work establishes a new paradigm for dataset enhancement that effectively bridges the gap between multimodal pre-training and fine-tuning. Our code is available at https://github.com/s-enmt/MMFT.

  </details>



- **From Implicit Ambiguity to Explicit Solidity: Diagnosing Interior Geometric Degradation in Neural Radiance Fields for Dense 3D Scene Understanding**  
  Jiangsan Zhao, Jakob Geipel, Kryzysztof Kusnierek  
  _2026-01-29_ · https://arxiv.org/abs/2601.21421v1  
  <details><summary>Abstract</summary>

  Neural Radiance Fields (NeRFs) have emerged as a powerful paradigm for multi-view reconstruction, complementing classical photogrammetric pipelines based on Structure-from-Motion (SfM) and Multi-View Stereo (MVS). However, their reliability for quantitative 3D analysis in dense, self-occluding scenes remains poorly understood. In this study, we identify a fundamental failure mode of implicit density fields under heavy occlusion, which we term Interior Geometric Degradation (IGD). We show that transmittance-based volumetric optimization satisfies photometric supervision by reconstructing hollow or fragmented structures rather than solid interiors, leading to systematic instance undercounting. Through controlled experiments on synthetic datasets with increasing occlusion, we demonstrate that state-of-the-art mask-supervised NeRFs saturate at approximately 89% instance recovery in dense scenes, despite improved surface coherence and mask quality. To overcome this limitation, we introduce an explicit geometric pipeline based on Sparse Voxel Rasterization (SVRaster), initialized from SfM feature geometry. By projecting 2D instance masks onto an explicit voxel grid and enforcing geometric separation via recursive splitting, our approach preserves physical solidity and achieves a 95.8% recovery rate in dense clusters. A sensitivity analysis using degraded segmentation masks further shows that explicit SfM-based geometry is substantially more robust to supervision failure, recovering 43% more instances than implicit baselines. These results demonstrate that explicit geometric priors are a prerequisite for reliable quantitative analysis in highly self-occluding 3D scenes.

  </details>



- **Spotlighting Task-Relevant Features: Object-Centric Representations for Better Generalization in Robotic Manipulation**  
  Alexandre Chapin, Bruno Machado, Emmanuel Dellandréa, Liming Chen  
  _2026-01-29_ · https://arxiv.org/abs/2601.21416v1  
  <details><summary>Abstract</summary>

  The generalization capabilities of robotic manipulation policies are heavily influenced by the choice of visual representations. Existing approaches typically rely on representations extracted from pre-trained encoders, using two dominant types of features: global features, which summarize an entire image via a single pooled vector, and dense features, which preserve a patch-wise embedding from the final encoder layer. While widely used, both feature types mix task-relevant and irrelevant information, leading to poor generalization under distribution shifts, such as changes in lighting, textures, or the presence of distractors. In this work, we explore an intermediate structured alternative: Slot-Based Object-Centric Representations (SBOCR), which group dense features into a finite set of object-like entities. This representation permits to naturally reduce the noise provided to the robotic manipulation policy while keeping enough information to efficiently perform the task. We benchmark a range of global and dense representations against intermediate slot-based representations, across a suite of simulated and real-world manipulation tasks ranging from simple to complex. We evaluate their generalization under diverse visual conditions, including changes in lighting, texture, and the presence of distractors. Our findings reveal that SBOCR-based policies outperform dense and global representation-based policies in generalization settings, even without task-specific pretraining. These insights suggest that SBOCR is a promising direction for designing visual systems that generalize effectively in dynamic, real-world robotic environments.

  </details>



- **Semantic-Guided Dynamic Sparsification for Pre-Trained Model-based Class-Incremental Learning**  
  Ruiqi Liu, Boyu Diao, Zijia An, Runjie Shao, Zhulin An, Fei Wang, Yongjun Xu  
  _2026-01-29_ · https://arxiv.org/abs/2601.21345v1  
  <details><summary>Abstract</summary>

  Class-Incremental Learning (CIL) requires a model to continually learn new classes without forgetting old ones. A common and efficient solution freezes a pre-trained model and employs lightweight adapters, whose parameters are often forced to be orthogonal to prevent inter-task interference. However, we argue that this parameter-constraining method is detrimental to plasticity. To this end, we propose Semantic-Guided Dynamic Sparsification (SGDS), a novel method that proactively guides the activation space by governing the orientation and rank of its subspaces through targeted sparsification. Specifically, SGDS promotes knowledge transfer by encouraging similar classes to share a compact activation subspace, while simultaneously preventing interference by assigning non-overlapping activation subspaces to dissimilar classes. By sculpting class-specific sparse subspaces in the activation space, SGDS effectively mitigates interference without imposing rigid constraints on the parameter space. Extensive experiments on various benchmark datasets demonstrate the state-of-the-art performance of SGDS.

  </details>


