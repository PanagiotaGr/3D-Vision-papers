# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-01-28 06:53 UTC_

Total papers shown: **50**


---

- **DuwatBench: Bridging Language and Visual Heritage through an Arabic Calligraphy Benchmark for Multimodal Understanding**  
  Shubham Patle, Sara Ghaboura, Hania Tariq, Mohammad Usman Khan, Omkar Thawakar, Rao Muhammad Anwer, Salman Khan  
  _2026-01-27_ · https://arxiv.org/abs/2601.19898v1  
  <details><summary>Abstract</summary>

  Arabic calligraphy represents one of the richest visual traditions of the Arabic language, blending linguistic meaning with artistic form. Although multimodal models have advanced across languages, their ability to process Arabic script, especially in artistic and stylized calligraphic forms, remains largely unexplored. To address this gap, we present DuwatBench, a benchmark of 1,272 curated samples containing about 1,475 unique words across six classical and modern calligraphic styles, each paired with sentence-level detection annotations. The dataset reflects real-world challenges in Arabic writing, such as complex stroke patterns, dense ligatures, and stylistic variations that often challenge standard text recognition systems. Using DuwatBench, we evaluated 13 leading Arabic and multilingual multimodal models and showed that while they perform well on clean text, they struggle with calligraphic variation, artistic distortions, and precise visual-text alignment. By publicly releasing DuwatBench and its annotations, we aim to advance culturally grounded multimodal research, foster fair inclusion of the Arabic language and visual heritage in AI systems, and support continued progress in this area. Our dataset (https://huggingface.co/datasets/MBZUAI/DuwatBench) and evaluation suit (https://github.com/mbzuai-oryx/DuwatBench) are publicly available.

  </details>



- **VGGT-SLAM 2.0: Real time Dense Feed-forward Scene Reconstruction**  
  Dominic Maggio, Luca Carlone  
  _2026-01-27_ · https://arxiv.org/abs/2601.19887v1  
  <details><summary>Abstract</summary>

  We present VGGT-SLAM 2.0, a real time RGB feed-forward SLAM system which substantially improves upon VGGT-SLAM for incrementally aligning submaps created from VGGT. Firstly, we remove high-dimensional 15-degree-of-freedom drift and planar degeneracy from VGGT-SLAM by creating a new factor graph design while still addressing the reconstruction ambiguity of VGGT given unknown camera intrinsics. Secondly, by studying the attention layers of VGGT, we show that one of the layers is well suited to assist in image retrieval verification for free without additional training, which enables both rejecting false positive matches and allows for completing more loop closures. Finally, we conduct a suite of experiments which includes showing VGGT-SLAM 2.0 can easily be adapted for open-set object detection and demonstrating real time performance while running online onboard a ground robot using a Jetson Thor. We also test in environments ranging from cluttered indoor apartments and office scenes to a 4,200 square foot barn, and we also demonstrate VGGT-SLAM 2.0 achieves the highest accuracy on the TUM dataset with about 23 percent less pose error than VGGT-SLAM. Code will be released upon publication.

  </details>



- **Information-Theoretic Detection of Bimanual Interactions for Dual-Arm Robot Plan Generation**  
  Elena Merlo, Marta Lagomarsino, Arash Ajoudani  
  _2026-01-27_ · https://arxiv.org/abs/2601.19832v1  
  <details><summary>Abstract</summary>

  Programming by demonstration is a strategy to simplify the robot programming process for non-experts via human demonstrations. However, its adoption for bimanual tasks is an underexplored problem due to the complexity of hand coordination, which also hinders data recording. This paper presents a novel one-shot method for processing a single RGB video of a bimanual task demonstration to generate an execution plan for a dual-arm robotic system. To detect hand coordination policies, we apply Shannon's information theory to analyze the information flow between scene elements and leverage scene graph properties. The generated plan is a modular behavior tree that assumes different structures based on the desired arms coordination. We validated the effectiveness of this framework through multiple subject video demonstrations, which we collected and made open-source, and exploiting data from an external, publicly available dataset. Comparisons with existing methods revealed significant improvements in generating a centralized execution plan for coordinating two-arm systems.

  </details>



- **Diffusion for De-Occlusion: Accessory-Aware Diffusion Inpainting for Robust Ear Biometric Recognition**  
  Deeksha Arun, Kevin W. Bowyer, Patrick Flynn  
  _2026-01-27_ · https://arxiv.org/abs/2601.19795v1  
  <details><summary>Abstract</summary>

  Ear occlusions (arising from the presence of ear accessories such as earrings and earphones) can negatively impact performance in ear-based biometric recognition systems, especially in unconstrained imaging circumstances. In this study, we assess the effectiveness of a diffusion-based ear inpainting technique as a pre-processing aid to mitigate the issues of ear accessory occlusions in transformer-based ear recognition systems. Given an input ear image and an automatically derived accessory mask, the inpainting model reconstructs clean and anatomically plausible ear regions by synthesizing missing pixels while preserving local geometric coherence along key ear structures, including the helix, antihelix, concha, and lobule. We evaluate the effectiveness of this pre-processing aid in transformer-based recognition systems for several vision transformer models and different patch sizes for a range of benchmark datasets. Experiments show that diffusion-based inpainting can be a useful pre-processing aid to alleviate ear accessory occlusions to improve overall recognition performance.

  </details>



- **WaterClear-GS: Optical-Aware Gaussian Splatting for Underwater Reconstruction and Restoration**  
  Xinrui Zhang, Yufeng Wang, Shuangkang Fang, Zesheng Wang, Dacheng Qi, Wenrui Ding  
  _2026-01-27_ · https://arxiv.org/abs/2601.19753v1  
  <details><summary>Abstract</summary>

  Underwater 3D reconstruction and appearance restoration are hindered by the complex optical properties of water, such as wavelength-dependent attenuation and scattering. Existing Neural Radiance Fields (NeRF)-based methods struggle with slow rendering speeds and suboptimal color restoration, while 3D Gaussian Splatting (3DGS) inherently lacks the capability to model complex volumetric scattering effects. To address these issues, we introduce WaterClear-GS, the first pure 3DGS-based framework that explicitly integrates underwater optical properties of local attenuation and scattering into Gaussian primitives, eliminating the need for an auxiliary medium network. Our method employs a dual-branch optimization strategy to ensure underwater photometric consistency while naturally recovering water-free appearances. This strategy is enhanced by depth-guided geometry regularization and perception-driven image loss, together with exposure constraints, spatially-adaptive regularization, and physically guided spectral regularization, which collectively enforce local 3D coherence and maintain natural visual perception. Experiments on standard benchmarks and our newly collected dataset demonstrate that WaterClear-GS achieves outstanding performance on both novel view synthesis (NVS) and underwater image restoration (UIR) tasks, while maintaining real-time rendering. The code will be available at https://buaaxrzhang.github.io/WaterClear-GS/.

  </details>



- **Benchmarking Multimodal Large Language Models for Missing Modality Completion in Product Catalogues**  
  Junchen Fu, Wenhao Deng, Kaiwen Zheng, Alexandros Karatzoglou, Ioannis Arapakis, Yu Ye, Yongxin Ni, Joemon M. Jose, Xuri Ge  
  _2026-01-27_ · https://arxiv.org/abs/2601.19750v1  
  <details><summary>Abstract</summary>

  Missing-modality information on e-commerce platforms, such as absent product images or textual descriptions, often arises from annotation errors or incomplete metadata, impairing both product presentation and downstream applications such as recommendation systems. Motivated by the multimodal generative capabilities of recent Multimodal Large Language Models (MLLMs), this work investigates a fundamental yet underexplored question: can MLLMs generate missing modalities for products in e-commerce scenarios? We propose the Missing Modality Product Completion Benchmark (MMPCBench), which consists of two sub-benchmarks: a Content Quality Completion Benchmark and a Recommendation Benchmark. We further evaluate six state-of-the-art MLLMs from the Qwen2.5-VL and Gemma-3 model families across nine real-world e-commerce categories, focusing on image-to-text and text-to-image completion tasks. Experimental results show that while MLLMs can capture high-level semantics, they struggle with fine-grained word-level and pixel- or patch-level alignment. In addition, performance varies substantially across product categories and model scales, and we observe no trivial correlation between model size and performance, in contrast to trends commonly reported in mainstream benchmarks. We also explore Group Relative Policy Optimization (GRPO) to better align MLLMs with this task. GRPO improves image-to-text completion but does not yield gains for text-to-image completion. Overall, these findings expose the limitations of current MLLMs in real-world cross-modal generation and represent an early step toward more effective missing-modality product completion.

  </details>



- **Interpretable and backpropagation-free Green Learning for efficient multi-task echocardiographic segmentation and classification**  
  Jyun-Ping Kao, Jiaxing Yang, C. -C. Jay Kuo, Jonghye Woo  
  _2026-01-27_ · https://arxiv.org/abs/2601.19743v1  
  <details><summary>Abstract</summary>

  Echocardiography is a cornerstone for managing heart failure (HF), with Left Ventricular Ejection Fraction (LVEF) being a critical metric for guiding therapy. However, manual LVEF assessment suffers from high inter-observer variability, while existing Deep Learning (DL) models are often computationally intensive and data-hungry "black boxes" that impede clinical trust and adoption. Here, we propose a backpropagation-free multi-task Green Learning (MTGL) framework that performs simultaneous Left Ventricle (LV) segmentation and LVEF classification. Our framework integrates an unsupervised VoxelHop encoder for hierarchical spatio-temporal feature extraction with a multi-level regression decoder and an XG-Boost classifier. On the EchoNet-Dynamic dataset, our MTGL model achieves state-of-the-art classification and segmentation performance, attaining a classification accuracy of 94.3% and a Dice Similarity Coefficient (DSC) of 0.912, significantly outperforming several advanced 3D DL models. Crucially, our model achieves this with over an order of magnitude fewer parameters, demonstrating exceptional computational efficiency. This work demonstrates that the GL paradigm can deliver highly accurate, efficient, and interpretable solutions for complex medical image analysis, paving the way for more sustainable and trustworthy artificial intelligence in clinical practice.

  </details>



- **A new Image Similarity Metric for a Perceptual and Transparent Geometric and Chromatic Assessment**  
  Antonio Di Marino, Vincenzo Bevilacqua, Emanuel Di Nardo, Angelo Ciaramella, Ivanoe De Falco, Giovanna Sannino  
  _2026-01-27_ · https://arxiv.org/abs/2601.19680v1  
  <details><summary>Abstract</summary>

  In the literature, several studies have shown that state-of-the-art image similarity metrics are not perceptual metrics; moreover, they have difficulty evaluating images, especially when texture distortion is also present. In this work, we propose a new perceptual metric composed of two terms. The first term evaluates the dissimilarity between the textures of two images using Earth Mover's Distance. The second term evaluates the chromatic dissimilarity between two images in the Oklab perceptual color space. We evaluated the performance of our metric on a non-traditional dataset, called Berkeley-Adobe Perceptual Patch Similarity, which contains a wide range of complex distortions in shapes and colors. We have shown that our metric outperforms the state of the art, especially when images contain shape distortions, confirming also its greater perceptiveness. Furthermore, although deep black-box metrics could be very accurate, they only provide similarity scores between two images, without explaining their main differences and similarities. Our metric, on the other hand, provides visual explanations to support the calculated score, making the similarity assessment transparent and justified.

  </details>



- **Towards Governance-Oriented Low-Altitude Intelligence: A Management-Centric Multi-Modal Benchmark With Implicitly Coordinated Vision-Language Reasoning Framework**  
  Hao Chang, Zhihui Wang, Lingxiang Wu, Peijin Wang, Wenhui Diao, Jinqiao Wang  
  _2026-01-27_ · https://arxiv.org/abs/2601.19640v1  
  <details><summary>Abstract</summary>

  Low-altitude vision systems are becoming a critical infrastructure for smart city governance. However, existing object-centric perception paradigms and loosely coupled vision-language pipelines are still difficult to support management-oriented anomaly understanding required in real-world urban governance. To bridge this gap, we introduce GovLA-10K, the first management-oriented multi-modal benchmark for low-altitude intelligence, along with GovLA-Reasoner, a unified vision-language reasoning framework tailored for governance-aware aerial perception. Unlike existing studies that aim to exhaustively annotate all visible objects, GovLA-10K is deliberately designed around functionally salient targets that directly correspond to practical management needs, and further provides actionable management suggestions grounded in these observations. To effectively coordinate the fine-grained visual grounding with high-level contextual language reasoning, GovLA-Reasoner introduces an efficient feature adapter that implicitly coordinates discriminative representation sharing between the visual detector and the large language model (LLM). Extensive experiments show that our method significantly improves performance while avoiding the need of fine-tuning for any task-specific individual components. We believe our work offers a new perspective and foundation for future studies on management-aware low-altitude vision-language systems.

  </details>



- **The role of self-supervised pretraining in differentially private medical image analysis**  
  Soroosh Tayebi Arasteh, Mina Farajiamiri, Mahshad Lotfinia, Behrus Hinrichs-Puladi, Jonas Bienzeisler, Mohamed Alhaskir, Mirabela Rusu, Christiane Kuhl, Sven Nebelung, Daniel Truhn  
  _2026-01-27_ · https://arxiv.org/abs/2601.19618v1  
  <details><summary>Abstract</summary>

  Differential privacy (DP) provides formal protection for sensitive data but typically incurs substantial losses in diagnostic performance. Model initialization has emerged as a critical factor in mitigating this degradation, yet the role of modern self-supervised learning under full-model DP remains poorly understood. Here, we present a large-scale evaluation of initialization strategies for differentially private medical image analysis, using chest radiograph classification as a representative benchmark with more than 800,000 images. Using state-of-the-art ConvNeXt models trained with DP-SGD across realistic privacy regimes, we compare non-domain-specific supervised ImageNet initialization, non-domain-specific self-supervised DINOv3 initialization, and domain-specific supervised pretraining on MIMIC-CXR, the largest publicly available chest radiograph dataset. Evaluations are conducted across five external datasets spanning diverse institutions and acquisition settings. We show that DINOv3 initialization consistently improves diagnostic utility relative to ImageNet initialization under DP, but remains inferior to domain-specific supervised pretraining, which achieves performance closest to non-private baselines. We further demonstrate that initialization choice strongly influences demographic fairness, cross-dataset generalization, and robustness to data scale and model capacity under privacy constraints. The results establish initialization strategy as a central determinant of utility, fairness, and generalization in differentially private medical imaging.

  </details>



- **Localized Latent Editing for Dose-Response Modeling in Botulinum Toxin Injection Planning**  
  Estèphe Arnaud, Mohamed Daoudi, Pierre Guerreschi  
  _2026-01-27_ · https://arxiv.org/abs/2601.19593v1  
  <details><summary>Abstract</summary>

  Botulinum toxin (Botox) injections are the gold standard for managing facial asymmetry and aesthetic rejuvenation, yet determining the optimal dosage remains largely intuitive, often leading to suboptimal outcomes. We propose a localized latent editing framework that simulates Botulinum Toxin injection effects for injection planning through dose-response modeling. Our key contribution is a Region-Specific Latent Axis Discovery method that learns localized muscle relaxation trajectories in StyleGAN2's latent space, enabling precise control over specific facial regions without global side effects. By correlating these localized latent trajectories with injected toxin units, we learn a predictive dose-response model. We rigorously compare two approaches: direct metric regression versus image-based generative simulation on a clinical dataset of N=360 images from 46 patients. On a hold-out test set, our framework demonstrates moderate-to-strong structural correlations for geometric asymmetry metrics, confirming that the generative model correctly captures the direction of morphological changes. While biological variability limits absolute precision, we introduce a hybrid "Human-in-the-Loop" workflow where clinicians interactively refine simulations, bridging the gap between pathological reconstruction and cosmetic planning.

  </details>



- **ScenePilot-Bench: A Large-Scale Dataset and Benchmark for Evaluation of Vision-Language Models in Autonomous Driving**  
  Yujin Wang, Yutong Zheng, Wenxian Fan, Tianyi Wang, Hongqing Chu, Daxin Tian, Bingzhao Gao, Jianqiang Wang, Hong Chen  
  _2026-01-27_ · https://arxiv.org/abs/2601.19582v1  
  <details><summary>Abstract</summary>

  In this paper, we introduce ScenePilot-Bench, a large-scale first-person driving benchmark designed to evaluate vision-language models (VLMs) in autonomous driving scenarios. ScenePilot-Bench is built upon ScenePilot-4K, a diverse dataset comprising 3,847 hours of driving videos, annotated with multi-granularity information including scene descriptions, risk assessments, key participant identification, ego trajectories, and camera parameters. The benchmark features a four-axis evaluation suite that assesses VLM capabilities in scene understanding, spatial perception, motion planning, and GPT-Score, with safety-aware metrics and cross-region generalization settings. We benchmark representative VLMs on ScenePilot-Bench, providing empirical analyses that clarify current performance boundaries and identify gaps for driving-oriented reasoning. ScenePilot-Bench offers a comprehensive framework for evaluating and advancing VLMs in safety-critical autonomous driving contexts.

  </details>



- **The S3LI Vulcano Dataset: A Dataset for Multi-Modal SLAM in Unstructured Planetary Environments**  
  Riccardo Giubilato, Marcus Gerhard Müller, Marco Sewtz, Laura Alejandra Encinar Gonzalez, John Folkesson, Rudolph Triebel  
  _2026-01-27_ · https://arxiv.org/abs/2601.19557v1  
  <details><summary>Abstract</summary>

  We release the S3LI Vulcano dataset, a multi-modal dataset towards development and benchmarking of Simultaneous Localization and Mapping (SLAM) and place recognition algorithms that rely on visual and LiDAR modalities. Several sequences are recorded on the volcanic island of Vulcano, from the Aeolian Islands in Sicily, Italy. The sequences provide users with data from a variety of environments, textures and terrains, including basaltic or iron-rich rocks, geological formations from old lava channels, as well as dry vegetation and water. The data (rmc.dlr.de/s3li_dataset) is accompanied by an open source toolkit (github.com/DLR-RM/s3li-toolkit) providing tools for generating ground truth poses as well as preparation of labelled samples for place recognition tasks.

  </details>



- **A Non-Invasive 3D Gait Analysis Framework for Quantifying Psychomotor Retardation in Major Depressive Disorder**  
  Fouad Boutaleb, Emery Pierson, Mohamed Daoudi, Clémence Nineuil, Ali Amad, Fabien D'Hondt  
  _2026-01-27_ · https://arxiv.org/abs/2601.19526v1  
  <details><summary>Abstract</summary>

  Predicting the status of Major Depressive Disorder (MDD) from objective, non-invasive methods is an active research field. Yet, extracting automatically objective, interpretable features for a detailed analysis of the patient state remains largely unexplored. Among MDD's symptoms, Psychomotor retardation (PMR) is a core item, yet its clinical assessment remains largely subjective. While 3D motion capture offers an objective alternative, its reliance on specialized hardware often precludes routine clinical use. In this paper, we propose a non-invasive computational framework that transforms monocular RGB video into clinically relevant 3D gait kinematics. Our pipeline uses Gravity-View Coordinates along with a novel trajectory-correction algorithm that leverages the closed-loop topology of our adapted Timed Up and Go (TUG) protocol to mitigate monocular depth errors. This novel pipeline enables the extraction of 297 explicit gait biomechanical biomarkers from a single camera capture. To address the challenges of small clinical datasets, we introduce a stability-based machine learning framework that identifies robust motor signatures while preventing overfitting. Validated on the CALYPSO dataset, our method achieves an 83.3% accuracy in detecting PMR and explains 64% of the variance in overall depression severity (R^2=0.64). Notably, our study reveals a strong link between reduced ankle propulsion and restricted pelvic mobility to the depressive motor phenotype. These results demonstrate that physical movement serves as a robust proxy for the cognitive state, offering a transparent and scalable tool for the objective monitoring of depression in standard clinical environments.

  </details>



- **ALRM: Agentic LLM for Robotic Manipulation**  
  Vitor Gaboardi dos Santos, Ibrahim Khadraoui, Ibrahim Farhat, Hamza Yous, Samy Teffahi, Hakim Hacid  
  _2026-01-27_ · https://arxiv.org/abs/2601.19510v1  
  <details><summary>Abstract</summary>

  Large Language Models (LLMs) have recently empowered agentic frameworks to exhibit advanced reasoning and planning capabilities. However, their integration in robotic control pipelines remains limited in two aspects: (1) prior \ac{llm}-based approaches often lack modular, agentic execution mechanisms, limiting their ability to plan, reflect on outcomes, and revise actions in a closed-loop manner; and (2) existing benchmarks for manipulation tasks focus on low-level control and do not systematically evaluate multistep reasoning and linguistic variation. In this paper, we propose Agentic LLM for Robot Manipulation (ALRM), an LLM-driven agentic framework for robotic manipulation. ALRM integrates policy generation with agentic execution through a ReAct-style reasoning loop, supporting two complementary modes: Code-asPolicy (CaP) for direct executable control code generation, and Tool-as-Policy (TaP) for iterative planning and tool-based action execution. To enable systematic evaluation, we also introduce a novel simulation benchmark comprising 56 tasks across multiple environments, capturing linguistically diverse instructions. Experiments with ten LLMs demonstrate that ALRM provides a scalable, interpretable, and modular approach for bridging natural language reasoning with reliable robotic execution. Results reveal Claude-4.1-Opus as the top closed-source model and Falcon-H1-7B as the top open-source model under CaP.

  </details>



- **Reinforcement Learning Goal-Reaching Control with Guaranteed Lyapunov-Like Stabilizer for Mobile Robots**  
  Mehdi Heydari Shahna, Seyed Adel Alizadeh Kolagar, Jouni Mattila  
  _2026-01-27_ · https://arxiv.org/abs/2601.19499v1  
  <details><summary>Abstract</summary>

  Reinforcement learning (RL) can be highly effective at learning goal-reaching policies, but it typically does not provide formal guarantees that the goal will always be reached. A common approach to provide formal goal-reaching guarantees is to introduce a shielding mechanism that restricts the agent to actions that satisfy predefined safety constraints. The main challenge here is integrating this mechanism with RL so that learning and exploration remain effective without becoming overly conservative. Hence, this paper proposes an RL-based control framework that provides formal goal-reaching guarantees for wheeled mobile robots operating in unstructured environments. We first design a real-time RL policy with a set of 15 carefully defined reward terms. These rewards encourage the robot to reach both static and dynamic goals while generating sufficiently smooth command signals that comply with predefined safety specifications, which is critical in practice. Second, a Lyapunov-like stabilizer layer is integrated into the benchmark RL framework as a policy supervisor to formally strengthen the goal-reaching control while preserving meaningful exploration of the state action space. The proposed framework is suitable for real-time deployment in challenging environments, as it provides a formal guarantee of convergence to the intended goal states and compensates for uncertainties by generating real-time control signals based on the current state, while respecting real-world motion constraints. The experimental results show that the proposed Lyapunov-like stabilizer consistently improves the benchmark RL policies, boosting the goal-reaching rate from 84.6% to 99.0%, sharply reducing failures, and improving efficiency.

  </details>



- **Cortex-Grounded Diffusion Models for Brain Image Generation**  
  Fabian Bongratz, Yitong Li, Sama Elbaroudy, Christian Wachinger  
  _2026-01-27_ · https://arxiv.org/abs/2601.19498v1  
  <details><summary>Abstract</summary>

  Synthetic neuroimaging data can mitigate critical limitations of real-world datasets, including the scarcity of rare phenotypes, domain shifts across scanners, and insufficient longitudinal coverage. However, existing generative models largely rely on weak conditioning signals, such as labels or text, which lack anatomical grounding and often produce biologically implausible outputs. To this end, we introduce Cor2Vox, a cortex-grounded generative framework for brain magnetic resonance image (MRI) synthesis that ties image generation to continuous structural priors of the cerebral cortex. It leverages high-resolution cortical surfaces to guide a 3D shape-to-image Brownian bridge diffusion process, enabling topologically faithful synthesis and precise control over underlying anatomies. To support the generation of new, realistic brain shapes, we developed a large-scale statistical shape model of cortical morphology derived from over 33,000 UK Biobank scans. We validated the fidelity of Cor2Vox based on traditional image quality metrics, advanced cortical surface reconstruction, and whole-brain segmentation quality, outperforming many baseline methods. Across three applications, namely (i) anatomically consistent synthesis, (ii) simulation of progressive gray matter atrophy, and (iii) harmonization of in-house frontotemporal dementia scans with public datasets, Cor2Vox preserved fine-grained cortical morphology at the sub-voxel level, exhibiting remarkable robustness to variations in cortical geometry and disease phenotype without retraining.

  </details>



- **Dynamic Worlds, Dynamic Humans: Generating Virtual Human-Scene Interaction Motion in Dynamic Scenes**  
  Yin Wang, Zhiying Leng, Haitian Liu, Frederick W. B. Li, Mu Li, Xiaohui Liang  
  _2026-01-27_ · https://arxiv.org/abs/2601.19484v1  
  <details><summary>Abstract</summary>

  Scenes are continuously undergoing dynamic changes in the real world. However, existing human-scene interaction generation methods typically treat the scene as static, which deviates from reality. Inspired by world models, we introduce Dyn-HSI, the first cognitive architecture for dynamic human-scene interaction, which endows virtual humans with three humanoid components. (1)Vision (human eyes): we equip the virtual human with a Dynamic Scene-Aware Navigation, which continuously perceives changes in the surrounding environment and adaptively predicts the next waypoint. (2)Memory (human brain): we equip the virtual human with a Hierarchical Experience Memory, which stores and updates experiential data accumulated during training. This allows the model to leverage prior knowledge during inference for context-aware motion priming, thereby enhancing both motion quality and generalization. (3) Control (human body): we equip the virtual human with Human-Scene Interaction Diffusion Model, which generates high-fidelity interaction motions conditioned on multimodal inputs. To evaluate performance in dynamic scenes, we extend the existing static human-scene interaction datasets to construct a dynamic benchmark, Dyn-Scenes. We conduct extensive qualitative and quantitative experiments to validate Dyn-HSI, showing that our method consistently outperforms existing approaches and generates high-quality human-scene interaction motions in both static and dynamic settings.

  </details>



- **Towards Gold-Standard Depth Estimation for Tree Branches in UAV Forestry: Benchmarking Deep Stereo Matching Methods**  
  Yida Lin, Bing Xue, Mengjie Zhang, Sam Schofield, Richard Green  
  _2026-01-27_ · https://arxiv.org/abs/2601.19461v1  
  <details><summary>Abstract</summary>

  Autonomous UAV forestry operations require robust depth estimation with strong cross-domain generalization, yet existing evaluations focus on urban and indoor scenarios, leaving a critical gap for vegetation-dense environments. We present the first systematic zero-shot evaluation of eight stereo methods spanning iterative refinement, foundation model, diffusion-based, and 3D CNN paradigms. All methods use officially released pretrained weights (trained on Scene Flow) and are evaluated on four standard benchmarks (ETH3D, KITTI 2012/2015, Middlebury) plus a novel 5,313-pair Canterbury Tree Branches dataset ($1920 \times 1080$). Results reveal scene-dependent patterns: foundation models excel on structured scenes (BridgeDepth: 0.23 px on ETH3D; DEFOM: 4.65 px on Middlebury), while iterative methods show variable cross-benchmark performance (IGEV++: 0.36 px on ETH3D but 6.77 px on Middlebury; IGEV: 0.33 px on ETH3D but 4.99 px on Middlebury). Qualitative evaluation on the Tree Branches dataset establishes DEFOM as the gold-standard baseline for vegetation depth estimation, with superior cross-domain consistency (consistently ranking 1st-2nd across benchmarks, average rank 1.75). DEFOM predictions will serve as pseudo-ground-truth for future benchmarking.

  </details>



- **RoamScene3D: Immersive Text-to-3D Scene Generation via Adaptive Object-aware Roaming**  
  Jisheng Chu, Wenrui Li, Rui Zhao, Wangmeng Zuo, Shifeng Chen, Xiaopeng Fan  
  _2026-01-27_ · https://arxiv.org/abs/2601.19433v1  
  <details><summary>Abstract</summary>

  Generating immersive 3D scenes from texts is a core task in computer vision, crucial for applications in virtual reality and game development. Despite the promise of leveraging 2D diffusion priors, existing methods suffer from spatial blindness and rely on predefined trajectories that fail to exploit the inner relationships among salient objects. Consequently, these approaches are unable to comprehend the semantic layout, preventing them from exploring the scene adaptively to infer occluded content. Moreover, current inpainting models operate in 2D image space, struggling to plausibly fill holes caused by camera motion. To address these limitations, we propose RoamScene3D, a novel framework that bridges the gap between semantic guidance and spatial generation. Our method reasons about the semantic relations among objects and produces consistent and photorealistic scenes. Specifically, we employ a vision-language model (VLM) to construct a scene graph that encodes object relations, guiding the camera to perceive salient object boundaries and plan an adaptive roaming trajectory. Furthermore, to mitigate the limitations of static 2D priors, we introduce a Motion-Injected Inpainting model that is fine-tuned on a synthetic panoramic dataset integrating authentic camera trajectories, making it adaptive to camera motion. Extensive experiments demonstrate that with semantic reasoning and geometric constraints, our method significantly outperforms state-of-the-art approaches in producing consistent and photorealistic scenes. Our code is available at https://github.com/JS-CHU/RoamScene3D.

  </details>



- **Unveiling Perceptual Artifacts: A Fine-Grained Benchmark for Interpretable AI-Generated Image Detection**  
  Yao Xiao, Weiyan Chen, Jiahao Chen, Zijie Cao, Weijian Deng, Binbin Yang, Ziyi Dong, Xiangyang Ji, Wei Ke, Pengxu Wei, et al.  
  _2026-01-27_ · https://arxiv.org/abs/2601.19430v1  
  <details><summary>Abstract</summary>

  Current AI-Generated Image (AIGI) detection approaches predominantly rely on binary classification to distinguish real from synthetic images, often lacking interpretable or convincing evidence to substantiate their decisions. This limitation stems from existing AIGI detection benchmarks, which, despite featuring a broad collection of synthetic images, remain restricted in their coverage of artifact diversity and lack detailed, localized annotations. To bridge this gap, we introduce a fine-grained benchmark towards eXplainable AI-Generated image Detection, named X-AIGD, which provides pixel-level, categorized annotations of perceptual artifacts, spanning low-level distortions, high-level semantics, and cognitive-level counterfactuals. These comprehensive annotations facilitate fine-grained interpretability evaluation and deeper insight into model decision-making processes. Our extensive investigation using X-AIGD provides several key insights: (1) Existing AIGI detectors demonstrate negligible reliance on perceptual artifacts, even at the most basic distortion level. (2) While AIGI detectors can be trained to identify specific artifacts, they still substantially base their judgment on uninterpretable features. (3) Explicitly aligning model attention with artifact regions can increase the interpretability and generalization of detectors. The data and code are available at: https://github.com/Coxy7/X-AIGD.

  </details>



- **Tri-Reader: An Open-Access, Multi-Stage AI Pipeline for First-Pass Lung Nodule Annotation in Screening CT**  
  Fakrul Islam Tushar, Joseph Y. Lo  
  _2026-01-27_ · https://arxiv.org/abs/2601.19380v1  
  <details><summary>Abstract</summary>

  Using multiple open-access models trained on public datasets, we developed Tri-Reader, a comprehensive, freely available pipeline that integrates lung segmentation, nodule detection, and malignancy classification into a unified tri-stage workflow. The pipeline is designed to prioritize sensitivity while reducing the candidate burden for annotators. To ensure accuracy and generalizability across diverse practices, we evaluated Tri-Reader on multiple internal and external datasets as compared with expert annotations and dataset-provided reference standards.

  </details>



- **Establishing dermatopathology encyclopedia DermpathNet with Artificial Intelligence-Based Workflow**  
  Ziyang Xu, Mingquan Lin, Yiliang Zhou, Zihan Xu, Seth J. Orlow, Zihan Xu, Shane A. Meehan, Alexandra Flamm, Ata S. Moshiri, Yifan Peng  
  _2026-01-27_ · https://arxiv.org/abs/2601.19378v1  
  <details><summary>Abstract</summary>

  Accessing high-quality, open-access dermatopathology image datasets for learning and cross-referencing is a common challenge for clinicians and dermatopathology trainees. To establish a comprehensive open-access dermatopathology dataset for educational, cross-referencing, and machine-learning purposes, we employed a hybrid workflow to curate and categorize images from the PubMed Central (PMC) repository. We used specific keywords to extract relevant images, and classified them using a novel hybrid method that combined deep learning-based image modality classification with figure caption analyses. Validation on 651 manually annotated images demonstrated the robustness of our workflow, with an F-score of 89.6\% for the deep learning approach, 61.0\% for the keyword-based retrieval method, and 90.4\% for the hybrid approach. We retrieved over 7,772 images across 166 diagnoses and released this fully annotated dataset, reviewed by board-certified dermatopathologists. Using our dataset as a challenging task, we found the current image analysis algorithm from OpenAI inadequate for analyzing dermatopathology images. In conclusion, we have developed a large, peer-reviewed, open-access dermatopathology image dataset, DermpathNet, which features a semi-automated curation workflow.

  </details>



- **Perception-to-Pursuit: Track-Centric Temporal Reasoning for Open-World Drone Detection and Autonomous Chasing**  
  Venkatakrishna Reddy Oruganti  
  _2026-01-27_ · https://arxiv.org/abs/2601.19318v1  
  <details><summary>Abstract</summary>

  Autonomous drone pursuit requires not only detecting drones but also predicting their trajectories in a manner that enables kinematically feasible interception. Existing tracking methods optimize for prediction accuracy but ignore pursuit feasibility, resulting in trajectories that are physically impossible to intercept 99.9% of the time. We propose Perception-to-Pursuit (P2P), a track-centric temporal reasoning framework that bridges detection and actionable pursuit planning. Our method represents drone motion as compact 8-dimensional tokens capturing velocity, acceleration, scale, and smoothness, enabling a 12-frame causal transformer to reason about future behavior. We introduce the Intercept Success Rate (ISR) metric to measure pursuit feasibility under realistic interceptor constraints. Evaluated on the Anti-UAV-RGBT dataset with 226 real drone sequences, P2P achieves 28.12 pixel average displacement error and 0.597 ISR, representing a 77% improvement in trajectory prediction and 597x improvement in pursuit feasibility over tracking-only baselines, while maintaining perfect drone classification accuracy (100%). Our work demonstrates that temporal reasoning over motion patterns enables both accurate prediction and actionable pursuit planning.

  </details>



- **Beyond Shadows: A Large-Scale Benchmark and Multi-Stage Framework for High-Fidelity Facial Shadow Removal**  
  Tailong Luo, Jiesong Bai, Jinyang Huang, Junyu Xia, Wangyu Wu, Xuhang Chen  
  _2026-01-27_ · https://arxiv.org/abs/2601.19309v1  
  <details><summary>Abstract</summary>

  Facial shadows often degrade image quality and the performance of vision algorithms. Existing methods struggle to remove shadows while preserving texture, especially under complex lighting conditions, and they lack real-world paired datasets for training. We present the Augmented Shadow Face in the Wild (ASFW) dataset, the first large-scale real-world dataset for facial shadow removal, containing 1,081 paired shadow and shadow-free images created via a professional Photoshop workflow. ASFW offers photorealistic shadow variations and accurate ground truths, bridging the gap between synthetic and real domains. Deep models trained on ASFW demonstrate improved shadow removal in real-world conditions. We also introduce the Face Shadow Eraser (FSE) method to showcase the effectiveness of the dataset. Experiments demonstrate that ASFW enhances the performance of facial shadow removal models, setting new standards for this task.

  </details>



- **ProMist-5K: A Comprehensive Dataset for Digital Emulation of Cinematic Pro-Mist Filter Effects**  
  Yingtie Lei, Zimeng Li, Chi-Man Pun, Wangyu Wu, Junke Yang, Xuhang Chen  
  _2026-01-27_ · https://arxiv.org/abs/2601.19295v1  
  <details><summary>Abstract</summary>

  Pro-Mist filters are widely used in cinematography for their ability to create soft halation, lower contrast, and produce a distinctive, atmospheric style. These effects are difficult to reproduce digitally due to the complex behavior of light diffusion. We present ProMist-5K, a dataset designed to support cinematic style emulation. It is built using a physically inspired pipeline in a scene-referred linear space and includes 20,000 high-resolution image pairs across four configurations, covering two filter densities (1/2 and 1/8) and two focal lengths (20mm and 50mm). Unlike general style datasets, ProMist-5K focuses on realistic glow and highlight diffusion effects. Multiple blur layers and carefully tuned weighting are used to model the varying intensity and spread of optical diffusion. The dataset provides a consistent and controllable target domain that supports various image translation models and learning paradigms. Experiments show that the dataset works well across different training settings and helps capture both subtle and strong cinematic appearances. ProMist-5K offers a practical and physically grounded resource for film-inspired image transformation, bridging the gap between digital flexibility and traditional lens aesthetics. The dataset is available at https://www.kaggle.com/datasets/yingtielei/promist5k.

  </details>



- **A Multi-View Consistency Framework with Semi-Supervised Domain Adaptation**  
  Yuting Hong, Li Dong, Xiaojie Qiu, Hui Xiao, Baochen Yao, Siming Zheng, Chengbin Peng  
  _2026-01-27_ · https://arxiv.org/abs/2601.19266v1  
  <details><summary>Abstract</summary>

  Semi-Supervised Domain Adaptation (SSDA) leverages knowledge from a fully labeled source domain to classify data in a partially labeled target domain. Due to the limited number of labeled samples in the target domain, there can be intrinsic similarity of classes in the feature space, which may result in biased predictions, even when the model is trained on a balanced dataset. To overcome this limitation, we introduce a multi-view consistency framework, which includes two views for training strongly augmented data. One is a debiasing strategy for correcting class-wise prediction probabilities according to the prediction performance of the model. The other involves leveraging pseudo-negative labels derived from the model predictions. Furthermore, we introduce a cross-domain affinity learning aimed at aligning features of the same class across different domains, thereby enhancing overall performance. Experimental results demonstrate that our method outperforms the competing methods on two standard domain adaptation datasets, DomainNet and Office-Home. Combining unsupervised domain adaptation and semi-supervised learning offers indispensable contributions to the industrial sector by enhancing model adaptability, reducing annotation costs, and improving performance.

  </details>



- **Handcrafted Feature Fusion for Reliable Detection of AI-Generated Images**  
  Syed Mehedi Hasan Nirob, Moqsadur Rahman, Shamim Ehsan, Summit Haque  
  _2026-01-27_ · https://arxiv.org/abs/2601.19262v1  
  <details><summary>Abstract</summary>

  The rapid progress of generative models has enabled the creation of highly realistic synthetic images, raising concerns about authenticity and trust in digital media. Detecting such fake content reliably is an urgent challenge. While deep learning approaches dominate current literature, handcrafted features remain attractive for their interpretability, efficiency, and generalizability. In this paper, we conduct a systematic evaluation of handcrafted descriptors, including raw pixels, color histograms, Discrete Cosine Transform (DCT), Histogram of Oriented Gradients (HOG), Local Binary Patterns (LBP), Gray-Level Co-occurrence Matrix (GLCM), and wavelet features, on the CIFAKE dataset of real versus synthetic images. Using 50,000 training and 10,000 test samples, we benchmark seven classifiers ranging from Logistic Regression to advanced gradient-boosted ensembles (LightGBM, XGBoost, CatBoost). Results demonstrate that LightGBM consistently outperforms alternatives, achieving PR-AUC 0.9879, ROC-AUC 0.9878, F1 0.9447, and a Brier score of 0.0414 with mixed features, representing strong gains in calibration and discrimination over simpler descriptors. Across three configurations (baseline, advanced, mixed), performance improves monotonically, confirming that combining diverse handcrafted features yields substantial benefit. These findings highlight the continued relevance of carefully engineered features and ensemble learning for detecting synthetic images, particularly in contexts where interpretability and computational efficiency are critical.

  </details>



- **VC-Bench: Pioneering the Video Connecting Benchmark with a Dataset and Evaluation Metrics**  
  Zhiyu Yin, Zhipeng Liu, Kehai Chen, Lemao Liu, Jin Liu, Hong-Dong Li, Yang Xiang, Min Zhang  
  _2026-01-27_ · https://arxiv.org/abs/2601.19236v1  
  <details><summary>Abstract</summary>

  While current video generation focuses on text or image conditions, practical applications like video editing and vlogging often need to seamlessly connect separate clips. In our work, we introduce Video Connecting, an innovative task that aims to generate smooth intermediate video content between given start and end clips. However, the absence of standardized evaluation benchmarks has hindered the development of this task. To bridge this gap, we proposed VC-Bench, a novel benchmark specifically designed for video connecting. It includes 1,579 high-quality videos collected from public platforms, covering 15 main categories and 72 subcategories to ensure diversity and structure. VC-Bench focuses on three core aspects: Video Quality Score VQS, Start-End Consistency Score SECS, and Transition Smoothness Score TSS. Together, they form a comprehensive framework that moves beyond conventional quality-only metrics. We evaluated multiple state-of-the-art video generation models on VC-Bench. Experimental results reveal significant limitations in maintaining start-end consistency and transition smoothness, leading to lower overall coherence and fluidity. We expect that VC-Bench will serve as a pioneering benchmark to inspire and guide future research in video connecting. The evaluation metrics and dataset are publicly available at: https://anonymous.4open.science/r/VC-Bench-1B67/.

  </details>



- **UniPCB: A Unified Vision-Language Benchmark for Open-Ended PCB Quality Inspection**  
  Fuxiang Sun, Xi Jiang, Jiansheng Wu, Haigang Zhang, Feng Zheng, Jinfeng Yang  
  _2026-01-27_ · https://arxiv.org/abs/2601.19222v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) show promise for general industrial quality inspection, but fall short in complex scenarios, such as Printed Circuit Board (PCB) inspection. PCB inspection poses unique challenges due to densely packed components, complex wiring structures, and subtle defect patterns that require specialized domain expertise. However, a high-quality, unified vision-language benchmark for quantitatively evaluating MLLMs across PCB inspection tasks remains absent, stemming not only from limited data availability but also from fragmented datasets and inconsistent standardization. To fill this gap, we propose UniPCB, the first unified vision-language benchmark for open-ended PCB quality inspection. UniPCB is built via a systematic pipeline that curates and standardizes data from disparate sources across three annotated scenarios. Furthermore, we introduce PCB-GPT, an MLLM trained on a new instruction dataset generated by this pipeline, utilizing a novel progressive curriculum that mimics the learning process of human experts. Evaluations on the UniPCB benchmark show that while existing MLLMs falter on domain-specific tasks, PCB-GPT establishes a new baseline. Notably, it more than doubles the performance on fine-grained defect localization compared to the strongest competitors, with significant advantages in localization and analysis. We will release the instruction data, benchmark, and model to facilitate future research.

  </details>



- **MATA: A Trainable Hierarchical Automaton System for Multi-Agent Visual Reasoning**  
  Zhixi Cai, Fucai Ke, Kevin Leo, Sukai Huang, Maria Garcia de la Banda, Peter J. Stuckey, Hamid Rezatofighi  
  _2026-01-27_ · https://arxiv.org/abs/2601.19204v1  
  <details><summary>Abstract</summary>

  Recent vision-language models have strong perceptual ability but their implicit reasoning is hard to explain and easily generates hallucinations on complex queries. Compositional methods improve interpretability, but most rely on a single agent or hand-crafted pipeline and cannot decide when to collaborate across complementary agents or compete among overlapping ones. We introduce MATA (Multi-Agent hierarchical Trainable Automaton), a multi-agent system presented as a hierarchical finite-state automaton for visual reasoning whose top-level transitions are chosen by a trainable hyper agent. Each agent corresponds to a state in the hyper automaton, and runs a small rule-based sub-automaton for reliable micro-control. All agents read and write a shared memory, yielding transparent execution history. To supervise the hyper agent's transition policy, we build transition-trajectory trees and transform to memory-to-next-state pairs, forming the MATA-SFT-90K dataset for supervised finetuning (SFT). The finetuned LLM as the transition policy understands the query and the capacity of agents, and it can efficiently choose the optimal agent to solve the task. Across multiple visual reasoning benchmarks, MATA achieves the state-of-the-art results compared with monolithic and compositional baselines. The code and dataset are available at https://github.com/ControlNet/MATA.

  </details>



- **LocationAgent: A Hierarchical Agent for Image Geolocation via Decoupling Strategy and Evidence from Parametric Knowledge**  
  Qiujun Li, Zijin Xiao, Xulin Wang, Zhidan Ma, Cheng Yang, Haifeng Li  
  _2026-01-27_ · https://arxiv.org/abs/2601.19155v1  
  <details><summary>Abstract</summary>

  Image geolocation aims to infer capture locations based on visual content. Fundamentally, this constitutes a reasoning process composed of \textit{hypothesis-verification cycles}, requiring models to possess both geospatial reasoning capabilities and the ability to verify evidence against geographic facts. Existing methods typically internalize location knowledge and reasoning patterns into static memory via supervised training or trajectory-based reinforcement fine-tuning. Consequently, these methods are prone to factual hallucinations and generalization bottlenecks in open-world settings or scenarios requiring dynamic knowledge. To address these challenges, we propose a Hierarchical Localization Agent, called LocationAgent. Our core philosophy is to retain hierarchical reasoning logic within the model while offloading the verification of geographic evidence to external tools. To implement hierarchical reasoning, we design the RER architecture (Reasoner-Executor-Recorder), which employs role separation and context compression to prevent the drifting problem in multi-step reasoning. For evidence verification, we construct a suite of clue exploration tools that provide diverse evidence to support location reasoning. Furthermore, to address data leakage and the scarcity of Chinese data in existing datasets, we introduce CCL-Bench (China City Location Bench), an image geolocation benchmark encompassing various scene granularities and difficulty levels. Extensive experiments demonstrate that LocationAgent significantly outperforms existing methods by at least 30\% in zero-shot settings.

  </details>



- **TFFM: Topology-Aware Feature Fusion Module via Latent Graph Reasoning for Retinal Vessel Segmentation**  
  Iftekhar Ahmed, Shakib Absar, Aftar Ahmad Sami, Shadman Sakib, Debojyoti Biswas, Seraj Al Mahmud Mostafa  
  _2026-01-27_ · https://arxiv.org/abs/2601.19136v1  
  <details><summary>Abstract</summary>

  Precise segmentation of retinal arteries and veins carries the diagnosis of systemic cardiovascular conditions. However, standard convolutional architectures often yield topologically disjointed segmentations, characterized by gaps and discontinuities that render reliable graph-based clinical analysis impossible despite high pixel-level accuracy. To address this, we introduce a topology-aware framework engineered to maintain vascular connectivity. Our architecture fuses a Topological Feature Fusion Module (TFFM) that maps local feature representations into a latent graph space, deploying Graph Attention Networks to capture global structural dependencies often missed by fixed receptive fields. Furthermore, we drive the learning process with a hybrid objective function, coupling Tversky loss for class imbalance with soft clDice loss to explicitly penalize topological disconnects. Evaluation on the Fundus-AVSeg dataset reveals state-of-the-art performance, achieving a combined Dice score of 90.97% and a 95% Hausdorff Distance of 3.50 pixels. Notably, our method decreases vessel fragmentation by approximately 38% relative to baselines, yielding topologically coherent vascular trees viable for automated biomarker quantification. We open-source our code at https://tffm-module.github.io/.

  </details>



- **Resolving Primitive-Sharing Ambiguity in Long-Tailed Industrial Point Cloud Segmentation via Spatial Context Constraints**  
  Chao Yin, Qing Han, Zhiwei Hou, Yue Liu, Anjin Dai, Hongda Hu, Ji Yang, Wei Yao  
  _2026-01-27_ · https://arxiv.org/abs/2601.19128v1  
  <details><summary>Abstract</summary>

  Industrial point cloud segmentation for Digital Twin construction faces a persistent challenge: safety-critical components such as reducers and valves are systematically misclassified. These failures stem from two compounding factors: such components are rare in training data, yet they share identical local geometry with dominant structures like pipes. This work identifies a dual crisis unique to industrial 3D data extreme class imbalance 215:1 ratio compounded by geometric ambiguity where most tail classes share cylindrical primitives with head classes. Existing frequency-based re-weighting methods address statistical imbalance but cannot resolve geometric ambiguity. We propose spatial context constraints that leverage neighborhood prediction consistency to disambiguate locally similar structures. Our approach extends the Class-Balanced (CB) Loss framework with two architecture-agnostic mechanisms: (1) Boundary-CB, an entropy-based constraint that emphasizes ambiguous boundaries, and (2) Density-CB, a density-based constraint that compensates for scan-dependent variations. Both integrate as plug-and-play modules without network modifications, requiring only loss function replacement. On the Industrial3D dataset (610M points from water treatment facilities), our method achieves 55.74% mIoU with 21.7% relative improvement on tail-class performance (29.59% vs. 24.32% baseline) while preserving head-class accuracy (88.14%). Components with primitive-sharing ambiguity show dramatic gains: reducer improves from 0% to 21.12% IoU; valve improves by 24.3% relative. This resolves geometric ambiguity without the typical head-tail trade-off, enabling reliable identification of safety-critical components for automated knowledge extraction in Digital Twin applications.

  </details>



- **Implicit Non-Causal Factors are Out via Dataset Splitting for Domain Generalization Object Detection**  
  Zhilong Zhang, Lei Zhang, Qing He, Shuyin Xia, Guoyin Wang, Fuxiang Huang  
  _2026-01-27_ · https://arxiv.org/abs/2601.19127v1  
  <details><summary>Abstract</summary>

  Open world object detection faces a significant challenge in domain-invariant representation, i.e., implicit non-causal factors. Most domain generalization (DG) methods based on domain adversarial learning (DAL) pay much attention to learn domain-invariant information, but often overlook the potential non-causal factors. We unveil two critical causes: 1) The domain discriminator-based DAL method is subject to the extremely sparse domain label, i.e., assigning only one domain label to each dataset, thus can only associate explicit non-causal factor, which is incredibly limited. 2) The non-causal factors, induced by unidentified data bias, are excessively implicit and cannot be solely discerned by conventional DAL paradigm. Based on these key findings, inspired by the Granular-Ball perspective, we propose an improved DAL method, i.e., GB-DAL. The proposed GB-DAL utilizes Prototype-based Granular Ball Splitting (PGBS) module to generate more dense domains from limited datasets, akin to more fine-grained granular balls, indicating more potential non-causal factors. Inspired by adversarial perturbations akin to non-causal factors, we propose a Simulated Non-causal Factors (SNF) module as a means of data augmentation to reduce the implicitness of non-causal factors, and facilitate the training of GB-DAL. Comparative experiments on numerous benchmarks demonstrate that our method achieves better generalization performance in novel circumstances.

  </details>



- **m2sv: A Scalable Benchmark for Map-to-Street-View Spatial Reasoning**  
  Yosub Shin, Michael Buriek, Igor Molybog  
  _2026-01-27_ · https://arxiv.org/abs/2601.19099v1  
  <details><summary>Abstract</summary>

  Vision--language models (VLMs) achieve strong performance on many multimodal benchmarks but remain brittle on spatial reasoning tasks that require aligning abstract overhead representations with egocentric views. We introduce m2sv, a scalable benchmark for map-to-street-view spatial reasoning that asks models to infer camera viewing direction by aligning a north-up overhead map with a Street View image captured at the same real-world intersection. We release m2sv-20k, a geographically diverse benchmark with controlled ambiguity, along with m2sv-sft-11k, a curated set of structured reasoning traces for supervised fine-tuning. Despite strong performance on existing multimodal benchmarks, the best evaluated VLM achieves only 65.2% accuracy on m2sv, far below the human baseline of 95%. While supervised fine-tuning and reinforcement learning yield consistent gains, cross-benchmark evaluations reveal limited transfer. Beyond aggregate accuracy, we systematically analyze difficulty in map-to-street-view reasoning using both structural signals and human effort, and conduct an extensive failure analysis of adapted open models. Our findings highlight persistent gaps in geometric alignment, evidence aggregation, and reasoning consistency, motivating future work on grounded spatial reasoning across viewpoints.

  </details>



- **On the Role of Depth in Surgical Vision Foundation Models: An Empirical Study of RGB-D Pre-training**  
  John J. Han, Adam Schmidt, Muhammad Abdullah Jamal, Chinedu Nwoye, Anita Rau, Jie Ying Wu, Omid Mohareri  
  _2026-01-26_ · https://arxiv.org/abs/2601.18929v1  
  <details><summary>Abstract</summary>

  Vision foundation models (VFMs) have emerged as powerful tools for surgical scene understanding. However, current approaches predominantly rely on unimodal RGB pre-training, overlooking the complex 3D geometry inherent to surgical environments. Although several architectures support multimodal or geometry-aware inputs in general computer vision, the benefits of incorporating depth information in surgical settings remain underexplored. We conduct a large-scale empirical study comparing eight ViT-based VFMs that differ in pre-training domain, learning objective, and input modality (RGB vs. RGB-D). For pre-training, we use a curated dataset of 1.4 million robotic surgical images paired with depth maps generated from an off-the-shelf network. We evaluate these models under both frozen-backbone and end-to-end fine-tuning protocols across eight surgical datasets spanning object detection, segmentation, depth estimation, and pose estimation. Our experiments yield several consistent findings. Models incorporating explicit geometric tokenization, such as MultiMAE, substantially outperform unimodal baselines across all tasks. Notably, geometric-aware pre-training enables remarkable data efficiency: models fine-tuned on just 25% of labeled data consistently surpass RGB-only models trained on the full dataset. Importantly, these gains require no architectural or runtime changes at inference; depth is used only during pre-training, making adoption straightforward. These findings suggest that multimodal pre-training offers a viable path towards building more capable surgical vision systems.

  </details>



- **DeFM: Learning Foundation Representations from Depth for Robotics**  
  Manthan Patel, Jonas Frey, Mayank Mittal, Fan Yang, Alexander Hansson, Amir Bar, Cesar Cadena, Marco Hutter  
  _2026-01-26_ · https://arxiv.org/abs/2601.18923v1  
  <details><summary>Abstract</summary>

  Depth sensors are widely deployed across robotic platforms, and advances in fast, high-fidelity depth simulation have enabled robotic policies trained on depth observations to achieve robust sim-to-real transfer for a wide range of tasks. Despite this, representation learning for depth modality remains underexplored compared to RGB, where large-scale foundation models now define the state of the art. To address this gap, we present DeFM, a self-supervised foundation model trained entirely on depth images for robotic applications. Using a DINO-style self-distillation objective on a curated dataset of 60M depth images, DeFM learns geometric and semantic representations that generalize to diverse environments, tasks, and sensors. To retain metric awareness across multiple scales, we introduce a novel input normalization strategy. We further distill DeFM into compact models suitable for resource-constrained robotic systems. When evaluated on depth-based classification, segmentation, navigation, locomotion, and manipulation benchmarks, DeFM achieves state-of-the-art performance and demonstrates strong generalization from simulation to real-world environments. We release all our pretrained models, which can be adopted off-the-shelf for depth-based robotic learning without task-specific fine-tuning. Webpage: https://de-fm.github.io/

  </details>



- **Weakly supervised framework for wildlife detection and counting in challenging Arctic environments: a case study on caribou (Rangifer tarandus)**  
  Ghazaleh Serati, Samuel Foucher, Jerome Theau  
  _2026-01-26_ · https://arxiv.org/abs/2601.18891v1  
  <details><summary>Abstract</summary>

  Caribou across the Arctic has declined in recent decades, motivating scalable and accurate monitoring approaches to guide evidence-based conservation actions and policy decisions. Manual interpretation from this imagery is labor-intensive and error-prone, underscoring the need for automatic and reliable detection across varying scenes. Yet, such automatic detection is challenging due to severe background heterogeneity, dominant empty terrain (class imbalance), small or occluded targets, and wide variation in density and scale. To make the detection model (HerdNet) more robust to these challenges, a weakly supervised patch-level pretraining based on a detection network's architecture is proposed. The detection dataset includes five caribou herds distributed across Alaska. By learning from empty vs. non-empty labels in this dataset, the approach produces early weakly supervised knowledge for enhanced detection compared to HerdNet, which is initialized from generic weights. Accordingly, the patch-based pretrain network attained high accuracy on multi-herd imagery (2017) and on an independent year's (2019) test sets (F1: 93.7%/92.6%, respectively), enabling reliable mapping of regions containing animals to facilitate manual counting on large aerial imagery. Transferred to detection, initialization from weakly supervised pretraining yielded consistent gains over ImageNet weights on both positive patches (F1: 92.6%/93.5% vs. 89.3%/88.6%), and full-image counting (F1: 95.5%/93.3% vs. 91.5%/90.4%). Remaining limitations are false positives from animal-like background clutter and false negatives related to low animal density occlusions. Overall, pretraining on coarse labels prior to detection makes it possible to rely on weakly-supervised pretrained weights even when labeled data are limited, achieving results comparable to generic-weight initialization.

  </details>



- **Trustworthy Evaluation of Robotic Manipulation: A New Benchmark and AutoEval Methods**  
  Mengyuan Liu, Juyi Sheng, Peiming Li, Ziyi Wang, Tianming Xu, Tiantian Xu, Hong Liu  
  _2026-01-26_ · https://arxiv.org/abs/2601.18723v1  
  <details><summary>Abstract</summary>

  Driven by the rapid evolution of Vision-Action and Vision-Language-Action models, imitation learning has significantly advanced robotic manipulation capabilities. However, evaluation methodologies have lagged behind, hindering the establishment of Trustworthy Evaluation for these behaviors. Current paradigms rely on binary success rates, failing to address the critical dimensions of trust: Source Authenticity (i.e., distinguishing genuine policy behaviors from human teleoperation) and Execution Quality (e.g., smoothness and safety). To bridge these gaps, we propose a solution that combines the Eval-Actions benchmark and the AutoEval architecture. First, we construct the Eval-Actions benchmark to support trustworthiness analysis. Distinct from existing datasets restricted to successful human demonstrations, Eval-Actions integrates VA and VLA policy execution trajectories alongside human teleoperation data, explicitly including failure scenarios. This dataset is structured around three core supervision signals: Expert Grading (EG), Rank-Guided preferences (RG), and Chain-of-Thought (CoT). Building on this, we propose the AutoEval architecture: AutoEval leverages Spatio-Temporal Aggregation for semantic assessment, augmented by an auxiliary Kinematic Calibration Signal to refine motion smoothness; AutoEval Plus (AutoEval-P) incorporates the Group Relative Policy Optimization (GRPO) paradigm to enhance logical reasoning capabilities. Experiments show AutoEval achieves Spearman's Rank Correlation Coefficients (SRCC) of 0.81 and 0.84 under the EG and RG protocols, respectively. Crucially, the framework possesses robust source discrimination capabilities, distinguishing between policy-generated and teleoperated videos with 99.6% accuracy, thereby establishing a rigorous standard for trustworthy robotic evaluation. Our project and code are available at https://term-bench.github.io/.

  </details>



- **Are Video Generation Models Geographically Fair? An Attraction-Centric Evaluation of Global Visual Knowledge**  
  Xiao Liu, Jiawei Zhang  
  _2026-01-26_ · https://arxiv.org/abs/2601.18698v1  
  <details><summary>Abstract</summary>

  Recent advances in text-to-video generation have produced visually compelling results, yet it remains unclear whether these models encode geographically equitable visual knowledge. In this work, we investigate the geo-equity and geographically grounded visual knowledge of text-to-video models through an attraction-centric evaluation. We introduce Geo-Attraction Landmark Probing (GAP), a systematic framework for assessing how faithfully models synthesize tourist attractions from diverse regions, and construct GEOATTRACTION-500, a benchmark of 500 globally distributed attractions spanning varied regions and popularity levels. GAP integrates complementary metrics that disentangle overall video quality from attraction-specific knowledge, including global structural alignment, fine-grained keypoint-based alignment, and vision-language model judgments, all validated against human evaluation. Applying GAP to the state-of-the-art text-to-video model Sora 2, we find that, contrary to common assumptions of strong geographic bias, the model exhibits a relatively uniform level of geographically grounded visual knowledge across regions, development levels, and cultural groupings, with only weak dependence on attraction popularity. These results suggest that current text-to-video models express global visual knowledge more evenly than expected, highlighting both their promise for globally deployed applications and the need for continued evaluation as such systems evolve.

  </details>



- **A Pragmatic VLA Foundation Model**  
  Wei Wu, Fan Lu, Yunnan Wang, Shuai Yang, Shi Liu, Fangjing Wang, Qian Zhu, He Sun, Yong Wang, Shuailei Ma, et al.  
  _2026-01-26_ · https://arxiv.org/abs/2601.18692v1  
  <details><summary>Abstract</summary>

  Offering great potential in robotic manipulation, a capable Vision-Language-Action (VLA) foundation model is expected to faithfully generalize across tasks and platforms while ensuring cost efficiency (e.g., data and GPU hours required for adaptation). To this end, we develop LingBot-VLA with around 20,000 hours of real-world data from 9 popular dual-arm robot configurations. Through a systematic assessment on 3 robotic platforms, each completing 100 tasks with 130 post-training episodes per task, our model achieves clear superiority over competitors, showcasing its strong performance and broad generalizability. We have also built an efficient codebase, which delivers a throughput of 261 samples per second per GPU with an 8-GPU training setup, representing a 1.5~2.8$\times$ (depending on the relied VLM base model) speedup over existing VLA-oriented codebases. The above features ensure that our model is well-suited for real-world deployment. To advance the field of robot learning, we provide open access to the code, base model, and benchmark data, with a focus on enabling more challenging tasks and promoting sound evaluation standards.

  </details>



- **Scale-Aware Self-Supervised Learning for Segmentation of Small and Sparse Structures**  
  Jorge Quesada, Ghassan AlRegib  
  _2026-01-26_ · https://arxiv.org/abs/2601.18619v1  
  <details><summary>Abstract</summary>

  Self-supervised learning (SSL) has emerged as a powerful strategy for representation learning under limited annotation regimes, yet its effectiveness remains highly sensitive to many factors, especially the nature of the target task. In segmentation, existing pipelines are typically tuned to large, homogeneous regions, but their performance drops when objects are small, sparse, or locally irregular. In this work, we propose a scale-aware SSL adaptation that integrates small-window cropping into the augmentation pipeline, zooming in on fine-scale structures during pretraining. We evaluate this approach across two domains with markedly different data modalities: seismic imaging, where the goal is to segment sparse faults, and neuroimaging, where the task is to delineate small cellular structures. In both settings, our method yields consistent improvements over standard and state-of-the-art baselines under label constraints, improving accuracy by up to 13% for fault segmentation and 5% for cell delineation. In contrast, large-scale features such as seismic facies or tissue regions see little benefit, underscoring that the value of SSL depends critically on the scale of the target objects. Our findings highlight the need to align SSL design with object size and sparsity, offering a general principle for buil ding more effective representation learning pipelines across scientific imaging domains.

  </details>



- **AGSP-DSA: An Adaptive Graph Signal Processing Framework for Robust Multimodal Fusion with Dynamic Semantic Alignment**  
  KV Karthikeya, Ashok Kumar Das, Shantanu Pal, Vivekananda Bhat K, Arun Sekar Rajasekaran  
  _2026-01-26_ · https://arxiv.org/abs/2601.18589v1  
  <details><summary>Abstract</summary>

  In this paper, we introduce an Adaptive Graph Signal Processing with Dynamic Semantic Alignment (AGSP DSA) framework to perform robust multimodal data fusion over heterogeneous sources, including text, audio, and images. The requested approach uses a dual-graph construction to learn both intra-modal and inter-modal relations, spectral graph filtering to boost the informative signals, and effective node embedding with Multi-scale Graph Convolutional Networks (GCNs). Semantic aware attention mechanism: each modality may dynamically contribute to the context with respect to contextual relevance. The experimental outcomes on three benchmark datasets, including CMU-MOSEI, AVE, and MM-IMDB, show that AGSP-DSA performs as the state of the art. More precisely, it achieves 95.3% accuracy, 0.936 F1-score, and 0.924 mAP on CMU-MOSEI, improving MM-GNN by 2.6 percent in accuracy. It gets 93.4% accuracy and 0.911 F1-score on AVE and 91.8% accuracy and 0.886 F1-score on MM-IMDB, which demonstrate good generalization and robustness in the missing modality setting. These findings verify the efficiency of AGSP-DSA in promoting multimodal learning in sentiment analysis, event recognition and multimedia classification.

  </details>



- **Self-Refining Video Sampling**  
  Sangwon Jang, Taekyung Ki, Jaehyeong Jo, Saining Xie, Jaehong Yoon, Sung Ju Hwang  
  _2026-01-26_ · https://arxiv.org/abs/2601.18577v1  
  <details><summary>Abstract</summary>

  Modern video generators still struggle with complex physical dynamics, often falling short of physical realism. Existing approaches address this using external verifiers or additional training on augmented data, which is computationally expensive and still limited in capturing fine-grained motion. In this work, we present self-refining video sampling, a simple method that uses a pre-trained video generator trained on large-scale datasets as its own self-refiner. By interpreting the generator as a denoising autoencoder, we enable iterative inner-loop refinement at inference time without any external verifier or additional training. We further introduce an uncertainty-aware refinement strategy that selectively refines regions based on self-consistency, which prevents artifacts caused by over-refinement. Experiments on state-of-the-art video generators demonstrate significant improvements in motion coherence and physics alignment, achieving over 70\% human preference compared to the default sampler and guidance-based sampler.

  </details>



- **From Cold Start to Active Learning: Embedding-Based Scan Selection for Medical Image Segmentation**  
  Devon Levy, Bar Assayag, Laura Gaspar, Ilan Shimshoni, Bella Specktor-Fadida  
  _2026-01-26_ · https://arxiv.org/abs/2601.18532v1  
  <details><summary>Abstract</summary>

  Accurate segmentation annotations are critical for disease monitoring, yet manual labeling remains a major bottleneck due to the time and expertise required. Active learning (AL) alleviates this burden by prioritizing informative samples for annotation, typically through a diversity-based cold-start phase followed by uncertainty-driven selection. We propose a novel cold-start sampling strategy that combines foundation-model embeddings with clustering, including automatic selection of the number of clusters and proportional sampling across clusters, to construct a diverse and representative initial training. This is followed by an uncertainty-based AL framework that integrates spatial diversity to guide sample selection. The proposed method is intuitive and interpretable, enabling visualization of the feature-space distribution of candidate samples. We evaluate our approach on three datasets spanning X-ray and MRI modalities. On the CheXmask dataset, the cold-start strategy outperforms random selection, improving Dice from 0.918 to 0.929 and reducing the Hausdorff distance from 32.41 to 27.66 mm. In the AL setting, combined entropy and diversity selection improves Dice from 0.919 to 0.939 and reduces the Hausdorff distance from 30.10 to 19.16 mm. On the Montgomery dataset, cold-start gains are substantial, with Dice improving from 0.928 to 0.950 and Hausdorff distance decreasing from 14.22 to 9.38 mm. On the SynthStrip dataset, cold-start selection slightly affects Dice but reduces the Hausdorff distance from 9.43 to 8.69 mm, while active learning improves Dice from 0.816 to 0.826 and reduces the Hausdorff distance from 7.76 to 6.38 mm. Overall, the proposed framework consistently outperforms baseline methods in low-data regimes, improving segmentation accuracy.

  </details>



- **DisasterInsight: A Multimodal Benchmark for Function-Aware and Grounded Disaster Assessment**  
  Sara Tehrani, Yonghao Xu, Leif Haglund, Amanda Berg, Michael Felsberg  
  _2026-01-26_ · https://arxiv.org/abs/2601.18493v1  
  <details><summary>Abstract</summary>

  Timely interpretation of satellite imagery is critical for disaster response, yet existing vision-language benchmarks for remote sensing largely focus on coarse labels and image-level recognition, overlooking the functional understanding and instruction robustness required in real humanitarian workflows. We introduce DisasterInsight, a multimodal benchmark designed to evaluate vision-language models (VLMs) on realistic disaster analysis tasks. DisasterInsight restructures the xBD dataset into approximately 112K building-centered instances and supports instruction-diverse evaluation across multiple tasks, including building-function classification, damage-level and disaster-type classification, counting, and structured report generation aligned with humanitarian assessment guidelines. To establish domain-adapted baselines, we propose DI-Chat, obtained by fine-tuning existing VLM backbones on disaster-specific instruction data using parameter-efficient Low-Rank Adaptation (LoRA). Extensive experiments on state-of-the-art generic and remote-sensing VLMs reveal substantial performance gaps across tasks, particularly in damage understanding and structured report generation. DI-Chat achieves significant improvements on damage-level and disaster-type classification as well as report generation quality, while building-function classification remains challenging for all evaluated models. DisasterInsight provides a unified benchmark for studying grounded multimodal reasoning in disaster imagery.

  </details>



- **AgentDoG: A Diagnostic Guardrail Framework for AI Agent Safety and Security**  
  Dongrui Liu, Qihan Ren, Chen Qian, Shuai Shao, Yuejin Xie, Yu Li, Zhonghao Yang, Haoyu Luo, Peng Wang, Qingyu Liu, et al.  
  _2026-01-26_ · https://arxiv.org/abs/2601.18491v1  
  <details><summary>Abstract</summary>

  The rise of AI agents introduces complex safety and security challenges arising from autonomous tool use and environmental interactions. Current guardrail models lack agentic risk awareness and transparency in risk diagnosis. To introduce an agentic guardrail that covers complex and numerous risky behaviors, we first propose a unified three-dimensional taxonomy that orthogonally categorizes agentic risks by their source (where), failure mode (how), and consequence (what). Guided by this structured and hierarchical taxonomy, we introduce a new fine-grained agentic safety benchmark (ATBench) and a Diagnostic Guardrail framework for agent safety and security (AgentDoG). AgentDoG provides fine-grained and contextual monitoring across agent trajectories. More Crucially, AgentDoG can diagnose the root causes of unsafe actions and seemingly safe but unreasonable actions, offering provenance and transparency beyond binary labels to facilitate effective agent alignment. AgentDoG variants are available in three sizes (4B, 7B, and 8B parameters) across Qwen and Llama model families. Extensive experimental results demonstrate that AgentDoG achieves state-of-the-art performance in agentic safety moderation in diverse and complex interactive scenarios. All models and datasets are openly released.

  </details>



- **3DGesPolicy: Phoneme-Aware Holistic Co-Speech Gesture Generation Based on Action Control**  
  Xuanmeng Sha, Liyun Zhang, Tomohiro Mashita, Naoya Chiba, Yuki Uranishi  
  _2026-01-26_ · https://arxiv.org/abs/2601.18451v1  
  <details><summary>Abstract</summary>

  Generating holistic co-speech gestures that integrate full-body motion with facial expressions suffers from semantically incoherent coordination on body motion and spatially unstable meaningless movements due to existing part-decomposed or frame-level regression methods, We introduce 3DGesPolicy, a novel action-based framework that reformulates holistic gesture generation as a continuous trajectory control problem through diffusion policy from robotics. By modeling frame-to-frame variations as unified holistic actions, our method effectively learns inter-frame holistic gesture motion patterns and ensures both spatially and semantically coherent movement trajectories that adhere to realistic motion manifolds. To further bridge the gap in expressive alignment, we propose a Gesture-Audio-Phoneme (GAP) fusion module that can deeply integrate and refine multi-modal signals, ensuring structured and fine-grained alignment between speech semantics, body motion, and facial expressions. Extensive quantitative and qualitative experiments on the BEAT2 dataset demonstrate the effectiveness of our 3DGesPolicy across other state-of-the-art methods in generating natural, expressive, and highly speech-aligned holistic gestures.

  </details>



- **Dynamic Mask-Based Backdoor Attack Against Vision AI Models: A Case Study on Mushroom Detection**  
  Zeineb Dridi, Jihen Bennaceur, Amine Ben Hassouna  
  _2026-01-26_ · https://arxiv.org/abs/2601.18845v1  
  <details><summary>Abstract</summary>

  Deep learning has revolutionized numerous tasks within the computer vision field, including image classification, image segmentation, and object detection. However, the increasing deployment of deep learning models has exposed them to various adversarial attacks, including backdoor attacks. This paper presents a novel dynamic mask-based backdoor attack method, specifically designed for object detection models. We exploit a dataset poisoning technique to embed a malicious trigger, rendering any models trained on this compromised dataset vulnerable to our backdoor attack. We particularly focus on a mushroom detection dataset to demonstrate the practical risks posed by such attacks on critical real-life domains. Our work also emphasizes the importance of creating a detailed backdoor attack scenario to illustrate the significant risks associated with the outsourcing practice. Our approach leverages SAM, a recent and powerful image segmentation AI model, to create masks for dynamic trigger placement, introducing a new and stealthy attack method. Through extensive experimentation, we show that our sophisticated attack scenario maintains high accuracy on clean data with the YOLOv7 object detection model while achieving high attack success rates on poisoned samples. Our approach surpasses traditional methods for backdoor injection, which are based on static and consistent patterns. Our findings underscore the urgent need for robust countermeasures to protect deep learning models from these evolving adversarial threats.

  </details>


