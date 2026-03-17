# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-03-17 07:17 UTC_

Total papers shown: **50**


---

- **Towards Generalizable Robotic Manipulation in Dynamic Environments**  
  Heng Fang, Shangru Li, Shuhan Wang, Xuanyang Xi, Dingkang Liang, Xiang Bai  
  _2026-03-16_ · https://arxiv.org/abs/2603.15620v1  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models excel in static manipulation but struggle in dynamic environments with moving targets. This performance gap primarily stems from a scarcity of dynamic manipulation datasets and the reliance of mainstream VLAs on single-frame observations, restricting their spatiotemporal reasoning capabilities. To address this, we introduce DOMINO, a large-scale dataset and benchmark for generalizable dynamic manipulation, featuring 35 tasks with hierarchical complexities, over 110K expert trajectories, and a multi-dimensional evaluation suite. Through comprehensive experiments, we systematically evaluate existing VLAs on dynamic tasks, explore effective training strategies for dynamic awareness, and validate the generalizability of dynamic data. Furthermore, we propose PUMA, a dynamics-aware VLA architecture. By integrating scene-centric historical optical flow and specialized world queries to implicitly forecast object-centric future states, PUMA couples history-aware perception with short-horizon prediction. Results demonstrate that PUMA achieves state-of-the-art performance, yielding a 6.3% absolute improvement in success rate over baselines. Moreover, we show that training on dynamic data fosters robust spatiotemporal representations that transfer to static tasks. All code and data are available at https://github.com/H-EmbodVis/DOMINO.

  </details>



- **GlyphPrinter: Region-Grouped Direct Preference Optimization for Glyph-Accurate Visual Text Rendering**  
  Xincheng Shuai, Ziye Li, Henghui Ding, Dacheng Tao  
  _2026-03-16_ · https://arxiv.org/abs/2603.15616v1  
  <details><summary>Abstract</summary>

  Generating accurate glyphs for visual text rendering is essential yet challenging. Existing methods typically enhance text rendering by training on a large amount of high-quality scene text images, but the limited coverage of glyph variations and excessive stylization often compromise glyph accuracy, especially for complex or out-of-domain characters. Some methods leverage reinforcement learning to alleviate this issue, yet their reward models usually depend on text recognition systems that are insensitive to fine-grained glyph errors, so images with incorrect glyphs may still receive high rewards. Inspired by Direct Preference Optimization (DPO), we propose GlyphPrinter, a preference-based text rendering method that eliminates reliance on explicit reward models. However, the standard DPO objective only models overall preference between two samples, which is insufficient for visual text rendering where glyph errors typically occur in localized regions. To address this issue, we construct the GlyphCorrector dataset with region-level glyph preference annotations and propose Region-Grouped DPO (R-GDPO), a region-based objective that optimizes inter- and intra-sample preferences over annotated regions, substantially enhancing glyph accuracy. Furthermore, we introduce Regional Reward Guidance, an inference strategy that samples from an optimal distribution with controllable glyph accuracy. Extensive experiments demonstrate that the proposed GlyphPrinter outperforms existing methods in glyph accuracy while maintaining a favorable balance between stylization and precision.

  </details>



- **HSImul3R: Physics-in-the-Loop Reconstruction of Simulation-Ready Human-Scene Interactions**  
  Yukang Cao, Haozhe Xie, Fangzhou Hong, Long Zhuo, Zhaoxi Chen, Liang Pan, Ziwei Liu  
  _2026-03-16_ · https://arxiv.org/abs/2603.15612v1  
  <details><summary>Abstract</summary>

  We present HSImul3R, a unified framework for simulation-ready 3D reconstruction of human-scene interactions (HSI) from casual captures, including sparse-view images and monocular videos. Existing methods suffer from a perception-simulation gap: visually plausible reconstructions often violate physical constraints, leading to instability in physics engines and failure in embodied AI applications. To bridge this gap, we introduce a physically-grounded bi-directional optimization pipeline that treats the physics simulator as an active supervisor to jointly refine human dynamics and scene geometry. In the forward direction, we employ Scene-targeted Reinforcement Learning to optimize human motion under dual supervision of motion fidelity and contact stability. In the reverse direction, we propose Direct Simulation Reward Optimization, which leverages simulation feedback on gravitational stability and interaction success to refine scene geometry. We further present HSIBench, a new benchmark with diverse objects and interaction scenarios. Extensive experiments demonstrate that HSImul3R produces the first stable, simulation-ready HSI reconstructions and can be directly deployed to real-world humanoid robots.

  </details>



- **From Passive Observer to Active Critic: Reinforcement Learning Elicits Process Reasoning for Robotic Manipulation**  
  Yibin Liu, Yaxing Lyu, Daqi Gao, Zhixuan Liang, Weiliang Tang, Shilong Mu, Xiaokang Yang, Yao Mu  
  _2026-03-16_ · https://arxiv.org/abs/2603.15600v1  
  <details><summary>Abstract</summary>

  Accurate process supervision remains a critical challenge for long-horizon robotic manipulation. A primary bottleneck is that current video MLLMs, trained primarily under a Supervised Fine-Tuning (SFT) paradigm, function as passive "Observers" that recognize ongoing events rather than evaluating the current state relative to the final task goal. In this paper, we introduce PRIMO R1 (Process Reasoning Induced Monitoring), a 7B framework that transforms video MLLMs into active "Critics". We leverage outcome-based Reinforcement Learning to incentivize explicit Chain-of-Thought generation for progress estimation. Furthermore, our architecture constructs a structured temporal input by explicitly anchoring the video sequence between initial and current state images. Supported by the proposed PRIMO Dataset and Benchmark, extensive experiments across diverse in-domain environments and out-of-domain real-world humanoid scenarios demonstrate that PRIMO R1 achieves state-of-the-art performance. Quantitatively, our 7B model achieves a 50% reduction in the mean absolute error of specialized reasoning baselines, demonstrating significant relative accuracy improvements over 72B-scale general MLLMs. Furthermore, PRIMO R1 exhibits strong zero-shot generalization on difficult failure detection tasks. We establish state-of-the-art performance on RoboFail benchmark with 67.0% accuracy, surpassing closed-source models like OpenAI o1 by 6.0%.

  </details>



- **Grounding World Simulation Models in a Real-World Metropolis**  
  Junyoung Seo, Hyunwook Choi, Minkyung Kwon, Jinhyeok Choi, Siyoon Jin, Gayoung Lee, Junho Kim, JoungBin Lee, Geonmo Gu, Dongyoon Han, et al.  
  _2026-03-16_ · https://arxiv.org/abs/2603.15583v1  
  <details><summary>Abstract</summary>

  What if a world simulation model could render not an imagined environment but a city that actually exists? Prior generative world models synthesize visually plausible yet artificial environments by imagining all content. We present Seoul World Model (SWM), a city-scale world model grounded in the real city of Seoul. SWM anchors autoregressive video generation through retrieval-augmented conditioning on nearby street-view images. However, this design introduces several challenges, including temporal misalignment between retrieved references and the dynamic target scene, limited trajectory diversity and data sparsity from vehicle-mounted captures at sparse intervals. We address these challenges through cross-temporal pairing, a large-scale synthetic dataset enabling diverse camera trajectories, and a view interpolation pipeline that synthesizes coherent training videos from sparse street-view images. We further introduce a Virtual Lookahead Sink to stabilize long-horizon generation by continuously re-grounding each chunk to a retrieved image at a future location. We evaluate SWM against recent video world models across three cities: Seoul, Busan, and Ann Arbor. SWM outperforms existing methods in generating spatially faithful, temporally consistent, long-horizon videos grounded in actual urban environments over trajectories reaching hundreds of meters, while supporting diverse camera movements and text-prompted scenario variations.

  </details>



- **Benchmarking Machine Learning Approaches for Polarization Mapping in Ferroelectrics Using 4D-STEM**  
  Matej Martinc, Goran Dražič, Anton Kokalj, Katarina Žiberna, Janina Roknić, Matic Poberžnik, Sašo Džeroski, Andreja Benčan Golob  
  _2026-03-16_ · https://arxiv.org/abs/2603.15582v1  
  <details><summary>Abstract</summary>

  Four-dimensional scanning transmission electron microscopy (4D-STEM) provides rich, atomic-scale insights into materials structures. However, extracting specific physical properties - such as polarization directions essential for understanding functional properties of ferroelectrics - remains a significant challenge. In this study, we systematically benchmark multiple machine learning models, namely ResNet, VGG, a custom convolutional neural network, and PCA-informed k-Nearest Neighbors, to automate the detection of polarization directions from 4D-STEM diffraction patterns in ferroelectric potassium sodium niobate. While models trained on synthetic data achieve high accuracy on idealized synthetic diffraction patterns of equivalent thickness, the domain gap between simulation and experiment remains a critical barrier to real-world deployment. In this context, a custom made prototype representation training regime and PCA-based methods, combined with data augmentation and filtering, can better bridge this gap. Error analysis reveals periodic missclassification patterns, indicating that not all diffraction patterns carry enough information for a successful classification. Additionally, our qualitative analysis demonstrates that irregularities in the model's prediction patterns correlate with defects in the crystal structure, suggesting that supervised models could be used for detecting structural defects. These findings guide the development of robust, transferable machine learning tools for electron microscopy analysis.

  </details>



- **Severe Domain Shift in Skeleton-Based Action Recognition:A Study of Uncertainty Failure in Real-World Gym Environments**  
  Aaditya Khanal, Junxiu Zhou  
  _2026-03-16_ · https://arxiv.org/abs/2603.15574v1  
  <details><summary>Abstract</summary>

  The practical deployment gap -- transitioning from controlled multi-view 3D skeleton capture to unconstrained monocular 2D pose estimation -- introduces a compound domain shift whose safety implications remain critically underexplored. We present a systematic study of this severe domain shift using a novel Gym2D dataset (style/viewpoint shift) and the UCF101 dataset (semantic shift). Our Skeleton Transformer achieves 63.2% cross-subject accuracy on NTU-120 but drops to 1.6% under zero-shot transfer to the Gym domain and 1.16% on UCF101. Critically, we demonstrate that high Out-Of-Distribution (OOD) detection AUROC does not guarantee safe selective classification. Standard uncertainty methods fail to detect this performance drop: the model remains confidently incorrect with 99.6% risk even at 50% coverage across both OOD datasets. While energy-based scoring (AUROC >= 0.91) and Mahalanobis distance provide reliable distributional detection signals, such high AUROC scores coexist with poor risk-coverage behavior when making decisions. A lightweight finetuned gating mechanism restores calibration and enables graceful abstention, substantially reducing the rate of confident wrong predictions. Our work challenges standard deployment assumptions, providing a principled safety analysis of both semantic and geometric skeleton recognition deployment.

  </details>



- **Panoramic Affordance Prediction**  
  Zixin Zhang, Chenfei Liao, Hongfei Zhang, Harold Haodong Chen, Kanghao Chen, Zichen Wen, Litao Guo, Bin Ren, Xu Zheng, Yinchuan Li, et al.  
  _2026-03-16_ · https://arxiv.org/abs/2603.15558v1  
  <details><summary>Abstract</summary>

  Affordance prediction serves as a critical bridge between perception and action in embodied AI. However, existing research is confined to pinhole camera models, which suffer from narrow Fields of View (FoV) and fragmented observations, often missing critical holistic environmental context. In this paper, we present the first exploration into Panoramic Affordance Prediction, utilizing 360-degree imagery to capture global spatial relationships and holistic scene understanding. To facilitate this novel task, we first introduce PAP-12K, a large-scale benchmark dataset containing over 1,000 ultra-high-resolution (12k, 11904 x 5952) panoramic images with over 12k carefully annotated QA pairs and affordance masks. Furthermore, we propose PAP, a training-free, coarse-to-fine pipeline inspired by the human foveal visual system to tackle the ultra-high resolution and severe distortion inherent in panoramic images. PAP employs recursive visual routing via grid prompting to progressively locate targets, applies an adaptive gaze mechanism to rectify local geometric distortions, and utilizes a cascaded grounding pipeline to extract precise instance-level masks. Experimental results on PAP-12K reveal that existing affordance prediction methods designed for standard perspective images suffer severe performance degradation and fail due to the unique challenges of panoramic vision. In contrast, PAP framework effectively overcomes these obstacles, significantly outperforming state-of-the-art baselines and highlighting the immense potential of panoramic perception for robust embodied intelligence.

  </details>



- **Learning Latent Proxies for Controllable Single-Image Relighting**  
  Haoze Zheng, Zihao Wang, Xianfeng Wu, Yajing Bai, Yexin Liu, Yun Li, Xiaogang Xu, Harry Yang  
  _2026-03-16_ · https://arxiv.org/abs/2603.15555v1  
  <details><summary>Abstract</summary>

  Single-image relighting is highly under-constrained: small illumination changes can produce large, nonlinear variations in shading, shadows, and specularities, while geometry and materials remain unobserved. Existing diffusion-based approaches either rely on intrinsic or G-buffer pipelines that require dense and fragile supervision, or operate purely in latent space without physical grounding, making fine-grained control of direction, intensity, and color unreliable. We observe that a full intrinsic decomposition is unnecessary and redundant for accurate relighting. Instead, sparse but physically meaningful cues, indicating where illumination should change and how materials should respond, are sufficient to guide a diffusion model. Based on this insight, we introduce LightCtrl that integrates physical priors at two levels: a few-shot latent proxy encoder that extracts compact material-geometry cues from limited PBR supervision, and a lighting-aware mask that identifies sensitive illumination regions and steers the denoiser toward shading relevant pixels. To compensate for scarce PBR data, we refine the proxy branch using a DPO-based objective that enforces physical consistency in the predicted cues. We also present ScaLight, a large-scale object-level dataset with systematically varied illumination and complete camera-light metadata, enabling physically consistent and controllable training. Across object and scene level benchmarks, our method achieves photometrically faithful relighting with accurate continuous control, surpassing prior diffusion and intrinsic-based baselines, including gains of up to +2.4 dB PSNR and 35% lower RMSE under controlled lighting shifts.

  </details>



- **Kimodo: Scaling Controllable Human Motion Generation**  
  Davis Rempe, Mathis Petrovich, Ye Yuan, Haotian Zhang, Xue Bin Peng, Yifeng Jiang, Tingwu Wang, Umar Iqbal, David Minor, Michael de Ruyter, et al.  
  _2026-03-16_ · https://arxiv.org/abs/2603.15546v1  
  <details><summary>Abstract</summary>

  High-quality human motion data is becoming increasingly important for applications in robotics, simulation, and entertainment. Recent generative models offer a potential data source, enabling human motion synthesis through intuitive inputs like text prompts or kinematic constraints on poses. However, the small scale of public mocap datasets has limited the motion quality, control accuracy, and generalization of these models. In this work, we introduce Kimodo, an expressive and controllable kinematic motion diffusion model trained on 700 hours of optical motion capture data. Our model generates high-quality motions while being easily controlled through text and a comprehensive suite of kinematic constraints including full-body keyframes, sparse joint positions/rotations, 2D waypoints, and dense 2D paths. This is enabled through a carefully designed motion representation and two-stage denoiser architecture that decomposes root and body prediction to minimize motion artifacts while allowing for flexible constraint conditioning. Experiments on the large-scale mocap dataset justify key design decisions and analyze how the scaling of dataset size and model size affect performance.

  </details>



- **Clinically Aware Synthetic Image Generation for Concept Coverage in Chest X-ray Models**  
  Amy Rafferty, Rishi Ramaesh, Ajitha Rajan  
  _2026-03-16_ · https://arxiv.org/abs/2603.15525v1  
  <details><summary>Abstract</summary>

  The clinical deployment of AI diagnostic models demands more than benchmark accuracy - it demands robustness across the full spectrum of disease presentations. However, publicly available chest radiographic datasets systematically underrepresent critical clinical feature combinations, leaving models under-trained precisely where clinical stakes are highest. We present CARS, a clinically aware and anatomically grounded framework that addresses this gap through principled synthetic image generation. CARS applies targeted perturbations to clinical feature vectors, enabling controlled insertion and deletion of pathological findings while explicitly preserving anatomical structure. We evaluate CARS across seven backbone architectures by fine-tuning models on synthetic subsets and testing on a held-out MIMIC-CXR benchmark. Compared to prior feature perturbation approaches, fine-tuning on CARS-generated images consistently improves precision-recall performance, reduces predictive uncertainty, and improves model calibration. Structural and semantic analyses demonstrate high anatomical fidelity, strong feature alignment, and low semantic uncertainty. Independent evaluation by two expert radiologists further confirms realism and clinical agreement. As the field moves toward regulated clinical AI, CARS demonstrates that anatomically faithful synthetic data generation for better feature space coverage is a viable and effective strategy for improving both the performance and trustworthiness of chest X-ray classification systems - without compromising clinical integrity.

  </details>



- **Federated Learning of Binary Neural Networks: Enabling Low-Cost Inference**  
  Nitin Priyadarshini Shankar, Soham Lahiri, Sheetal Kalyani, Saurav Prakash  
  _2026-03-16_ · https://arxiv.org/abs/2603.15507v1  
  <details><summary>Abstract</summary>

  Federated Learning (FL) preserves privacy by distributing training across devices. However, using DNNs is computationally intensive at the low-powered edge during inference. Edge deployment demands models that simultaneously optimize memory footprint and computational efficiency, a dilemma where conventional DNNs fail by exceeding resource limits. Traditional post-training binarization reduces model size but suffers from severe accuracy loss due to quantization errors. To address these challenges, we propose FedBNN, a rotation-aware binary neural network framework that learns binary representations directly during local training. By encoding each weight as a single bit $\{+1, -1\}$ instead of a $32$-bit float, FedBNN shrinks the model footprint, significantly reducing runtime (during inference) FLOPs and memory requirements in comparison to federated methods using real models. Evaluations across multiple benchmark datasets demonstrate that FedBNN significantly reduces resource consumption while performing similarly to existing federated methods using real-valued models.

  </details>



- **RSGen: Enhancing Layout-Driven Remote Sensing Image Generation with Diverse Edge Guidance**  
  Xianbao Hou, Yonghao He, Zeyd Boukhers, John See, Hu Su, Wei Sui, Cong Yang  
  _2026-03-16_ · https://arxiv.org/abs/2603.15484v1  
  <details><summary>Abstract</summary>

  Diffusion models have significantly mitigated the impact of annotated data scarcity in remote sensing (RS). Although recent approaches have successfully harnessed these models to enable diverse and controllable Layout-to-Image (L2I) synthesis, they still suffer from limited fine-grained control and fail to strictly adhere to bounding box constraints. To address these limitations, we propose RSGen, a plug-and-play framework that leverages diverse edge guidance to enhance layout-driven RS image generation. Specifically, RSGen employs a progressive enhancement strategy: 1) it first enriches the diversity of edge maps composited from retrieved training instances via Image-to-Image generation; and 2) subsequently utilizes these diverse edge maps as conditioning for existing L2I models to enforce pixel-level control within bounding boxes, ensuring the generated instances strictly adhere to the layout. Extensive experiments across three baseline models demonstrate that RSGen significantly boosts the capabilities of existing L2I models. For instance, with CC-Diff on the DOTA dataset for oriented object detection, we achieve remarkable gains of +9.8/+12.0 in YOLOScore mAP50/mAP50-95 and +1.6 in mAP on the downstream detection task. Our code will be publicly available: https://github.com/D-Robotics-AI-Lab/RSGen

  </details>



- **Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation**  
  Yuanfan Zheng, Kunyu Peng, Xu Zheng, Kailun Yang  
  _2026-03-16_ · https://arxiv.org/abs/2603.15475v1  
  <details><summary>Abstract</summary>

  Cross-domain panoramic semantic segmentation has attracted growing interest as it enables comprehensive 360° scene understanding for real-world applications. However, it remains particularly challenging due to severe geometric Field of View (FoV) distortions and inconsistent open-set semantics across domains. In this work, we formulate an open-set domain adaptation setting, and propose Extrapolative Domain Adaptive Panoramic Segmentation (EDA-PSeg) framework that trains on local perspective views and tests on full 360° panoramic images, explicitly tackling both geometric FoV shifts across domains and semantic uncertainty arising from previously unseen classes. To this end, we propose the Euler-Margin Attention (EMA), which introduces an angular margin to enhance viewpoint-invariant semantic representation, while performing amplitude and phase modulation to improve generalization toward unseen classes. Additionally, we design the Graph Matching Adapter (GMA), which builds high-order graph relations to align shared semantics across FoV shifts while effectively separating novel categories through structural adaptation. Extensive experiments on four benchmark datasets under camera-shift, weather-condition, and open-set scenarios demonstrate that EDA-PSeg achieves state-of-the-art performance, robust generalization to diverse viewing geometries, and resilience under varying environmental conditions. The code is available at https://github.com/zyfone/EDA-PSeg.

  </details>



- **RoCo Challenge at AAAI 2026: Benchmarking Robotic Collaborative Manipulation for Assembly Towards Industrial Automation**  
  Haichao Liu, Yuheng Zhou, Zhenyu Wu, Ziheng Ji, Ziyu Shan, Qianzhun Wang, Ruixuan Liu, Zhiyuan Yang, Yejun Gu, Shalman Khan, et al.  
  _2026-03-16_ · https://arxiv.org/abs/2603.15469v1  
  <details><summary>Abstract</summary>

  Embodied Artificial Intelligence (EAI) is rapidly developing, gradually subverting previous autonomous systems' paradigms from isolated perception to integrated, continuous action. This transition is highly significant for industrial robotic manipulation, promising to free human workers from repetitive, dangerous daily labor. To benchmark and advance this capability, we introduce the Robotic Collaborative Assembly Assistance (RoCo) Challenge with a dataset towards simulation and real-world assembly manipulation. Set against the backdrop of human-centered manufacturing, this challenge focuses on a high-precision planetary gearbox assembly task, a demanding yet highly representative operation in modern industry. Built upon a self-developed data collection, training, and evaluation system in Isaac Sim, and utilizing a dual-arm robot for real-world deployment, the challenge operates in two phases. The Simulation Round defines fine-grained task phases for step-wise scoring to handle the long-horizon nature of the assembly. The Real-World Round mirrors this evaluation with physical gearbox components and high-quality teleoperated datasets. The core tasks require assembling an epicyclic gearbox from scratch, including mounting three planet gears, a sun gear, and a ring gear. Attracting over 60 teams and 170+ participants from more than 10 countries, the challenge yielded highly effective solutions, most notably ARC-VLA and RoboCola. Results demonstrate that a dual-model framework for long-horizon multi-task learning is highly effective, and the strategic utilization of recovery-from-failure curriculum data is a critical insight for successful deployment. This report outlines the competition setup, evaluation approach, key findings, and future directions for industrial EAI. Our dataset, CAD files, code, and evaluation results can be found at: https://rocochallenge.github.io/RoCo2026/.

  </details>



- **Evaluating Time Awareness and Cross-modal Active Perception of Large Models via 4D Escape Room Task**  
  Yurui Dong, Ziyue Wang, Shuyun Lu, Dairu Liu, Xuechen Liu, Fuwen Luo, Peng Li, Yang Liu  
  _2026-03-16_ · https://arxiv.org/abs/2603.15467v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have recently made rapid progress toward unified Omni models that integrate vision, language, and audio. However, existing environments largely focus on 2D or 3D visual context and vision-language tasks, offering limited support for temporally dependent auditory signals and selective cross-modal integration, where different modalities may provide complementary or interfering information, which are essential capabilities for realistic multimodal reasoning. As a result, whether models can actively coordinate modalities and reason under time-varying, irreversible conditions remains underexplored. To this end, we introduce \textbf{EscapeCraft-4D}, a customizable 4D environment for assessing selective cross-modal perception and time awareness in Omni models. It incorporates trigger-based auditory sources, temporally transient evidence, and location-dependent cues, requiring agents to perform spatio-temporal reasoning and proactive multimodal integration under time constraints. Building on this environment, we curate a benchmark to evaluate corresponding abilities across powerful models. Evaluation results suggest that models struggle with modality bias, and reveal significant gaps in current model's ability to integrate multiple modalities under time constraints. Further in-depth analysis uncovers how multiple modalities interact and jointly influence model decisions in complex multimodal reasoning environments.

  </details>



- **End-to-End Dexterous Grasp Learning from Single-View Point Clouds via a Multi-Object Scene Dataset**  
  Tao Geng, Dapeng Yang, Ziwei Liu, Le Zhang, Le Qi, WangYang Li, Yi Ren, Shan Luo, Fenglei Ni  
  _2026-03-16_ · https://arxiv.org/abs/2603.15410v1  
  <details><summary>Abstract</summary>

  Dexterous grasping in multi-object scene constitutes a fundamental challenge in robotic manipulation. Current mainstream grasping datasets predominantly focus on single-object scenarios and predefined grasp configurations, often neglecting environmental interference and the modeling of dexterous pre-grasp gesture, thereby limiting their generalizability in real-world applications. To address this, we propose DGS-Net, an end-to-end grasp prediction network capable of learning dense grasp configurations from single-view point clouds in multi-object scene. Furthermore, we propose a two-stage grasp data generation strategy that progresses from dense single-object grasp synthesis to dense scene-level grasp generation. Our dataset comprises 307 objects, 240 multi-object scenes, and over 350k validated grasps. By explicitly modeling grasp offsets and pre-grasp configurations, the dataset provides more robust and accurate supervision for dexterous grasp learning. Experimental results show that DGS-Net achieves grasp success rates of 88.63\% in simulation and 78.98\% on a real robotic platform, while exhibiting lower penetration with a mean penetration depth of 0.375 mm and penetration volume of 559.45 mm^3, outperforming existing methods and demonstrating strong effectiveness and generalization capability. Our dataset is available at https://github.com/4taotao8/DGS-Net.

  </details>



- **Detection of Autonomous Shuttles in Urban Traffic Images Using Adaptive Residual Context**  
  Mohamed Aziz Younes, Nicolas Saunier, Guillaume-Alexandre Bilodeau  
  _2026-03-16_ · https://arxiv.org/abs/2603.15404v1  
  <details><summary>Abstract</summary>

  The progressive automation of transport promises to enhance safety and sustainability through shared mobility. Like other vehicles and road users, and even more so for such a new technology, it requires monitoring to understand how it interacts in traffic and to evaluate its safety. This can be done with fixed cameras and video object detection. However, the addition of new detection targets generally requires a fine-tuning approach for regular detection methods. Unfortunately, this implementation strategy will lead to a phenomenon known as catastrophic forgetting, which causes a degradation in scene understanding. In road safety applications, preserving contextual scene knowledge is of the utmost importance for protecting road users. We introduce the Adaptive Residual Context (ARC) architecture to address this. ARC links a frozen context branch and trainable task-specific branches through a Context-Guided Bridge, utilizing attention to transfer spatial features while preserving pre-trained representations. Experiments on a custom dataset show that ARC matches fine-tuned baselines while significantly improving knowledge retention, offering a data-efficient solution to add new vehicle categories for complex urban environments.

  </details>



- **Pointing-Based Object Recognition**  
  Lukáš Hajdúch, Viktor Kocur  
  _2026-03-16_ · https://arxiv.org/abs/2603.15403v1  
  <details><summary>Abstract</summary>

  This paper presents a comprehensive pipeline for recognizing objects targeted by human pointing gestures using RGB images. As human-robot interaction moves toward more intuitive interfaces, the ability to identify targets of non-verbal communication becomes crucial. Our proposed system integrates several existing state-of-the-art methods, including object detection, body pose estimation, monocular depth estimation, and vision-language models. We evaluate the impact of 3D spatial information reconstructed from a single image and the utility of image captioning models in correcting classification errors. Experimental results on a custom dataset show that incorporating depth information significantly improves target identification, especially in complex scenes with overlapping objects. The modularity of the approach allows for deployment in environments where specialized depth sensors are unavailable.

  </details>



- **Spectral Rectification for Parameter-Efficient Adaptation of Foundation Models in Colonoscopy Depth Estimation**  
  Xiaoxian Zhang, Minghai Shi, Lei Li  
  _2026-03-16_ · https://arxiv.org/abs/2603.15374v1  
  <details><summary>Abstract</summary>

  Accurate monocular depth estimation is critical in colonoscopy for lesion localization and navigation. Foundation models trained on natural images fail to generalize directly to colonoscopy. We identify the core issue not as a semantic gap, but as a statistical shift in the frequency domain: colonoscopy images lack the strong high-frequency edge and texture gradients that these models rely on for geometric reasoning. To address this, we propose SpecDepth, a parameter-efficient adaptation framework that preserves the robust geometric representations of the pre-trained models while adapting to the colonoscopy domain. Its key innovation is an adaptive spectral rectification module, which uses a learnable wavelet decomposition to explicitly model and amplify the attenuated high-frequency components in feature maps. Different from conventional fine-tuning that risks distorting high-level semantic features, this targeted, low-level adjustment realigns the input signal with the original inductive bias of the foundational model. On the public C3VD and SimCol3D datasets, SpecDepth achieved state-of-the-art performance with an absolute relative error of 0.022 and 0.027, respectively. Our work demonstrates that directly addressing spectral mismatches is a highly effective strategy for adapting vision foundation models to specialized medical imaging tasks. The code will be released publicly after the manuscript is accepted for publication.

  </details>



- **A PPO-Based Bitrate Allocation Conditional Diffusion Model for Remote Sensing Image Compression**  
  Yuming Han, Jooho Kim, Anish Shakya  
  _2026-03-16_ · https://arxiv.org/abs/2603.15365v1  
  <details><summary>Abstract</summary>

  Existing remote sensing image compression methods still explore to balance high compression efficiency with the preservation of fine details and task-relevant information. Meanwhile, high-resolution drone imagery offers valuable structural details for urban monitoring and disaster assessment, but large-area datasets can easily reach hundreds of gigabytes, creating significant challenges for storage and long-term management. In this paper, we propose a PPO-based bitrate allocation Conditional Diffusion Compression (PCDC) framework. PCDC integrates a conditional diffusion decoder with a PPO-based block-wise bitrate allocation strategy to achieve high compression ratios while maintaining strong perceptual performance. We also release a high-resolution drone image dataset with richer structural details at a consistent low altitude over residential neighborhoods in coastal urban areas. Experimental results show compression ratios of 19.3x on DIV2K and 21.2x on the drone image dataset. Moreover, downstream object detection experiments demonstrate that the reconstructed images preserve task-relevant information with negligible performance loss.

  </details>



- **Generative Video Compression with One-Dimensional Latent Representation**  
  Zihan Zheng, Zhaoyang Jia, Naifu Xue, Jiahao Li, Bin Li, Zongyu Guo, Xiaoyi Zhang, Zhenghao Chen, Houqiang Li, Yan Lu  
  _2026-03-16_ · https://arxiv.org/abs/2603.15302v1  
  <details><summary>Abstract</summary>

  Recent advancements in generative video codec (GVC) typically encode video into a 2D latent grid and employ high-capacity generative decoders for reconstruction. However, this paradigm still leaves two key challenges in fully exploiting spatial-temporal redundancy: Spatially, the 2D latent grid inevitably preserves intra-frame redundancy due to its rigid structure, where adjacent patches remain highly similar, thereby necessitating a higher bitrate. Temporally, the 2D latent grid is less effective for modeling long-term correlations in a compact and semantically coherent manner, as it hinders the aggregation of common contents across frames. To address these limitations, we introduce Generative Video Compression with One-Dimensional (1D) Latent Representation (GVC1D). GVC1D encodes the video data into extreme compact 1D latent tokens conditioned on both short- and long-term contexts. Without the rigid 2D spatial correspondence, these 1D latent tokens can adaptively attend to semantic regions and naturally facilitate token reduction, thereby reducing spatial redundancy. Furthermore, the proposed 1D memory provides semantically rich long-term context while maintaining low computational cost, thereby further reducing temporal redundancy. Experimental results indicate that GVC1D attains superior compression efficiency, where it achieves bitrate reductions of 60.4\% under LPIPS and 68.8\% under DISTS on the HEVC Class B dataset, surpassing the previous video compression methods.Project: https://gvc1d.github.io/

  </details>



- **GNIO: Gated Neural Inertial Odometry**  
  Dapeng Feng, Yizhen Yin, Zhiqiang Chen, Yuhua Qi, Hongbo Chen  
  _2026-03-16_ · https://arxiv.org/abs/2603.15281v1  
  <details><summary>Abstract</summary>

  Inertial navigation using low-cost MEMS sensors is plagued by rapid drift due to sensor noise and bias instability. While recent data-driven approaches have made significant strides, they often struggle with micro-drifts during stationarity and mode fusion during complex motion transitions due to their reliance on fixed-window regression. In this work, we introduce Gated Neural Inertial Odometry (GNIO), a novel learning-based framework that explicitly models motion validity and context. We propose two key architectural innovations: \ding{182} a learnable Motion Bank that queries a global dictionary of motion patterns to provide semantic context beyond the local receptive field, and \ding{183} a Gated Prediction Head that decomposes displacement into magnitude and direction. This gating mechanism acts as a soft, differentiable Zero-Velocity Update (ZUPT), dynamically suppressing sensor noise during stationary periods while scaling predictions during dynamic motion. Extensive experiments across four public benchmarks demonstrate that GNIO significantly reduces position drift compared to state-of-the-art CNN and Transformer-based baselines. Notably, GNIO achieves a $60.21\%$ reduction in trajectory error on the OxIOD dataset and exhibits superior generalization in challenging scenarios involving frequent stops and irregular motion speeds.

  </details>



- **Dataset Diversity Metrics and Impact on Classification Models**  
  Théo Sourget, Niclas Claßen, Jack Junchi Xu, Rob van der Goot, Veronika Cheplygina  
  _2026-03-16_ · https://arxiv.org/abs/2603.15276v1  
  <details><summary>Abstract</summary>

  The diversity of training datasets is usually perceived as an important aspect to obtain a robust model. However, the definition of diversity is often not defined or differs across papers, and while some metrics exist, the quantification of this diversity is often overlooked when developing new algorithms. In this work, we study the behaviour of multiple dataset diversity metrics for image, text and metadata using MorphoMNIST, a toy dataset with controlled perturbations, and PadChest, a publicly available chest X-ray dataset. We evaluate whether these metrics correlate with each other but also with the intuition of a clinical expert. We also assess whether they correlate with downstream-task performance and how they impact the training dynamic of the models. We find limited correlations between the AUC and image or metadata reference-free diversity metrics, but higher correlations with the FID and the semantic diversity metrics. Finally, the clinical expert indicates that scanners are the main source of diversity in practice. However, we find that the addition of another scanner to the training set leads to shortcut learning. The code used in this study is available at https://github.com/TheoSourget/dataset_diversity_evaluation

  </details>



- **Exemplar Diffusion: Improving Medical Object Detection with Opportunistic Labels**  
  Victor Wåhlstrand, Jennifer Alvén, Ida Häggström  
  _2026-03-16_ · https://arxiv.org/abs/2603.15267v1  
  <details><summary>Abstract</summary>

  We present a framework to take advantage of existing labels at inference, called \textit{exemplars}, in order to improve the performance of object detection in medical images. The method, \textit{exemplar diffusion}, leverages existing diffusion methods for object detection to enable a training-free approach to adding information of known bounding boxes at test time. We demonstrate that for medical image datasets with clear spatial structure, the method yields an across-the-board increase in average precision and recall, and a robustness to exemplar quality, enabling non-expert annotation. Moreover, we demonstrate how our method may also be used to quantify predictive uncertainty in diffusion detection methods. Source code and data splits openly available online: https://github.com/waahlstrand/ExemplarDiffusion

  </details>



- **IConE: Batch Independent Collapse Prevention for Self-Supervised Representation Learning**  
  Konstantinos Almpanakis, Anna Kreshuk  
  _2026-03-16_ · https://arxiv.org/abs/2603.15263v1  
  <details><summary>Abstract</summary>

  Self-supervised learning (SSL) has revolutionized representation learning, with Joint-Embedding Architectures (JEAs) emerging as an effective approach for capturing semantic features. Existing JEAs rely on implicit or explicit batch interaction -- via negative sampling or statistical regularization -- to prevent representation collapse. This reliance becomes problematic in regimes where batch sizes must be small, such as high-dimensional scientific data, where memory constraints and class imbalance make large, well-balanced batches infeasible. We introduce IConE (Instance-Contrasted Embeddings), a framework that decouples collapse prevention from the training batch size. Rather than enforcing diversity through batch statistics, IConE maintains a global set of learnable auxiliary instance embeddings regularized by an explicit diversity objective. This transfers the anti-collapse mechanism from the transient batch to a dataset-level embedding space, allowing stable training even when batch statistics are unreliable, down to batch size 1. Across diverse 2D and 3D biomedical modalities, IConE outperforms strong contrastive and non-contrastive baselines throughout the small-batch regime (from B=1 to B=64) and demonstrates marked robustness to severe class imbalance. Geometric analysis shows that IConE preserves high intrinsic dimensionality in the learned representations, preventing the collapse observed in existing JEAs as batch sizes shrink.

  </details>



- **HalDec-Bench: Benchmarking Hallucination Detector in Image Captioning**  
  Kuniaki Saito, Risa Shinoda, Shohei Tanaka, Tosho Hirasawa, Fumio Okura, Yoshitaka Ushiku  
  _2026-03-16_ · https://arxiv.org/abs/2603.15253v1  
  <details><summary>Abstract</summary>

  Hallucination detection in captions (HalDec) assesses a vision-language model's ability to correctly align image content with text by identifying errors in captions that misrepresent the image. Beyond evaluation, effective hallucination detection is also essential for curating high-quality image-caption pairs used to train VLMs. However, the generalizability of VLMs as hallucination detectors across different captioning models and hallucination types remains unclear due to the lack of a comprehensive benchmark. In this work, we introduce HalDec-Bench, a benchmark designed to evaluate hallucination detectors in a principled and interpretable manner. HalDec-Bench contains captions generated by diverse VLMs together with human annotations indicating the presence of hallucinations, detailed hallucination-type categories, and segment-level labels. The benchmark provides tasks with a wide range of difficulty levels and reveals performance differences across models that are not visible in existing multimodal reasoning or alignment benchmarks. Our analysis further uncovers two key findings. First, detectors tend to recognize sentences appearing at the beginning of a response as correct, regardless of their actual correctness. Second, our experiments suggest that dataset noise can be substantially reduced by using strong VLMs as filters while employing recent VLMs as caption generators. Our project page is available at https://dahlian00.github.io/HalDec-Bench-Page/.

  </details>



- **Multi-turn Physics-informed Vision-language Model for Physics-grounded Anomaly Detection**  
  Yao Gu, Xiaohao Xu, Yingna Wu  
  _2026-03-16_ · https://arxiv.org/abs/2603.15237v1  
  <details><summary>Abstract</summary>

  Vision-Language Models (VLMs) demonstrate strong general-purpose reasoning but remain limited in physics-grounded anomaly detection, where causal understanding of dynamics is essential. Existing VLMs, trained predominantly on appearance-centric correlations, fail to capture kinematic constraints, leading to poor performance on anomalies such as irregular rotations or violated mechanical motions. We introduce a physics-informed instruction tuning framework that explicitly encodes object properties, motion paradigms, and dynamic constraints into structured prompts. By delivering these physical priors through multi-turn dialogues, our method decomposes causal reasoning into incremental steps, enabling robust internal representations of normal and abnormal dynamics. Evaluated on the Phys-AD benchmark, our approach achieves 96.7% AUROC in video-level detection--substantially outperforming prior SOTA (66.9%)--and yields superior causal explanations (0.777 LLM score). This work highlights how structured physics priors can transform VLMs into reliable detectors of dynamic anomalies.

  </details>



- **HYDRA: Unifying Multi-modal Generation and Understanding via Representation-Harmonized Tokenization**  
  Xuerui Qiu, Yutao Cui, Guozhen Zhang, Junzhe Li, JiaKui Hu, Xiao Zhang, Yang Li, Songtao Liu, Miles Yang, Yu Shi, et al.  
  _2026-03-16_ · https://arxiv.org/abs/2603.15228v1  
  <details><summary>Abstract</summary>

  Unified Multimodal Models struggle to bridge the fundamental gap between the abstract representations needed for visual understanding and the detailed primitives required for generation. Existing approaches typically compromise by employing decoupled encoders, stacking representation encoder atop VAEs, or utilizing discrete quantization. However, these methods often disrupt information coherence and lead to optimization conflicts. To this end, we introduce HYDRA-TOK, a representation-harmonized pure ViT in the insight that visual modeling should evolve from generation to understanding. HYDRA-TOK reformulates the standard backbone into a progressive learner that transitions from a Gen-ViT, which captures structure-preserving primitives, to a Sem-ViT for semantic encoding. Crucially, this transition is mediated by a Generation-Semantic Bottleneck (GSB), which compresses features into a low-dimensional space to filter noise for robust synthesis, then restores dimensionality to empower complex semantic comprehension. Built upon this foundation, we present HYDRA, a native unified framework integrating perception and generation within a single parameter space. Extensive experiments establish HYDRA as a new state-of-the-art. It sets a benchmark in visual reconstruction (rFID 0.08) and achieves top-tier generation performance on GenEval (0.86), DPG-Bench (86.4), and WISE (0.53), while simultaneously outperforming previous native UMMs by an average of 10.0 points across eight challenging understanding benchmarks.

  </details>



- **Coupled Particle Filters for Robust Affordance Estimation**  
  Patrick Lowin, Vito Mengers, Oliver Brock  
  _2026-03-16_ · https://arxiv.org/abs/2603.15223v1  
  <details><summary>Abstract</summary>

  Robotic affordance estimation is challenging due to visual, geometric, and semantic ambiguities in sensory input. We propose a method that disambiguates these signals using two coupled recursive estimators for sub-aspects of affordances: graspable and movable regions. Each estimator encodes property-specific regularities to reduce uncertainty, while their coupling enables bidirectional information exchange that focuses attention on regions where both agree, i.e., affordances. Evaluated on a real-world dataset, our method outperforms three recent affordance estimators (Where2Act, Hands-as-Probes, and HRP) by 308%, 245%, and 257% in precision, and remains robust under challenging conditions such as low light or cluttered environments. Furthermore, our method achieves a 70% success rate in our real-world evaluation. These results demonstrate that coupling complementary estimators yields precise, robust, and embodiment-appropriate affordance predictions.

  </details>



- **What Matters for Scalable and Robust Learning in End-to-End Driving Planners?**  
  David Holtz, Niklas Hanselmann, Simon Doll, Marius Cordts, Bernt Schiele  
  _2026-03-16_ · https://arxiv.org/abs/2603.15185v1  
  <details><summary>Abstract</summary>

  End-to-end autonomous driving has gained significant attention for its potential to learn robust behavior in interactive scenarios and scale with data. Popular architectures often build on separate modules for perception and planning connected through latent representations, such as bird's eye view feature grids, to maintain end-to-end differentiability. This paradigm emerged mostly on open-loop datasets, with evaluation focusing not only on driving performance, but also intermediate perception tasks. Unfortunately, architectural advances that excel in open-loop often fail to translate to scalable learning of robust closed-loop driving. In this paper, we systematically re-examine the impact of common architectural patterns on closed-loop performance: (1) high-resolution perceptual representations, (2) disentangled trajectory representations, and (3) generative planning. Crucially, our analysis evaluates the combined impact of these patterns, revealing both unexpected limitations as well as underexplored synergies. Building on these insights, we introduce BevAD, a novel lightweight and highly scalable end-to-end driving architecture. BevAD achieves 72.7% success rate on the Bench2Drive benchmark and demonstrates strong data-scaling behavior using pure imitation learning. Our code and models are publicly available here: https://dmholtz.github.io/bevad/

  </details>



- **KiRAS: Keyframe Guided Self-Imitation for Robust and Adaptive Skill Learning in Quadruped Robots**  
  Xiaoyi Wei, Peng Zhai, Jiaxin Tu, Yueqi Zhang, Yuqi Li, Zonghao Zhang, Hu Zhou, Lihua Zhang  
  _2026-03-16_ · https://arxiv.org/abs/2603.15179v1  
  <details><summary>Abstract</summary>

  With advances in reinforcement learning and imitation learning, quadruped robots can acquire diverse skills within a single policy by imitating multiple skill-specific datasets. However, the lack of datasets on complex terrains limits the ability of such multi-skill policies to generalize effectively in unstructured environments. Inspired by animation, we adopt keyframes as minimal and universal skill representations, relaxing dataset constraints and enabling the integration of terrain adaptability with skill diversity. We propose Keyframe Guided Self-Imitation for Robust and Adaptive Skill Learning (KiRAS), an end-to-end framework for acquiring and transitioning between diverse skill primitives on complex terrains. KiRAS first learns diverse skills on flat terrain through keyframe-guided self-imitation, eliminating the need for expert datasets; then continues training the same policy network on rough terrains to enhance robustness. To eliminate catastrophic forgetting, a proficiency-based Skill Initialization Technique is introduced. Experiments on Solo-8 and Unitree Go1 robots show that KiRAS enables robust skill acquisition and smooth transitions across challenging terrains. This framework demonstrates its potential as a lightweight platform for multi-skill generation and dataset collection. It further enables flexible skill transitions that enhance locomotion on challenging terrains.

  </details>



- **ForceVLA2: Unleashing Hybrid Force-Position Control with Force Awareness for Contact-Rich Manipulation**  
  Yang Li, Zhaxizhuoma, Hongru Jiang, Junjie Xia, Hongquan Zhang, Jinda Du, Yunsong Zhou, Jia Zeng, Ce Hao, Jieji Ren, et al.  
  _2026-03-16_ · https://arxiv.org/abs/2603.15169v1  
  <details><summary>Abstract</summary>

  Embodied intelligence for contact-rich manipulation has predominantly relied on position control, while explicit awareness and regulation of interaction forces remain under-explored, limiting stability, precision, and robustness in real-world tasks. We propose ForceVLA2, an end-to-end vision-language-action framework that equips robots with hybrid force-position control and explicit force awareness. ForceVLA2 introduces force-based prompts into the VLM expert to construct force-aware task concepts across stages, and employs a Cross-Scale Mixture-of-Experts (MoE) in the action expert to adaptively fuse these concepts with real-time interaction forces for closed-loop hybrid force-position regulation. To support learning and evaluation, we construct ForceVLA2-Dataset, containing 1,000 trajectories over 5 contact-rich tasks, including wiping, pressing, and assembling, with multi-view images, task prompts, proprioceptive state, and force signals. Extensive experiments show that ForceVLA2 substantially improves success rates and reliability in contact-rich manipulation, outperforming pi0 and pi0.5 by 48.0% and 35.0%, respectively, across the 5 tasks, and mitigating common failure modes such as arm overload and unstable contact, thereby actively advancing force-aware interactive physical intelligence in VLAs. The project page is available at https://sites.google.com/view/force-vla2/home.

  </details>



- **Multimodal Connectome Fusion via Cross-Attention for Autism Spectrum Disorder Classification Using Graph Learning**  
  Ansar Rahman, Hassan Shojaee-Mend, Sepideh Hatamikia  
  _2026-03-16_ · https://arxiv.org/abs/2603.15168v1  
  <details><summary>Abstract</summary>

  Autism spectrum disorder (ASD) is a complex neurodevelopmental condition characterized by atypical functional brain connectivity and subtle structural alterations. rs-fMRI has been widely used to identify disruptions in large-scale brain networks, while structural MRI provides complementary information about morphological organization. Despite their complementary nature, effectively integrating these heterogeneous imaging modalities within a unified framework remains challenging. This study proposes a multimodal graph learning framework that preserves the dominant role of functional connectivity while integrating structural imaging and phenotypic information for ASD classification. The proposed framework is evaluated on ABIDE-I dataset. Each subject is represented as a node within a population graph. Functional and structural features are extracted as modality-specific node attributes, while inter-subject relationships are modeled using a pairwise association encoder (PAE) based on phenotypic information. Two Edge Variational GCNs are trained to learn subject-level embeddings. To enable effective multimodal integration, we introduce a novel asymmetric transformer-based cross-attention mechanism that allows functional embeddings to selectively incorporate complementary structural information while preserving functional dominance. The fused embeddings are then passed to a MLP for ASD classification. Using stratified 10-fold cross-validation, the framework achieved an AUC of 87.3% and an accuracy of 84.4%. Under leave-one-site-out cross-validation (LOSO-CV), the model achieved an average cross-site accuracy of 82.0%, outperforming existing methods by approximately 3% under 10-fold cross-validation and 7% under LOSO-CV. The proposed framework effectively integrates heterogeneous multimodal data from the multi-site ABIDE-I dataset, improving automated ASD classification across imaging sites.

  </details>



- **TextOVSR: Text-Guided Real-World Opera Video Super-Resolution**  
  Hua Chang, Xin Xu, Wei Liu, Jiayi Wu, Kui Jiang, Fei Ma, Qi Tian  
  _2026-03-16_ · https://arxiv.org/abs/2603.15153v1  
  <details><summary>Abstract</summary>

  Many classic opera videos exhibit poor visual quality due to the limitations of early filming equipment and long-term degradation during storage. Although real-world video super-resolution (RWVSR) has achieved significant advances in recent years, directly applying existing methods to degraded opera videos remains challenging. The difficulties are twofold. First, accurately modeling real-world degradations is complex: simplistic combinations of classical degradation kernels fail to capture the authentic noise distribution, while methods that extract real noise patches from external datasets are prone to style mismatches that introduce visual artifacts. Second, current RWVSR methods, which rely solely on degraded image features, struggle to reconstruct realistic and detailed textures due to a lack of high-level semantic guidance. To address these issues, we propose a Text-guided Dual-Branch Opera Video Super-Resolution (TextOVSR) network, which introduces two types of textual prompts to guide the super-resolution process. Specifically, degradation-descriptive text, derived from the degradation process, is incorporated into the negative branch to constrain the solution space. Simultaneously, content-descriptive text is incorporated into a positive branch and our proposed Text-Enhanced Discriminator (TED) to provide semantic guidance for enhanced texture reconstruction. Furthermore, we design a Degradation-Robust Feature Fusion (DRF) module to facilitate cross-modal feature fusion while suppressing degradation interference. Experiments on our OperaLQ benchmark show that TextOVSR outperforms state-of-the-art methods both qualitatively and quantitatively. The code is available at https://github.com/ChangHua0/TextOVSR.

  </details>



- **Low-light Image Enhancement with Retinex Decomposition in Latent Space**  
  Bolun Zheng, Qingshan Lei, Quan Chen, Qianyu Zhang, Kainan Yu, Xu Jia, Lingyu Zhu  
  _2026-03-16_ · https://arxiv.org/abs/2603.15131v1  
  <details><summary>Abstract</summary>

  Retinex theory provides a principled foundation for low-light image enhancement, inspiring numerous learning-based methods that integrate its principles. However, existing methods exhibits limitations in accurately decomposing reflectance and illumination components. To address this, we propose a Retinex-Guided Transformer~(RGT) model, which is a two-stage model consisting of decomposition and enhancement phases. First, we propose a latent space decomposition strategy to separate reflectance and illumination components. By incorporating the log transformation and 1-pixel offset, we convert the intrinsically multiplicative relationship into an additive formulation, enhancing decomposition stability and precision. Subsequently, we construct a U-shaped component refiner incorporating the proposed guidance fusion transformer block. The component refiner refines reflectance component to preserve texture details and optimize illumination distribution, effectively transforming low-light inputs to normal-light counterparts. Experimental evaluations across four benchmark datasets validate that our method achieves competitive performance in low-light enhancement and a more stable training process.

  </details>



- **VAREX: A Benchmark for Multi-Modal Structured Extraction from Documents**  
  Udi Barzelay, Ophir Azulai, Inbar Shapira, Idan Friedman, Foad Abo Dahood, Madison Lee, Abraham Daniels  
  _2026-03-16_ · https://arxiv.org/abs/2603.15118v1  
  <details><summary>Abstract</summary>

  We introduce VAREX (VARied-schema EXtraction), a benchmark for evaluating multimodal foundation models on structured data extraction from government forms. VAREX employs a Reverse Annotation pipeline that programmatically fills PDF templates with synthetic values, producing deterministic ground truth validated through three-phase quality assurance. The benchmark comprises 1,777 documents with 1,771 unique schemas across three structural categories, each provided in four input modalities: plain text, layout-preserving text (whitespace-aligned to approximate column positions), document image, or both text and image combined. Unlike existing benchmarks that evaluate from a single input representation, VAREX provides four controlled modalities per document, enabling systematic ablation of how input format affects extraction accuracy -- a capability absent from prior benchmarks. We evaluate 20 models from frontier proprietary models to small open models, with particular attention to models <=4B parameters suitable for cost-sensitive and latency-constrained deployment. Results reveal that (1) below 4B parameters, structured output compliance -- not extraction capability -- is a dominant bottleneck; in particular, schema echo (models producing schema-conforming structure instead of extracted values) depresses scores by 45-65 pp (percentage points) in affected models; (2) extraction-specific fine-tuning at 2B yields +81 pp gains, demonstrating that the instruction-following deficit is addressable without scale; (3) layout-preserving text provides the largest accuracy gain (+3-18 pp), exceeding pixel-level visual cues; and (4) the benchmark most effectively discriminates models in the 60-95% accuracy band. Dataset and evaluation code are publicly available.

  </details>



- **ReactMotion: Generating Reactive Listener Motions from Speaker Utterance**  
  Cheng Luo, Bizhu Wu, Bing Li, Jianfeng Ren, Ruibin Bai, Rong Qu, Linlin Shen, Bernard Ghanem  
  _2026-03-16_ · https://arxiv.org/abs/2603.15083v1  
  <details><summary>Abstract</summary>

  In this paper, we introduce a new task, Reactive Listener Motion Generation from Speaker Utterance, which aims to generate naturalistic listener body motions that appropriately respond to a speaker's utterance. However, modeling such nonverbal listener behaviors remains underexplored and challenging due to the inherently non-deterministic nature of human reactions. To facilitate this task, we present ReactMotionNet, a large-scale dataset that pairs speaker utterances with multiple candidate listener motions annotated with varying degrees of appropriateness. This dataset design explicitly captures the one-to-many nature of listener behavior and provides supervision beyond a single ground-truth motion. Building on this dataset design, we develop preference-oriented evaluation protocols tailored to evaluate reactive appropriateness, where conventional motion metrics focusing on input-motion alignment ignore. We further propose ReactMotion, a unified generative framework that jointly models text, audio, emotion, and motion, and is trained with preference-based objectives to encourage both appropriate and diverse listener responses. Extensive experiments show that ReactMotion outperforms retrieval baselines and cascaded LLM-based pipelines, generating more natural, diverse, and appropriate listener motions.

  </details>



- **GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents**  
  Yang Li, Yuchen Liu, Haoyu Lu, Zhiqiang Xia, Hongzhen Wang, Kaiyang Han, Changpeng Yang, Jinyang Wu, Jiaming Xu, Runyu Shi, et al.  
  _2026-03-16_ · https://arxiv.org/abs/2603.15039v1  
  <details><summary>Abstract</summary>

  Recent progress in Multimodal Large Language Models (MLLMs) has enabled mobile GUI agents capable of visual perception, cross-modal reasoning, and interactive control. However, existing benchmarks are largely English-centric and fail to capture the linguistic and interaction characteristics of the Chinese mobile ecosystem. They also focus on isolated skills such as GUI grounding or offline agent, lacking a unified and fine-grained framework to assess the full capability chain from perception to execution. To address this gap, we introduce GUI-CEval, the first comprehensive benchmark for Chinese mobile GUI agents, built entirely on physical device environments. GUI-CEval spans 201 mainstream apps across four device types and adopts a two-level structure that evaluates both atomic abilities and realistic application-level performance along five dimensions: perception, planning, reflection, execution, and evaluation. All data are collected and verified through multi-stage manual processes to ensure authenticity and reproducibility. Extensive experiments on 20 representative MLLMs and multi-agent systems show that while models such as Qwen2.5-VL and UI-TARS perform competitively, most MLLMs still exhibit clear weaknesses in reflective decision-making and post-action self-evaluation, limiting their reliability in real-world interactions. We hope GUI-CEval provides a comprehensive and interpretable benchmark to guide capability diagnosis and advance the development of Chinese mobile GUI agents.

  </details>



- **Training-free Detection of Generated Videos via Spatial-Temporal Likelihoods**  
  Omer Ben Hayun, Roy Betser, Meir Yossef Levi, Levi Kassel, Guy Gilboa  
  _2026-03-16_ · https://arxiv.org/abs/2603.15026v1  
  <details><summary>Abstract</summary>

  Following major advances in text and image generation, the video domain has surged, producing highly realistic and controllable sequences. Along with this progress, these models also raise serious concerns about misinformation, making reliable detection of synthetic videos increasingly crucial. Image-based detectors are fundamentally limited because they operate per frame and ignore temporal dynamics, while supervised video detectors generalize poorly to unseen generators, a critical drawback given the rapid emergence of new models. These challenges motivate zero-shot approaches, which avoid synthetic data and instead score content against real-data statistics, enabling training-free, model-agnostic detection. We introduce \emph{STALL}, a simple, training-free, theoretically justified detector that provides likelihood-based scoring for videos, jointly modeling spatial and temporal evidence within a probabilistic framework. We evaluate STALL on two public benchmarks and introduce ComGenVid, a new benchmark with state-of-the-art generative models. STALL consistently outperforms prior image- and video-based baselines. Code and data are available at https://omerbenhayun.github.io/stall-video.

  </details>



- **MER-Bench: A Comprehensive Benchmark for Multimodal Meme Reappraisal**  
  Yiqi Nie, Fei Wang, Junjie Chen, Kun Li, Yudi Cai, Dan Guo, Chenglong Li, Meng Wang  
  _2026-03-16_ · https://arxiv.org/abs/2603.15020v1  
  <details><summary>Abstract</summary>

  Memes represent a tightly coupled, multimodal form of social expression, in which visual context and overlaid text jointly convey nuanced affect and commentary. Inspired by cognitive reappraisal in psychology, we introduce Meme Reappraisal, a novel multimodal generation task that aims to transform negatively framed memes into constructive ones while preserving their underlying scenario, entities, and structural layout. Unlike prior works on meme understanding or generation, Meme Reappraisal requires emotion-controllable, structure-preserving multimodal transformation under multiple semantic and stylistic constraints. To support this task, we construct MER-Bench, a benchmark of real-world memes with fine-grained multimodal annotations, including source and target emotions, positively rewritten meme text, visual editing specifications, and taxonomy labels covering visual type, sentiment polarity, and layout structure. We further propose a structured evaluation framework based on a multimodal large language model (MLLM)-as-a-Judge paradigm, decomposing performance into modality-level generation quality, affect controllability, structural fidelity, and global affective alignment. Extensive experiments across representative image-editing and multimodal-generation systems reveal substantial gaps in satisfying the constraints of structural preservation, semantic consistency, and affective transformation. We believe MER-Bench establishes a foundation for research on controllable meme editing and emotion-aware multimodal generation. Our code is available at: https://github.com/one-seven17/MER-Bench.

  </details>



- **Reference-Free Omnidirectional Stereo Matching via Multi-View Consistency Maximization**  
  Lehuai Xu, Weiming Zhang, Yang Li, Sidan Du, Lin Wang  
  _2026-03-16_ · https://arxiv.org/abs/2603.15019v1  
  <details><summary>Abstract</summary>

  Reliable omnidirectional depth estimation from multi-fisheye stereo matching is pivotal to many applications, such as embodied robotics. Existing approaches either rely on spherical sweeping with heuristic fusion strategies to build the cost columns or perform reference-centric stereo matching based on rectified views. However, these methods fail to explicitly exploit geometric relationships between multiple views, rendering them less capable of capturing the global dependencies, visibility, or scale changes. In this paper, we shift to a new perspective and propose a novel reference-free framework, dubbed FreeOmniMVS, via multi-view consistency maximization. The highlight of FreeOmniMVS is that it can aggregate pair-wise correlations into a robust, visibility-aware, and global consensus. As such, it is tolerant to occlusions, partial overlaps, and varying baselines. Specifically, to achieve global coherence, we introduce a novel View-pair Correlation Transformer (VCT) that explicitly models pairwise correlation volumes across all camera view pairs, allowing us to drop unreliable pairs caused by occlusion or out-of-focus observations. To realize scalable and visibility-aware consensus, we propose a lightweight attention mechanism that adaptively fuses the correlation vectors, eliminating the need for a designated reference view and allowing all cameras to contribute equally to the stereo matching process. Extensive experiments on diverse benchmark datasets demonstrate the superiority of our method for globally consistent, visibility-aware, and scale-aware omnidirectional depth estimation.

  </details>



- **Molecular Identifier Visual Prompt and Verifiable Reinforcement Learning for Chemical Reaction Diagram Parsing**  
  Jiahe Song, Chuang Wang, Yinfan Wang, Hao Zheng, Rui Nie, Bowen Jiang, Xingjian Wei, Junyuan Gao, Yubin Wang, Bin Wang, et al.  
  _2026-03-16_ · https://arxiv.org/abs/2603.15011v1  
  <details><summary>Abstract</summary>

  Reaction diagram parsing (RxnDP) is critical for extracting chemical synthesis information from literature. Although recent Vision-Language Models (VLMs) have emerged as a promising paradigm to automate this complex visual reasoning task, their application is fundamentally bottlenecked by the inability to align visual chemical entities with pre-trained knowledge, alongside the inherent discrepancy between token-level training and reaction-level evaluation. To address these dual challenges, this work enhances VLM-based RxnDP from two complementary perspectives: prompting representation and learning paradigms. First, we propose Identifier as Visual Prompting (IdtVP), which leverages naturally occurring molecule identifiers (e.g., bold numerals like 1a) to activate the chemical knowledge acquired during VLM pre-training. IdtVP enables powerful zero-shot and out-of-distribution capabilities, outperforming existing prompting strategies. Second, to further optimize performance within fine-tuning paradigms, we introduce Re3-DAPO, a reinforcement learning algorithm that leverages verifiable rewards to directly optimize reaction-level metrics, thereby achieving consistent gains over standard supervised fine-tuning. Additionally, we release the ScannedRxn benchmark, comprising scanned historical reaction diagrams with real-world artifacts, to rigorously assess model robustness and out-of-distribution ability. Our contributions advance the accuracy and generalization of VLM-based reaction diagram parsing. We will release data, models, and code on GitHub.

  </details>



- **Thermal Image Refinement with Depth Estimation using Recurrent Networks for Monocular ORB-SLAM3**  
  Hürkan Şahin, Huy Xuan Pham, Van Huyen Dang, Alper Yegenoglu, Erdal Kayacan  
  _2026-03-16_ · https://arxiv.org/abs/2603.14998v1  
  <details><summary>Abstract</summary>

  Autonomous navigation in GPS-denied and visually degraded environments remains challenging for unmanned aerial vehicles (UAVs). To this end, we investigate the use of a monocular thermal camera as a standalone sensor on a UAV platform for real-time depth estimation and simultaneous localization and mapping (SLAM). To extract depth information from thermal images, we propose a novel pipeline employing a lightweight supervised network with recurrent blocks (RBs) integrated to capture temporal dependencies, enabling more robust predictions. The network combines lightweight convolutional backbones with a thermal refinement network (T-RefNet) to refine raw thermal inputs and enhance feature visibility. The refined thermal images and predicted depth maps are integrated into ORB-SLAM3, enabling thermal-only localization. Unlike previous methods, the network is trained on a custom non-radiometric dataset, obviating the need for high-cost radiometric thermal cameras. Experimental results on datasets and UAV flights demonstrate competitive depth accuracy and robust SLAM performance under low-light conditions. On the radiometric VIVID++ (indoor-dark) dataset, our method achieves an absolute relative error of approximately 0.06, compared to baselines exceeding 0.11. In our non-radiometric indoor set, baseline errors remain above 0.24, whereas our approach remains below 0.10. Thermal-only ORB-SLAM3 maintains a mean trajectory error under 0.4 m.

  </details>



- **MMSpec: Benchmarking Speculative Decoding for Vision-Language Models**  
  Hui Shen, Xin Wang, Ping Zhang, Yunta Hsieh, Qi Han, Zhongwei Wan, Ziheng Zhang, Jingxuan Zhang, Jing Xiong, Ziyuan Liu, et al.  
  _2026-03-16_ · https://arxiv.org/abs/2603.14989v1  
  <details><summary>Abstract</summary>

  Vision-language models (VLMs) achieve strong performance on multimodal tasks but suffer from high inference latency due to large model sizes and long multimodal contexts. Speculative decoding has recently emerged as an effective acceleration technique, yet its behavior in VLMs remains insufficiently understood. We introduce MMSpec, the first benchmark for evaluating speculative decoding in vision-language models. MMSpec contains 600 multimodal samples across six task categories and integrates ten representative speculative decoding algorithms under a unified evaluation framework. Our study reveals three key findings: (1) methods designed for text-only LLMs degrade in multimodal scenarios, (2) vision awareness becomes increasingly important at larger batch sizes, and (3) throughput speedup alone does not reliably reflect latency performance. Motivated by these findings, we propose ViSkip, a plug-and-play speculative decoding method that dynamically adapts speculation to vision tokens and achieves state-of-the-art performance.

  </details>



- **Anchoring Emotions in Text: Robust Multimodal Fusion for Mimicry Intensity Estimation**  
  Lingsi Zhu, Yuefeng Zou, Yunxiang Zhang, Naixiang Zheng, Guoyuan Wang, Jun Yu, Jiaen Liang, Wei Huang, Shengping Liu, Ximin Zheng  
  _2026-03-16_ · https://arxiv.org/abs/2603.14976v1  
  <details><summary>Abstract</summary>

  Estimating Emotional Mimicry Intensity (EMI) in naturalistic environments is a critical yet challenging task in affective computing. The primary difficulty lies in effectively modeling the complex, nonlinear temporal dynamics across highly heterogeneous modalities, especially when physical signals are corrupted or missing. To tackle this, we propose TAEMI (Text-Anchored Emotional Mimicry Intensity estimation), a novel multimodal framework designed for the 10th ABAW Competition. Motivated by the observation that continuous visual and acoustic signals are highly susceptible to transient environmental noise, we break the traditional symmetric fusion paradigm. Instead, we leverage textual transcript--which inherently encode a stable, time-independent semantic prior--as central anchors. Specifically, we introduce a Text-Anchored Dual Cross-Attention mechanism that utilizes these robust textual queries to actively filter out frame-level redundancies and align the noisy physical streams. Furthermore, to prevent catastrophic performance degradation caused by inevitably missing data in unconstrained real-world scenarios, we integrate Learnable Missing-Modality Tokens and a Modality Dropout strategy during training. Extensive experiments on the Hume-Vidmimic2 dataset demonstrate that TAEMI effectively captures fine-grained emotional variations and maintains robust predictive resilience under imperfect conditions. Our framework achieves a state-of-the-art mean Pearson correlation coefficient across six continuous emotional dimensions, significantly outperforming existing baseline methods.

  </details>



- **Learning from Mistakes: Post-Training for Driving VLA with Takeover Data**  
  Yinfeng Gao, Deqing Liu, Qichao Zhang, Yupeng Zheng, Haochen Tian, Guang Li, Hangjun Ye, Long Chen, Da-Wei Ding, Dongbin Zhao  
  _2026-03-16_ · https://arxiv.org/abs/2603.14972v1  
  <details><summary>Abstract</summary>

  Current Vision-Language-Action (VLA) paradigms in end-to-end autonomous driving rely on offline training from static datasets, leaving them vulnerable to distribution shift. Recent post-training methods use takeover data to mitigate this by augmenting the dataset with high-quality expert takeover samples, yet they suffer from two key limitations: supervision restricted to the period after the takeover moments leads to policies with limited safety margins, and passive preference optimization lacks active exploration for optimal performance. In this paper, we propose TakeVLA, a novel VLA post-training framework that overcomes these shortcomings through two complementary innovations. First, we introduce pre-takeover language supervision, which allows the VLA to learn from mistakes proactively. By explicitly teaching the model about what to do in error-prone situations, we cultivate a precautionary mindset that anticipates hazards early and substantially enlarges safety margins. Second, we propose Scenario Dreaming, a reinforcement fine-tuning paradigm that operates in reconstruceted takeover scenarios, encouraging active exploration beyond mere preference fitting. Experiments on the Bench2Drive benchmark demonstrate that TakeVLA achieves state-of-the-art closed-loop performance, surpassing the strong VLA baseline SimLingo by 4.93 in driving score, with an enhanced safety margin as evidenced by an 11.76% increase in average TTC.

  </details>



- **Pansharpening for Thin-Cloud Contaminated Remote Sensing Images: A Unified Framework and Benchmark Dataset**  
  Songcheng Du, Yang Zou, Jiaxin Li, Mingxuan Liu, Ying Li, Changjing Shang, Qiang Shen  
  _2026-03-16_ · https://arxiv.org/abs/2603.14952v1  
  <details><summary>Abstract</summary>

  Pansharpening under thin cloudy conditions is a practically significant yet rarely addressed task, challenged by simultaneous spatial resolution degradation and cloud-induced spectral distortions. Existing methods often address cloud removal and pansharpening sequentially, leading to cumulative errors and suboptimal performance due to the lack of joint degradation modeling. To address these challenges, we propose a Unified Pansharpening Model with Thin Cloud Removal (Pan-TCR), an end-to-end framework that integrates physical priors. Motivated by theoretical analysis in the frequency domain, we design a frequency-decoupled restoration (FDR) block that disentangles the restoration of multispectral image (MSI) features into amplitude and phase components, each guided by complementary degradation-robust prompts: the near-infrared (NIR) band amplitude for cloud-resilient restoration, and the panchromatic (PAN) phase for high-resolution structural enhancement. To ensure coherence between the two components, we further introduce an interactive inter-frequency consistency (IFC) module, enabling cross-modal refinement that enforces consistency and robustness across frequency cues. Furthermore, we introduce the first real-world thin-cloud contaminated pansharpening dataset (PanTCR-GF2), comprising paired clean and cloudy PAN-MSI images, to enable robust benchmarking under realistic conditions. Extensive experiments on real-world and synthetic datasets demonstrate the superiority and robustness of Pan-TCR, establishing a new benchmark for pansharpening under realistic atmospheric degradations.

  </details>



- **FAR-Drive: Frame-AutoRegressive Video Generation in Closed-Loop Autonomous Driving**  
  Yaoru Li, Federico Landi, Marco Godi, Xin Jin, Ruiju Fu, Yufei Ma, Muyang Sun, Heyu Si, Qi Guo  
  _2026-03-16_ · https://arxiv.org/abs/2603.14938v1  
  <details><summary>Abstract</summary>

  Despite rapid progress in autonomous driving, reliable training and evaluation of driving systems remain fundamentally constrained by the lack of scalable and interactive simulation environments. Recent generative video models achieve remarkable visual fidelity, yet most operate in open-loop settings and fail to support fine-grained frame-level interaction between agent actions and environment evolution. Building a learning-based closed-loop simulator for autonomous driving poses three major challenges: maintaining long-horizon temporal and cross-view consistency, mitigating autoregressive degradation under iterative self-conditioning, and satisfying low-latency inference constraints. In this work, we propose FAR-Drive, a frame-level autoregressive video generation framework for autonomous driving. We introduce a multi-view diffusion transformer with fine-grained structured control, enabling geometrically consistent multi-camera generation. To address long-horizon consistency and iterative degradation, we design a two-stage training strategy consisting of adaptive reference horizon conditioning and blend-forcing autoregressive training, which progressively improves consistency and robustness under self-conditioning. To meet low-latency interaction requirements, we further integrate system-level efficiency optimizations for inference acceleration. Experiments on the nuScenes dataset demonstrate that our method achieves state-of-the-art performance among existing closed-loop autonomous driving simulation approaches, while maintaining sub-second latency on a single GPU.

  </details>



- **Workflow-Aware Structured Layer Decomposition for Illustration Production**  
  Tianyu Zhang, Dongchi Li, Keiichi Sawada, Haoran Xie  
  _2026-03-16_ · https://arxiv.org/abs/2603.14925v1  
  <details><summary>Abstract</summary>

  Recent generative image editing methods adopt layered representations to mitigate the entangled nature of raster images and improve controllability, typically relying on object-based segmentation. However, such strategies may fail to capture the structural and stylized properties of human-created images, such as anime illustrations. To solve this issue, we propose a workflow-aware structured layer decomposition framework tailored to the illustration production of anime artwork. Inspired by the creation pipeline of anime production, our method decomposes the illustration into semantically meaningful production layers, including line art, flat color, shadow, and highlight. To decouple all these layers, we introduce lightweight layer semantic embeddings to provide specific task guidance for each layer. Furthermore, a set of layer-wise losses is incorporated to supervise the training process of individual layers. To overcome the lack of ground-truth layered data, we construct a high-quality illustration dataset that simulated the standard anime production workflow. Experiments demonstrate that the accurate and visually coherent layer decompositions were achieved by using our method. We believe that the resulting layered representation further enables downstream tasks such as recoloring and embedding texture, supporting content creation, and illustration editing. Code is available at: https://github.com/zty0304/Anime-layer-decomposition

  </details>


