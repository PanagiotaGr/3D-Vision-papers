# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-03-31 07:42 UTC_

Total papers shown: **50**


---

- **Gen-Searcher: Reinforcing Agentic Search for Image Generation**  
  Kaituo Feng, Manyuan Zhang, Shuang Chen, Yunlong Lin, Kaixuan Fan, Yilei Jiang, Hongyu Li, Dian Zheng, Chenyang Wang, Xiangyu Yue  
  _2026-03-30_ · https://arxiv.org/abs/2603.28767v1  
  <details><summary>Abstract</summary>

  Recent image generation models have shown strong capabilities in generating high-fidelity and photorealistic images. However, they are fundamentally constrained by frozen internal knowledge, thus often failing on real-world scenarios that are knowledge-intensive or require up-to-date information. In this paper, we present Gen-Searcher, as the first attempt to train a search-augmented image generation agent, which performs multi-hop reasoning and search to collect the textual knowledge and reference images needed for grounded generation. To achieve this, we construct a tailored data pipeline and curate two high-quality datasets, Gen-Searcher-SFT-10k and Gen-Searcher-RL-6k, containing diverse search-intensive prompts and corresponding ground-truth synthesis images. We further introduce KnowGen, a comprehensive benchmark that explicitly requires search-grounded external knowledge for image generation and evaluates models from multiple dimensions. Based on these resources, we train Gen-Searcher with SFT followed by agentic reinforcement learning with dual reward feedback, which combines text-based and image-based rewards to provide more stable and informative learning signals for GRPO training. Experiments show that Gen-Searcher brings substantial gains, improving Qwen-Image by around 16 points on KnowGen and 15 points on WISE. We hope this work can serve as an open foundation for search agents in image generation, and we fully open-source our data, models, and code.

  </details>



- **HandX: Scaling Bimanual Motion and Interaction Generation**  
  Zimu Zhang, Yucheng Zhang, Xiyan Xu, Ziyin Wang, Sirui Xu, Kai Zhou, Bing Zhou, Chuan Guo, Jian Wang, Yu-Xiong Wang, et al.  
  _2026-03-30_ · https://arxiv.org/abs/2603.28766v1  
  <details><summary>Abstract</summary>

  Synthesizing human motion has advanced rapidly, yet realistic hand motion and bimanual interaction remain underexplored. Whole-body models often miss the fine-grained cues that drive dexterous behavior, finger articulation, contact timing, and inter-hand coordination, and existing resources lack high-fidelity bimanual sequences that capture nuanced finger dynamics and collaboration. To fill this gap, we present HandX, a unified foundation spanning data, annotation, and evaluation. We consolidate and filter existing datasets for quality, and collect a new motion-capture dataset targeting underrepresented bimanual interactions with detailed finger dynamics. For scalable annotation, we introduce a decoupled strategy that extracts representative motion features, e.g., contact events and finger flexion, and then leverages reasoning from large language models to produce fine-grained, semantically rich descriptions aligned with these features. Building on the resulting data and annotations, we benchmark diffusion and autoregressive models with versatile conditioning modes. Experiments demonstrate high-quality dexterous motion generation, supported by our newly proposed hand-focused metrics. We further observe clear scaling trends: larger models trained on larger, higher-quality datasets produce more semantically coherent bimanual motion. Our dataset is released to support future research.

  </details>



- **PoseDreamer: Scalable and Photorealistic Human Data Generation Pipeline with Diffusion Models**  
  Lorenza Prospero, Orest Kupyn, Ostap Viniavskyi, João F. Henriques, Christian Rupprecht  
  _2026-03-30_ · https://arxiv.org/abs/2603.28763v1  
  <details><summary>Abstract</summary>

  Acquiring labeled datasets for 3D human mesh estimation is challenging due to depth ambiguities and the inherent difficulty of annotating 3D geometry from monocular images. Existing datasets are either real, with manually annotated 3D geometry and limited scale, or synthetic, rendered from 3D engines that provide precise labels but suffer from limited photorealism, low diversity, and high production costs. In this work, we explore a third path: generated data. We introduce PoseDreamer, a novel pipeline that leverages diffusion models to generate large-scale synthetic datasets with 3D mesh annotations. Our approach combines controllable image generation with Direct Preference Optimization for control alignment, curriculum-based hard sample mining, and multi-stage quality filtering. Together, these components naturally maintain correspondence between 3D labels and generated images, while prioritizing challenging samples to maximize dataset utility. Using PoseDreamer, we generate more than 500,000 high-quality synthetic samples, achieving a 76% improvement in image-quality metrics compared to rendering-based datasets. Models trained on PoseDreamer achieve performance comparable to or superior to those trained on real-world and traditional synthetic datasets. In addition, combining PoseDreamer with synthetic datasets results in better performance than combining real-world and synthetic datasets, demonstrating the complementary nature of our dataset. We will release the full dataset and generation code.

  </details>



- **SHOW3D: Capturing Scenes of 3D Hands and Objects in the Wild**  
  Patrick Rim, Kevin Harris, Braden Copple, Shangchen Han, Xu Xie, Ivan Shugurov, Sizhe An, He Wen, Alex Wong, Tomas Hodan, et al.  
  _2026-03-30_ · https://arxiv.org/abs/2603.28760v1  
  <details><summary>Abstract</summary>

  Accurate 3D understanding of human hands and objects during manipulation remains a significant challenge for egocentric computer vision. Existing hand-object interaction datasets are predominantly captured in controlled studio settings, which limits both environmental diversity and the ability of models trained on such data to generalize to real-world scenarios. To address this challenge, we introduce a novel marker-less multi-camera system that allows for nearly unconstrained mobility in genuinely in-the-wild conditions, while still having the ability to generate precise 3D annotations of hands and objects. The capture system consists of a lightweight, back-mounted, multi-camera rig that is synchronized and calibrated with a user-worn VR headset. For 3D ground-truth annotation of hands and objects, we develop an ego-exo tracking pipeline and rigorously evaluate its quality. Finally, we present SHOW3D, the first large-scale dataset with 3D annotations that show hands interacting with objects in diverse real-world environments, including outdoor settings. Our approach significantly reduces the fundamental trade-off between environmental realism and accuracy of 3D annotations, which we validate with experiments on several downstream tasks. show3d-dataset.github.io

  </details>



- **FlowIt: Global Matching for Optical Flow with Confidence-Guided Refinement**  
  Sadra Safadoust, Fabio Tosi, Matteo Poggi, Fatma Güney  
  _2026-03-30_ · https://arxiv.org/abs/2603.28759v1  
  <details><summary>Abstract</summary>

  We present FlowIt, a novel architecture for optical flow estimation designed to robustly handle large pixel displacements. At its core, FlowIt leverages a hierarchical transformer architecture that captures extensive global context, enabling the model to effectively model long-range correspondences. To overcome the limitations of localized matching, we formulate the flow initialization as an optimal transport problem. This formulation yields a highly robust initial flow field, alongside explicitly derived occlusion and confidence maps. These cues are then seamlessly integrated into a guided refinement stage, where the network actively propagates reliable motion estimates from high-confidence regions into ambiguous, low-confidence areas. Extensive experiments across the Sintel, KITTI, Spring, and LayeredFlow datasets validate the efficacy of our approach. FlowIt achieves state-of-the-art results on the competitive Sintel and KITTI benchmarks, while simultaneously establishing new state-of-the-art cross-dataset zero-shot generalization performance on Sintel, Spring, and LayeredFlow.

  </details>



- **SonoWorld: From One Image to a 3D Audio-Visual Scene**  
  Derong Jin, Xiyi Chen, Ming C. Lin, Ruohan Gao  
  _2026-03-30_ · https://arxiv.org/abs/2603.28757v1  
  <details><summary>Abstract</summary>

  Tremendous progress in visual scene generation now turns a single image into an explorable 3D world, yet immersion remains incomplete without sound. We introduce Image2AVScene, the task of generating a 3D audio-visual scene from a single image, and present SonoWorld, the first framework to tackle this challenge. From one image, our pipeline outpaints a 360° panorama, lifts it into a navigable 3D scene, places language-guided sound anchors, and renders ambisonics for point, areal, and ambient sources, yielding spatial audio aligned with scene geometry and semantics. Quantitative evaluations on a newly curated real-world dataset and a controlled user study confirm the effectiveness of our approach. Beyond free-viewpoint audio-visual rendering, we also demonstrate applications to one-shot acoustic learning and audio-visual spatial source separation. Project website: https://humathe.github.io/sonoworld/

  </details>



- **Sim-to-Real Fruit Detection Using Synthetic Data: Quantitative Evaluation and Embedded Deployment with Isaac Sim**  
  Martina Hutter-Mironovova  
  _2026-03-30_ · https://arxiv.org/abs/2603.28670v1  
  <details><summary>Abstract</summary>

  This study investigates the effectiveness of synthetic data for sim-to-real transfer in object detection under constrained data conditions and embedded deployment requirements. Synthetic datasets were generated in NVIDIA Isaac Sim and combined with limited real-world fruit images to train YOLO-based detection models under real-only, synthetic-only, and hybrid regimes. Performance was evaluated on two test datasets: an in-domain dataset with conditions matching the training data and a domain shift dataset containing real fruit and different background conditions. Results show that models trained exclusively on real data achieve the highest accuracy, while synthetic-only models exhibit reduced performance due to a domain gap. Hybrid training strategies significantly improve performance compared to synthetic-only approaches and achieve results close to real-only training while reducing the need for manual annotation. Under domain shift conditions, all models show performance degradation, with hybrid models providing improved robustness. The trained models were successfully deployed on a Jetson Orin NX using TensorRT optimization, achieving real-time inference performance. The findings highlight that synthetic data is most effective when used in combination with real data and that deployment constraints must be considered alongside detection accuracy.

  </details>



- **Industrial3D: A Terrestrial LiDAR Point Cloud Dataset and CrossParadigm Benchmark for Industrial Infrastructure**  
  Chao Yin, Hongzhe Yue, Qing Han, Difeng Hu, Zhenyu Liang, Fangzhou Lin, Bing Sun, Boyu Wang, Mingkai Li, Wei Yao, et al.  
  _2026-03-30_ · https://arxiv.org/abs/2603.28660v1  
  <details><summary>Abstract</summary>

  Automated semantic understanding of dense point clouds is a prerequisite for Scan-to-BIM pipelines, digital twin construction, and as-built verification--core tasks in the digital transformation of the construction industry. Yet for industrial mechanical, electrical, and plumbing (MEP) facilities, this challenge remains largely unsolved: TLS acquisitions of water treatment plants, chiller halls, and pumping stations exhibit extreme geometric ambiguity, severe occlusion, and extreme class imbalance that architectural benchmarks (e.g., S3DIS or ScanNet) cannot adequately represent. We present Industrial3D, a terrestrial LiDAR dataset comprising 612 million expertly labelled points at 6 mm resolution from 13 water treatment facilities. At 6.6x the scale of the closest comparable MEP dataset, Industrial3D provides the largest and most demanding testbed for industrial 3D scene understanding to date. We further establish the first industrial cross-paradigm benchmark, evaluating nine representative methods across fully supervised, weakly supervised, unsupervised, and foundation model settings under a unified benchmark protocol. The best supervised method achieves 55.74% mIoU, whereas zero-shot Point-SAM reaches only 15.79%--a 39.95 percentage-point gap that quantifies the unresolved domain-transfer challenge for industrial TLS data. Systematic analysis reveals that this gap originates from a dual crisis: statistical rarity (215:1 imbalance, 3.5x more severe than S3DIS) and geometric ambiguity (tail-class points share cylindrical primitives with head-class pipes) that frequency-based re-weighting alone cannot resolve. Industrial3D, along with benchmark code and pre-trained models, will be publicly available at https://github.com/pointcloudyc/Industrial3D.

  </details>



- **TGIF2: Extended Text-Guided Inpainting Forgery Dataset & Benchmark**  
  Hannes Mareen, Dimitrios Karageorgiou, Paschalis Giakoumoglou, Peter Lambert, Symeon Papadopoulos, Glenn Van Wallendael  
  _2026-03-30_ · https://arxiv.org/abs/2603.28613v1  
  <details><summary>Abstract</summary>

  Generative AI has made text-guided inpainting a powerful image editing tool, but at the same time a growing challenge for media forensics. Existing benchmarks, including our text-guided inpainting forgery (TGIF) dataset, show that image forgery localization (IFL) methods can localize manipulations in spliced images but struggle not in fully regenerated (FR) images, while synthetic image detection (SID) methods can detect fully regenerated images but cannot perform localization. With new generative inpainting models emerging and the open problem of localization in FR images remaining, updated datasets and benchmarks are needed. We introduce TGIF2, an extended version of TGIF, that captures recent advances in text-guided inpainting and enables a deeper analysis of forensic robustness. TGIF2 augments the original dataset with edits generated by FLUX.1 models, as well as with random non-semantic masks. Using the TGIF2 dataset, we conduct a forensic evaluation spanning IFL and SID, including fine-tuning IFL methods on FR images and generative super-resolution attacks. Our experiments show that both IFL and SID methods degrade on FLUX.1 manipulations, highlighting limited generalization. Additionally, while fine-tuning improves localization on FR images, evaluation with random non-semantic masks reveals object bias. Furthermore, generative super-resolution significantly weakens forensic traces, demonstrating that common image enhancement operations can undermine current forensic pipelines. In summary, TGIF2 provides an updated dataset and benchmark, which enables new insights into the challenges posed by modern inpainting and AI-based image enhancements. TGIF2 is available at https://github.com/IDLabMedia/tgif-dataset.

  </details>



- **ELViS: Efficient Visual Similarity from Local Descriptors that Generalizes Across Domains**  
  Pavel Suma, Giorgos Kordopatis-Zilos, Yannis Kalantidis, Giorgos Tolias  
  _2026-03-30_ · https://arxiv.org/abs/2603.28603v1  
  <details><summary>Abstract</summary>

  Large-scale instance-level training data is scarce, so models are typically trained on domain-specific datasets. Yet in real-world retrieval, they must handle diverse domains, making generalization to unseen data critical. We introduce ELViS, an image-to-image similarity model that generalizes effectively to unseen domains. Unlike conventional approaches, our model operates in similarity space rather than representation space, promoting cross-domain transfer. It leverages local descriptor correspondences, refines their similarities through an optimal transport step with data-dependent gains that suppress uninformative descriptors, and aggregates strong correspondences via a voting process into an image-level similarity. This design injects strong inductive biases, yielding a simple, efficient, and interpretable model. To assess generalization, we compile a benchmark of eight datasets spanning landmarks, artworks, products, and multi-domain collections, and evaluate ELViS as a re-ranking method. Our experiments show that ELViS outperforms competing methods by a large margin in out-of-domain scenarios and on average, while requiring only a fraction of their computational cost. Code available at: https://github.com/pavelsuma/ELViS/

  </details>



- **XSPA: Crafting Imperceptible X-Shaped Sparse Adversarial Perturbations for Transferable Attacks on VLMs**  
  Chengyin Hu, Jiaju Han, Xuemeng Sun, Qike Zhang, Yiwei Wei, Ang Li, Chunlei Meng, Xiang Chen, Jiahuan Long  
  _2026-03-30_ · https://arxiv.org/abs/2603.28568v1  
  <details><summary>Abstract</summary>

  Vision-language models (VLMs) rely on a shared visual-textual representation space to perform tasks such as zero-shot classification, image captioning, and visual question answering (VQA). While this shared space enables strong cross-task generalization, it may also introduce a common vulnerability: small visual perturbations can propagate through the shared embedding space and cause correlated semantic failures across tasks. This risk is particularly important in interactive and decision-support settings, yet it remains unclear whether VLMs are robust to highly constrained, sparse, and geometrically fixed perturbations. To address this question, we propose X-shaped Sparse Pixel Attack (XSPA), an imperceptible structured attack that restricts perturbations to two intersecting diagonal lines. Compared with dense perturbations or flexible localized patches, XSPA operates under a much stricter attack budget and thus provides a more stringent test of VLM robustness. Within this sparse support, XSPA jointly optimizes a classification objective, cross-task semantic guidance, and regularization on perturbation magnitude and along-line smoothness, inducing transferable misclassification as well as semantic drift in captioning and VQA while preserving visual subtlety. Under the default setting, XSPA modifies only about 1.76% of image pixels. Experiments on the COCO dataset show that XSPA consistently degrades performance across all three tasks. Zero-shot accuracy drops by 52.33 points on OpenAI CLIP ViT-L/14 and 67.00 points on OpenCLIP ViT-B/16, while GPT-4-evaluated caption consistency decreases by up to 58.60 points and VQA correctness by up to 44.38 points. These results suggest that even highly sparse and visually subtle perturbations with fixed geometric priors can substantially disrupt cross-task semantics in VLMs, revealing a notable robustness gap in current multimodal systems.

  </details>



- **MarkushGrapher-2: End-to-end Multimodal Recognition of Chemical Structures**  
  Tim Strohmeyer, Lucas Morin, Gerhard Ingmar Meijer, Valéry Weber, Ahmed Nassar, Peter Staar  
  _2026-03-30_ · https://arxiv.org/abs/2603.28550v1  
  <details><summary>Abstract</summary>

  Automatically extracting chemical structures from documents is essential for the large-scale analysis of the literature in chemistry. Automatic pipelines have been developed to recognize molecules represented either in figures or in text independently. However, methods for recognizing chemical structures from multimodal descriptions (Markush structures) lag behind in precision and cannot be used for automatic large-scale processing. In this work, we present MarkushGrapher-2, an end-to-end approach for the multimodal recognition of chemical structures in documents. First, our method employs a dedicated OCR model to extract text from chemical images. Second, the text, image, and layout information are jointly encoded through a Vision-Text-Layout encoder and an Optical Chemical Structure Recognition vision encoder. Finally, the resulting encodings are effectively fused through a two-stage training strategy and used to auto-regressively generate a representation of the Markush structure. To address the lack of training data, we introduce an automatic pipeline for constructing a large-scale dataset of real-world Markush structures. In addition, we present IP5-M, a large manually-annotated benchmark of real-world Markush structures, designed to advance research on this challenging task. Extensive experiments show that our approach substantially outperforms state-of-the-art models in multimodal Markush structure recognition, while maintaining strong performance in molecule structure recognition. Code, models, and datasets are released publicly.

  </details>



- **GEditBench v2: A Human-Aligned Benchmark for General Image Editing**  
  Zhangqi Jiang, Zheng Sun, Xianfang Zeng, Yufeng Yang, Xuanyang Zhang, Yongliang Wu, Wei Cheng, Gang Yu, Xu Yang, Bihan Wen  
  _2026-03-30_ · https://arxiv.org/abs/2603.28547v1  
  <details><summary>Abstract</summary>

  Recent advances in image editing have enabled models to handle complex instructions with impressive realism. However, existing evaluation frameworks lag behind: current benchmarks suffer from narrow task coverage, while standard metrics fail to adequately capture visual consistency, i.e., the preservation of identity, structure and semantic coherence between edited and original images. To address these limitations, we introduce GEditBench v2, a comprehensive benchmark with 1,200 real-world user queries spanning 23 tasks, including a dedicated open-set category for unconstrained, out-of-distribution editing instructions beyond predefined tasks. Furthermore, we propose PVC-Judge, an open-source pairwise assessment model for visual consistency, trained via two novel region-decoupled preference data synthesis pipelines. Besides, we construct VCReward-Bench using expert-annotated preference pairs to assess the alignment of PVC-Judge with human judgments on visual consistency evaluation. Experiments show that our PVC-Judge achieves state-of-the-art evaluation performance among open-source models and even surpasses GPT-5.1 on average. Finally, by benchmarking 16 frontier editing models, we show that GEditBench v2 enables more human-aligned evaluation, revealing critical limitations of current models, and providing a reliable foundation for advancing precise image editing.

  </details>



- **ManipArena: Comprehensive Real-world Evaluation of Reasoning-Oriented Generalist Robot Manipulation**  
  Yu Sun, Meng Cao, Ping Yang, Rongtao Xu, Yunxiao Yan, Runze Xu, Liang Ma, Roy Gan, Andy Zhai, Qingxuan Chen, et al.  
  _2026-03-30_ · https://arxiv.org/abs/2603.28545v1  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models and world models have recently emerged as promising paradigms for general-purpose robotic intelligence, yet their progress is hindered by the lack of reliable evaluation protocols that reflect real-world deployment. Existing benchmarks are largely simulator-centric, which provide controllability but fail to capture the reality gap caused by perception noise, complex contact dynamics, hardware constraints, and system latency. Moreover, fragmented real-world evaluations across different robot platforms prevent fair and reproducible comparison. To address these challenges, we introduce ManipArena, a standardized evaluation framework designed to bridge simulation and real-world execution. ManipArena comprises 20 diverse tasks across 10,812 expert trajectories emphasizing reasoning-oriented manipulation tasks requiring semantic and spatial reasoning, supports multi-level generalization through controlled out-of-distribution settings, and incorporates long-horizon mobile manipulation beyond tabletop scenarios. The framework further provides rich sensory diagnostics, including low-level motor signals, and synchronized real-to-sim environments constructed via high-quality 3D scanning. Together, these features enable fair, realistic, and reproducible evaluation for both VLA and world model approaches, providing a scalable foundation for diagnosing and advancing embodied intelligence systems.

  </details>



- **MRI-to-CT synthesis using drifting models**  
  Qing Lyu, Jianxu Wang, Jeremy Hudson, Ge Wang, Chirstopher T. Whitlow  
  _2026-03-30_ · https://arxiv.org/abs/2603.28498v1  
  <details><summary>Abstract</summary>

  Accurate MRI-to-CT synthesis could enable MR-only pelvic workflows by providing CT-like images with bone details while avoiding additional ionizing radiation. In this work, we investigate recently proposed drifting models for synthesizing pelvis CT images from MRI and benchmark them against convolutional neural networks (UNet, VAE), a generative adversarial network (WGAN-GP), a physics-inspired probabilistic model (PPFM), and diffusion-based methods (FastDDPM, DDIM, DDPM). Experiments are performed on two complementary datasets: Gold Atlas Male Pelvis and the SynthRAD2023 pelvis subset. Image fidelity and structural consistency are evaluated with SSIM, PSNR, and RMSE, complemented by qualitative assessment of anatomically critical regions such as cortical bone and pelvic soft-tissue interfaces. Across both datasets, the proposed drifting model achieves high SSIM and PSNR and low RMSE, surpassing strong diffusion baselines and conventional CNN-, VAE-, GAN-, and PPFM-based methods. Visual inspection shows sharper cortical bone edges, improved depiction of sacral and femoral head geometry, and reduced artifacts or over-smoothing, particularly at bone-air-soft tissue boundaries. Moreover, the drifting model attains these gains with one-step inference and inference times on the order of milliseconds, yielding a more favorable accuracy-efficiency trade-off than iterative diffusion sampling while remaining competitive in image quality. These findings suggest that drifting models are a promising direction for fast, high-quality pelvic synthetic CT generation from MRI and warrant further investigation for downstream applications such as MRI-only radiotherapy planning and PET/MR attenuation correction.

  </details>



- **CiQi-Agent: Aligning Vision, Tools and Aesthetics in Multimodal Agent for Cultural Reasoning on Chinese Porcelains**  
  Wenhan Wang, Zhixiang Zhou, Zhongtian Ma, Yanzhu Chen, Ziyu Lin, Hao Sheng, Pengfei Liu, Honglin Ma, Wenqi Shao, Qiaosheng Zhang, et al.  
  _2026-03-30_ · https://arxiv.org/abs/2603.28474v1  
  <details><summary>Abstract</summary>

  The connoisseurship of antique Chinese porcelain demands extensive historical expertise, material understanding, and aesthetic sensitivity, making it difficult for non-specialists to engage. To democratize cultural-heritage understanding and assist expert connoisseurship, we introduce CiQi-Agent -- a domain-specific Porcelain Connoisseurship Agent for intelligent analysis of antique Chinese porcelain. CiQi-Agent supports multi-image porcelain inputs and enables vision tool invocation and multimodal retrieval-augmented generation, performing fine-grained connoisseurship analysis across six attributes: dynasty, reign period, kiln site, glaze color, decorative motif, and vessel shape. Beyond attribute classification, it captures subtle visual details, retrieves relevant domain knowledge, and integrates visual and textual evidence to produce coherent, explainable connoisseurship descriptions. To achieve this capability, we construct a large-scale, expert-annotated dataset CiQi-VQA, comprising 29,596 porcelain specimens, 51,553 images, and 557,940 visual question--answering pairs, and further establish a comprehensive benchmark CiQi-Bench aligned with the previously mentioned six attributes. CiQi-Agent is trained through supervised fine-tuning, reinforcement learning, and a tool-augmented reasoning framework that integrates two categories of tools: a vision tool and multimodal retrieval tools. Experimental results show that CiQi-Agent (7B) outperforms all competitive open- and closed-source models across all six attributes on CiQi-Bench, achieving on average 12.2\% higher accuracy than GPT-5. The model and dataset have been released and are publicly available at https://huggingface.co/datasets/SII-Monument-Valley/CiQi-VQA.

  </details>



- **Active Stereo-Camera Outperforms Multi-Sensor Setup in ACT Imitation Learning for Humanoid Manipulation**  
  Robin Kühn, Moritz Schappler, Thomas Seel, Dennis Bank  
  _2026-03-30_ · https://arxiv.org/abs/2603.28422v1  
  <details><summary>Abstract</summary>

  The complexity of teaching humanoid robots new tasks is one of the major reasons hindering their widespread adoption in the industry. While Imitation Learning (IL), particularly Action Chunking with Transformers (ACT), enables rapid task acquisition, there is no consensus yet on the optimal sensory hardware required for manipulation tasks. This paper benchmarks 14 sensor combinations on the Unitree G1 humanoid robot equipped with three-finger hands for two manipulation tasks. We explicitly evaluate the integration of tactile and proprioceptive modalities alongside active vision. Our analysis demonstrates that strategic sensor selection can outperform complex configurations in data-limited regimes while reducing computational overhead. We develop an open-source Unified Ablation Framework that utilizes sensor masking on a comprehensive master dataset. Results indicate that additional modalities often degrade performance for IL with limited data. A minimal active stereo-camera setup outperformed complex multi-sensor configurations, achieving 87.5% success in a spatial generalization task and 94.4% in a structured manipulation task. Conversely, adding pressure sensors to this setup reduced success to 67.3% in the latter task due to a low signal-to-noise ratio. We conclude that in data-limited regimes, active vision offers a superior trade-off between robustness and complexity. While tactile modalities may require larger datasets to be effective, our findings validate that strategic sensor selection is critical for designing an efficient learning process.

  </details>



- **Unified Restoration-Perception Learning: Maritime Infrared-Visible Image Fusion and Segmentation**  
  Weichao Cai, Weiliang Huang, Biao Xue, Chao Huang, Fei Yuan, Bob Zhang  
  _2026-03-30_ · https://arxiv.org/abs/2603.28414v1  
  <details><summary>Abstract</summary>

  Marine scene understanding and segmentation plays a vital role in maritime monitoring and navigation safety. However, prevalent factors like fog and strong reflections in maritime environments cause severe image degradation, significantly compromising the stability of semantic perception. Existing restoration and enhancement methods typically target specific degradations or focus solely on visual quality, lacking end-to-end collaborative mechanisms that simultaneously improve structural recovery and semantic effectiveness. Moreover, publicly available infrared-visible datasets are predominantly collected from urban scenes, failing to capture the authentic characteristics of coupled degradations in marine environments. To address these challenges, the Infrared-Visible Maritime Ship Dataset (IVMSD) is proposed to cover various maritime scenarios under diverse weather and illumination conditions. Building upon this dataset, a Multi-task Complementary Learning Framework (MCLF) is proposed to collaboratively perform image restoration, multimodal fusion, and semantic segmentation within a unified architecture. The framework includes a Frequency-Spatial Enhancement Complementary (FSEC) module for degradation suppression and structural enhancement, a Semantic-Visual Consistency Attention (SVCA) module for semantic-consistent guidance, and a cross-modality guided attention mechanism for selective fusion. Experimental results on IVMSD demonstrate that the proposed method achieves state-of-the-art segmentation performance, significantly enhancing robustness and perceptual quality under complex maritime conditions.

  </details>



- **SVH-BD : Synthetic Vegetation Hyperspectral Benchmark Dataset for Emulation of Remote Sensing Images**  
  Chedly Ben Azizi, Claire Guilloteau, Gilles Roussel, Matthieu Puigt  
  _2026-03-30_ · https://arxiv.org/abs/2603.28390v1  
  <details><summary>Abstract</summary>

  This dataset provides a large collection of 10,915 synthetic hyperspectral image cubes paired with pixel-level vegetation trait maps, designed to support research in radiative transfer emulation, vegetation trait retrieval, and uncertainty quantification. Each hyperspectral cube contains 211 bands spanning 400--2500 nm at 10 nm resolution and a fixed spatial layout of 64 \times 64 pixels, offering continuous simulated surface reflectance spectra suitable for emulator development and machine-learning tasks requiring high spectral detail. Vegetation traits were derived by inverting Sentinel-2 Level-2A surface reflectance using a PROSAIL-based lookup-table approach, followed by forward PROSAIL simulations to generate hyperspectral reflectance under physically consistent canopy and illumination conditions. The dataset covers four ecologically diverse regions -- East Africa, Northern France, Eastern India, and Southern Spain -- and includes 5th and 95th percentile uncertainty maps as well as Sentinel-2 scene classification layers. This resource enables benchmarking of inversion methods, development of fast radiative transfer emulators, and studies of spectral--biophysical relationships under controlled yet realistic environmental variability.

  </details>



- **SEA: Evaluating Sketch Abstraction Efficiency via Element-level Commonsense Visual Question Answering**  
  Jiho Park, Sieun Choi, Jaeyoon Seo, Minho Sohn, Yeana Kim, Jihie Kim  
  _2026-03-30_ · https://arxiv.org/abs/2603.28363v1  
  <details><summary>Abstract</summary>

  A sketch is a distilled form of visual abstraction that conveys core concepts through simplified yet purposeful strokes while omitting extraneous detail. Despite its expressive power, quantifying the efficiency of semantic abstraction in sketches remains challenging. Existing evaluation methods that rely on reference images, low-level visual features, or recognition accuracy do not capture abstraction, the defining property of sketches. To address these limitations, we introduce SEA (Sketch Evaluation metric for Abstraction efficiency), a reference-free metric that assesses how economically a sketch represents class-defining visual elements while preserving semantic recognizability. These elements are derived per class from commonsense knowledge about features typically depicted in sketches. SEA leverages a visual question answering model to determine the presence of each element and returns a quantitative score that reflects semantic retention under visual economy. To support this metric, we present CommonSketch, the first semantically annotated sketch dataset, comprising 23,100 human-drawn sketches across 300 classes, each paired with a caption and element-level annotations. Experiments show that SEA aligns closely with human judgments and reliably discriminates levels of abstraction efficiency, while CommonSketch serves as a benchmark providing systematic evaluation of element-level sketch understanding across various vision-language models.

  </details>



- **Beyond Scanpaths: Graph-Based Gaze Simulation in Dynamic Scenes**  
  Luke Palmer, Petar Palasek, Hazem Abdelkawy  
  _2026-03-30_ · https://arxiv.org/abs/2603.28319v1  
  <details><summary>Abstract</summary>

  Accurately modelling human attention is essential for numerous computer vision applications, particularly in the domain of automotive safety. Existing methods typically collapse gaze into saliency maps or scanpaths, treating gaze dynamics only implicitly. We instead formulate gaze modelling as an autoregressive dynamical system and explicitly unroll raw gaze trajectories over time, conditioned on both gaze history and the evolving environment. Driving scenes are represented as gaze-centric graphs processed by the Affinity Relation Transformer (ART), a heterogeneous graph transformer that models interactions between driver gaze, traffic objects, and road structure. We further introduce the Object Density Network (ODN) to predict next-step gaze distributions, capturing the stochastic and object-centric nature of attentional shifts in complex environments. We also release Focus100, a new dataset of raw gaze data from 30 participants viewing egocentric driving footage. Trained directly on raw gaze, without fixation filtering, our unified approach produces more natural gaze trajectories, scanpath dynamics, and saliency maps than existing attention models, offering valuable insights for the temporal modelling of human attention in dynamic environments.

  </details>



- **DinoDental: Benchmarking DINOv3 as a Unified Vision Encoder for Dental Image Analysis**  
  Kun Tang, Xinquan Yang, Mianjie Zheng, Xuefen Liu, Xuguang Li, Xiaoqi Guo, Ruihan Chen, Linlin Shen, He Meng  
  _2026-03-30_ · https://arxiv.org/abs/2603.28297v1  
  <details><summary>Abstract</summary>

  The scarcity and high cost of expert annotations in dental imaging present a significant challenge for the development of AI in dentistry. DINOv3, a state-of-the-art, self-supervised vision foundation model pre-trained on 1.7 billion images, offers a promising pathway to mitigate this issue. However, its reliability when transferred to the dental domain, with its unique imaging characteristics and clinical subtleties, remains unclear. To address this, we introduce DinoDental, a unified benchmark designed to systematically evaluate whether DINOv3 can serve as a reliable, off-the-shelf encoder for comprehensive dental image analysis without requiring domain-specific pre-training. Constructed from multiple public datasets, DinoDental covers a wide range of tasks, including classification, detection, and instance segmentation on both panoramic radiographs and intraoral photographs. We further analyze the model's transfer performance by scaling its size and input resolution, and by comparing different adaptation strategies, including frozen features, full fine-tuning, and the parameter-efficient Low-Rank Adaptation (LoRA) method. Our experiments show that DINOv3 can serve as a strong unified encoder for dental image analysis across both panoramic radiographs and intraoral photographs, remaining competitive across tasks while showing particularly clear advantages for intraoral image understanding and boundary-sensitive dense prediction. Collectively, DinoDental provides a systematic framework for comprehensively evaluating DINOv3 in dental analysis, establishing a foundational benchmark to guide efficient and effective model selection and adaptation for the dental AI community.

  </details>



- **TerraSky3D: Multi-View Reconstructions of European Landmarks in 4K**  
  Mattia D'Urso, Yuxi Hu, Christian Sormann, Mattia Rossi, Friedrich Fraundorfer  
  _2026-03-30_ · https://arxiv.org/abs/2603.28287v1  
  <details><summary>Abstract</summary>

  Despite the growing need for data of more and more sophisticated 3D reconstruction pipelines, we can still observe a scarcity of suitable public datasets. Existing 3D datasets are either low resolution, limited to a small amount of scenes, based on images of varying quality because retrieved from the internet, or limited to specific capturing scenarios. Motivated by this lack of suitable 3D datasets, we captured TerraSky3D, a high-resolution large-scale 3D reconstruction dataset comprising 50,000 images divided into 150 ground, aerial, and mixed scenes. The dataset focuses on European landmarks and comes with curated calibration data, camera poses, and depth maps. TerraSky3D tries to answer the need for challenging dataset that can be used to train and evaluate 3D reconstruction-related pipelines.

  </details>



- **osmAG-Nav: A Hierarchical Semantic Topometric Navigation Stack for Robust Lifelong Indoor Autonomy**  
  Yongqi Zhang, Jiajie Zhang, Chengqian Li, Fujing Xie, Sören Schwertfeger  
  _2026-03-30_ · https://arxiv.org/abs/2603.28271v1  
  <details><summary>Abstract</summary>

  The deployment of mobile robots in large-scale, multi-floor environments demands navigation systems that achieve spatial scalability without compromising local kinematic precision. Traditional navigation stacks, reliant on monolithic occupancy grid maps, face severe bottlenecks in storage efficiency, cross-floor reasoning, and long-horizon planning. To address these limitations, this paper presents osmAG-Nav, a complete, open-source ROS2 navigation stack built upon the hierarchical semantic topometric OpenStreetMap Area Graph (osmAG) map standard. The system follows a "System of Systems" architecture that decouples global topological reasoning from local metric execution. A Hierarchical osmAG planner replaces dense grid searches with an LCA-anchored pipeline on a passage-centric graph whose edge costs derive from local raster traversability rather than Euclidean distance, yielding low-millisecond planning on long campus-scale routes. A Rolling Window mechanism rasterizes a fixed-size local metric grid around the robot, keeping the local costmap memory footprint independent of the total mapped area, while a Segmented Execution strategy dispatches intermediate goals to standard ROS2 controllers for smooth handoffs. System robustness is reinforced by a structure-aware LiDAR localization framework that filters dynamic clutter against permanent architectural priors. Extensive experiments on a real-world multi-story indoor-outdoor campus (>11,025 m^2) show that, on the same-floor benchmark subset, osmAG-Nav delivers up to 7816x lower planning latency than a grid-based baseline on long routes while maintaining low path-length overhead and lifelong localization stability. A single-floor long-range robot mission further validates the integrated stack reliability. The full stack is released as modular ROS2 Lifecycle Nodes.

  </details>



- **TwinMixing: A Shuffle-Aware Feature Interaction Model for Multi-Task Segmentation**  
  Minh-Khoi Do, Huy Che, Dinh-Duy Phan, Duc-Khai Lam, Duc-Lung Vu  
  _2026-03-30_ · https://arxiv.org/abs/2603.28233v1  
  <details><summary>Abstract</summary>

  Accurate and efficient perception is essential for autonomous driving, where segmentation tasks such as drivable-area and lane segmentation provide critical cues for motion planning and control. However, achieving high segmentation accuracy while maintaining real-time performance on low-cost hardware remains a challenging problem. To address this issue, we introduce TwinMixing, a lightweight multi-task segmentation model designed explicitly for drivable-area and lane segmentation. The proposed network features a shared encoder and task-specific decoders, enabling both feature sharing and task specialization. Within the encoder, we propose an Efficient Pyramid Mixing (EPM) module that enhances multi-scale feature extraction through a combination of grouped convolutions, depthwise dilated convolutions and channel shuffle operations, effectively expanding the receptive field while minimizing computational cost. Each decoder adopts a Dual-Branch Upsampling (DBU) Block composed of a learnable transposed convolution-based Fine detailed branch and a parameter-free bilinear interpolation-based Coarse grained branch, achieving detailed yet spatially consistent feature reconstruction. Extensive experiments on the BDD100K dataset validate the effectiveness of TwinMixing across three configurations - tiny, base, and large. Among them, the base configuration achieves the best trade-off between accuracy and computational efficiency, reaching 92.0% mIoU for drivable-area segmentation and 32.3% IoU for lane segmentation with only 0.43M parameters and 3.95 GFLOPs. Moreover, TwinMixing consistently outperforms existing segmentation models on the same tasks, as illustrated in Fig. 1. Thanks to its compact and modular design, TwinMixing demonstrates strong potential for real-time deployment in autonomous driving and embedded perception systems. The source code: https://github.com/Jun0se7en/TwinMixing.

  </details>



- **Ghost-FWL: A Large-Scale Full-Waveform LiDAR Dataset for Ghost Detection and Removal**  
  Kazuma Ikeda, Ryosei Hara, Rokuto Nagata, Ozora Sako. Zihao Ding, Takahiro Kado, Ibuki Fujioka, Taro Beppu, Mariko Isogawa, Kentaro Yoshioka  
  _2026-03-30_ · https://arxiv.org/abs/2603.28224v1  
  <details><summary>Abstract</summary>

  LiDAR has become an essential sensing modality in autonomous driving, robotics, and smart-city applications. However, ghost points (or ghosts), which are false reflections caused by multi-path laser returns from glass and reflective surfaces, severely degrade 3D mapping and localization accuracy. Prior ghost removal relies on geometric consistency in dense point clouds, failing on mobile LiDAR's sparse, dynamic data. We address this by exploiting full-waveform LiDAR (FWL), which captures complete temporal intensity profiles rather than just peak distances, providing crucial cues for distinguishing ghosts from genuine reflections in mobile scenarios. As this is a new task, we present Ghost-FWL, the first and largest annotated mobile FWL dataset for ghost detection and removal. Ghost-FWL comprises 24K frames across 10 diverse scenes with 7.5 billion peak-level annotations, which is 100x larger than existing annotated FWL datasets. Benefiting from this large-scale dataset, we establish a FWL-based baseline model for ghost detection and propose FWL-MAE, a masked autoencoder for efficient self-supervised representation learning on FWL data. Experiments show that our baseline outperforms existing methods in ghost removal accuracy, and our ghost removal further enhances downstream tasks such as LiDAR-based SLAM (66% trajectory error reduction) and 3D object detection (50x false positive reduction). The dataset and code is publicly available and can be accessed via the project page: https://keio-csg.github.io/Ghost-FWL

  </details>



- **Explaining CLIP Zero-shot Predictions Through Concepts**  
  Onat Ozdemir, Anders Christensen, Stephan Alaniz, Zeynep Akata, Emre Akbas  
  _2026-03-30_ · https://arxiv.org/abs/2603.28211v1  
  <details><summary>Abstract</summary>

  Large-scale vision-language models such as CLIP have achieved remarkable success in zero-shot image recognition, yet their predictions remain largely opaque to human understanding. In contrast, Concept Bottleneck Models provide interpretable intermediate representations by reasoning through human-defined concepts, but they rely on concept supervision and lack the ability to generalize to unseen classes. We introduce EZPC that bridges these two paradigms by explaining CLIP's zero-shot predictions through human-understandable concepts. Our method projects CLIP's joint image-text embeddings into a concept space learned from language descriptions, enabling faithful and transparent explanations without additional supervision. The model learns this projection via a combination of alignment and reconstruction objectives, ensuring that concept activations preserve CLIP's semantic structure while remaining interpretable. Extensive experiments on five benchmark datasets, CIFAR-100, CUB-200-2011, Places365, ImageNet-100, and ImageNet-1k, demonstrate that our approach maintains CLIP's strong zero-shot classification accuracy while providing meaningful concept-level explanations. By grounding open-vocabulary predictions in explicit semantic concepts, our method offers a principled step toward interpretable and trustworthy vision-language models. Code is available at https://github.com/oonat/ezpc.

  </details>



- **ToLL: Topological Layout Learning with Structural Multi-view Augmentation for 3D Scene Graph Pretraining**  
  Yucheng Huang, Luping Ji, Xiangwei Jiang, Wen Li, Mao Ye  
  _2026-03-30_ · https://arxiv.org/abs/2603.28178v1  
  <details><summary>Abstract</summary>

  3D Scene Graph (3DSG) generation plays a pivotal role in spatial understanding and semantic-affordance perception. However, its generalizability is often constrained by data scarcity. Current solutions primarily focus on cross-modal assisted representation learning and object-centric generation pre-training. The former relies heavily on predicate annotations, while the latter's predicate learning may be bypassed due to strong object priors. Consequently, they could not often provide a label-free and robust self-supervised proxy task for 3DSG fine-tuning. To bridge this gap, we propose a Topological Layout Learning (ToLL) for 3DSG pretraining framework. In detail, we design an Anchor-Conditioned Topological Geometry Reasoning, with a GNN to recover the global layout of zero-centered subgraphs by the spatial priors from sparse anchors. This process is strictly modulated by predicate features, thereby enforcing the predicate relation learning. Furthermore, we construct a Structural Multi-view Augmentation to avoid semantic corruption, and enhancing representations via self-distillation. The extensive experiments on 3DSSG dataset demonstrate that our ToLL could improve representation quality, outperforming state-of-the-art baselines.

  </details>



- **BlankSkip: Early-exit Object Detection onboard Nano-drones**  
  Carlo Marra, Beatrice Alessandra Motetti, Alessio Burrello, Enrico Macii, Massimo Poncino, Daniele Jahier Pagliari  
  _2026-03-30_ · https://arxiv.org/abs/2603.28149v1  
  <details><summary>Abstract</summary>

  Deploying tiny computer vision Deep Neural Networks (DNNs) on-board nano-sized drones is key for achieving autonomy, but is complicated by the extremely tight constraints of their computational platforms (approximately 10 MiB memory, 1 W power budget). Early-exit adaptive DNNs that dial down the computational effort for "easy-to-process" input frames represent a promising way to reduce the average inference latency. However, while this approach is extensively studied for classification, its application to dense tasks like object detection (OD) is not straightforward. In this paper, we propose BlankSkip, an adaptive network for on-device OD that leverages a simple auxiliary classification task for early exit, i.e., identifying frames with no objects of interest. With experiments using a real-world nano-drone platform, the Bitcraze Crazyflie 2.1, we achieve up to 24% average throughput improvement with a limited 0.015 mean Average Precision (mAP) drop compared to a static MobileNet-SSD detector, on a state-of-the-art nano-drones OD dataset.

  </details>



- **Intelligent Road Condition Monitoring using 3D In-Air SONAR Sensing**  
  Amber Cassimon, Robin Kerstens, Walter Daems, Jan Steckel  
  _2026-03-30_ · https://arxiv.org/abs/2603.28141v1  
  <details><summary>Abstract</summary>

  In this paper, we investigate the capabilities of in-air 3D SONAR sensors for the monitoring of road surface conditions. Concretely, we consider two applications: Road material classification and Road damage detection and classification. While such tasks can be performed with other sensor modalities, such as camera sensors and LiDAR sensors, these sensor modalities tend to fail in harsh sensing conditions, such as heavy rain, smoke or fog. By using a sensing modality that is robust to such interference, we enable the creation of opportunistic sensing applications, where vehicles performing other tasks (garbage collection, mail delivery, etc.) can also be used to monitor the condition of the road. For these tasks, we use a single dataset, in which different types of damages are annotated, with labels including the material of the road surface. In the material classification task, we differentiate between three different road materials: Asphalt, Concrete and Element roads. In the damage detection and classification task, we determine if there is damage, and what type of damage (independent of material type), without localizing the damage. We are succesful in determining the road surface type from SONAR sensor data, with F1 scores approaching 90% on the test set, but find that for the detection of damages performace lags, with F1 score around 75%. From this, we conclude that SONAR sensing is a promising modality to include in opportunistic sensing-based pavement management systems, but that further research is needed to reach the desired accuracy.

  </details>



- **Robust Remote Sensing Image-Text Retrieval with Noisy Correspondence**  
  Qiya Song, Yiqiang Xie, Yuan Sun, Renwei Dian, Xudong Kang  
  _2026-03-30_ · https://arxiv.org/abs/2603.28134v1  
  <details><summary>Abstract</summary>

  As a pivotal task that bridges remote visual and linguistic understanding, Remote Sensing Image-Text Retrieval (RSITR) has attracted considerable research interest in recent years. However, almost all RSITR methods implicitly assume that image-text pairs are matched perfectly. In practice, acquiring a large set of well-aligned data pairs is often prohibitively expensive or even infeasible. In addition, we also notice that the remote sensing datasets (e.g., RSITMD) truly contain some inaccurate or mismatched image text descriptions. Based on the above observations, we reveal an important but untouched problem in RSITR, i.e., Noisy Correspondence (NC). To overcome these challenges, we propose a novel Robust Remote Sensing Image-Text Retrieval (RRSITR) paradigm that designs a self-paced learning strategy to mimic human cognitive learning patterns, thereby learning from easy to hard from multi-modal data with NC. Specifically, we first divide all training sample pairs into three categories based on the loss magnitude of each pair, i.e., clean sample pairs, ambiguous sample pairs, and noisy sample pairs. Then, we respectively estimate the reliability of each training pair by assigning a weight to each pair based on the values of the loss. Further, we respectively design a new multi-modal self-paced function to dynamically regulate the training sequence and weights of the samples, thus establishing a progressive learning process. Finally, for noisy sample pairs, we present a robust triplet loss to dynamically adjust the soft margin based on semantic similarity, thereby enhancing the robustness against noise. Extensive experiments on three popular benchmark datasets demonstrate that the proposed RRSITR significantly outperforms the state-of-the-art methods, especially in high noise rates. The code is available at: https://github.com/MSFLabX/RRSITR

  </details>



- **MDPBench: A Benchmark for Multilingual Document Parsing in Real-World Scenarios**  
  Zhang Li, Zhibo Lin, Qiang Liu, Ziyang Zhang, Shuo Zhang, Zidun Guo, Jiajun Song, Jiarui Zhang, Xiang Bai, Yuliang Liu  
  _2026-03-30_ · https://arxiv.org/abs/2603.28130v1  
  <details><summary>Abstract</summary>

  We introduce Multilingual Document Parsing Benchmark, the first benchmark for multilingual digital and photographed document parsing. Document parsing has made remarkable strides, yet almost exclusively on clean, digital, well-formatted pages in a handful of dominant languages. No systematic benchmark exists to evaluate how models perform on digital and photographed documents across diverse scripts and low-resource languages. MDPBench comprises 3,400 document images spanning 17 languages, diverse scripts, and varied photographic conditions, with high-quality annotations produced through a rigorous pipeline of expert model labeling, manual correction, and human verification. To ensure fair comparison and prevent data leakage, we maintain separate public and private evaluation splits. Our comprehensive evaluation of both open-source and closed-source models uncovers a striking finding: while closed-source models (notably Gemini3-Pro) prove relatively robust, open-source alternatives suffer dramatic performance collapse, particularly on non-Latin scripts and real-world photographed documents, with an average drop of 17.8% on photographed documents and 14.0% on non-Latin scripts. These results reveal significant performance imbalances across languages and conditions, and point to concrete directions for building more inclusive, deployment-ready parsing systems. Source available at https://github.com/Yuliang-Liu/MultimodalOCR.

  </details>



- **$AutoDrive\text{-}P^3$: Unified Chain of Perception-Prediction-Planning Thought via Reinforcement Fine-Tuning**  
  Yuqi Ye, Zijian Zhang, Junhong Lin, Shangkun Sun, Changhao Peng, Wei Gao  
  _2026-03-30_ · https://arxiv.org/abs/2603.28116v1  
  <details><summary>Abstract</summary>

  Vision-language models (VLMs) are increasingly being adopted for end-to-end autonomous driving systems due to their exceptional performance in handling long-tail scenarios. However, current VLM-based approaches suffer from two major limitations: 1) Some VLMs directly output planning results without chain-of-thought (CoT) reasoning, bypassing crucial perception and prediction stages which creates a significant domain gap and compromises decision-making capability; 2) Other VLMs can generate outputs for perception, prediction, and planning tasks but employ a fragmented decision-making approach where these modules operate separately, leading to a significant lack of synergy that undermines true planning performance. To address these limitations, we propose ${AutoDrive\text{-}P^3}$, a novel framework that seamlessly integrates $\textbf{P}$erception, $\textbf{P}$rediction, and $\textbf{P}$lanning through structured reasoning. We introduce the ${P^3\text{-}CoT}$ dataset to facilitate coherent reasoning and propose ${P^3\text{-}GRPO}$, a hierarchical reinforcement learning algorithm that provides progressive supervision across all three tasks. Specifically, ${AutoDrive\text{-}P^3}$ progressively generates CoT reasoning and answers for perception, prediction, and planning, where perception provides essential information for subsequent prediction and planning, while both perception and prediction collectively contribute to the final planning decisions, enabling safer and more interpretable autonomous driving. Additionally, to balance inference efficiency with performance, we introduce dual thinking modes: detailed thinking and fast thinking. Extensive experiments on both open-loop (nuScenes) and closed-loop (NAVSIMv1/v2) benchmarks demonstrate that our approach achieves state-of-the-art performance in planning tasks. Code is available at https://github.com/haha-yuki-haha/AutoDrive-P3.

  </details>



- **Contour-Guided Query-Based Feature Fusion for Boundary-Aware and Generalizable Cardiac Ultrasound Segmentation**  
  Zahid Ullah, Sieun Choi, Jihie Kim  
  _2026-03-30_ · https://arxiv.org/abs/2603.28110v1  
  <details><summary>Abstract</summary>

  Accurate cardiac ultrasound segmentation is essential for reliable assessment of ventricular function in intelligent healthcare systems. However, echocardiographic images are challenging due to low contrast, speckle noise, irregular boundaries, and domain shifts across devices and patient populations. Existing methods, largely based on appearance-driven learning, often fail to preserve boundary precision and structural consistency under these conditions. To address these issues, we propose a Contour-Guided Query Refinement Network (CGQR-Net) for boundary-aware cardiac ultrasound segmentation. The framework integrates multi-resolution feature representations with contour-derived structural priors. An HRNet backbone preserves high-resolution spatial details while capturing multi-scale context. A coarse segmentation is first generated, from which anatomical contours are extracted and encoded into learnable query embeddings. These contour-guided queries interact with fused feature maps via cross-attention, enabling structure-aware refinement that improves boundary delineation and reduces noise artifacts. A dual-head supervision strategy jointly optimizes segmentation and boundary prediction to enforce structural consistency. The proposed method is evaluated on the CAMUS dataset and further validated on the CardiacNet dataset to assess cross-dataset generalization. Experimental results demonstrate improved segmentation accuracy, enhanced boundary precision, and robust performance across varying imaging conditions. These results highlight the effectiveness of integrating contour-level structural information with feature-level representations for reliable cardiac ultrasound segmentation.

  </details>



- **SHARP: Short-Window Streaming for Accurate and Robust Prediction in Motion Forecasting**  
  Alexander Prutsch, Christian Fruhwirth-Reisinger, David Schinagl, Horst Possegger  
  _2026-03-30_ · https://arxiv.org/abs/2603.28091v1  
  <details><summary>Abstract</summary>

  In dynamic traffic environments, motion forecasting models must be able to accurately estimate future trajectories continuously. Streaming-based methods are a promising solution, but despite recent advances, their performance often degrades when exposed to heterogeneous observation lengths. To address this, we propose a novel streaming-based motion forecasting framework that explicitly focuses on evolving scenes. Our method incrementally processes incoming observation windows and leverages an instance-aware context streaming to maintain and update latent agent representations across inference steps. A dual training objective further enables consistent forecasting accuracy across diverse observation horizons. Extensive experiments on Argoverse 2, nuScenes, and Argoverse 1 demonstrate the robustness of our approach under evolving scene conditions and also on the single-agent benchmarks. Our model achieves state-of-the-art performance in streaming inference on the Argoverse 2 multi-agent benchmark, while maintaining minimal latency, highlighting its suitability for real-world deployment.

  </details>



- **To View Transform or Not to View Transform: NeRF-based Pre-training Perspective**  
  Hyeonjun Jeong, Juyeb Shin, Dongsuk Kum  
  _2026-03-30_ · https://arxiv.org/abs/2603.28090v1  
  <details><summary>Abstract</summary>

  Neural radiance fields (NeRFs) have emerged as a prominent pre-training paradigm for vision-centric autonomous driving, which enhances 3D geometry and appearance understanding in a fully self-supervised manner. To apply NeRF-based pretraining to 3D perception models, recent approaches have simply applied NeRFs to volumetric features obtained from view transformation. However, coupling NeRFs with view transformation inherits conflicting priors; view transformation imposes discrete and rigid representations, whereas radiance fields assume continuous and adaptive functions. When these opposing assumptions are forced into a single pipeline, the misalignment surfaces as blurry and ambiguous 3D representations that ultimately limit 3D scene understanding. Moreover, the NeRF network for pre-training is discarded during downstream tasks, resulting in inefficient utilization of enhanced 3D representations through NeRF. In this paper, we propose a novel NeRF-Resembled Point-based 3D detector that can learn continuous 3D representation and thus avoid the misaligned priors from view transformation. NeRP3D preserves the pre-trained NeRF network regardless of the tasks, inheriting the principle of continuous 3D representation learning and leading to greater potentials for both scene reconstruction and detection tasks. Experiments on nuScenes dataset demonstrate that our proposed approach significantly improves previous state-of-the-art methods, outperforming not only pretext scene reconstruction tasks but also downstream detection tasks.

  </details>



- **LogiStory: A Logic-Aware Framework for Multi-Image Story Visualization**  
  Chutian Meng, Fan Ma, Chi Zhang, Jiaxu Miao, Yi Yang, Yueting Zhuang  
  _2026-03-30_ · https://arxiv.org/abs/2603.28082v1  
  <details><summary>Abstract</summary>

  Generating coherent and communicative visual sequences, such as image sequences and videos, remains a significant challenge for current multimodal systems. Despite advances in visual quality and the integration of world knowledge, existing models still struggle to maintain logical flow, often resulting in disjointed actions, fragmented narratives, and unclear storylines. We attribute these issues to the lack of attention to visual logic, a critical yet underexplored dimension of visual sequence generation that we define as the perceptual and causal coherence among characters, actions, and scenes over time. To bridge this gap, we propose a logic-aware multi-image story visualization framework, LogiStory. The framework is built around the central innovation of explicitly modeling visual logic in story visualization. To realize this idea, we design a multi-agent system that grounds roles, extracts causal chains, and verifies story-level consistency, transforming narrative coherence from an implicit byproduct of image generation into an explicit modeling objective. This design effectively bridges structured story planning with visual generation, enhancing both narrative clarity and visual quality in story visualization. Furthermore, to evaluate the generation capacity, we construct LogicTale, a benchmark comprising richly annotated stories, emphasizing causal reasoning, and visual logic interpretability. We establish comprehensive automatic and human evaluation protocols designed to measure both visual logic and perceptual quality. Experiments demonstrate that our approach significantly improves the narrative logic of generated visual stories. This work provides a foundational step towards modeling and enforcing visual logic in general image sequence and video generation tasks.

  </details>



- **AIBench: Evaluating Visual-Logical Consistency in Academic Illustration Generation**  
  Zhaohe Liao, Kaixun Jiang, Zhihang Liu, Yujie Wei, Junqiu Yu, Quanhao Li, Hong-Tao Yu, Pandeng Li, Yuzheng Wang, Zhen Xing, et al.  
  _2026-03-30_ · https://arxiv.org/abs/2603.28068v1  
  <details><summary>Abstract</summary>

  Although image generation has boosted various applications via its rapid evolution, whether the state-of-the-art models are able to produce ready-to-use academic illustrations for papers is still largely unexplored.Directly comparing or evaluating the illustration with VLM is native but requires oracle multi-modal understanding ability, which is unreliable for long and complex texts and illustrations. To address this, we propose AIBench, the first benchmark using VQA for evaluating logic correctness of the academic illustrations and VLMs for assessing aesthetics. In detail, we designed four levels of questions proposed from a logic diagram summarized from the method part of the paper, which query whether the generated illustration aligns with the paper on different scales. Our VQA-based approach raises more accurate and detailed evaluations on visual-logical consistency while relying less on the ability of the judger VLM. With our high-quality AIBench, we conduct extensive experiments and conclude that the performance gap between models on this task is significantly larger than general ones, reflecting their various complex reasoning and high-density generation ability. Further, the logic and aesthetics are hard to optimize simultaneously as in handcrafted illustrations. Additional experiments further state that test-time scaling on both abilities significantly boosts the performance on this task.

  </details>



- **Event6D: Event-based Novel Object 6D Pose Tracking**  
  Jae-Young Kang, Hoonehee Cho, Taeyeop Lee, Minjun Kang, Bowen Wen, Youngho Kim, Kuk-Jin Yoon  
  _2026-03-30_ · https://arxiv.org/abs/2603.28045v1  
  <details><summary>Abstract</summary>

  Event cameras provide microsecond latency, making them suitable for 6D object pose tracking in fast, dynamic scenes where conventional RGB and depth pipelines suffer from motion blur and large pixel displacements. We introduce EventTrack6D, an event-depth tracking framework that generalizes to novel objects without object-specific training by reconstructing both intensity and depth at arbitrary timestamps between depth frames. Conditioned on the most recent depth measurement, our dual reconstruction recovers dense photometric and geometric cues from sparse event streams. Our EventTrack6D operates at over 120 FPS and maintains temporal consistency under rapid motion. To support training and evaluation, we introduce a comprehensive benchmark suite: a large-scale synthetic dataset for training and two complementary evaluation sets, including real and simulated event datasets. Trained exclusively on synthetic data, EventTrack6D generalizes effectively to real-world scenarios without fine-tuning, maintaining accurate tracking across diverse objects and motion patterns. Our method and datasets validate the effectiveness of event cameras for event-based 6D pose tracking of novel objects. Code and datasets are publicly available at https://chohoonhee.github.io/Event6D.

  </details>



- **CARLA-Air: Fly Drones Inside a CARLA World -- A Unified Infrastructure for Air-Ground Embodied Intelligence**  
  Tianle Zeng, Hanxuan Chen, Yanci Wen, Hong Zhang  
  _2026-03-30_ · https://arxiv.org/abs/2603.28032v1  
  <details><summary>Abstract</summary>

  The convergence of low-altitude economies, embodied intelligence, and air-ground cooperative systems creates growing demand for simulation infrastructure capable of jointly modeling aerial and ground agents within a single physically coherent environment. Existing open-source platforms remain domain-segregated: driving simulators lack aerial dynamics, while multirotor simulators lack realistic ground scenes. Bridge-based co-simulation introduces synchronization overhead and cannot guarantee strict spatial-temporal consistency. We present CARLA-Air, an open-source infrastructure that unifies high-fidelity urban driving and physics-accurate multirotor flight within a single Unreal Engine process. The platform preserves both CARLA and AirSim native Python APIs and ROS 2 interfaces, enabling zero-modification code reuse. Within a shared physics tick and rendering pipeline, CARLA-Air delivers photorealistic environments with rule-compliant traffic, socially-aware pedestrians, and aerodynamically consistent UAV dynamics, synchronously capturing up to 18 sensor modalities across all platforms at each tick. The platform supports representative air-ground embodied intelligence workloads spanning cooperation, embodied navigation and vision-language action, multi-modal perception and dataset construction, and reinforcement-learning-based policy training. An extensible asset pipeline allows integration of custom robot platforms into the shared world. By inheriting AirSim's aerial capabilities -- whose upstream development has been archived -- CARLA-Air ensures this widely adopted flight stack continues to evolve within a modern infrastructure. Released with prebuilt binaries and full source: https://github.com/louiszengCN/CarlaAir

  </details>



- **Efficient Domain Adaptation for Text Line Recognition via Decoupled Language Models**  
  Arundhathi Dev, Justin Zhan  
  _2026-03-30_ · https://arxiv.org/abs/2603.28028v1  
  <details><summary>Abstract</summary>

  Optical character recognition remains critical infrastructure for document digitization, yet state-of-the-art performance is often restricted to well-resourced institutions by prohibitive computational barriers. End-to-end transformer architectures achieve strong accuracy but demand hundreds of GPU hours for domain adaptation, limiting accessibility for practitioners and digital humanities scholars. We present a modular detection-and-correction framework that achieves near-SOTA accuracy with single-GPU training. Our approach decouples lightweight visual character detection (domain-agnostic) from domain-specific linguistic correction using pretrained sequence models including T5, ByT5, and BART. By training the correctors entirely on synthetic noise, we enable annotation-free domain adaptation without requiring labeled target images. Evaluating across modern clean handwriting, cursive script, and historical documents, we identify a critical "Pareto frontier" in architecture selection: T5-Base excels on modern text with standard vocabulary, whereas ByT5-Base dominates on historical documents by reconstructing archaic spellings at the byte level. Our results demonstrate that this decoupled paradigm matches end-to-end transformer accuracy while reducing compute by approximately 95%, establishing a viable, resource-efficient alternative to monolithic OCR architectures.

  </details>



- **Physically Inspired Gaussian Splatting for HDR Novel View Synthesis**  
  Huimin Zeng, Yue Bai, Hailing Wang, Yun Fu  
  _2026-03-30_ · https://arxiv.org/abs/2603.28020v1  
  <details><summary>Abstract</summary>

  High dynamic range novel view synthesis (HDR-NVS) reconstructs scenes with dynamic details by fusing multi-exposure low dynamic range (LDR) views, yet it struggles to capture ambient illumination-dependent appearance. Implicitly supervising HDR content by constraining tone-mapped results fails in correcting abnormal HDR values, and results in limited gradients for Gaussians in under/over-exposed regions. To this end, we introduce PhysHDR-GS, a physically inspired HDR-NVS framework that models scene appearance via intrinsic reflectance and adjustable ambient illumination. PhysHDR-GS employs a complementary image-exposure (IE) branch and Gaussian-illumination (GI) branch to faithfully reproduce standard camera observations and capture illumination-dependent appearance changes, respectively. During training, the proposed cross-branch HDR consistency loss provides explicit supervision for HDR content, while an illumination-guided gradient scaling strategy mitigates exposure-biased gradient starvation and reduces under-densified representations. Experimental results across realistic and synthetic datasets demonstrate our superiority in reconstructing HDR details (e.g., a PSNR gain of 2.04 dB over HDR-GS), while maintaining real-time rendering speed (up to 76 FPS). Code and models are available at https://huimin-zeng.github.io/PhysHDR-GS/.

  </details>



- **Energy-Aware Imitation Learning for Steering Prediction Using Events and Frames**  
  Hu Cao, Jiong Liu, Xingzhuo Yan, Rui Song, Yan Xia, Walter Zimmer, Guang Chen, Alois Knoll  
  _2026-03-30_ · https://arxiv.org/abs/2603.28008v1  
  <details><summary>Abstract</summary>

  In autonomous driving, relying solely on frame-based cameras can lead to inaccuracies caused by factors like long exposure times, high-speed motion, and challenging lighting conditions. To address these issues, we introduce a bio-inspired vision sensor known as the event camera. Unlike conventional cameras, event cameras capture sparse, asynchronous events that provide a complementary modality to mitigate these challenges. In this work, we propose an energy-aware imitation learning framework for steering prediction that leverages both events and frames. Specifically, we design an Energy-driven Cross-modality Fusion Module (ECFM) and an energy-aware decoder to produce reliable and safe predictions. Extensive experiments on two public real-world datasets, DDD20 and DRFuser, demonstrate that our method outperforms existing state-of-the-art (SOTA) approaches. The codes and trained models will be released upon acceptance.

  </details>



- **UniDA3D: A Unified Domain-Adaptive Framework for Multi-View 3D Object Detection**  
  Hongjing Wu, Cheng Chi, Jinlin Wu, Yanzhao Su, Zhen Lei, Wenqi Ren  
  _2026-03-30_ · https://arxiv.org/abs/2603.27995v1  
  <details><summary>Abstract</summary>

  Camera-only 3D object detection is critical for autonomous driving, offering a cost-effective alternative to LiDAR based methods. In particular, multi-view 3D object detection has emerged as a promising direction due to its balanced trade-off between performance and cost. However, existing methods often suffer significant performance degradation under complex environmental conditions such as nighttime, fog, and rain, primarily due to their reliance on training data collected mostly in ideal conditions. To address this challenge, we propose UniDA3D, a unified domain-adaptive multi-view 3D object detector designed for robust perception under diverse adverse conditions. UniDA3D formulates nighttime, rainy, and foggy scenes as a unified multi target domain adaptation problem and leverages a novel query guided domain discrepancy mitigation (QDDM) module to align object features between source and target domains at both batch and global levels via query-centric adversarial and contrastive learning. Furthermore, we introduce a domain-adaptive teacher student training pipeline with an exponential-moving-average teacher and dynamically updated high-quality pseudo labels to enhance consistency learning and suppress background noise in unlabeled target domains. In contrast to prior approaches that require separate training for each condition, UniDA3D performs a single unified training process across multiple domains, enabling robust all-weather 3D perception. On a synthesized multi-view 3D benchmark constructed by generating nighttime, rainy, and foggy counterparts from nuScenes (nuScenes-Night, nuScenes-Rain, and nuScenes-Haze), UniDA3D consistently outperforms state of-the-art camera-only multi-view 3D detectors under extreme conditions, achieving substantial gains in mAP and NDS while maintaining real-time inference efficiency.

  </details>



- **Beyond Dataset Distillation: Lossless Dataset Concentration via Diffusion-Assisted Distribution Alignment**  
  Tongfei Liu, Yufan Liu, Bing Li, Weiming Hu  
  _2026-03-30_ · https://arxiv.org/abs/2603.27987v1  
  <details><summary>Abstract</summary>

  The high cost and accessibility problem associated with large datasets hinder the development of large-scale visual recognition systems. Dataset Distillation addresses these problems by synthesizing compact surrogate datasets for efficient training, storage, transfer, and privacy preservation. The existing state-of-the-art diffusion-based dataset distillation methods face three issues: lack of theoretical justification, poor efficiency in scaling to high data volumes, and failure in data-free scenarios. To address these issues, we establish a theoretical framework that justifies the use of diffusion models by proving the equivalence between dataset distillation and distribution matching, and reveals an inherent efficiency limit in the dataset distillation paradigm. We then propose a Dataset Concentration (DsCo) framework that uses a diffusion-based Noise-Optimization (NOpt) method to synthesize a small yet representative set of samples, and optionally augments the synthetic data via "Doping", which mixes selected samples from the original dataset with the synthetic samples to overcome the efficiency limit of dataset distillation. DsCo is applicable in both data-accessible and data-free scenarios, achieving SOTA performances for low data volumes, and it extends well to high data volumes, where it nearly reduces the dataset size by half with no performance degradation.

  </details>



- **CDH-Bench: A Commonsense-Driven Hallucination Benchmark for Evaluating Visual Fidelity in Vision-Language Models**  
  Kesheng Chen, Yamin Hu, Qi Zhou, Zhenqian Zhu, Wenjian Luo  
  _2026-03-30_ · https://arxiv.org/abs/2603.27982v1  
  <details><summary>Abstract</summary>

  Vision-language models (VLMs) achieve strong performance on many benchmarks, yet a basic reliability question remains underexplored: when visual evidence conflicts with commonsense, do models follow what is shown or what commonsense suggests? A characteristic failure in this setting is that the model overrides visual evidence and outputs the commonsense alternative. We term this phenomenon \textbf{commonsense-driven hallucination} (CDH). To evaluate it, we introduce \textbf{CDH-Bench}, a benchmark designed to create explicit \textbf{visual evidence--commonsense conflicts}. CDH-Bench covers three dimensions: \textit{counting anomalies}, \textit{relational anomalies}, and \textit{attribute anomalies}. We evaluate frontier VLMs under \textit{binary Question Answering (QA)} and \textit{multiple-choice QA}, and report metrics including \textit{Counterfactual Accuracy} (CF-Acc), \textit{Commonsense Accuracy} (CS-Acc), \textit{Counterfactual Accuracy Drop} (CFAD), \textit{Commonsense Collapse Rate} (CCR), and \textit{Relative Prior Dependency} (RPD). Results show that even strong models remain vulnerable to prior-driven normalization under visual evidence--commonsense conflict. CDH-Bench provides a controlled diagnostic of visual fidelity under visual evidence--commonsense conflict.

  </details>



- **AffordMatcher: Affordance Learning in 3D Scenes from Visual Signifiers**  
  Nghia Vu, Tuong Do, Khang Nguyen, Baoru Huang, Nhat Le, Binh Xuan Nguyen, Erman Tjiputra, Quang D. Tran, Ravi Prakash, Te-Chuan Chiu, et al.  
  _2026-03-30_ · https://arxiv.org/abs/2603.27970v1  
  <details><summary>Abstract</summary>

  Affordance learning is a complex challenge in many applications, where existing approaches primarily focus on the geometric structures, visual knowledge, and affordance labels of objects to determine interactable regions. However, extending this learning capability to a scene is significantly more complicated, as incorporating object- and scene-level semantics is not straightforward. In this work, we introduce AffordBridge, a large-scale dataset with 291,637 functional interaction annotations across 685 high-resolution indoor scenes in the form of point clouds. Our affordance annotations are complemented by RGB images that are linked to the same instances within the scenes. Building upon our dataset, we propose AffordMatcher, an affordance learning method that establishes coherent semantic correspondences between image-based and point cloud-based instances for keypoint matching, enabling a more precise identification of affordance regions based on cues, so-called visual signifiers. Experimental results on our dataset demonstrate the effectiveness of our approach compared to other methods.

  </details>



- **Learning Multi-View Spatial Reasoning from Cross-View Relations**  
  Suchae Jeong, Jaehwi Song, Haeone Lee, Hanna Kim, Jian Kim, Dongjun Lee, Dong Kyu Shin, Changyeon Kim, Dongyoon Hahm, Woogyeol Jin, et al.  
  _2026-03-30_ · https://arxiv.org/abs/2603.27967v1  
  <details><summary>Abstract</summary>

  Vision-language models (VLMs) have achieved impressive results on single-view vision tasks, but lack the multi-view spatial reasoning capabilities essential for embodied AI systems to understand 3D environments and manipulate objects across different viewpoints. In this work, we introduce Cross-View Relations (XVR), a large-scale dataset designed to teach VLMs spatial reasoning across multiple views. XVR comprises 100K vision-question-answer samples derived from 18K diverse 3D scenes and 70K robotic manipulation trajectories, spanning three fundamental spatial reasoning tasks: Correspondence (matching objects across views), Verification (validating spatial relationships), and Localization (identifying object positions). VLMs fine-tuned on XVR achieve substantial improvements on established multi-view and robotic spatial reasoning benchmarks (MindCube and RoboSpatial). When integrated as backbones in Vision-Language-Action models, XVR-trained representations improve success rates on RoboCasa. Our results demonstrate that explicit training on cross-view spatial relations significantly enhances multi-view reasoning and transfers effectively to real-world robotic manipulation.

  </details>



- **MathGen: Revealing the Illusion of Mathematical Competence through Text-to-Image Generation**  
  Ruiyao Liu, Hui Shen, Ping Zhang, Yunta Hsieh, Yifan Zhang, Jing Xu, Sicheng Chen, Junchen Li, Jiawei Lu, Jianing Ma, et al.  
  _2026-03-30_ · https://arxiv.org/abs/2603.27959v1  
  <details><summary>Abstract</summary>

  Modern generative models have demonstrated the ability to solve challenging mathematical problems. In many real-world settings, however, mathematical solutions must be expressed visually through diagrams, plots, geometric constructions, and structured symbolic layouts, where correctness depends on precise visual composition. Can generative models still do so when the answer must be rendered visually rather than written in text? To study this problem, we introduce MathGen, a rigorous benchmark of 900 problems spanning seven core domains, each paired with an executable verifier under a Script-as-a-Judge protocol for deterministic and objective evaluation. Experiments on representative open-source and proprietary text-to-image models show that mathematical fidelity remains a major bottleneck: even the best closed-source model reaches only 42.0% overall accuracy, while open-source models achieve just ~ 1-11%, often near 0% on structured tasks. Overall, current T2I models remain far from competent at even elementary mathematical visual generation.

  </details>



- **JaWildText: A Benchmark for Vision-Language Models on Japanese Scene Text Understanding**  
  Koki Maeda, Naoaki Okazaki  
  _2026-03-30_ · https://arxiv.org/abs/2603.27942v1  
  <details><summary>Abstract</summary>

  Japanese scene text poses challenges that multilingual benchmarks often fail to capture, including mixed scripts, frequent vertical writing, and a character inventory far larger than the Latin alphabet. Although Japanese is included in several multilingual benchmarks, these resources do not adequately capture the language-specific complexities. Meanwhile, existing Japanese visual text datasets have primarily focused on scanned documents, leaving in-the-wild scene text underexplored. To fill this gap, we introduce JaWildText, a diagnostic benchmark for evaluating vision-language models (VLMs) on Japanese scene text understanding. JaWildText contains 3,241 instances from 2,961 images newly captured in Japan, with 1.12 million annotated characters spanning 3,643 unique character types. It comprises three complementary tasks that vary in visual organization, output format, and writing style: (i) Dense Scene Text Visual Question Answering (STVQA), which requires reasoning over multiple pieces of visual text evidence; (ii) Receipt Key Information Extraction (KIE), which tests layout-aware structured extraction from mobile-captured receipts; and (iii) Handwriting OCR, which evaluates page-level transcription across various media and writing directions. We evaluate 14 open-weight VLMs and find that the best model achieves an average score of 0.64 across the three tasks. Error analyses show recognition remains the dominant bottleneck, especially for kanji. JaWildText enables fine-grained, script-aware diagnosis of Japanese scene text capabilities, and will be released with evaluation code.

  </details>


