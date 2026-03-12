# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-03-12 07:11 UTC_

Total papers shown: **50**


---

- **Agentar-Fin-OCR**  
  Siyi Qian, Xiongfei Bai, Bingtao Fu, Yichen Lu, Gaoyang Zhang, Xudong Yang, Peng Zhang  
  _2026-03-11_ · https://arxiv.org/abs/2603.11044v1  
  <details><summary>Abstract</summary>

  In this paper, we propose Agentar-Fin-OCR, a document parsing system tailored to financial-domain documents, transforming ultra-long financial PDFs into semantically consistent, highly accurate, structured outputs with auditing-grade provenance. To address finance-specific challenges such as complex layouts, cross-page structural discontinuities, and cell-level referencing capability, Agentar-Fin-OCR combines (1) a Cross-page Contents Consolidation algorithm to restore continuity across pages and a Document-level Heading Hierarchy Reconstruction (DHR) module to build a globally consistent Table of Contents (TOC) tree for structure-aware retrieval, and (2) a difficulty-adaptive curriculum learning training strategy for table parsing, together with a CellBBoxRegressor module that uses structural anchor tokens to localize table cells from decoder hidden states without external detectors. Experiments demonstrate that our model shows high performance on the table parsing metrics of OmniDocBench. To enable realistic evaluation in the financial vertical, we further introduce FinDocBench, a benchmark that includes six financial document categories with expert-verified annotations and evaluation metrics including Table of Contents edit-distance-based similarity (TocEDS), cross-page concatenated TEDS, and Table Cell Intersection over Union (C-IoU). We evaluate a wide range of state-of-the-art models on FinDocBench to assess their capabilities and remaining limitations on financial documents. Overall, Agentar-Fin-OCR and FinDocBench provide a practical foundation for reliable downstream financial document applications.

  </details>



- **DynVLA: Learning World Dynamics for Action Reasoning in Autonomous Driving**  
  Shuyao Shang, Bing Zhan, Yunfei Yan, Yuqi Wang, Yingyan Li, Yasong An, Xiaoman Wang, Jierui Liu, Lu Hou, Lue Fan, et al.  
  _2026-03-11_ · https://arxiv.org/abs/2603.11041v1  
  <details><summary>Abstract</summary>

  We propose DynVLA, a driving VLA model that introduces a new CoT paradigm termed Dynamics CoT. DynVLA forecasts compact world dynamics before action generation, enabling more informed and physically grounded decision-making. To obtain compact dynamics representations, DynVLA introduces a Dynamics Tokenizer that compresses future evolution into a small set of dynamics tokens. Considering the rich environment dynamics in interaction-intensive driving scenarios, DynVLA decouples ego-centric and environment-centric dynamics, yielding more accurate world dynamics modeling. We then train DynVLA to generate dynamics tokens before actions through SFT and RFT, improving decision quality while maintaining latency-efficient inference. Compared to Textual CoT, which lacks fine-grained spatiotemporal understanding, and Visual CoT, which introduces substantial redundancy due to dense image prediction, Dynamics CoT captures the evolution of the world in a compact, interpretable, and efficient form. Extensive experiments on NAVSIM, Bench2Drive, and a large-scale in-house dataset demonstrate that DynVLA consistently outperforms Textual CoT and Visual CoT methods, validating the effectiveness and practical value of Dynamics CoT.

  </details>



- **Too Vivid to Be Real? Benchmarking and Calibrating Generative Color Fidelity**  
  Zhengyao Fang, Zexi Jia, Yijia Zhong, Pengcheng Luo, Jinchao Zhang, Guangming Lu, Jun Yu, Wenjie Pei  
  _2026-03-11_ · https://arxiv.org/abs/2603.10990v1  
  <details><summary>Abstract</summary>

  Recent advances in text-to-image (T2I) generation have greatly improved visual quality, yet producing images that appear visually authentic to real-world photography remains challenging. This is partly due to biases in existing evaluation paradigms: human ratings and preference-trained metrics often favor visually vivid images with exaggerated saturation and contrast, which make generations often too vivid to be real even when prompted for realistic-style images. To address this issue, we present Color Fidelity Dataset (CFD) and Color Fidelity Metric (CFM) for objective evaluation of color fidelity in realistic-style generations. CFD contains over 1.3M real and synthetic images with ordered levels of color realism, while CFM employs a multimodal encoder to learn perceptual color fidelity. In addition, we propose a training-free Color Fidelity Refinement (CFR) that adaptively modulates spatial-temporal guidance scale in generation, thereby enhancing color authenticity. Together, CFD supports CFM for assessment, whose learned attention further guides CFR to refine T2I fidelity, forming a progressive framework for assessing and improving color fidelity in realistic-style T2I generation. The dataset and code are available at https://github.com/ZhengyaoFang/CFM.

  </details>



- **Contrastive learning-based video quality assessment-jointed video vision transformer for video recognition**  
  Jian Sun, Mohammad H. Mahoor  
  _2026-03-11_ · https://arxiv.org/abs/2603.10965v1  
  <details><summary>Abstract</summary>

  Video quality significantly affects video classification. We found this problem when we classified Mild Cognitive Impairment well from clear videos, but worse from blurred ones. From then, we realized that referring to Video Quality Assessment (VQA) may improve video classification. This paper proposed Self-Supervised Learning-based Video Vision Transformer combined with No-reference VQA for video classification (SSL-V3) to fulfill the goal. SSL-V3 leverages Combined-SSL mechanism to join VQA into video classification and address the label shortage of VQA, which commonly occurs in video datasets, making it impossible to provide an accurate Video Quality Score. In brief, Combined-SSL takes video quality score as a factor to directly tune the feature map of the video classification. Then, the score, as an intersected point, links VQA and classification, using the supervised classification task to tune the parameters of VQA. SSL-V3 achieved robust experimental results on two datasets. For example, it reached an accuracy of 94.87% on some interview videos in the I-CONECT (a facial video-involved healthcare dataset), verifying SSL-V3's effectiveness.

  </details>



- **Historical Consensus: Preventing Posterior Collapse via Iterative Selection of Gaussian Mixture Priors**  
  Zegu Zhang, Jian Zhang  
  _2026-03-11_ · https://arxiv.org/abs/2603.10935v1  
  <details><summary>Abstract</summary>

  Variational autoencoders (VAEs) frequently suffer from posterior collapse, where latent variables become uninformative and the approximate posterior degenerates to the prior. Recent work has characterized this phenomenon as a phase transition governed by the spectral properties of the data covariance matrix. In this paper, we propose a fundamentally different approach: instead of avoiding collapse through architectural constraints or hyperparameter tuning, we eliminate the possibility of collapse altogether by leveraging the multiplicity of Gaussian mixture model (GMM) clusterings. We introduce Historical Consensus Training, an iterative selection procedure that progressively refines a set of candidate GMM priors through alternating optimization and selection. The key insight is that models trained to satisfy multiple distinct clustering constraints develop a historical barrier -- a region in parameter space that remains stable even when subsequently trained with a single objective. We prove that this barrier excludes the collapsed solution, and demonstrate through extensive experiments on synthetic and real-world datasets that our method achieves non-collapsed representations regardless of decoder variance or regularization strength. Our approach requires no explicit stability conditions (e.g., $σ^{\prime 2} < λ_{\max}$) and works with arbitrary neural architectures. The code is available at https://github.com/tsegoochang/historical-consensus-vae.

  </details>



- **Bridging the Skill Gap in Clinical CBCT Interpretation with CBCTRepD**  
  Qinxin Wu, Fucheng Niu, Hengchuan Zhu, Yifan Sun, Ye Shen, Xu Li, Han Wu, Leqi Liu, Zhiwen Pan, Zuozhu Liu, et al.  
  _2026-03-11_ · https://arxiv.org/abs/2603.10933v1  
  <details><summary>Abstract</summary>

  Generative AI has advanced rapidly in medical report generation; however, its application to oral and maxillofacial CBCT reporting remains limited, largely because of the scarcity of high-quality paired CBCT-report data and the intrinsic complexity of volumetric CBCT interpretation. To address this, we introduce CBCTRepD, a bilingual oral and maxillofacial CBCT report-generation system designed for integration into routine radiologist-AI co-authoring workflows. We curated a large-scale, high-quality paired CBCT-report dataset comprising approximately 7,408 studies, covering 55 oral disease entities across diverse acquisition settings, and used it to develop the system. We further established a clinically grounded, multi-level evaluation framework that assesses both direct AI-generated drafts and radiologist-edited collaboration reports using automatic metrics together with radiologist- and clinician-centered evaluation. Using this framework, we show that CBCTRepD achieves superior report-generation performance and produces drafts with writing quality and standardization comparable to those of intermediate radiologists. More importantly, in radiologist-AI collaboration, CBCTRepD provides consistent and clinically meaningful benefits across experience levels: it helps novice radiologists improve toward intermediate-level reporting, enables intermediate radiologists to approach senior-level performance, and even assists senior radiologists by reducing omission-related errors, including clinically important missed lesions. By improving report structure, reducing omissions, and promoting attention to co-existing lesions across anatomical regions, CBCTRepD shows strong and reliable potential as a practical assistant for real-world CBCT reporting across multi-level care settings.

  </details>



- **FG-CLTP: Fine-Grained Contrastive Language Tactile Pretraining for Robotic Manipulation**  
  Wenxuan Ma, Chaofan Zhang, Yinghao Cai, Guocai Yao, Shaowei Cui, Shuo Wang  
  _2026-03-11_ · https://arxiv.org/abs/2603.10871v1  
  <details><summary>Abstract</summary>

  Recent advancements in integrating tactile sensing into vision-language-action (VLA) models have demonstrated transformative potential for robotic perception. However, existing tactile representations predominantly rely on qualitative descriptors (e.g., texture), neglecting quantitative contact states such as force magnitude, contact geometry, and principal axis orientation, which are indispensable for fine-grained manipulation. To bridge this gap, we propose FG-CLTP, a fine-grained contrastive language tactile pretraining framework. We first introduce a novel dataset comprising over 100k tactile 3D point cloud-language pairs that explicitly capture multidimensional contact states from the sensor's perspective. We then implement a discretized numerical tokenization mechanism to achieve quantitative-semantic alignment, effectively injecting explicit physical metrics into the multimodal feature space. The proposed FG-CLTP model yields a 95.9% classification accuracy and reduces the regression error (MAE) by 52.6% compared to state-of-the-art methods. Furthermore, the integration of 3D point cloud representations establishes a sensor-agnostic foundation with a minimal sim-to-real gap of 3.5%. Building upon this fine-grained representation, we develop a 3D tactile-language-action (3D-TLA) architecture driven by a flow matching policy to enable multimodal reasoning and control. Extensive experiments demonstrate that our framework significantly outperforms strong baselines in contact-rich manipulation tasks, providing a robust and generalizable foundation for tactile-language-action models.

  </details>



- **GRACE: A Unified 2D Multi-Robot Path Planning Simulator & Benchmark for Grid, Roadmap, And Continuous Environments**  
  Chuanlong Zang, Anna Mannucci, Isabelle Barz, Philipp Schillinger, Florian Lier, Wolfgang Hönig  
  _2026-03-11_ · https://arxiv.org/abs/2603.10858v1  
  <details><summary>Abstract</summary>

  Advancing Multi-Agent Pathfinding (MAPF) and Multi-Robot Motion Planning (MRMP) requires platforms that enable transparent, reproducible comparisons across modeling choices. Existing tools either scale under simplifying assumptions (grids, homogeneous agents) or offer higher fidelity with less comparable instrumentation. We present GRACE, a unified 2D simulator+benchmark that instantiates the same task at multiple abstraction levels (grid, roadmap, continuous) via explicit, reproducible operators and a common evaluation protocol. Our empirical results on public maps and representative planners enable commensurate comparisons on a shared instance set. Furthermore, we quantify the expected representation-fidelity trade-offs (MRMP solves instances at higher fidelity but lower speed, while grid/roadmap planners scale farther). By consolidating representation, execution, and evaluation, GRACE thereby aims to make cross-representation studies more comparable and provides a means to advance multi-robot planning research and its translation to practice.

  </details>



- **On the Reliability of Cue Conflict and Beyond**  
  Pum Jun Kim, Seung-Ah Lee, Seongho Park, Dongyoon Han, Jaejun Yoo  
  _2026-03-11_ · https://arxiv.org/abs/2603.10834v1  
  <details><summary>Abstract</summary>

  Understanding how neural networks rely on visual cues offers a human-interpretable view of their internal decision processes. The cue-conflict benchmark has been influential in probing shape-texture preference and in motivating the insight that stronger, human-like shape bias is often associated with improved in-domain performance. However, we find that the current stylization-based instantiation can yield unstable and ambiguous bias estimates. Specifically, stylization may not reliably instantiate perceptually valid and separable cues nor control their relative informativeness, ratio-based bias can obscure absolute cue sensitivity, and restricting evaluation to preselected classes can distort model predictions by ignoring the full decision space. Together, these factors can confound preference with cue validity, cue balance, and recognizability artifacts. We introduce REFINED-BIAS, an integrated dataset and evaluation framework for reliable and interpretable shape-texture bias diagnosis. REFINED-BIAS constructs balanced, human- and model- recognizable cue pairs using explicit definitions of shape and texture, and measures cue-specific sensitivity over the full label space via a ranking-based metric, enabling fairer cross-model comparisons. Across diverse training regimes and architectures, REFINED-BIAS enables fairer cross-model comparison, more faithful diagnosis of shape and texture biases, and clearer empirical conclusions, resolving inconsistencies that prior cue-conflict evaluations could not reliably disambiguate.

  </details>



- **Evaluating Few-Shot Pill Recognition Under Visual Domain Shift**  
  W. I. Chu, G. Tarroni, L. Li  
  _2026-03-11_ · https://arxiv.org/abs/2603.10833v1  
  <details><summary>Abstract</summary>

  Adverse drug events are a significant source of preventable harm, which has led to the development of automated pill recognition systems to enhance medication safety. Real-world deployment of these systems is hindered by visually complex conditions, including cluttered scenes, overlapping pills, reflections, and diverse acquisition environments. This study investigates few-shot pill recognition from a deployment-oriented perspective, prioritizing generalization under realistic cross-dataset domain shifts over architectural innovation. A two-stage object detection framework is employed, involving base training followed by few-shot fine-tuning. Models are adapted to novel pill classes using one, five, or ten labeled examples per class and are evaluated on a separate deployment dataset featuring multi-object, cluttered scenes. The evaluation focuses on classification-centric and error-based metrics to address heterogeneous annotation strategies. Findings indicate that semantic pill recognition adapts rapidly with few-shot supervision, with classification performance reaching saturation even with a single labeled example. However, stress testing under overlapping and occluded conditions demonstrates a marked decline in localization and recall, despite robust semantic classification. Models trained on visually realistic, multi-pill data consistently exhibit greater robustness in low-shot scenarios, underscoring the importance of training data realism and the diagnostic utility of few-shot fine-tuning for deployment readiness.

  </details>



- **BALD-SAM: Disagreement-based Active Prompting in Interactive Segmentation**  
  Prithwijit Chowdhury, Mohit Prabhushankar, Ghassan AlRegib  
  _2026-03-11_ · https://arxiv.org/abs/2603.10828v1  
  <details><summary>Abstract</summary>

  The Segment Anything Model (SAM) has revolutionized interactive segmentation through spatial prompting. While existing work primarily focuses on automating prompts in various settings, real-world annotation workflows involve iterative refinement where annotators observe model outputs and strategically place prompts to resolve ambiguities. Current pipelines typically rely on the annotator's visual assessment of the predicted mask quality. We postulate that a principled approach for automated interactive prompting is to use a model-derived criterion to identify the most informative region for the next prompt. In this work, we establish active prompting: a spatial active learning approach where locations within images constitute an unlabeled pool and prompts serve as queries to prioritize information-rich regions, increasing the utility of each interaction. We further present BALD-SAM: a principled framework adapting Bayesian Active Learning by Disagreement (BALD) to spatial prompt selection by quantifying epistemic uncertainty. To do so, we freeze the entire model and apply Bayesian uncertainty modeling only to a small learned prediction head, making intractable uncertainty estimation practical for large multi-million parameter foundation models. Across 16 datasets spanning natural, medical, underwater, and seismic domains, BALD-SAM demonstrates strong cross-domain performance, ranking first or second on 14 of 16 benchmarks. We validate these gains through a comprehensive ablation suite covering 3 SAM backbones and 35 Laplace posterior configurations, amounting to 38 distinct ablation settings. Beyond strong average performance, BALD-SAM surpasses human prompting and, in several categories, even oracle prompting, while consistently outperforming one-shot baselines in final segmentation quality, particularly on thin and structurally complex objects.

  </details>



- **A dataset of medication images with instance segmentation masks for preventing adverse drug events**  
  W. I. Chu, S. Hirani, G. Tarroni, L. Li  
  _2026-03-11_ · https://arxiv.org/abs/2603.10825v1  
  <details><summary>Abstract</summary>

  Medication errors and adverse drug events (ADEs) pose significant risks to patient safety, often arising from difficulties in reliably identifying pharmaceuticals in real-world settings. AI-based pill recognition models offer a promising solution, but the lack of comprehensive datasets hinders their development. Existing pill image datasets rarely capture real-world complexities such as overlapping pills, varied lighting, and occlusions. MEDISEG addresses this gap by providing instance segmentation annotations for 32 distinct pill types across 8262 images, encompassing diverse conditions from individual pill images to cluttered dosette boxes. We trained YOLOv8 and YOLOv9 on MEDISEG to demonstrate their usability, achieving mean average precision at IoU 0.5 of 99.5 percent on the 3-Pills subset and 80.1 percent on the 32-Pills subset. We further evaluate MEDISEG under a few-shot detection protocol, demonstrating that base training on MEDISEG significantly improves recognition of unseen pill classes in occluded multi-pill scenarios compared to existing datasets. These results highlight the dataset's ability not only to support robust supervised training but also to promote transferable representations under limited supervision, making it a valuable resource for developing and benchmarking AI-driven systems for medication safety.

  </details>



- **HanMoVLM: Large Vision-Language Models for Professional Artistic Painting Evaluation**  
  Hongji Yang, Yucheng Zhou, Wencheng Han, Songlian Li, Xiaotong Zhao, Jianbing Shen  
  _2026-03-11_ · https://arxiv.org/abs/2603.10814v1  
  <details><summary>Abstract</summary>

  While Large Vision-Language Models (VLMs) demonstrate impressive general visual capabilities, they remain artistically blind and unable to offer professional evaluation of artworks within specific artistic domains like human experts. To bridge this gap, we transform VLMs into experts capable of professional-grade painting evaluation in the Chinese Artistic Domain, which is more abstract and demands extensive artistic training for evaluation. We introduce HanMo-Bench, a new dataset that features authentic auction-grade masterpieces and AI-generated works, grounded in real-world market valuations. To realize the rigorous judgment, we propose the HanMoVLM and construct a Chain-of-Thought (CoT) validated by experts. This CoT guides the model to perform expert-level reasoning: from content identification and Region of Interest (RoI) localization to professional evaluation, guided by both theme-specific evaluation and typical three-tier evaluation in Chinese paintings. Furthermore, we design a reward function to refine the reasoning process of the HanMoVLM to improve the accuracy. We demonstrate that HanMoVLM can serve as a critical backbone for Test-time Scaling in image generation. By acting as a high-quality verifier, HanMoVLM enables generative models to select the most artistically superior outputs from multiple candidates. Experimental results and human studies confirm that the proposed HanMoVLM effectively bridges the gap, achieving a high consistency with professional experts and significantly improving the quality of Chinese Painting generation.

  </details>



- **PolGS++: Physically-Guided Polarimetric Gaussian Splatting for Fast Reflective Surface Reconstruction**  
  Yufei Han, Chu Zhou, Youwei Lyu, Qi Chen, Si Li, Boxin Shi, Yunpeng Jia, Heng Guo, Zhanyu Ma  
  _2026-03-11_ · https://arxiv.org/abs/2603.10801v1  
  <details><summary>Abstract</summary>

  Accurate reconstruction of reflective surfaces remains a fundamental challenge in computer vision, with broad applications in real-time virtual reality and digital content creation. Although 3D Gaussian Splatting (3DGS) enables efficient novel-view rendering with explicit representations, its performance on reflective surfaces still lags behind implicit neural methods, especially in recovering fine geometry and surface normals. To address this gap, we propose PolGS++, a physically-guided polarimetric Gaussian Splatting framework for fast reflective surface reconstruction. Specifically, we integrate a polarized BRDF (pBRDF) model into 3DGS to explicitly decouple diffuse and specular components, providing physically grounded reflectance modeling and stronger geometric cues for reflective surface recovery. Furthermore, we introduce a depth-guided visibility mask acquisition mechanism that enables angle-of-polarization (AoP)-based tangent-space consistency constraints in Gaussian Splatting without costly ray-tracing intersections. This physically guided design improves reconstruction quality and efficiency, requiring only about 10 minutes of training. Extensive experiments on both synthetic and real-world datasets validate the effectiveness of our method.

  </details>



- **Phase-Interface Instance Segmentation as a Visual Sensor for Laboratory Process Monitoring**  
  Mingyue Li, Xin Yang, Shilin Yan, Jinye Ran, Morui Zhu, Zirui Peng, Huanqing Peng, Wei Peng, Guanghua Zhang, Shuo Li, et al.  
  _2026-03-11_ · https://arxiv.org/abs/2603.10782v1  
  <details><summary>Abstract</summary>

  Reliable visual monitoring of chemical experiments remains challenging in transparent glassware, where weak phase boundaries and optical artifacts degrade conventional segmentation. We formulate laboratory phenomena as the time evolution of phase interfaces and introduce the Chemical Transparent Glasses dataset 2.0 (CTG 2.0), a vessel-aware benchmark with 3,668 images, 23 glassware categories, and five multiphase interface types for phase-interface instance segmentation. Building on YOLO11m-seg, we propose LGA-RCM-YOLO, which combines Local-Global Attention (LGA) for robust semantic representation and a Rectangular Self-Calibration Module (RCM) for boundary refinement of thin, elongated interfaces. On CTG 2.0, the proposed model achieves 84.4% AP@0.5 and 58.43% AP@0.5-0.95, improving over the YOLO11m baseline by 6.42 and 8.75 AP points, respectively, while maintaining near real-time inference (13.67 FPS, RTX 3060). An auxiliary color-attribute head further labels liquid instances as colored or colorless with 98.71% precision and 98.32% recall. Finally, we demonstrate continuous process monitoring in separatory-funnel phase separation and crystallization, showing that phase-interface instance segmentation can serve as a practical visual sensor for laboratory automation.

  </details>



- **CodePercept: Code-Grounded Visual STEM Perception for MLLMs**  
  Tongkun Guan, Zhibo Yang, Jianqiang Wan, Mingkun Yang, Zhengtao Guo, Zijian Hu, Ruilin Luo, Ruize Chen, Songtao Jiang, Peng Wang, et al.  
  _2026-03-11_ · https://arxiv.org/abs/2603.10757v1  
  <details><summary>Abstract</summary>

  When MLLMs fail at Science, Technology, Engineering, and Mathematics (STEM) visual reasoning, a fundamental question arises: is it due to perceptual deficiencies or reasoning limitations? Through systematic scaling analysis that independently scales perception and reasoning components, we uncover a critical insight: scaling perception consistently outperforms scaling reasoning. This reveals perception as the true lever limiting current STEM visual reasoning. Motivated by this insight, our work focuses on systematically enhancing the perception capabilities of MLLMs by establishing code as a powerful perceptual medium--executable code provides precise semantics that naturally align with the structured nature of STEM visuals. Specifically, we construct ICC-1M, a large-scale dataset comprising 1M Image-Caption-Code triplets that materializes this code-as-perception paradigm through two complementary approaches: (1) Code-Grounded Caption Generation treats executable code as ground truth for image captions, eliminating the hallucinations inherent in existing knowledge distillation methods; (2) STEM Image-to-Code Translation prompts models to generate reconstruction code, mitigating the ambiguity of natural language for perception enhancement. To validate this paradigm, we further introduce STEM2Code-Eval, a novel benchmark that directly evaluates visual perception in STEM domains. Unlike existing work relying on problem-solving accuracy as a proxy that only measures problem-relevant understanding, our benchmark requires comprehensive visual comprehension through executable code generation for image reconstruction, providing deterministic and verifiable assessment. Code is available at https://github.com/TongkunGuan/Qwen-CodePercept.

  </details>



- **Event-based Photometric Stereo via Rotating Illumination and Per-Pixel Learning**  
  Hyunwoo Kim, Won-Hoe Kim, Sanghoon Lee, Jianfei Cai, Giljoo Nam, Jae-Sang Hyun  
  _2026-03-11_ · https://arxiv.org/abs/2603.10748v1  
  <details><summary>Abstract</summary>

  Photometric stereo is a technique for estimating surface normals using images captured under varying illumination. However, conventional frame-based photometric stereo methods are limited in real-world applications due to their reliance on controlled lighting, and susceptibility to ambient illumination. To address these limitations, we propose an event-based photometric stereo system that leverages an event camera, which is effective in scenarios with continuously varying scene radiance and high dynamic range conditions. Our setup employs a single light source moving along a predefined circular trajectory, eliminating the need for multiple synchronized light sources and enabling a more compact and scalable design. We further introduce a lightweight per-pixel multi-layer neural network that directly predicts surface normals from event signals generated by intensity changes as the light source rotates, without system calibration. Experimental results on benchmark datasets and real-world data collected with our data acquisition system demonstrate the effectiveness of our method, achieving a 7.12\% reduction in mean angular error compared to existing event-based photometric stereo methods. In addition, our method demonstrates robustness in regions with sparse event activity, strong ambient illumination, and scenes affected by specularities.

  </details>



- **eLasmobranc Dataset: An Image Dataset for Elasmobranch Species Recognition and Biodiversity Monitoring**  
  Ismael Beviá-Ballesteros, Mario Jerez-Tallón, Nieves Aranda-Garrido, Isabel Abel-Abellán, Irene Antón-Linares, Jorge Azorín-López, Marcelo Saval-Calvo, Andres Fuster-Guilló, Francisca Giménez-Casalduero  
  _2026-03-11_ · https://arxiv.org/abs/2603.10724v1  
  <details><summary>Abstract</summary>

  Elasmobranch populations are experiencing significant global declines, and several species are currently classified as threatened. Reliable monitoring and species-level identification are essential to support conservation and spatial planning initiatives such as Important Shark and Ray Areas (ISRAs). However, existing visual datasets are predominantly detection-oriented, underwater-acquired, or limited to coarse-grained categories, restricting their applicability to fine-grained morphological classification. We present the eLasmobranc Dataset, a curated and publicly available image collection from seven ecologically relevant elasmobranch species inhabiting the eastern Spanish Mediterranean coast, a region where two ISRAs have been identified. Images were obtained through dedicated data collection, including field campaigns and collaborations with local fish markets and projects, as well as from open-access public sources. The dataset was constructed predominantly from images acquired outside the aquatic environment under standardized protocols to ensure clear visualization of diagnostic morphological traits. It integrates expert-validated species annotations, structured spatial and temporal metadata, and complementary species-level information. The eLasmobranc Dataset is specifically designed to support supervised species-level classification, population studies, and the development of artificial intelligence systems for biodiversity monitoring. By combining morphological clarity, taxonomic reliability, and public accessibility, the dataset addresses a critical gap in fine-grained elasmobranch identification and promotes reproducible research in conservation-oriented computer vision. The dataset is publicly available at https://zenodo.org/records/18549737.

  </details>



- **UAV traffic scene understanding: A cross-spectral guided approach and a unified benchmark**  
  Yu Zhang, Zhicheng Zhao, Ze Luo, Chenglong Li, Jin Tang  
  _2026-03-11_ · https://arxiv.org/abs/2603.10722v1  
  <details><summary>Abstract</summary>

  Traffic scene understanding from unmanned aerial vehicle (UAV) platforms is crucial for intelligent transportation systems due to its flexible deployment and wide-area monitoring capabilities. However, existing methods face significant challenges in real-world surveillance, as their heavy reliance on optical imagery leads to severe performance degradation under adverse illumination conditions like nighttime and fog. Furthermore, current Visual Question Answering (VQA) models are restricted to elementary perception tasks, lacking the domain-specific regulatory knowledge required to assess complex traffic behaviors. To address these limitations, we propose a novel Cross-spectral Traffic Cognition Network (CTCNet) for robust UAV traffic scene understanding. Specifically, we design a Prototype-Guided Knowledge Embedding (PGKE) module that leverages high-level semantic prototypes from an external Traffic Regulation Memory (TRM) to anchor domain-specific knowledge into visual representations, enabling the model to comprehend complex behaviors and distinguish fine-grained traffic violations. Moreover, we develop a Quality-Aware Spectral Compensation (QASC) module that exploits the complementary characteristics of optical and thermal modalities to perform bidirectional context exchange, effectively compensating for degraded features to ensure robust representation in complex environments. In addition, we construct Traffic-VQA, the first large-scale optical-thermal infrared benchmark for cognitive UAV traffic understanding, comprising 8,180 aligned image pairs and 1.3 million question-answer pairs across 31 diverse types. Extensive experiments demonstrate that CTCNet significantly outperforms state-of-the-art methods in both cognition and perception scenarios. The dataset is available at https://github.com/YuZhang-2004/UAV-traffic-scene-understanding.

  </details>



- **WalkGPT: Grounded Vision-Language Conversation with Depth-Aware Segmentation for Pedestrian Navigation**  
  Rafi Ibn Sultan, Hui Zhu, Xiangyu Zhou, Chengyin Li, Prashant Khanduri, Marco Brocanelli, Dongxiao Zhu  
  _2026-03-11_ · https://arxiv.org/abs/2603.10703v1  
  <details><summary>Abstract</summary>

  Ensuring accessible pedestrian navigation requires reasoning about both semantic and spatial aspects of complex urban scenes, a challenge that existing Large Vision-Language Models (LVLMs) struggle to meet. Although these models can describe visual content, their lack of explicit grounding leads to object hallucinations and unreliable depth reasoning, limiting their usefulness for accessibility guidance. We introduce WalkGPT, a pixel-grounded LVLM for the new task of Grounded Navigation Guide, unifying language reasoning and segmentation within a single architecture for depth-aware accessibility guidance. Given a pedestrian-view image and a navigation query, WalkGPT generates a conversational response with segmentation masks that delineate accessible and harmful features, along with relative depth estimation. The model incorporates a Multi-Scale Query Projector (MSQP) that shapes the final image tokens by aggregating them along text tokens across spatial hierarchies, and a Calibrated Text Projector (CTP), guided by a proposed Region Alignment Loss, that maps language embeddings into segmentation-aware representations. These components enable fine-grained grounding and depth inference without user-provided cues or anchor points, allowing the model to generate complete and realistic navigation guidance. We also introduce PAVE, a large-scale benchmark of 41k pedestrian-view images paired with accessibility-aware questions and depth-grounded answers. Experiments show that WalkGPT achieves strong grounded reasoning and segmentation performance. The source code and dataset are available on the \href{https://sites.google.com/view/walkgpt-26/home}{project website}.

  </details>



- **Bioinspired CNNs for border completion in occluded images**  
  Catarina P. Coutinho, Aneeqa Merhab, Janko Petkovic, Ferdinando Zanchetta, Rita Fioresi  
  _2026-03-11_ · https://arxiv.org/abs/2603.10694v1  
  <details><summary>Abstract</summary>

  We exploit the mathematical modeling of the border completion problem in the visual cortex to design convolutional neural network (CNN) filters that enhance robustness to image occlusions. We evaluate our CNN architecture, BorderNet, on three occluded datasets (MNIST, Fashion-MNIST, and EMNIST) under two types of occlusions: stripes and grids. In all cases, BorderNet demonstrates improved performance, with gains varying depending on the severity of the occlusions and the dataset.

  </details>



- **MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction**  
  Jonas Merkert, Alexander Blumberg, Jan-Hendrik Pauls, Christoph Stiller  
  _2026-03-11_ · https://arxiv.org/abs/2603.10688v1  
  <details><summary>Abstract</summary>

  Autonomous vehicles rely on map information to understand the world around them. However, the creation and maintenance of offline high-definition (HD) maps remains costly. A more scalable alternative lies in online HD map construction, which only requires map annotations at training time. To further reduce the need for annotating vast training labels, self-supervised training provides an alternative. This work focuses on improving the latent birds-eye-view (BEV) feature grid representation within a vectorized online HD map construction model by enforcing geospatial consistency between overlapping BEV feature grids as part of a contrastive loss function. To ensure geospatial overlap for contrastive pairs, we introduce an approach to analyze the overlap between traversals within a given dataset and generate subsidiary dataset splits following adjustable multi-traversal requirements. We train the same model supervised using a reduced set of single-traversal labeled data and self-supervised on a broader unlabeled set of data following our multi-traversal requirements, effectively implementing a semi-supervised approach. Our approach outperforms the supervised baseline across the board, both quantitatively in terms of the downstream tasks vectorized map perception performance and qualitatively in terms of segmentation in the principal component analysis (PCA) visualization of the BEV feature space.

  </details>



- **A$^2$-Edit: Precise Reference-Guided Image Editing of Arbitrary Objects and Ambiguous Masks**  
  Huayu Zheng, Guangzhao Li, Baixuan Zhao, Siqi Luo, Hantao Jiang, Guangtao Zhai, Xiaohong Liu  
  _2026-03-11_ · https://arxiv.org/abs/2603.10685v1  
  <details><summary>Abstract</summary>

  We propose \textbf{A$^2$-Edit}, a unified inpainting framework for arbitrary object categories, which allows users to replace any target region with a reference object using only a coarse mask. To address the issues of severe homogenization and limited category coverage in existing datasets, we construct a large-scale, multi-category dataset \textbf{UniEdit-500K}, which includes 8 major categories, 209 fine-grained subcategories, and a total of 500,104 image pairs. Such rich category diversity poses new challenges for the model, requiring it to automatically learn semantic relationships and distinctions across categories. To this end, we introduce the \textbf{Mixture of Transformer} module, which performs differentiated modeling of various object categories through dynamic expert selection, and further enhances cross-category semantic transfer and generalization through collaboration among experts. In addition, we propose a \textbf{Mask Annealing Training Strategy} (MATS) that progressively relaxes mask precision during training, reducing the model's reliance on accurate masks and improving robustness across diverse editing tasks. Extensive experiments on benchmarks such as VITON-HD and AnyInsertion demonstrate that A$^2$-Edit consistently outperforms existing approaches across all metrics, providing a new and efficient solution for arbitrary object editing.

  </details>



- **Are Video Reasoning Models Ready to Go Outside?**  
  Yangfan He, Changgyu Boo, Jaehong Yoon  
  _2026-03-11_ · https://arxiv.org/abs/2603.10652v1  
  <details><summary>Abstract</summary>

  In real-world deployment, vision-language models often encounter disturbances such as weather, occlusion, and camera motion. Under such conditions, their understanding and reasoning degrade substantially, revealing a gap between clean, controlled (i.e., unperturbed) evaluation settings and real-world robustness. To address this limitation, we propose ROVA, a novel training framework that improves robustness by modeling a robustness-aware consistency reward under spatio-temporal corruptions. ROVA introduces a difficulty-aware online training strategy that prioritizes informative samples based on the model's evolving capability. Specifically, it continuously re-estimates sample difficulty via self-reflective evaluation, enabling adaptive training with a robustness-aware consistency reward. We also introduce PVRBench, a new benchmark that injects real-world perturbations into embodied video datasets to assess both accuracy and reasoning quality under realistic disturbances. We evaluate ROVA and baselines on PVRBench, UrbanVideo, and VisBench, where open-source and proprietary models suffer up to 35% and 28% drops in accuracy and reasoning under realistic perturbations. ROVA effectively mitigates performance degradation, boosting relative accuracy by at least 24% and reasoning by over 9% compared with baseline models (QWen2.5/3-VL, InternVL2.5, Embodied-R). These gains transfer to clean standard benchmarks, yielding consistent improvements.

  </details>



- **AdaClearGrasp: Learning Adaptive Clearing for Zero-Shot Robust Dexterous Grasping in Densely Cluttered Environments**  
  Zixuan Chen, Wenquan Zhang, Jing Fang, Ruiming Zeng, Zhixuan Xu, Yiwen Hou, Xinke Wang, Jieqi Shi, Jing Huo, Yang Gao  
  _2026-03-11_ · https://arxiv.org/abs/2603.10616v1  
  <details><summary>Abstract</summary>

  In densely cluttered environments, physical interference, visual occlusions, and unstable contacts often cause direct dexterous grasping to fail, while aggressive singulation strategies may compromise safety. Enabling robots to adaptively decide whether to clear surrounding objects or directly grasp the target is therefore crucial for robust manipulation. We propose AdaClearGrasp, a closed-loop decision-execution framework for adaptive clearing and zero-shot dexterous grasping in densely cluttered environments. The framework formulates manipulation as a controllable high-level decision process that determines whether to directly grasp the target or first clear surrounding objects. A pretrained vision-language model (VLM) interprets visual observations and language task descriptions to reason about grasp interference and generate a high-level planning skeleton, which invokes structured atomic skills through a unified action interface. For dexterous grasping, we train a reinforcement learning policy with a relative hand-object distance representation, enabling zero-shot generalization across diverse object geometries and physical properties. During execution, visual feedback monitors outcomes and triggers replanning upon failures, forming a closed-loop correction mechanism. To evaluate language-conditioned dexterous grasping in clutter, we introduce Clutter-Bench, the first simulation benchmark with graded clutter complexity. It includes seven target objects across three clutter levels, yielding 210 task scenarios. We further perform sim-to-real experiments on three objects under three clutter levels (18 scenarios). Results demonstrate that AdaClearGrasp significantly improves grasp success rates in densely cluttered environments. For more videos and code, please visit our project website: https://chenzixuan99.github.io/adaclear-grasp.github.io/.

  </details>



- **MUNIChus: Multilingual News Image Captioning Benchmark**  
  Yuji Chen, Alistair Plum, Hansi Hettiarachchi, Diptesh Kanojia, Saroj Basnet, Marcos Zampieri, Tharindu Ranasinghe  
  _2026-03-11_ · https://arxiv.org/abs/2603.10613v1  
  <details><summary>Abstract</summary>

  The goal of news image captioning is to generate captions by integrating news article content with corresponding images, highlighting the relationship between textual context and visual elements. The majority of research on news image captioning focuses on English, primarily because datasets in other languages are scarce. To address this limitation, we create the first multilingual news image captioning benchmark, MUNIChus, comprising 9 languages, including several low-resource languages such as Sinhala and Urdu. We evaluate various state-of-the-art neural news image captioning models on MUNIChus and find that news image captioning remains challenging. We also make MUNIChus publicly available with over 20 models already benchmarked. MUNIChus opens new avenues for further advancements in developing and evaluating multilingual news image captioning models.

  </details>



- **Learning Bimanual Cloth Manipulation with Vision-based Tactile Sensing via Single Robotic Arm**  
  Dongmyoung Lee, Wei Chen, Xiaoshuai Chen, Rui Zong, Petar Kormushev  
  _2026-03-11_ · https://arxiv.org/abs/2603.10609v1  
  <details><summary>Abstract</summary>

  Robotic cloth manipulation remains challenging due to the high-dimensional state space of fabrics, their deformable nature, and frequent occlusions that limit vision-based sensing. Although dual-arm systems can mitigate some of these issues, they increase hardware and control complexity. This paper presents Touch G.O.G., a compact vision-based tactile gripper and perception/control framework for single-arm bimanual cloth manipulation. The proposed framework combines three key components: (1) a novel gripper design and control strategy for in-gripper cloth sliding with a single robot arm, (2) a Vision Foundation Model-backboned Vision Transformer pipeline for cloth part classification (PC-Net) and edge pose estimation (PE-Net) using real and synthetic tactile images, and (3) an encoder-decoder synthetic data generator (SD-Net) that reduces manual annotation by producing high-fidelity tactile images. Experiments show 96% accuracy in distinguishing edges, corners, interior regions, and grasp failures, together with sub-millimeter edge localization and 4.5° orientation error. Real-world results demonstrate reliable cloth unfolding, even for crumpled fabrics, using only a single robotic arm. These results highlight Touch G.O.G. as a compact and cost-effective solution for deformable object manipulation.

  </details>



- **Need for Speed: Zero-Shot Depth Completion with Single-Step Diffusion**  
  Jakub Gregorek, Paraskevas Pegios, Nando Metzger, Konrad Schindler, Theodora Kontogianni, Lazaros Nalpantidis  
  _2026-03-11_ · https://arxiv.org/abs/2603.10584v1  
  <details><summary>Abstract</summary>

  We introduce Marigold-SSD, a single-step, late-fusion depth completion framework that leverages strong diffusion priors while eliminating the costly test-time optimization typically associated with diffusion-based methods. By shifting computational burden from inference to finetuning, our approach enables efficient and robust 3D perception under real-world latency constraints. Marigold-SSD achieves significantly faster inference with a training cost of only 4.5 GPU days. We evaluate our method across four indoor and two outdoor benchmarks, demonstrating strong cross-domain generalization and zero-shot performance compared to existing depth completion approaches. Our approach significantly narrows the efficiency gap between diffusion-based and discriminative models. Finally, we challenge common evaluation protocols by analyzing performance under varying input sparsity levels. Page: https://dtu-pas.github.io/marigold-ssd/

  </details>



- **R4-CGQA: Retrieval-based Vision Language Models for Computer Graphics Image Quality Assessment**  
  Zhuangzi Li, Jian Jin, Shilv Cai, Weisi Lin  
  _2026-03-11_ · https://arxiv.org/abs/2603.10578v1  
  <details><summary>Abstract</summary>

  Immersive Computer Graphics (CGs) rendering has become ubiquitous in modern daily life. However, comprehensively evaluating CG quality remains challenging for two reasons: First, existing CG datasets lack systematic descriptions of rendering quality; and second existing CG quality assessment methods cannot provide reasonable text-based explanations. To address these issues, we first identify six key perceptual dimensions of CG quality from the user perspective and construct a dataset of 3500 CG images with corresponding quality descriptions. Each description covers CG style, content, and perceived quality along the selected dimensions. Furthermore, we use a subset of the dataset to build several question-answer benchmarks based on the descriptions in order to evaluate the responses of existing Vision Language Models (VLMs). We find that current VLMs are not sufficiently accurate in judging fine-grained CG quality, but that descriptions of visually similar images can significantly improve a VLM's understanding of a given CG image. Motivated by this observation, we adopt retrieval-augmented generation and propose a two-stream retrieval framework that effectively enhances the CG quality assessment capabilities of VLMs. Experiments on several representative VLMs demonstrate that our method substantially improves their performance on CG quality assessment.

  </details>



- **TacLoc: Global Tactile Localization on Objects from a Registration Perspective**  
  Zirui Zhang, Boyang Zhang, Fumin Zhang, Huan Yin  
  _2026-03-11_ · https://arxiv.org/abs/2603.10565v1  
  <details><summary>Abstract</summary>

  Pose estimation is essential for robotic manipulation, particularly when visual perception is occluded during gripper-object interactions. Existing tactile-based methods generally rely on tactile simulation or pre-trained models, which limits their generalizability and efficiency. In this study, we propose TacLoc, a novel tactile localization framework that formulates the problem as a one-shot point cloud registration task. TacLoc introduces a graph-theoretic partial-to-full registration method, leveraging dense point clouds and surface normals from tactile sensing for efficient and accurate pose estimation. Without requiring rendered data or pre-trained models, TacLoc achieves improved performance through normal-guided graph pruning and a hypothesis-and-verification pipeline. TacLoc is evaluated extensively on the YCB dataset. We further demonstrate TacLoc on real-world objects across two different visual-tactile sensors.

  </details>



- **PET-F2I: A Comprehensive Benchmark and Parameter-Efficient Fine-Tuning of LLMs for PET/CT Report Impression Generation**  
  Yuchen Liu, Wenbo Zhang, Liling Peng, Yichi Zhang, Yu Fu, Xin Guo, Chao Qu, Yuan Qi, Le Xue  
  _2026-03-11_ · https://arxiv.org/abs/2603.10560v1  
  <details><summary>Abstract</summary>

  PET/CT imaging is pivotal in oncology and nuclear medicine, yet summarizing complex findings into precise diagnostic impressions is labor-intensive. While LLMs have shown promise in medical text generation, their capability in the highly specialized domain of PET/CT remains underexplored. We introduce PET-F2I-41K (PET Findings-to-Impression Benchmark), a large-scale benchmark for PET/CT impression generation using LLMs, constructed from over 41k real-world reports. Using PET-F2I-41K, we conduct a comprehensive evaluation of 27 models across proprietary frontier LLMs, open-source generalist models, and medical-domain LLMs, and we develop a domain-adapted 7B model (PET-F2I-7B) fine-tuned from Qwen2.5-7B-Instruct via LoRA. Beyond standard NLG metrics (e.g., BLEU-4, ROUGE-L, BERTScore), we propose three clinically grounded metrics - Entity Coverage Rate (ECR), Uncovered Entity Rate (UER), and Factual Consistency Rate (FCR) - to assess diagnostic completeness and factual reliability. Experiments reveal that neither frontier nor medical-domain LLMs perform adequately in zero-shot settings. In contrast, PET-F2I-7B achieves substantial gains (e.g., 0.708 BLEU-4) and a 3.0x improvement in entity coverage over the strongest baseline, while offering advantages in cost, latency, and privacy. Beyond this modeling contribution, PET-F2I-41K establishes a standardized evaluation framework to accelerate the development of reliable and clinically deployable reporting systems for PET/CT.

  </details>



- **Prompting with the human-touch: evaluating model-sensitivity of foundation models for musculoskeletal CT segmentation**  
  Caroline Magg, Maaike A. ter Wee, Johannes G. G. Dobbe, Geert J. Streekstra, Leendert Blankevoort, Clara I. Sánchez, Hoel Kervadec  
  _2026-03-11_ · https://arxiv.org/abs/2603.10541v1  
  <details><summary>Abstract</summary>

  Promptable Foundation Models (FMs), initially introduced for natural image segmentation, have also revolutionized medical image segmentation. The increasing number of models, along with evaluations varying in datasets, metrics, and compared models, makes direct performance comparison between models difficult and complicates the selection of the most suitable model for specific clinical tasks. In our study, 11 promptable FMs are tested using non-iterative 2D and 3D prompting strategies on a private and public dataset focusing on bone and implant segmentation in four anatomical regions (wrist, shoulder, hip and lower leg). The Pareto-optimal models are identified and further analyzed using human prompts collected through a dedicated observer study. Our findings are: 1) The segmentation performance varies a lot between FMs and prompting strategies; 2) The Pareto-optimal models in 2D are SAM and SAM2.1, in 3D nnInteractive and Med-SAM2; 3) Localization accuracy and rater consistency vary with anatomical structures, with higher consistency for simple structures (wrist bones) and lower consistency for complex structures (pelvis, tibia, implants); 4) The segmentation performance drops using human prompts, suggesting that performance reported on "ideal" prompts extracted from reference labels might overestimate the performance in a human-driven setting; 5) All models were sensitive to prompt variations. While two models demonstrated intra-rater robustness, it did not scale to inter-rater settings. We conclude that the selection of the most optimal FM for a human-driven setting remains challenging, with even high-performing FMs being sensitive to variations in human input prompts. Our code base for prompt extraction and model inference is available: https://github.com/CarolineMagg/segmentation-FM-benchmark/

  </details>



- **IMTBench: A Multi-Scenario Cross-Modal Collaborative Evaluation Benchmark for In-Image Machine Translation**  
  Jiahao Lyu, Pei Fu, Zhenhang Li, Weichao Zeng, Shaojie Zhan, Jiahui Yang, Can Ma, Yu Zhou, Zhenbo Luo, Jian Luan  
  _2026-03-11_ · https://arxiv.org/abs/2603.10495v1  
  <details><summary>Abstract</summary>

  End-to-end In-Image Machine Translation (IIMT) aims to convert text embedded within an image into a target language while preserving the original visual context, layout, and rendering style. However, existing IIMT benchmarks are largely synthetic and thus fail to reflect real-world complexity, while current evaluation protocols focus on single-modality metrics and overlook cross-modal faithfulness between rendered text and model outputs. To address these shortcomings, we present In-image Machine Translation Benchmark (IMTBench), a new benchmark of 2,500 image translation samples covering four practical scenarios and nine languages. IMTBench supports multi-aspect evaluation, including translation quality, background preservation, overall image quality, and a cross-modal alignment score that measures consistency between the translated text produced by the model and the text rendered in the translated image. We benchmark strong commercial cascade systems, and both closed- and open-source unified multi-modal models, and observe large performance gaps across scenarios and languages, especially on natural scenes and resource-limited languages, highlighting substantial headroom for end-to-end image text translation. We hope IMTBench establishes a standardized benchmark to accelerate progress in this emerging task.

  </details>



- **StructDamage:A Large Scale Unified Crack and Surface Defect Dataset for Robust Structural Damage Detection**  
  Misbah Ijaz, Saif Ur Rehman Khan, Abd Ur Rehman, Sebastian Vollmer, Andreas Dengel, Muhammad Nabeel Asim  
  _2026-03-11_ · https://arxiv.org/abs/2603.10484v1  
  <details><summary>Abstract</summary>

  Automated detection and classification of structural cracks and surface defects is a critical challenge in civil engineering, infrastructure maintenance, and heritage preservation. Recent advances in Computer Vision (CV) and Deep Learning (DL) have significantly improved automatic crack detection. However, these methods rely heavily on large, diverse, and carefully curated datasets that include various crack types across different surface materials. Many existing public crack datasets lack geographic diversity, surface types, scale, and labeling consistency, making it challenging for trained algorithms to generalize effectively in real world conditions. We provide a novel dataset, StructDamage, a curated collection of approximately 78,093 images spanning nine surface types: walls, tile, stone, road, pavement, deck, concrete, and brick. The dataset was constructed by systematically aggregating, harmonizing, and reannotating images from 32 publicly available datasets covering concrete structures, asphalt pavements, masonry walls, bridges, and historic buildings. All images are organized in a folder level classification hierarchy suitable for training Convolutional Neural Networks (CNNs) and Vision Transformers. To highlight the practical value of the dataset, we present baseline classification results using fifteen DL architectures from six model families, with twelve achieving macro F1-scores over 0.96. The best performing model DenseNet201 achieves 98.62% accuracy. The proposed dataset provides a comprehensive and versatile resource suitable for classification tasks. With thorough documentation and a standard structure, it is designed to promote reproducible research and support the development and fair evaluation of robust crack damage detection approaches.

  </details>



- **Fighting Hallucinations with Counterfactuals: Diffusion-Guided Perturbations for LVLM Hallucination Suppression**  
  Hamidreza Dastmalchi, Aijun An, Ali Cheraghian, Hamed Barzamini  
  _2026-03-11_ · https://arxiv.org/abs/2603.10470v1  
  <details><summary>Abstract</summary>

  While large vision-language models (LVLMs) achieve strong performance on multimodal tasks, they frequently generate hallucinations -- unfaithful outputs misaligned with the visual input. To address this issue, we introduce CIPHER (Counterfactual Image Perturbations for Hallucination Extraction and Removal), a training-free method that suppresses vision-induced hallucinations via lightweight feature-level correction. Unlike prior training-free approaches that primarily focus on text-induced hallucinations, CIPHER explicitly targets hallucinations arising from the visual modality. CIPHER operates in two phases. In the offline phase, we construct OHC-25K (Object-Hallucinated Counterfactuals, 25,000 samples), a counterfactual dataset consisting of diffusion-edited images that intentionally contradict the original ground-truth captions. We pair these edited images with the unchanged ground-truth captions and process them through an LVLM to extract hallucination-related representations. Contrasting these representations with those from authentic (image, caption) pairs reveals structured, systematic shifts spanning a low-rank subspace characterizing vision-induced hallucination. In the inference phase, CIPHER suppresses hallucinations by projecting intermediate hidden states away from this subspace. Experiments across multiple benchmarks show that CIPHER significantly reduces hallucination rates while preserving task performance, demonstrating the effectiveness of counterfactual visual perturbations for improving LVLM faithfulness. Code and additional materials are available at https://hamidreza-dastmalchi.github.io/cipher-cvpr2026/.

  </details>



- **DepthCache: Depth-Guided Training-Free Visual Token Merging for Vision-Language-Action Model Inference**  
  Yuquan Li, Lianjie Ma, Han Ding, Lijun Zhu  
  _2026-03-11_ · https://arxiv.org/abs/2603.10469v1  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models enable generalist robotic manipulation but suffer from high inference latency. This bottleneck stems from the massive number of visual tokens processed by large language backbones. Existing methods either prune or merge tokens uniformly, degrading the spatial reasoning essential for robotic control. We present DepthCache, a training-free framework that leverages depth as a structural prior for visual token compression. It partitions observations into depth-based regions and applies spatially differentiated merge ratios, preserving the near-field workspace while compressing the distant background. To exploit temporal redundancy, DepthCache distributes the merging process across consecutive frames, ensuring consistent representations while reducing per-step computation. A motion-adaptive pipeline further optimizes auxiliary view compression based on end-effector dynamics. The framework requires no model modification, generalizing across diverse VLA architectures. On the LIBERO benchmark, DepthCache achieves up to 1.28x inference speedup with less than 1% average success rate degradation across three VLA models (pi_0.5, OpenVLA, GR00T), whereas pruning and merging baselines incur 4--24% degradation at comparable compression. Real-world experiments on a physical manipulator demonstrate that DepthCache enables faster task throughput and more responsive closed-loop control in latency-sensitive scenarios.

  </details>



- **MoXaRt: Audio-Visual Object-Guided Sound Interaction for XR**  
  Tianyu Xu, Sieun Kim, Qianhui Zheng, Ruoyu Xu, Tejasvi Ravi, Anuva Kulkarni, Katrina Passarella-Ward, Junyi Zhu, Adarsh Kowdle  
  _2026-03-11_ · https://arxiv.org/abs/2603.10465v1  
  <details><summary>Abstract</summary>

  In Extended Reality (XR), complex acoustic environments often overwhelm users, compromising both scene awareness and social engagement due to entangled sound sources. We introduce MoXaRt, a real-time XR system that uses audio-visual cues to separate these sources and enable fine-grained sound interaction. MoXaRt's core is a cascaded architecture that performs coarse, audio-only separation in parallel with visual detection of sources (e.g., faces, instruments). These visual anchors then guide refinement networks to isolate individual sources, separating complex mixes of up to 5 concurrent sources (e.g., 2 voices + 3 instruments) with ~2 second processing latency. We validate MoXaRt through a technical evaluation on a new dataset of 30 one-minute recordings featuring concurrent speech and music, and a 22-participant user study. Empirical results indicate that our system significantly enhances speech intelligibility, yielding a 36.2% (p < 0.01) increase in listening comprehension within adversarial acoustic environments while substantially reducing cognitive load (p < 0.001), thereby paving the way for more perceptive and socially adept XR experiences.

  </details>



- **Learning to Wander: Improving the Global Image Geolocation Ability of LMMs via Actionable Reasoning**  
  Yushuo Zheng, Huiyu Duan, Zicheng Zhang, Xiaohong Liu, Xiongkuo Min  
  _2026-03-11_ · https://arxiv.org/abs/2603.10463v1  
  <details><summary>Abstract</summary>

  Geolocation, the task of identifying the geographic location of an image, requires abundant world knowledge and complex reasoning abilities. Though advanced large multimodal models (LMMs) have shown superior aforementioned capabilities, their performance on the geolocation task remains unexplored. To this end, we introduce \textbf{WanderBench}, the first open access global geolocation benchmark designed for actionable geolocation reasoning in embodied scenarios. WanderBench contains over 32K panoramas across six continents, organized as navigable graphs that enable physical actions such as rotation and movement, transforming geolocation from static recognition into interactive exploration. Building on this foundation, we propose \textbf{GeoAoT} (Action of Thought), a \underline{Geo}location framework with \underline{A}ction of \underline{T}hough, which couples reasoning with embodied actions. Instead of generating textual reasoning chains, GeoAoT produces actionable plans such as, approaching landmarks or adjusting viewpoints, to actively reduce uncertainty. We further establish an evaluation protocol that jointly measures geolocation accuracy and difficulty-aware geolocation questioning ability. Experiments on 19 large multimodal models show that GeoAoT achieves superior fine-grained localization and stronger generalization in dynamic environments. WanderBench and GeoAoT define a new paradigm for actionable, reasoning driven geolocation in embodied visual understanding.

  </details>



- **KnowDiffuser: A Knowledge-Guided Diffusion Planner with LM Reasoning and Prior-Informed Trajectory Initialization**  
  Fan Ding, Xuewen Luo, Fengze Yang, Bo Yu, HwaHui Tew, Ganesh Krishnasamy, Junn Yong Loo  
  _2026-03-11_ · https://arxiv.org/abs/2603.10441v1  
  <details><summary>Abstract</summary>

  Recent advancements in Language Models (LMs) have demonstrated strong semantic reasoning capabilities, enabling their application in high-level decision-making for autonomous driving (AD). However, LMs operate over discrete token spaces and lack the ability to generate continuous, physically feasible trajectories required for motion planning. Meanwhile, diffusion models have proven effective at generating reliable and dynamically consistent trajectories, but often lack semantic interpretability and alignment with scene-level understanding. To address these limitations, we propose \textbf{KnowDiffuser}, a knowledge-guided motion planning framework that tightly integrates the semantic understanding of language models with the generative power of diffusion models. The framework employs a language model to infer context-aware meta-actions from structured scene representations, which are then mapped to prior trajectories that anchor the subsequent denoising process. A two-stage truncated denoising mechanism refines these trajectories efficiently, preserving both semantic alignment and physical feasibility. Experiments on the nuPlan benchmark demonstrate that KnowDiffuser significantly outperforms existing planners in both open-loop and closed-loop evaluations, establishing a robust and interpretable framework that effectively bridges the semantic-to-physical gap in AD systems.

  </details>



- **COHORT: Hybrid RL for Collaborative Large DNN Inference on Multi-Robot Systems Under Real-Time Constraints**  
  Mohammad Saeid Anwar, Anuradha Ravi, Indrajeet Ghosh, Gaurav Shinde, Carl Busart, Nirmalya Roy  
  _2026-03-11_ · https://arxiv.org/abs/2603.10436v1  
  <details><summary>Abstract</summary>

  Large deep neural networks (DNNs), especially transformer-based and multimodal architectures, are computationally demanding and challenging to deploy on resource-constrained edge platforms like field robots. These challenges intensify in mission-critical scenarios (e.g., disaster response), where robots must collaborate under tight constraints on bandwidth, latency, and battery life, often without infrastructure or server support. To address these limitations, we present COHORT, a collaborative DNN inference and task-execution framework for multi-robot systems built on the Robotic Operating System (ROS). COHORT employs a hybrid offline-online reinforcement learning (RL) strategy to dynamically schedule and distribute DNN module execution across robots. Our key contributions are threefold: (a) Offline RL policy learning combined with Advantage-Weighted Regression (AWR), trained on auction-based task allocation data from heterogeneous DNN workloads across distributed robots, (b) Online policy adaptation via Multi-Agent PPO (MAPPO), initialized from the offline policy and fine-tuned in real time, and (c) comprehensive evaluation of COHORT on vision-language model (VLM) inference tasks such as CLIP and SAM, analyzing scalability with increasing robot/workload and robustness under . We benchmark COHORT against genetic algorithms and multiple RL baselines. Experimental results demonstrate that COHORT reduces battery consumption by 15.4% and increases GPU utilization by 51.67%, while satisfying frame-rate and deadline constraints 2.55 times of the time.

  </details>



- **Rethinking Gaussian Trajectory Predictors: Calibrated Uncertainty for Safe Planning**  
  Fatemeh Cheraghi Pouria, Mahsa Golchoubian, Katherine Driggs-Campbell  
  _2026-03-11_ · https://arxiv.org/abs/2603.10407v1  
  <details><summary>Abstract</summary>

  Accurate trajectory prediction is critical for safe autonomous navigation in crowded environments. While many trajectory predictors output Gaussian distributions to represent the multi-modal distribution over future pedestrian positions, the reliability of their confidence levels often remains unaddressed. This limitation can lead to unsafe or overly conservative motion planning when the predictor is integrated with an uncertainty-aware planner. Existing Gaussian trajectory predictors primarily rely on the Negative Log-Likelihood loss, which is prone to predict over- or under-confident distributions, and may compromise downstream planner safety. This paper introduces a novel loss function for calibrating prediction uncertainty which leverages Kernel Density Estimation to estimate the empirical distribution of confidence levels. The proposed formulation enforces consistency with the properties of a Gaussian assumption by explicitly matching the estimated empirical distribution to the Chi-squared distribution. To ensure accurate mean prediction, a Mean Squared Error term is also incorporated in the final loss formulation. Experimental results on real-world trajectory datasets show that our method significantly improves the reliability of confidence levels predicted by different State-Of-The-Art Gaussian trajectory predictors. We also demonstrate the importance of providing planners with reliable probabilistic insights (i.e. calibrated confidence levels) for collision-free navigation in complex scenarios. For this purpose, we integrate Gaussian trajectory predictors trained with our loss function with an uncertainty-aware Model Predictive Control on scenarios extracted from real-world datasets, achieving improved planning performance through calibrated confidence levels.

  </details>



- **GeoSense: Internalizing Geometric Necessity Perception for Multimodal Reasoning**  
  Ruiheng Liu, Haihong Hao, Mingfei Han, Xin Gu, Kecheng Zhang, Changlin Li, Xiaojun Chang  
  _2026-03-11_ · https://arxiv.org/abs/2603.10370v1  
  <details><summary>Abstract</summary>

  Advancing towards artificial superintelligence requires rich and intelligent perceptual capabilities. A critical frontier in this pursuit is overcoming the limited spatial understanding of Multimodal Large Language Models (MLLMs), where geometry information is essential. Existing methods often address this by rigidly injecting geometric signals into every input, while ignoring their necessity and adding computation overhead. Contrary to this paradigm, our framework endows the model with an awareness of perceptual insufficiency, empowering it to autonomously engage geometric features in reasoning when 2D cues are deemed insufficient. To achieve this, we first introduce an independent geometry input channel to the model architecture and conduct alignment training, enabling the effective utilization of geometric features. Subsequently, to endow the model with perceptual awareness, we curate a dedicated spatial-aware supervised fine-tuning dataset. This serves to activate the model's latent internal cues, empowering it to autonomously determine the necessity of geometric information. Experiments across multiple spatial reasoning benchmarks validate this approach, demonstrating significant spatial gains without compromising 2D visual reasoning capabilities, offering a path toward more robust, efficient and self-aware multi-modal intelligence.

  </details>



- **Geometric Autoencoder for Diffusion Models**  
  Hangyu Liu, Jianyong Wang, Yutao Sun  
  _2026-03-11_ · https://arxiv.org/abs/2603.10365v1  
  <details><summary>Abstract</summary>

  Latent diffusion models have established a new state-of-the-art in high-resolution visual generation. Integrating Vision Foundation Model priors improves generative efficiency, yet existing latent designs remain largely heuristic. These approaches often struggle to unify semantic discriminability, reconstruction fidelity, and latent compactness. In this paper, we propose Geometric Autoencoder (GAE), a principled framework that systematically addresses these challenges. By analyzing various alignment paradigms, GAE constructs an optimized low-dimensional semantic supervision target from VFMs to provide guidance for the autoencoder. Furthermore, we leverage latent normalization that replaces the restrictive KL-divergence of standard VAEs, enabling a more stable latent manifold specifically optimized for diffusion learning. To ensure robust reconstruction under high-intensity noise, GAE incorporates a dynamic noise sampling mechanism. Empirically, GAE achieves compelling performance on the ImageNet-1K $256 \times 256$ benchmark, reaching a gFID of 1.82 at only 80 epochs and 1.31 at 800 epochs without Classifier-Free Guidance, significantly surpassing existing state-of-the-art methods. Beyond generative quality, GAE establishes a superior equilibrium between compression, semantic depth and robust reconstruction stability. These results validate our design considerations, offering a promising paradigm for latent diffusion modeling. Code and models are publicly available at https://github.com/freezing-index/Geometric-Autoencoder-for-Diffusion-Models.

  </details>



- **StyleGallery: Training-free and Semantic-aware Personalized Style Transfer from Arbitrary Image References**  
  Boyu He, Yunfan Ye, Chang Liu, Weishang Wu, Fang Liu, Zhiping Cai  
  _2026-03-11_ · https://arxiv.org/abs/2603.10354v1  
  <details><summary>Abstract</summary>

  Despite the advancements in diffusion-based image style transfer, existing methods are commonly limited by 1) semantic gap: the style reference could miss proper content semantics, causing uncontrollable stylization; 2) reliance on extra constraints (e.g., semantic masks) restricting applicability; 3) rigid feature associations lacking adaptive global-local alignment, failing to balance fine-grained stylization and global content preservation. These limitations, particularly the inability to flexibly leverage style inputs, fundamentally restrict style transfer in terms of personalization, accuracy, and adaptability. To address these, we propose StyleGallery, a training-free and semantic-aware framework that supports arbitrary reference images as input and enables effective personalized customization. It comprises three core stages: semantic region segmentation (adaptive clustering on latent diffusion features to divide regions without extra inputs); clustered region matching (block filtering on extracted features for precise alignment); and style transfer optimization (energy function-guided diffusion sampling with regional style loss to optimize stylization). Experiments on our introduced benchmark demonstrate that StyleGallery outperforms state-of-the-art methods in content structure preservation, regional stylization, interpretability, and personalized customization, particularly when leveraging multiple style references.

  </details>



- **EmoStory: Emotion-Aware Story Generation**  
  Jingyuan Yang, Rucong Chen, Hui Huang  
  _2026-03-11_ · https://arxiv.org/abs/2603.10349v1  
  <details><summary>Abstract</summary>

  Story generation aims to produce image sequences that depict coherent narratives while maintaining subject consistency across frames. Although existing methods have excelled in producing coherent and expressive stories, they remain largely emotion-neutral, focusing on what subject appears in a story while overlooking how emotions shape narrative interpretation and visual presentation. As stories are intended to engage audiences emotionally, we introduce emotion-aware story generation, a new task that aims to generate subject-consistent visual stories with explicit emotional directions. This task is challenging due to the abstract nature of emotions, which must be grounded in concrete visual elements and consistently expressed across a narrative through visual composition. To address these challenges, we propose EmoStory, a two-stage framework that integrates agent-based story planning and region-aware story generation. The planning stage transforms target emotions into coherent story prompts with emotion agent and writer agent, while the generation stage preserves subject consistency and injects emotion-related elements through region-aware composition. We evaluate EmoStory on a newly constructed dataset covering 25 subjects and 600 emotional stories. Extensive quantitative and qualitative results, along with user studies, show that EmoStory outperforms state-of-the-art story generation methods in emotion accuracy, prompt alignment, and subject consistency.

  </details>



- **Fuel Gauge: Estimating Chain-of-Thought Length Ahead of Time in Large Multimodal Models**  
  Yuedong Yang, Xiwen Wei, Mustafa Munir, Radu Marculescu  
  _2026-03-11_ · https://arxiv.org/abs/2603.10335v1  
  <details><summary>Abstract</summary>

  Reasoning Large Multi-modality Models (LMMs) have become the de facto choice for many applications. However, these models rely on a Chain-of-Thought (CoT) process that is lengthy and unpredictable at runtime, often resulting in inefficient use of computational resources (due to memory fragmentation) and sub-optimal accuracy (due to under- and over-thinking). We observe empirically that the CoT process follows a very simple form, whose behavior is independent of the specific generated samples. This suggests that the CoT length can be estimated ahead of time based on a hidden parameter representing the amount of "fuel" available to support the reasoning process. Based on this insight, we propose Fuel Gauge, the first method which extracts this hidden signal and predicts CoT length ahead of time. We demonstrate the utility on the Fuel Gauge on two downstream tasks: predictive KV cache allocation, which addresses memory fragmentation in LMM serving systems, and CoT length modulation, which mitigates under-thinking and over-thinking. Extensive experiments on LMMs across text-only, image-text, and video-text question answering benchmarks demonstrate the effectiveness, generalizability, and practical value of our Fuel Gauge. For example, on the GPQA-Diamond benchmark, our Fuel Gauge achieves less than half the CoT length prediction error compared to the baseline; this translates into a 13.37x reduction in the memory allocation frequency.

  </details>



- **The Orthogonal Vulnerabilities of Generative AI Watermarks: A Comparative Empirical Benchmark of Spatial and Latent Provenance**  
  Jesse Yu, Nicholas Wei  
  _2026-03-11_ · https://arxiv.org/abs/2603.10323v1  
  <details><summary>Abstract</summary>

  As open-weights generative AI rapidly proliferates, the ability to synthesize hyper-realistic media has introduced profound challenges to digital trust. Automated disinformation and AI-generated imagery have made robust digital provenance a critical cybersecurity imperative. Currently, state-of-the-art invisible watermarks operate within one of two primary mathematical manifolds: the spatial domain (post-generation pixel embedding) or the latent domain (pre-generation frequency embedding). While existing literature frequently evaluates these models against isolated, classical distortions, there is a critical lack of rigorous, comparative benchmarking against modern generative AI editing tools. In this study, we empirically evaluate two leading representative paradigms, RivaGAN (Spatial) and Tree-Ring (Latent), utilizing an automated Attack Simulation Engine across 30 intensity intervals of geometric and generative perturbations. We formalize an "Adversarial Evasion Region" (AER) framework to measure cryptographic degradation against semantic visual retention (OpenCLIP > 70.0). Our statistical analysis ($n=100$ per interval, $MOE = \pm 3.92\%$) reveals that these domains possess mutually exclusive, mathematically orthogonal vulnerabilities. Spatial watermarks experience severe cryptographic degradation under algorithmic pixel-rewriting (exhibiting a 67.47% AER evasion rate under Img2Img translation), whereas latent watermarks exhibit profound fragility against geometric misalignment (yielding a 43.20% AER evasion rate under static cropping). By proving that single-domain watermarking is fundamentally insufficient against modern adversarial toolsets, this research exposes a systemic vulnerability in current digital provenance standards and establishes the foundational exigence for future multi-domain cryptographic architectures.

  </details>



- **SteadyTray: Learning Object Balancing Tasks in Humanoid Tray Transport via Residual Reinforcement Learning**  
  Anlun Huang, Zhenyu Wu, Soofiyan Atar, Yuheng Zhi, Michael Yip  
  _2026-03-11_ · https://arxiv.org/abs/2603.10306v1  
  <details><summary>Abstract</summary>

  Stabilizing unsecured payloads against the inherent oscillations of dynamic bipedal locomotion remains a critical engineering bottleneck for humanoids in unstructured environments. To solve this, we introduce ReST-RL, a hierarchical reinforcement learning architecture that explicitly decouples locomotion from payload stabilization, evaluated via the SteadyTray benchmark. Rather than relying on monolithic end-to-end learning, our framework integrates a robust base locomotion policy with a dynamic residual module engineered to actively cancel gait-induced perturbations at the end-effector. This architectural separation ensures steady tray transport without degrading the underlying bipedal stability. In simulation, the residual design significantly outperforms end-to-end baselines in gait smoothness and orientation accuracy, achieving a 96.9% success rate in variable velocity tracking and 74.5% robustness against external force disturbances. Successfully deployed on the Unitree G1 humanoid hardware, this modular approach demonstrates highly reliable zero-shot sim-to-real generalization across various objects and external force disturbances.

  </details>



- **A Robust Deep Learning Framework for Bangla License Plate Recognition Using YOLO and Vision-Language OCR**  
  Nayeb Hasin, Md. Arafath Rahman Nishat, Mainul Islam, Khandakar Shakib Al Hasan, Asif Newaz  
  _2026-03-10_ · https://arxiv.org/abs/2603.10267v1  
  <details><summary>Abstract</summary>

  An Automatic License Plate Recognition (ALPR) system constitutes a crucial element in an intelligent traffic management system. However, the detection of Bangla license plates remains challenging because of the complicated character scheme and uneven layouts. This paper presents a robust Bangla License Plate Recognition system that integrates a deep learning-based object detection model for license plate localization with Optical Character Recognition for text extraction. Multiple object detection architectures, including U-Net and several YOLO (You Only Look Once) variants, are compared for license plate localization. This study proposes a novel two-stage adaptive training strategy built upon the YOLOv8 architecture to improve localization performance. The proposed approach outperforms the established models, achieving an accuracy of 97.83% and an Intersection over Union (IoU) of 91.3%. The text recognition problem is phrased as a sequence generation problem with a VisionEncoderDecoder architecture, with a combination of encoder-decoders evaluated. It was demonstrated that the ViT + BanglaBERT model gives better results at the character level, with a Character Error Rate of 0.1323 and Word Error Rate of 0.1068. The proposed system also shows a consistent performance when tested on an external dataset that has been curated for this study purpose. The dataset offers completely different environment and lighting conditions compared to the training sample, indicating the robustness of the proposed framework. Overall, our proposed system provides a robust and reliable solution for Bangla license plate recognition and performs effectively across diverse real-world scenarios, including variations in lighting, noise, and plate styles. These strengths make it well suited for deployment in intelligent transportation applications such as automated law enforcement and access control.

  </details>



- **ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation**  
  Sofia Iliopoulou, Dimitris Ampeliotis, Athanassios Skodras  
  _2026-03-10_ · https://arxiv.org/abs/2603.10188v1  
  <details><summary>Abstract</summary>

  Recent progress in learning-based image compression has demonstrated that end-to-end optimization can substantially outperform traditional codecs by jointly learning compact latent representations and probabilistic entropy models. However, many existing approaches achieve high rate-distortion efficiency at the expense of increased computational cost and limited parallelism. This paper presents ARCHE - Autoregressive Residual Compression with Hyperprior and Excitation, an end-to-end learned image compression framework that balances modeling accuracy and computational efficiency. The proposed architecture unifies hierarchical, spatial, and channel-based priors within a single probabilistic framework, capturing both global and local dependencies in the latent representation of the image, while employing adaptive feature recalibration and residual refinement to enhance latent representation quality. Without relying on recurrent or transformer-based components, ARCHE attains state-of-the-art rate-distortion efficiency: it reduces the BD-Rate by approximately 48% relative to the commonly used benchmark model of Balle et al., 30% relative to the channel-wise autoregressive model of Minnen & Singh and 5% against the VVC Intra codec on the Kodak benchmark dataset. The framework maintains computational efficiency with 95M parameters and 222ms running time per image. Visual comparisons confirm sharper textures and improved color fidelity, particularly at lower bit rates, demonstrating that accurate entropy modeling can be achieved through efficient convolutional designs suitable for practical deployment.

  </details>


