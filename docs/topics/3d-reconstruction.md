# 3D Reconstruction

_Updated: 2026-02-06 07:11 UTC_

Total papers shown: **25**


---

- **Splat and Distill: Augmenting Teachers with Feed-Forward 3D Reconstruction For 3D-Aware Distillation**  
  David Shavin, Sagie Benaim  
  _2026-02-05_ · https://arxiv.org/abs/2602.06032v1  
  <details><summary>Abstract</summary>

  Vision Foundation Models (VFMs) have achieved remarkable success when applied to various downstream 2D tasks. Despite their effectiveness, they often exhibit a critical lack of 3D awareness. To this end, we introduce Splat and Distill, a framework that instills robust 3D awareness into 2D VFMs by augmenting the teacher model with a fast, feed-forward 3D reconstruction pipeline. Given 2D features produced by a teacher model, our method first lifts these features into an explicit 3D Gaussian representation, in a feedforward manner. These 3D features are then ``splatted" onto novel viewpoints, producing a set of novel 2D feature maps used to supervise the student model, ``distilling" geometrically grounded knowledge. By replacing slow per-scene optimization of prior work with our feed-forward lifting approach, our framework avoids feature-averaging artifacts, creating a dynamic learning process where the teacher's consistency improves alongside that of the student. We conduct a comprehensive evaluation on a suite of downstream tasks, including monocular depth estimation, surface normal estimation, multi-view correspondence, and semantic segmentation. Our method significantly outperforms prior works, not only achieving substantial gains in 3D awareness but also enhancing the underlying semantic richness of 2D features. Project page is available at https://davidshavin4.github.io/Splat-and-Distill/

  </details>



- **Neural Implicit 3D Cardiac Shape Reconstruction from Sparse CT Angiography Slices Mimicking 2D Transthoracic Echocardiography Views**  
  Gino E. Jansen, Carolina Brás, R. Nils Planken, Mark J. Schuuring, Berto J. Bouma, Ivana Išgum  
  _2026-02-05_ · https://arxiv.org/abs/2602.05884v1  
  <details><summary>Abstract</summary>

  Accurate 3D representations of cardiac structures allow quantitative analysis of anatomy and function. In this work, we propose a method for reconstructing complete 3D cardiac shapes from segmentations of sparse planes in CT angiography (CTA) for application in 2D transthoracic echocardiography (TTE). Our method uses a neural implicit function to reconstruct the 3D shape of the cardiac chambers and left-ventricle myocardium from sparse CTA planes. To investigate the feasibility of achieving 3D reconstruction from 2D TTE, we select planes that mimic the standard apical 2D TTE views. During training, a multi-layer perceptron learns shape priors from 3D segmentations of the target structures in CTA. At test time, the network reconstructs 3D cardiac shapes from segmentations of TTE-mimicking CTA planes by jointly optimizing the latent code and the rigid transforms that map the observed planes into 3D space. For each heart, we simulate four realistic apical views, and we compare reconstructed multi-class volumes with the reference CTA volumes. On a held-out set of CTA segmentations, our approach achieves an average Dice coefficient of 0.86 $\pm$ 0.04 across all structures. Our method also achieves markedly lower volume errors than the clinical standard, Simpson's biplane rule: 4.88 $\pm$ 4.26 mL vs. 8.14 $\pm$ 6.04 mL, respectively, for the left ventricle; and 6.40 $\pm$ 7.37 mL vs. 37.76 $\pm$ 22.96 mL, respectively, for the left atrium. This suggests that our approach offers a viable route to more accurate 3D chamber quantification in 2D transthoracic echocardiography.

  </details>



- **NVS-HO: A Benchmark for Novel View Synthesis of Handheld Objects**  
  Musawar Ali, Manuel Carranza-García, Nicola Fioraio, Samuele Salti, Luigi Di Stefano  
  _2026-02-05_ · https://arxiv.org/abs/2602.05822v1  
  <details><summary>Abstract</summary>

  We propose NVS-HO, the first benchmark designed for novel view synthesis of handheld objects in real-world environments using only RGB inputs. Each object is recorded in two complementary RGB sequences: (1) a handheld sequence, where the object is manipulated in front of a static camera, and (2) a board sequence, where the object is fixed on a ChArUco board to provide accurate camera poses via marker detection. The goal of NVS-HO is to learn a NVS model that captures the full appearance of an object from (1), whereas (2) provides the ground-truth images used for evaluation. To establish baselines, we consider both a classical SfM pipeline and a state-of-the-art pre-trained feed-forward neural network (VGGT) as pose estimators, and train NVS models based on NeRF and Gaussian Splatting. Our experiments reveal significant performance gaps in current methods under unconstrained handheld conditions, highlighting the need for more robust approaches. NVS-HO thus offers a challenging real-world benchmark to drive progress in RGB-based novel view synthesis of handheld objects.

  </details>



- **Depth as Prior Knowledge for Object Detection**  
  Moussa Kassem Sbeyti, Nadja Klein  
  _2026-02-05_ · https://arxiv.org/abs/2602.05730v1  
  <details><summary>Abstract</summary>

  Detecting small and distant objects remains challenging for object detectors due to scale variation, low resolution, and background clutter. Safety-critical applications require reliable detection of these objects for safe planning. Depth information can improve detection, but existing approaches require complex, model-specific architectural modifications. We provide a theoretical analysis followed by an empirical investigation of the depth-detection relationship. Together, they explain how depth causes systematic performance degradation and why depth-informed supervision mitigates it. We introduce DepthPrior, a framework that uses depth as prior knowledge rather than as a fused feature, providing comparable benefits without modifying detector architectures. DepthPrior consists of Depth-Based Loss Weighting (DLW) and Depth-Based Loss Stratification (DLS) during training, and Depth-Aware Confidence Thresholding (DCT) during inference. The only overhead is the initial cost of depth estimation. Experiments across four benchmarks (KITTI, MS COCO, VisDrone, SUN RGB-D) and two detectors (YOLOv11, EfficientDet) demonstrate the effectiveness of DepthPrior, achieving up to +9% mAP$_S$ and +7% mAR$_S$ for small objects, with inference recovery rates as high as 95:1 (true vs. false detections). DepthPrior offers these benefits without additional sensors, architectural changes, or performance costs. Code is available at https://github.com/mos-ks/DepthPrior.

  </details>



- **UniSurg: A Video-Native Foundation Model for Universal Understanding of Surgical Videos**  
  Jinlin Wu, Felix Holm, Chuxi Chen, An Wang, Yaxin Hu, Xiaofan Ye, Zelin Zang, Miao Xu, Lihua Zhou, Huai Liao, et al.  
  _2026-02-05_ · https://arxiv.org/abs/2602.05638v1  
  <details><summary>Abstract</summary>

  While foundation models have advanced surgical video analysis, current approaches rely predominantly on pixel-level reconstruction objectives that waste model capacity on low-level visual details - such as smoke, specular reflections, and fluid motion - rather than semantic structures essential for surgical understanding. We present UniSurg, a video-native foundation model that shifts the learning paradigm from pixel-level reconstruction to latent motion prediction. Built on the Video Joint Embedding Predictive Architecture (V-JEPA), UniSurg introduces three key technical innovations tailored to surgical videos: 1) motion-guided latent prediction to prioritize semantically meaningful regions, 2) spatiotemporal affinity self-distillation to enforce relational consistency, and 3) feature diversity regularization to prevent representation collapse in texture-sparse surgical scenes. To enable large-scale pretraining, we curate UniSurg-15M, the largest surgical video dataset to date, comprising 3,658 hours of video from 50 sources across 13 anatomical regions. Extensive experiments across 17 benchmarks demonstrate that UniSurg significantly outperforms state-of-the-art methods on surgical workflow recognition (+14.6% F1 on EgoSurgery, +10.3% on PitVis), action triplet recognition (39.54% mAP-IVT on CholecT50), skill assessment, polyp segmentation, and depth estimation. These results establish UniSurg as a new standard for universal, motion-oriented surgical video understanding.

  </details>



- **PIRATR: Parametric Object Inference for Robotic Applications with Transformers in 3D Point Clouds**  
  Michael Schwingshackl, Fabio F. Oberweger, Mario Niedermeyer, Huemer Johannes, Markus Murschitz  
  _2026-02-05_ · https://arxiv.org/abs/2602.05557v1  
  <details><summary>Abstract</summary>

  We present PIRATR, an end-to-end 3D object detection framework for robotic use cases in point clouds. Extending PI3DETR, our method streamlines parametric 3D object detection by jointly estimating multi-class 6-DoF poses and class-specific parametric attributes directly from occlusion-affected point cloud data. This formulation enables not only geometric localization but also the estimation of task-relevant properties for parametric objects, such as a gripper's opening, where the 3D model is adjusted according to simple, predefined rules. The architecture employs modular, class-specific heads, making it straightforward to extend to novel object types without re-designing the pipeline. We validate PIRATR on an automated forklift platform, focusing on three structurally and functionally diverse categories: crane grippers, loading platforms, and pallets. Trained entirely in a synthetic environment, PIRATR generalizes effectively to real outdoor LiDAR scans, achieving a detection mAP of 0.919 without additional fine-tuning. PIRATR establishes a new paradigm of pose-aware, parameterized perception. This bridges the gap between low-level geometric reasoning and actionable world models, paving the way for scalable, simulation-trained perception systems that can be deployed in dynamic robotic environments. Code available at https://github.com/swingaxe/piratr.

  </details>



- **Mapper-GIN: Lightweight Structural Graph Abstraction for Corrupted 3D Point Cloud Classification**  
  Jeongbin You, Donggun Kim, Sejun Park, Seungsang Oh  
  _2026-02-05_ · https://arxiv.org/abs/2602.05522v1  
  <details><summary>Abstract</summary>

  Robust 3D point cloud classification is often pursued by scaling up backbones or relying on specialized data augmentation. We instead ask whether structural abstraction alone can improve robustness, and study a simple topology-inspired decomposition based on the Mapper algorithm. We propose Mapper-GIN, a lightweight pipeline that partitions a point cloud into overlapping regions using Mapper (PCA lens, cubical cover, and followed by density-based clustering), constructs a region graph from their overlaps, and performs graph classification with a Graph Isomorphism Network. On the corruption benchmark ModelNet40-C, Mapper-GIN achieves competitive and stable accuracy under Noise and Transformation corruptions with only 0.5M parameters. In contrast to prior approaches that require heavier architectures or additional mechanisms to gain robustness, Mapper-GIN attains strong corruption robustness through simple region-level graph abstraction and GIN message passing. Overall, our results suggest that region-graph structure offers an efficient and interpretable source of robustness for 3D visual recognition.

  </details>



- **NeVStereo: A NeRF-Driven NVS-Stereo Architecture for High-Fidelity 3D Tasks**  
  Pengcheng Chen, Yue Hu, Wenhao Li, Nicole M Gunderson, Andrew Feng, Zhenglong Sun, Peter Beerel, Eric J Seibel  
  _2026-02-05_ · https://arxiv.org/abs/2602.05423v1  
  <details><summary>Abstract</summary>

  In modern dense 3D reconstruction, feed-forward systems (e.g., VGGT, pi3) focus on end-to-end matching and geometry prediction but do not explicitly output the novel view synthesis (NVS). Neural rendering-based approaches offer high-fidelity NVS and detailed geometry from posed images, yet they typically assume fixed camera poses and can be sensitive to pose errors. As a result, it remains non-trivial to obtain a single framework that can offer accurate poses, reliable depth, high-quality rendering, and accurate 3D surfaces from casually captured views. We present NeVStereo, a NeRF-driven NVS-stereo architecture that aims to jointly deliver camera poses, multi-view depth, novel view synthesis, and surface reconstruction from multi-view RGB-only inputs. NeVStereo combines NeRF-based NVS for stereo-friendly renderings, confidence-guided multi-view depth estimation, NeRF-coupled bundle adjustment for pose refinement, and an iterative refinement stage that updates both depth and the radiance field to improve geometric consistency. This design mitigated the common NeRF-based issues such as surface stacking, artifacts, and pose-depth coupling. Across indoor, outdoor, tabletop, and aerial benchmarks, our experiments indicate that NeVStereo achieves consistently strong zero-shot performance, with up to 36% lower depth error, 10.4% improved pose accuracy, 4.5% higher NVS fidelity, and state-of-the-art mesh quality (F1 91.93%, Chamfer 4.35 mm) compared to existing prestigious methods.

  </details>



- **Wid3R: Wide Field-of-View 3D Reconstruction via Camera Model Conditioning**  
  Dongki Jung, Jaehoon Choi, Adil Qureshi, Somi Jeong, Dinesh Manocha, Suyong Yeon  
  _2026-02-05_ · https://arxiv.org/abs/2602.05321v1  
  <details><summary>Abstract</summary>

  We present Wid3R, a feed-forward neural network for visual geometry reconstruction that supports wide field-of-view camera models. Prior methods typically assume that input images are rectified or captured with pinhole cameras, since both their architectures and training datasets are tailored to perspective images only. These assumptions limit their applicability in real-world scenarios that use fisheye or panoramic cameras and often require careful calibration and undistortion. In contrast, Wid3R is a generalizable multi-view 3D estimation method that can model wide field-of-view camera types. Our approach leverages a ray representation with spherical harmonics and a novel camera model token within the network, enabling distortion-aware 3D reconstruction. Furthermore, Wid3R is the first multi-view foundation model to support feed-forward 3D reconstruction directly from 360 imagery. It demonstrates strong zero-shot robustness and consistently outperforms prior methods, achieving improvements of up to +77.33 on Stanford2D3D.

  </details>



- **Fast-SAM3D: 3Dfy Anything in Images but Faster**  
  Weilun Feng, Mingqiang Wu, Zhiliang Chen, Chuanguang Yang, Haotong Qin, Yuqi Li, Xiaokun Liu, Guoxin Fan, Zhulin An, Libo Huang, et al.  
  _2026-02-05_ · https://arxiv.org/abs/2602.05293v1  
  <details><summary>Abstract</summary>

  SAM3D enables scalable, open-world 3D reconstruction from complex scenes, yet its deployment is hindered by prohibitive inference latency. In this work, we conduct the \textbf{first systematic investigation} into its inference dynamics, revealing that generic acceleration strategies are brittle in this context. We demonstrate that these failures stem from neglecting the pipeline's inherent multi-level \textbf{heterogeneity}: the kinematic distinctiveness between shape and layout, the intrinsic sparsity of texture refinement, and the spectral variance across geometries. To address this, we present \textbf{Fast-SAM3D}, a training-free framework that dynamically aligns computation with instantaneous generation complexity. Our approach integrates three heterogeneity-aware mechanisms: (1) \textit{Modality-Aware Step Caching} to decouple structural evolution from sensitive layout updates; (2) \textit{Joint Spatiotemporal Token Carving} to concentrate refinement on high-entropy regions; and (3) \textit{Spectral-Aware Token Aggregation} to adapt decoding resolution. Extensive experiments demonstrate that Fast-SAM3D delivers up to \textbf{2.67$\times$} end-to-end speedup with negligible fidelity loss, establishing a new Pareto frontier for efficient single-view 3D generation. Our code is released in https://github.com/wlfeng0509/Fast-SAM3D.

  </details>



- **PoseGaussian: Pose-Driven Novel View Synthesis for Robust 3D Human Reconstruction**  
  Ju Shen, Chen Chen, Tam V. Nguyen, Vijayan K. Asari  
  _2026-02-05_ · https://arxiv.org/abs/2602.05190v1  
  <details><summary>Abstract</summary>

  We propose PoseGaussian, a pose-guided Gaussian Splatting framework for high-fidelity human novel view synthesis. Human body pose serves a dual purpose in our design: as a structural prior, it is fused with a color encoder to refine depth estimation; as a temporal cue, it is processed by a dedicated pose encoder to enhance temporal consistency across frames. These components are integrated into a fully differentiable, end-to-end trainable pipeline. Unlike prior works that use pose only as a condition or for warping, PoseGaussian embeds pose signals into both geometric and temporal stages to improve robustness and generalization. It is specifically designed to address challenges inherent in dynamic human scenes, such as articulated motion and severe self-occlusion. Notably, our framework achieves real-time rendering at 100 FPS, maintaining the efficiency of standard Gaussian Splatting pipelines. We validate our approach on ZJU-MoCap, THuman2.0, and in-house datasets, demonstrating state-of-the-art performance in perceptual quality and structural accuracy (PSNR 30.86, SSIM 0.979, LPIPS 0.028).

  </details>



- **LitS: A novel Neighborhood Descriptor for Point Clouds**  
  Jonatan B. Bastos, Francisco F. Rivera, Oscar G. Lorenzo, David L. Vilariño, José C. Cabaleiro, Alberto M. Esmorís, Tomás F. Pena  
  _2026-02-04_ · https://arxiv.org/abs/2602.04838v1  
  <details><summary>Abstract</summary>

  With the advancement of 3D scanning technologies, point clouds have become fundamental for representing 3D spatial data, with applications that span across various scientific and technological fields. Practical analysis of this data depends crucially on available neighborhood descriptors to accurately characterize the local geometries of the point cloud. This paper introduces LitS, a novel neighborhood descriptor for 2D and 3D point clouds. LitS are piecewise constant functions on the unit circle that allow points to keep track of their surroundings. Each element in LitS' domain represents a direction with respect to a local reference system. Once constructed, evaluating LitS at any given direction gives us information about the number of neighbors in a cone-like region centered around that same direction. Thus, LitS conveys a lot of information about the local neighborhood of a point, which can be leveraged to gain global structural understanding by analyzing how LitS changes between close points. In addition, LitS comes in two versions ('regular' and 'cumulative') and has two parameters, allowing them to adapt to various contexts and types of point clouds. Overall, they are a versatile neighborhood descriptor, capable of capturing the nuances of local point arrangements and resilient to common point cloud data issues such as variable density and noise.

  </details>



- **How to rewrite the stars: Mapping your orchard over time through constellations of fruits**  
  Gonçalo P. Matos, Carlos Santiago, João P. Costeira, Ricardo L. Saldanha, Ernesto M. Morgado  
  _2026-02-04_ · https://arxiv.org/abs/2602.04722v1  
  <details><summary>Abstract</summary>

  Following crop growth through the vegetative cycle allows farmers to predict fruit setting and yield in early stages, but it is a laborious and non-scalable task if performed by a human who has to manually measure fruit sizes with a caliper or dendrometers. In recent years, computer vision has been used to automate several tasks in precision agriculture, such as detecting and counting fruits, and estimating their size. However, the fundamental problem of matching the exact same fruits from one video, collected on a given date, to the fruits visible in another video, collected on a later date, which is needed to track fruits' growth through time, remains to be solved. Few attempts were made, but they either assume that the camera always starts from the same known position and that there are sufficiently distinct features to match, or they used other sources of data like GPS. Here we propose a new paradigm to tackle this problem, based on constellations of 3D centroids, and introduce a descriptor for very sparse 3D point clouds that can be used to match fruits across videos. Matching constellations instead of individual fruits is key to deal with non-rigidity, occlusions and challenging imagery with few distinct visual features to track. The results show that the proposed method can be successfully used to match fruits across videos and through time, and also to build an orchard map and later use it to locate the camera pose in 6DoF, thus providing a method for autonomous navigation of robots in the orchard and for selective fruit picking, for example.

  </details>



- **AGILE: Hand-Object Interaction Reconstruction from Video via Agentic Generation**  
  Jin-Chuan Shi, Binhong Ye, Tao Liu, Junzhe He, Yangjinhui Xu, Xiaoyang Liu, Zeju Li, Hao Chen, Chunhua Shen  
  _2026-02-04_ · https://arxiv.org/abs/2602.04672v1  
  <details><summary>Abstract</summary>

  Reconstructing dynamic hand-object interactions from monocular videos is critical for dexterous manipulation data collection and creating realistic digital twins for robotics and VR. However, current methods face two prohibitive barriers: (1) reliance on neural rendering often yields fragmented, non-simulation-ready geometries under heavy occlusion, and (2) dependence on brittle Structure-from-Motion (SfM) initialization leads to frequent failures on in-the-wild footage. To overcome these limitations, we introduce AGILE, a robust framework that shifts the paradigm from reconstruction to agentic generation for interaction learning. First, we employ an agentic pipeline where a Vision-Language Model (VLM) guides a generative model to synthesize a complete, watertight object mesh with high-fidelity texture, independent of video occlusions. Second, bypassing fragile SfM entirely, we propose a robust anchor-and-track strategy. We initialize the object pose at a single interaction onset frame using a foundation model and propagate it temporally by leveraging the strong visual similarity between our generated asset and video observations. Finally, a contact-aware optimization integrates semantic, geometric, and interaction stability constraints to enforce physical plausibility. Extensive experiments on HO3D, DexYCB, and in-the-wild videos reveal that AGILE outperforms baselines in global geometric accuracy while demonstrating exceptional robustness on challenging sequences where prior art frequently collapses. By prioritizing physical validity, our method produces simulation-ready assets validated via real-to-sim retargeting for robotic applications.

  </details>



- **S-MUSt3R: Sliding Multi-view 3D Reconstruction**  
  Leonid Antsfeld, Boris Chidlovskii, Yohann Cabon, Vincent Leroy, Jerome Revaud  
  _2026-02-04_ · https://arxiv.org/abs/2602.04517v1  
  <details><summary>Abstract</summary>

  The recent paradigm shift in 3D vision led to the rise of foundation models with remarkable capabilities in 3D perception from uncalibrated images. However, extending these models to large-scale RGB stream 3D reconstruction remains challenging due to memory limitations. This work proposes S-MUSt3R, a simple and efficient pipeline that extends the limits of foundation models for monocular 3D reconstruction. Our approach addresses the scalability bottleneck of foundation models through a simple strategy of sequence segmentation followed by segment alignment and lightweight loop closure optimization. Without model retraining, we benefit from remarkable 3D reconstruction capacities of MUSt3R model and achieve trajectory and reconstruction performance comparable to traditional methods with more complex architecture. We evaluate S-MUSt3R on TUM, 7-Scenes and proprietary robot navigation datasets and show that S-MUSt3R runs successfully on long RGB sequences and produces accurate and consistent 3D reconstruction. Our results highlight the potential of leveraging the MUSt3R model for scalable monocular 3D scene in real-world settings, with an important advantage of making predictions directly in the metric space.

  </details>



- **TrajVG: 3D Trajectory-Coupled Visual Geometry Learning**  
  Xingyu Miao, Weiguang Zhao, Tao Lu, Linning Xu, Mulin Yu, Yang Long, Jiangmiao Pang, Junting Dong  
  _2026-02-04_ · https://arxiv.org/abs/2602.04439v2  
  <details><summary>Abstract</summary>

  Feed-forward multi-frame 3D reconstruction models often degrade on videos with object motion. Global-reference becomes ambiguous under multiple motions, while the local pointmap relies heavily on estimated relative poses and can drift, causing cross-frame misalignment and duplicated structures. We propose TrajVG, a reconstruction framework that makes cross-frame 3D correspondence an explicit prediction by estimating camera-coordinate 3D trajectories. We couple sparse trajectories, per-frame local point maps, and relative camera poses with geometric consistency objectives: (i) bidirectional trajectory-pointmap consistency with controlled gradient flow, and (ii) a pose consistency objective driven by static track anchors that suppresses gradients from dynamic regions. To scale training to in-the-wild videos where 3D trajectory labels are scarce, we reformulate the same coupling constraints into self-supervised objectives using only pseudo 2D tracks, enabling unified training with mixed supervision. Extensive experiments across 3D tracking, pose estimation, pointmap reconstruction, and video depth show that TrajVG surpasses the current feedforward performance baseline.

  </details>



- **Finding NeMO: A Geometry-Aware Representation of Template Views for Few-Shot Perception**  
  Sebastian Jung, Leonard Klüpfel, Rudolph Triebel, Maximilian Durner  
  _2026-02-04_ · https://arxiv.org/abs/2602.04343v1  
  <details><summary>Abstract</summary>

  We present Neural Memory Object (NeMO), a novel object-centric representation that can be used to detect, segment and estimate the 6DoF pose of objects unseen during training using RGB images. Our method consists of an encoder that requires only a few RGB template views depicting an object to generate a sparse object-like point cloud using a learned UDF containing semantic and geometric information. Next, a decoder takes the object encoding together with a query image to generate a variety of dense predictions. Through extensive experiments, we show that our method can be used for few-shot object perception without requiring any camera-specific parameters or retraining on target data. Our proposed concept of outsourcing object information in a NeMO and using a single network for multiple perception tasks enhances interaction with novel objects, improving scalability and efficiency by enabling quick object onboarding without retraining or extensive pre-processing. We report competitive and state-of-the-art results on various datasets and perception tasks of the BOP benchmark, demonstrating the versatility of our approach. https://github.com/DLR-RM/nemo

  </details>



- **Improving 2D Diffusion Models for 3D Medical Imaging with Inter-Slice Consistent Stochasticity**  
  Chenhe Du, Qing Wu, Xuanyu Tian, Jingyi Yu, Hongjiang Wei, Yuyao Zhang  
  _2026-02-04_ · https://arxiv.org/abs/2602.04162v1  
  <details><summary>Abstract</summary>

  3D medical imaging is in high demand and essential for clinical diagnosis and scientific research. Currently, diffusion models (DMs) have become an effective tool for medical imaging reconstruction thanks to their ability to learn rich, high-quality data priors. However, learning the 3D data distribution with DMs in medical imaging is challenging, not only due to the difficulties in data collection but also because of the significant computational burden during model training. A common compromise is to train the DMs on 2D data priors and reconstruct stacked 2D slices to address 3D medical inverse problems. However, the intrinsic randomness of diffusion sampling causes severe inter-slice discontinuities of reconstructed 3D volumes. Existing methods often enforce continuity regularizations along the z-axis, which introduces sensitive hyper-parameters and may lead to over-smoothing results. In this work, we revisit the origin of stochasticity in diffusion sampling and introduce Inter-Slice Consistent Stochasticity (ISCS), a simple yet effective strategy that encourages interslice consistency during diffusion sampling. Our key idea is to control the consistency of stochastic noise components during diffusion sampling, thereby aligning their sampling trajectories without adding any new loss terms or optimization steps. Importantly, the proposed ISCS is plug-and-play and can be dropped into any 2D trained diffusion based 3D reconstruction pipeline without additional computational cost. Experiments on several medical imaging problems show that our method can effectively improve the performance of medical 3D imaging problems based on 2D diffusion models. Our findings suggest that controlling inter-slice stochasticity is a principled and practically attractive route toward high-fidelity 3D medical imaging with 2D diffusion priors. The code is available at: https://github.com/duchenhe/ISCS

  </details>



- **SuperPoint-E: local features for 3D reconstruction via tracking adaptation in endoscopy**  
  O. Leon Barbed, José M. M. Montiel, Pascal Fua, Ana C. Murillo  
  _2026-02-04_ · https://arxiv.org/abs/2602.04108v1  
  <details><summary>Abstract</summary>

  In this work, we focus on boosting the feature extraction to improve the performance of Structure-from-Motion (SfM) in endoscopy videos. We present SuperPoint-E, a new local feature extraction method that, using our proposed Tracking Adaptation supervision strategy, significantly improves the quality of feature detection and description in endoscopy. Extensive experimentation on real endoscopy recordings studies our approach's most suitable configuration and evaluates SuperPoint-E feature quality. The comparison with other baselines also shows that our 3D reconstructions are denser and cover more and longer video segments because our detector fires more densely and our features are more likely to survive (i.e. higher detection precision). In addition, our descriptor is more discriminative, making the guided matching step almost redundant. The presented approach brings significant improvements in the 3D reconstructions obtained, via SfM on endoscopy videos, compared to the original SuperPoint and the gold standard SfM COLMAP pipeline.

  </details>



- **Seeing Through Clutter: Structured 3D Scene Reconstruction via Iterative Object Removal**  
  Rio Aguina-Kang, Kevin James Blackburn-Matzen, Thibault Groueix, Vladimir Kim, Matheus Gadelha  
  _2026-02-03_ · https://arxiv.org/abs/2602.04053v1  
  <details><summary>Abstract</summary>

  We present SeeingThroughClutter, a method for reconstructing structured 3D representations from single images by segmenting and modeling objects individually. Prior approaches rely on intermediate tasks such as semantic segmentation and depth estimation, which often underperform in complex scenes, particularly in the presence of occlusion and clutter. We address this by introducing an iterative object removal and reconstruction pipeline that decomposes complex scenes into a sequence of simpler subtasks. Using VLMs as orchestrators, foreground objects are removed one at a time via detection, segmentation, object removal, and 3D fitting. We show that removing objects allows for cleaner segmentations of subsequent objects, even in highly occluded scenes. Our method requires no task-specific training and benefits directly from ongoing advances in foundation models. We demonstrate stateof-the-art robustness on 3D-Front and ADE20K datasets. Project Page: https://rioak.github.io/seeingthroughclutter/

  </details>



- **AnyStyle: Single-Pass Multimodal Stylization for 3D Gaussian Splatting**  
  Joanna Kaleta, Bartosz Świrta, Kacper Kania, Przemysław Spurek, Marek Kowalski  
  _2026-02-03_ · https://arxiv.org/abs/2602.04043v1  
  <details><summary>Abstract</summary>

  The growing demand for rapid and scalable 3D asset creation has driven interest in feed-forward 3D reconstruction methods, with 3D Gaussian Splatting (3DGS) emerging as an effective scene representation. While recent approaches have demonstrated pose-free reconstruction from unposed image collections, integrating stylization or appearance control into such pipelines remains underexplored. Existing attempts largely rely on image-based conditioning, which limits both controllability and flexibility. In this work, we introduce AnyStyle, a feed-forward 3D reconstruction and stylization framework that enables pose-free, zero-shot stylization through multimodal conditioning. Our method supports both textual and visual style inputs, allowing users to control the scene appearance using natural language descriptions or reference images. We propose a modular stylization architecture that requires only minimal architectural modifications and can be integrated into existing feed-forward 3D reconstruction backbones. Experiments demonstrate that AnyStyle improves style controllability over prior feed-forward stylization methods while preserving high-quality geometric reconstruction. A user study further confirms that AnyStyle achieves superior stylization quality compared to an existing state-of-the-art approach. Repository: https://github.com/joaxkal/AnyStyle.

  </details>



- **EventNeuS: 3D Mesh Reconstruction from a Single Event Camera**  
  Shreyas Sachan, Viktor Rudnev, Mohamed Elgharib, Christian Theobalt, Vladislav Golyanik  
  _2026-02-03_ · https://arxiv.org/abs/2602.03847v1  
  <details><summary>Abstract</summary>

  Event cameras offer a considerable alternative to RGB cameras in many scenarios. While there are recent works on event-based novel-view synthesis, dense 3D mesh reconstruction remains scarcely explored and existing event-based techniques are severely limited in their 3D reconstruction accuracy. To address this limitation, we present EventNeuS, a self-supervised neural model for learning 3D representations from monocular colour event streams. Our approach, for the first time, combines 3D signed distance function and density field learning with event-based supervision. Furthermore, we introduce spherical harmonics encodings into our model for enhanced handling of view-dependent effects. EventNeuS outperforms existing approaches by a significant margin, achieving 34% lower Chamfer distance and 31% lower mean absolute error on average compared to the best previous method.

  </details>



- **Beyond the Vehicle: Cooperative Localization by Fusing Point Clouds for GPS-Challenged Urban Scenarios**  
  Kuo-Yi Chao, Ralph Rasshofer, Alois Christian Knoll  
  _2026-02-03_ · https://arxiv.org/abs/2602.03908v1  
  <details><summary>Abstract</summary>

  Accurate vehicle localization is a critical challenge in urban environments where GPS signals are often unreliable. This paper presents a cooperative multi-sensor and multi-modal localization approach to address this issue by fusing data from vehicle-to-vehicle (V2V) and vehicle-to-infrastructure (V2I) systems. Our approach integrates cooperative data with a point cloud registration-based simultaneous localization and mapping (SLAM) algorithm. The system processes point clouds generated from diverse sensor modalities, including vehicle-mounted LiDAR and stereo cameras, as well as sensors deployed at intersections. By leveraging shared data from infrastructure, our method significantly improves localization accuracy and robustness in complex, GPS-noisy urban scenarios.

  </details>



- **AffordanceGrasp-R1:Leveraging Reasoning-Based Affordance Segmentation with Reinforcement Learning for Robotic Grasping**  
  Dingyi Zhou, Mu He, Zhuowei Fang, Xiangtong Yao, Yinlong Liu, Alois Knoll, Hu Cao  
  _2026-02-03_ · https://arxiv.org/abs/2602.03547v1  
  <details><summary>Abstract</summary>

  We introduce AffordanceGrasp-R1, a reasoning-driven affordance segmentation framework for robotic grasping that combines a chain-of-thought (CoT) cold-start strategy with reinforcement learning to enhance deduction and spatial grounding. In addition, we redesign the grasping pipeline to be more context-aware by generating grasp candidates from the global scene point cloud and subsequently filtering them using instruction-conditioned affordance masks. Extensive experiments demonstrate that AffordanceGrasp-R1 consistently outperforms state-of-the-art (SOTA) methods on benchmark datasets, and real-world robotic grasping evaluations further validate its robustness and generalization under complex language-conditioned manipulation scenarios.

  </details>



- **PWAVEP: Purifying Imperceptible Adversarial Perturbations in 3D Point Clouds via Spectral Graph Wavelets**  
  Haoran Li, Renyang Liu, Hongjia Liu, Chen Wang, Long Yin, Jian Xu  
  _2026-02-03_ · https://arxiv.org/abs/2602.03333v1  
  <details><summary>Abstract</summary>

  Recent progress in adversarial attacks on 3D point clouds, particularly in achieving spatial imperceptibility and high attack performance, presents significant challenges for defenders. Current defensive approaches remain cumbersome, often requiring invasive model modifications, expensive training procedures or auxiliary data access. To address these threats, in this paper, we propose a plug-and-play and non-invasive defense mechanism in the spectral domain, grounded in a theoretical and empirical analysis of the relationship between imperceptible perturbations and high-frequency spectral components. Building upon these insights, we introduce a novel purification framework, termed PWAVEP, which begins by computing a spectral graph wavelet domain saliency score and local sparsity score for each point. Guided by these values, PWAVEP adopts a hierarchical strategy, it eliminates the most salient points, which are identified as hardly recoverable adversarial outliers. Simultaneously, it applies a spectral filtering process to a broader set of moderately salient points. This process leverages a graph wavelet transform to attenuate high-frequency coefficients associated with the targeted points, thereby effectively suppressing adversarial noise. Extensive evaluations demonstrate that the proposed PWAVEP achieves superior accuracy and robustness compared to existing approaches, advancing the state-of-the-art in 3D point cloud purification. Code and datasets are available at https://github.com/a772316182/pwavep

  </details>


