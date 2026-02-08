# 3D Reconstruction

_Updated: 2026-02-08 07:05 UTC_

Total papers shown: **8**


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


