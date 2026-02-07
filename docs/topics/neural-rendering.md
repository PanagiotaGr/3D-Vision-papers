# Neural Rendering & View Synthesis

_Updated: 2026-02-07 06:59 UTC_

Total papers shown: **8**


---

- **Splat and Distill: Augmenting Teachers with Feed-Forward 3D Reconstruction For 3D-Aware Distillation**  
  David Shavin, Sagie Benaim  
  _2026-02-05_ · https://arxiv.org/abs/2602.06032v1  
  <details><summary>Abstract</summary>

  Vision Foundation Models (VFMs) have achieved remarkable success when applied to various downstream 2D tasks. Despite their effectiveness, they often exhibit a critical lack of 3D awareness. To this end, we introduce Splat and Distill, a framework that instills robust 3D awareness into 2D VFMs by augmenting the teacher model with a fast, feed-forward 3D reconstruction pipeline. Given 2D features produced by a teacher model, our method first lifts these features into an explicit 3D Gaussian representation, in a feedforward manner. These 3D features are then ``splatted" onto novel viewpoints, producing a set of novel 2D feature maps used to supervise the student model, ``distilling" geometrically grounded knowledge. By replacing slow per-scene optimization of prior work with our feed-forward lifting approach, our framework avoids feature-averaging artifacts, creating a dynamic learning process where the teacher's consistency improves alongside that of the student. We conduct a comprehensive evaluation on a suite of downstream tasks, including monocular depth estimation, surface normal estimation, multi-view correspondence, and semantic segmentation. Our method significantly outperforms prior works, not only achieving substantial gains in 3D awareness but also enhancing the underlying semantic richness of 2D features. Project page is available at https://davidshavin4.github.io/Splat-and-Distill/

  </details>



- **NVS-HO: A Benchmark for Novel View Synthesis of Handheld Objects**  
  Musawar Ali, Manuel Carranza-García, Nicola Fioraio, Samuele Salti, Luigi Di Stefano  
  _2026-02-05_ · https://arxiv.org/abs/2602.05822v1  
  <details><summary>Abstract</summary>

  We propose NVS-HO, the first benchmark designed for novel view synthesis of handheld objects in real-world environments using only RGB inputs. Each object is recorded in two complementary RGB sequences: (1) a handheld sequence, where the object is manipulated in front of a static camera, and (2) a board sequence, where the object is fixed on a ChArUco board to provide accurate camera poses via marker detection. The goal of NVS-HO is to learn a NVS model that captures the full appearance of an object from (1), whereas (2) provides the ground-truth images used for evaluation. To establish baselines, we consider both a classical SfM pipeline and a state-of-the-art pre-trained feed-forward neural network (VGGT) as pose estimators, and train NVS models based on NeRF and Gaussian Splatting. Our experiments reveal significant performance gaps in current methods under unconstrained handheld conditions, highlighting the need for more robust approaches. NVS-HO thus offers a challenging real-world benchmark to drive progress in RGB-based novel view synthesis of handheld objects.

  </details>



- **NeVStereo: A NeRF-Driven NVS-Stereo Architecture for High-Fidelity 3D Tasks**  
  Pengcheng Chen, Yue Hu, Wenhao Li, Nicole M Gunderson, Andrew Feng, Zhenglong Sun, Peter Beerel, Eric J Seibel  
  _2026-02-05_ · https://arxiv.org/abs/2602.05423v1  
  <details><summary>Abstract</summary>

  In modern dense 3D reconstruction, feed-forward systems (e.g., VGGT, pi3) focus on end-to-end matching and geometry prediction but do not explicitly output the novel view synthesis (NVS). Neural rendering-based approaches offer high-fidelity NVS and detailed geometry from posed images, yet they typically assume fixed camera poses and can be sensitive to pose errors. As a result, it remains non-trivial to obtain a single framework that can offer accurate poses, reliable depth, high-quality rendering, and accurate 3D surfaces from casually captured views. We present NeVStereo, a NeRF-driven NVS-stereo architecture that aims to jointly deliver camera poses, multi-view depth, novel view synthesis, and surface reconstruction from multi-view RGB-only inputs. NeVStereo combines NeRF-based NVS for stereo-friendly renderings, confidence-guided multi-view depth estimation, NeRF-coupled bundle adjustment for pose refinement, and an iterative refinement stage that updates both depth and the radiance field to improve geometric consistency. This design mitigated the common NeRF-based issues such as surface stacking, artifacts, and pose-depth coupling. Across indoor, outdoor, tabletop, and aerial benchmarks, our experiments indicate that NeVStereo achieves consistently strong zero-shot performance, with up to 36% lower depth error, 10.4% improved pose accuracy, 4.5% higher NVS fidelity, and state-of-the-art mesh quality (F1 91.93%, Chamfer 4.35 mm) compared to existing prestigious methods.

  </details>



- **PoseGaussian: Pose-Driven Novel View Synthesis for Robust 3D Human Reconstruction**  
  Ju Shen, Chen Chen, Tam V. Nguyen, Vijayan K. Asari  
  _2026-02-05_ · https://arxiv.org/abs/2602.05190v1  
  <details><summary>Abstract</summary>

  We propose PoseGaussian, a pose-guided Gaussian Splatting framework for high-fidelity human novel view synthesis. Human body pose serves a dual purpose in our design: as a structural prior, it is fused with a color encoder to refine depth estimation; as a temporal cue, it is processed by a dedicated pose encoder to enhance temporal consistency across frames. These components are integrated into a fully differentiable, end-to-end trainable pipeline. Unlike prior works that use pose only as a condition or for warping, PoseGaussian embeds pose signals into both geometric and temporal stages to improve robustness and generalization. It is specifically designed to address challenges inherent in dynamic human scenes, such as articulated motion and severe self-occlusion. Notably, our framework achieves real-time rendering at 100 FPS, maintaining the efficiency of standard Gaussian Splatting pipelines. We validate our approach on ZJU-MoCap, THuman2.0, and in-house datasets, demonstrating state-of-the-art performance in perceptual quality and structural accuracy (PSNR 30.86, SSIM 0.979, LPIPS 0.028).

  </details>



- **QuantumGS: Quantum Encoding Framework for Gaussian Splatting**  
  Grzegorz Wilczyński, Rafał Tobiasz, Paweł Gora, Marcin Mazur, Przemysław Spurek  
  _2026-02-04_ · https://arxiv.org/abs/2602.05047v1  
  <details><summary>Abstract</summary>

  Recent advances in neural rendering, particularly 3D Gaussian Splatting (3DGS), have enabled real-time rendering of complex scenes. However, standard 3DGS relies on spherical harmonics, which often struggle to accurately capture high-frequency view-dependent effects such as sharp reflections and transparency. While hybrid approaches like Viewing Direction Gaussian Splatting (VDGS) mitigate this limitation using classical Multi-Layer Perceptrons (MLPs), they remain limited by the expressivity of classical networks in low-parameter regimes. In this paper, we introduce QuantumGS, a novel hybrid framework that integrates Variational Quantum Circuits (VQC) into the Gaussian Splatting pipeline. We propose a unique encoding strategy that maps the viewing direction directly onto the Bloch sphere, leveraging the natural geometry of qubits to represent 3D directional data. By replacing classical color-modulating networks with quantum circuits generated via a hypernetwork or conditioning mechanism, we achieve higher expressivity and better generalization. Source code is available in the supplementary material. Code is available at https://github.com/gwilczynski95/QuantumGS

  </details>



- **Differentiable Inverse Graphics for Zero-shot Scene Reconstruction and Robot Grasping**  
  Octavio Arriaga, Proneet Sharma, Jichen Guo, Marc Otto, Siddhant Kadwe, Rebecca Adam  
  _2026-02-04_ · https://arxiv.org/abs/2602.05029v1  
  <details><summary>Abstract</summary>

  Operating effectively in novel real-world environments requires robotic systems to estimate and interact with previously unseen objects. Current state-of-the-art models address this challenge by using large amounts of training data and test-time samples to build black-box scene representations. In this work, we introduce a differentiable neuro-graphics model that combines neural foundation models with physics-based differentiable rendering to perform zero-shot scene reconstruction and robot grasping without relying on any additional 3D data or test-time samples. Our model solves a series of constrained optimization problems to estimate physically consistent scene parameters, such as meshes, lighting conditions, material properties, and 6D poses of previously unseen objects from a single RGBD image and bounding boxes. We evaluated our approach on standard model-free few-shot benchmarks and demonstrated that it outperforms existing algorithms for model-free few-shot pose estimation. Furthermore, we validated the accuracy of our scene reconstructions by applying our algorithm to a zero-shot grasping task. By enabling zero-shot, physically-consistent scene reconstruction and grasping without reliance on extensive datasets or test-time sampling, our approach offers a pathway towards more data efficient, interpretable and generalizable robot autonomy in novel environments.

  </details>



- **AGILE: Hand-Object Interaction Reconstruction from Video via Agentic Generation**  
  Jin-Chuan Shi, Binhong Ye, Tao Liu, Junzhe He, Yangjinhui Xu, Xiaoyang Liu, Zeju Li, Hao Chen, Chunhua Shen  
  _2026-02-04_ · https://arxiv.org/abs/2602.04672v1  
  <details><summary>Abstract</summary>

  Reconstructing dynamic hand-object interactions from monocular videos is critical for dexterous manipulation data collection and creating realistic digital twins for robotics and VR. However, current methods face two prohibitive barriers: (1) reliance on neural rendering often yields fragmented, non-simulation-ready geometries under heavy occlusion, and (2) dependence on brittle Structure-from-Motion (SfM) initialization leads to frequent failures on in-the-wild footage. To overcome these limitations, we introduce AGILE, a robust framework that shifts the paradigm from reconstruction to agentic generation for interaction learning. First, we employ an agentic pipeline where a Vision-Language Model (VLM) guides a generative model to synthesize a complete, watertight object mesh with high-fidelity texture, independent of video occlusions. Second, bypassing fragile SfM entirely, we propose a robust anchor-and-track strategy. We initialize the object pose at a single interaction onset frame using a foundation model and propagate it temporally by leveraging the strong visual similarity between our generated asset and video observations. Finally, a contact-aware optimization integrates semantic, geometric, and interaction stability constraints to enforce physical plausibility. Extensive experiments on HO3D, DexYCB, and in-the-wild videos reveal that AGILE outperforms baselines in global geometric accuracy while demonstrating exceptional robustness on challenging sequences where prior art frequently collapses. By prioritizing physical validity, our method produces simulation-ready assets validated via real-to-sim retargeting for robotic applications.

  </details>



- **Nix and Fix: Targeting 1000x Compression of 3D Gaussian Splatting with Diffusion Models**  
  Cem Eteke, Enzo Tartaglione  
  _2026-02-04_ · https://arxiv.org/abs/2602.04549v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) revolutionized novel view rendering. Instead of inferring from dense spatial points, as implicit representations do, 3DGS uses sparse Gaussians. This enables real-time performance but increases space requirements, hindering applications such as immersive communication. 3DGS compression emerged as a field aimed at alleviating this issue. While impressive progress has been made, at low rates, compression introduces artifacts that degrade visual quality significantly. We introduce NiFi, a method for extreme 3DGS compression through restoration via artifact-aware, diffusion-based one-step distillation. We show that our method achieves state-of-the-art perceptual quality at extremely low rates, down to 0.1 MB, and towards 1000x rate improvement over 3DGS at comparable perceptual performance. The code will be open-sourced upon acceptance.

  </details>


