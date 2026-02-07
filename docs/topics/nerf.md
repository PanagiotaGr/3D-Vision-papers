# NeRF & Neural Radiance Fields

_Updated: 2026-02-07 06:59 UTC_

Total papers shown: **7**


---

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



- **NeVStereo: A NeRF-Driven NVS-Stereo Architecture for High-Fidelity 3D Tasks**  
  Pengcheng Chen, Yue Hu, Wenhao Li, Nicole M Gunderson, Andrew Feng, Zhenglong Sun, Peter Beerel, Eric J Seibel  
  _2026-02-05_ · https://arxiv.org/abs/2602.05423v1  
  <details><summary>Abstract</summary>

  In modern dense 3D reconstruction, feed-forward systems (e.g., VGGT, pi3) focus on end-to-end matching and geometry prediction but do not explicitly output the novel view synthesis (NVS). Neural rendering-based approaches offer high-fidelity NVS and detailed geometry from posed images, yet they typically assume fixed camera poses and can be sensitive to pose errors. As a result, it remains non-trivial to obtain a single framework that can offer accurate poses, reliable depth, high-quality rendering, and accurate 3D surfaces from casually captured views. We present NeVStereo, a NeRF-driven NVS-stereo architecture that aims to jointly deliver camera poses, multi-view depth, novel view synthesis, and surface reconstruction from multi-view RGB-only inputs. NeVStereo combines NeRF-based NVS for stereo-friendly renderings, confidence-guided multi-view depth estimation, NeRF-coupled bundle adjustment for pose refinement, and an iterative refinement stage that updates both depth and the radiance field to improve geometric consistency. This design mitigated the common NeRF-based issues such as surface stacking, artifacts, and pose-depth coupling. Across indoor, outdoor, tabletop, and aerial benchmarks, our experiments indicate that NeVStereo achieves consistently strong zero-shot performance, with up to 36% lower depth error, 10.4% improved pose accuracy, 4.5% higher NVS fidelity, and state-of-the-art mesh quality (F1 91.93%, Chamfer 4.35 mm) compared to existing prestigious methods.

  </details>



- **Dual-Representation Image Compression at Ultra-Low Bitrates via Explicit Semantics and Implicit Textures**  
  Chuqin Zhou, Xiaoyue Ling, Yunuo Chen, Jincheng Dai, Guo Lu, Wenjun Zhang  
  _2026-02-05_ · https://arxiv.org/abs/2602.05213v1  
  <details><summary>Abstract</summary>

  While recent neural codecs achieve strong performance at low bitrates when optimized for perceptual quality, their effectiveness deteriorates significantly under ultra-low bitrate conditions. To mitigate this, generative compression methods leveraging semantic priors from pretrained models have emerged as a promising paradigm. However, existing approaches are fundamentally constrained by a tradeoff between semantic faithfulness and perceptual realism. Methods based on explicit representations preserve content structure but often lack fine-grained textures, whereas implicit methods can synthesize visually plausible details at the cost of semantic drift. In this work, we propose a unified framework that bridges this gap by coherently integrating explicit and implicit representations in a training-free manner. Specifically, We condition a diffusion model on explicit high-level semantics while employing reverse-channel coding to implicitly convey fine-grained details. Moreover, we introduce a plug-in encoder that enables flexible control of the distortion-perception tradeoff by modulating the implicit information. Extensive experiments demonstrate that the proposed framework achieves state-of-the-art rate-perception performance, outperforming existing methods and surpassing DiffC by 29.92%, 19.33%, and 20.89% in DISTS BD-Rate on the Kodak, DIV2K, and CLIC2020 datasets, respectively.

  </details>



- **PoseGaussian: Pose-Driven Novel View Synthesis for Robust 3D Human Reconstruction**  
  Ju Shen, Chen Chen, Tam V. Nguyen, Vijayan K. Asari  
  _2026-02-05_ · https://arxiv.org/abs/2602.05190v1  
  <details><summary>Abstract</summary>

  We propose PoseGaussian, a pose-guided Gaussian Splatting framework for high-fidelity human novel view synthesis. Human body pose serves a dual purpose in our design: as a structural prior, it is fused with a color encoder to refine depth estimation; as a temporal cue, it is processed by a dedicated pose encoder to enhance temporal consistency across frames. These components are integrated into a fully differentiable, end-to-end trainable pipeline. Unlike prior works that use pose only as a condition or for warping, PoseGaussian embeds pose signals into both geometric and temporal stages to improve robustness and generalization. It is specifically designed to address challenges inherent in dynamic human scenes, such as articulated motion and severe self-occlusion. Notably, our framework achieves real-time rendering at 100 FPS, maintaining the efficiency of standard Gaussian Splatting pipelines. We validate our approach on ZJU-MoCap, THuman2.0, and in-house datasets, demonstrating state-of-the-art performance in perceptual quality and structural accuracy (PSNR 30.86, SSIM 0.979, LPIPS 0.028).

  </details>



- **Gabor Fields: Orientation-Selective Level-of-Detail for Volume Rendering**  
  Jorge Condor, Nicolai Hermann, Mehmet Ata Yurtsever, Piotr Didyk  
  _2026-02-04_ · https://arxiv.org/abs/2602.05081v1  
  <details><summary>Abstract</summary>

  Gaussian-based representations have enabled efficient physically-based volume rendering at a fraction of the memory cost of regular, discrete, voxel-based distributions. However, several remaining issues hamper their widespread use. One of the advantages of classic voxel grids is the ease of constructing hierarchical representations by either storing volumetric mipmaps or selectively pruning branches of an already hierarchical voxel grid. Such strategies reduce rendering time and eliminate aliasing when lower levels of detail are required. Constructing similar strategies for Gaussian-based volumes is not trivial. Straightforward solutions, such as prefiltering or computing mipmap-style representations, lead to increased memory requirements or expensive re-fitting of each level separately. Additionally, such solutions do not guarantee a smooth transition between different hierarchy levels. To address these limitations, we propose Gabor Fields, an orientation-selective mixture of Gabor kernels that enables continuous frequency filtering at no cost. The frequency content of the asset is reduced by selectively pruning primitives, directly benefiting rendering performance. Beyond filtering, we demonstrate that stochastically sampling from different frequencies and orientations at each ray recursion enables masking substantial portions of the volume, accelerating ray traversal time in single- and multiple-scattering settings. Furthermore, inspired by procedural volumes, we present an application for efficient design and rendering of procedural clouds as Gabor-noise-modulated Gaussians.

  </details>



- **Nix and Fix: Targeting 1000x Compression of 3D Gaussian Splatting with Diffusion Models**  
  Cem Eteke, Enzo Tartaglione  
  _2026-02-04_ · https://arxiv.org/abs/2602.04549v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) revolutionized novel view rendering. Instead of inferring from dense spatial points, as implicit representations do, 3DGS uses sparse Gaussians. This enables real-time performance but increases space requirements, hindering applications such as immersive communication. 3DGS compression emerged as a field aimed at alleviating this issue. While impressive progress has been made, at low rates, compression introduces artifacts that degrade visual quality significantly. We introduce NiFi, a method for extreme 3DGS compression through restoration via artifact-aware, diffusion-based one-step distillation. We show that our method achieves state-of-the-art perceptual quality at extremely low rates, down to 0.1 MB, and towards 1000x rate improvement over 3DGS at comparable perceptual performance. The code will be open-sourced upon acceptance.

  </details>


