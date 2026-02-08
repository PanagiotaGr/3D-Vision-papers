# NeRF & Neural Radiance Fields

_Updated: 2026-02-08 07:05 UTC_

Total papers shown: **3**


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


