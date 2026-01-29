# Neural Rendering & View Synthesis

_Updated: 2026-01-29 07:04 UTC_

Total papers shown: **6**


---

- **FreeFix: Boosting 3D Gaussian Splatting via Fine-Tuning-Free Diffusion Models**  
  Hongyu Zhou, Zisen Shao, Sheng Miao, Pan Wang, Dongfeng Bai, Bingbing Liu, Yiyi Liao  
  _2026-01-28_ · https://arxiv.org/abs/2601.20857v1  
  <details><summary>Abstract</summary>

  Neural Radiance Fields and 3D Gaussian Splatting have advanced novel view synthesis, yet still rely on dense inputs and often degrade at extrapolated views. Recent approaches leverage generative models, such as diffusion models, to provide additional supervision, but face a trade-off between generalization and fidelity: fine-tuning diffusion models for artifact removal improves fidelity but risks overfitting, while fine-tuning-free methods preserve generalization but often yield lower fidelity. We introduce FreeFix, a fine-tuning-free approach that pushes the boundary of this trade-off by enhancing extrapolated rendering with pretrained image diffusion models. We present an interleaved 2D-3D refinement strategy, showing that image diffusion models can be leveraged for consistent refinement without relying on costly video diffusion models. Furthermore, we take a closer look at the guidance signal for 2D refinement and propose a per-pixel confidence mask to identify uncertain regions for targeted improvement. Experiments across multiple datasets show that FreeFix improves multi-frame consistency and achieves performance comparable to or surpassing fine-tuning-based methods, while retaining strong generalization ability.

  </details>



- **Graphical X Splatting (GraphiXS): A Graphical Model for 4D Gaussian Splatting under Uncertainty**  
  Doga Yilmaz, Jialin Zhu, Deshan Gong, He Wang  
  _2026-01-27_ · https://arxiv.org/abs/2601.19843v1  
  <details><summary>Abstract</summary>

  We propose a new framework to systematically incorporate data uncertainty in Gaussian Splatting. Being the new paradigm of neural rendering, Gaussian Splatting has been investigated in many applications, with the main effort in extending its representation, improving its optimization process, and accelerating its speed. However, one orthogonal, much needed, but under-explored area is data uncertainty. In standard 4D Gaussian Splatting, data uncertainty can manifest as view sparsity, missing frames, camera asynchronization, etc. So far, there has been little research to holistically incorporating various types of data uncertainty under a single framework. To this end, we propose Graphical X Splatting, or GraphiXS, a new probabilistic framework that considers multiple types of data uncertainty, aiming for a fundamental augmentation of the current 4D Gaussian Splatting paradigm into a probabilistic setting. GraphiXS is general and can be instantiated with a range of primitives, e.g. Gaussians, Student's-t. Furthermore, GraphiXS can be used to `upgrade' existing methods to accommodate data uncertainty. Through exhaustive evaluation and comparison, we demonstrate that GraphiXS can systematically model various uncertainties in data, outperform existing methods in many settings where data are missing or polluted in space and time, and therefore is a major generalization of the current 4D Gaussian Splatting research.

  </details>



- **WaterClear-GS: Optical-Aware Gaussian Splatting for Underwater Reconstruction and Restoration**  
  Xinrui Zhang, Yufeng Wang, Shuangkang Fang, Zesheng Wang, Dacheng Qi, Wenrui Ding  
  _2026-01-27_ · https://arxiv.org/abs/2601.19753v1  
  <details><summary>Abstract</summary>

  Underwater 3D reconstruction and appearance restoration are hindered by the complex optical properties of water, such as wavelength-dependent attenuation and scattering. Existing Neural Radiance Fields (NeRF)-based methods struggle with slow rendering speeds and suboptimal color restoration, while 3D Gaussian Splatting (3DGS) inherently lacks the capability to model complex volumetric scattering effects. To address these issues, we introduce WaterClear-GS, the first pure 3DGS-based framework that explicitly integrates underwater optical properties of local attenuation and scattering into Gaussian primitives, eliminating the need for an auxiliary medium network. Our method employs a dual-branch optimization strategy to ensure underwater photometric consistency while naturally recovering water-free appearances. This strategy is enhanced by depth-guided geometry regularization and perception-driven image loss, together with exposure constraints, spatially-adaptive regularization, and physically guided spectral regularization, which collectively enforce local 3D coherence and maintain natural visual perception. Experiments on standard benchmarks and our newly collected dataset demonstrate that WaterClear-GS achieves outstanding performance on both novel view synthesis (NVS) and underwater image restoration (UIR) tasks, while maintaining real-time rendering. The code will be available at https://buaaxrzhang.github.io/WaterClear-GS/.

  </details>



- **Pay Attention to Where You Look**  
  Alex Beriand, JhihYang Wu, Daniel Brignac, Natnael Daba, Abhijit Mahalanobis  
  _2026-01-26_ · https://arxiv.org/abs/2601.18970v1  
  <details><summary>Abstract</summary>

  Novel view synthesis (NVS) has advanced with generative modeling, enabling photorealistic image generation. In few-shot NVS, where only a few input views are available, existing methods often assume equal importance for all input views relative to the target, leading to suboptimal results. We address this limitation by introducing a camera-weighting mechanism that adjusts the importance of source views based on their relevance to the target. We propose two approaches: a deterministic weighting scheme leveraging geometric properties like Euclidean distance and angular differences, and a cross-attention-based learning scheme that optimizes view weighting. Additionally, models can be further trained with our camera-weighting scheme to refine their understanding of view relevance and enhance synthesis quality. This mechanism is adaptable and can be integrated into various NVS algorithms, improving their ability to synthesize high-quality novel views. Our results demonstrate that adaptive view weighting enhances accuracy and realism, offering a promising direction for improving NVS.

  </details>



- **Splat-Portrait: Generalizing Talking Heads with Gaussian Splatting**  
  Tong Shi, Melonie de Almeida, Daniela Ivanova, Nicolas Pugeault, Paul Henderson  
  _2026-01-26_ · https://arxiv.org/abs/2601.18633v1  
  <details><summary>Abstract</summary>

  Talking Head Generation aims at synthesizing natural-looking talking videos from speech and a single portrait image. Previous 3D talking head generation methods have relied on domain-specific heuristics such as warping-based facial motion representation priors to animate talking motions, yet still produce inaccurate 3D avatar reconstructions, thus undermining the realism of generated animations. We introduce Splat-Portrait, a Gaussian-splatting-based method that addresses the challenges of 3D head reconstruction and lip motion synthesis. Our approach automatically learns to disentangle a single portrait image into a static 3D reconstruction represented as static Gaussian Splatting, and a predicted whole-image 2D background. It then generates natural lip motion conditioned on input audio, without any motion driven priors. Training is driven purely by 2D reconstruction and score-distillation losses, without 3D supervision nor landmarks. Experimental results demonstrate that Splat-Portrait exhibits superior performance on talking head generation and novel view synthesis, achieving better visual quality compared to previous works. Our project code and supplementary documents are public available at https://github.com/stonewalking/Splat-portrait.

  </details>



- **PPISP: Physically-Plausible Compensation and Control of Photometric Variations in Radiance Field Reconstruction**  
  Isaac Deutsch, Nicolas Moënne-Loccoz, Gavriel State, Zan Gojcic  
  _2026-01-26_ · https://arxiv.org/abs/2601.18336v1  
  <details><summary>Abstract</summary>

  Multi-view 3D reconstruction methods remain highly sensitive to photometric inconsistencies arising from camera optical characteristics and variations in image signal processing (ISP). Existing mitigation strategies such as per-frame latent variables or affine color corrections lack physical grounding and generalize poorly to novel views. We propose the Physically-Plausible ISP (PPISP) correction module, which disentangles camera-intrinsic and capture-dependent effects through physically based and interpretable transformations. A dedicated PPISP controller, trained on the input views, predicts ISP parameters for novel viewpoints, analogous to auto exposure and auto white balance in real cameras. This design enables realistic and fair evaluation on novel views without access to ground-truth images. PPISP achieves SoTA performance on standard benchmarks, while providing intuitive control and supporting the integration of metadata when available. The source code is available at: https://github.com/nv-tlabs/ppisp

  </details>


