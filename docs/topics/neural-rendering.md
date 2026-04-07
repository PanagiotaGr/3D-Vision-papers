# Neural Rendering & View Synthesis

_Updated: 2026-04-07 07:47 UTC_

Total papers shown: **4**


---

- **PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis**  
  Inseong Choi, Siwoo Lee, Seung-Hun Nam, Soohwan Song  
  _2026-04-06_ · https://arxiv.org/abs/2604.04576v1  
  <details><summary>Abstract</summary>

  Diffusion models are promising for sparse-view novel view synthesis (NVS), as they can generate pseudo-ground-truth views to aid 3D reconstruction pipelines like 3D Gaussian Splatting (3DGS). However, these synthesized images often contain photometric and geometric inconsistencies, and their direct use for supervision can impair reconstruction. To address this, we propose Partial-Reference Image Quality Assessment (PR-IQA), a framework that evaluates diffusion-generated views using reference images from different poses, eliminating the need for ground truth. PR-IQA first computes a geometrically consistent partial quality map in overlapping regions. It then performs quality completion to inpaint this partial map into a dense, full-image map. This completion is achieved via a cross-attention mechanism that incorporates reference-view context, ensuring cross-view consistency and enabling thorough quality assessment. When integrated into a diffusion-augmented 3DGS pipeline, PR-IQA restricts supervision to high-confidence regions identified by its quality maps. Experiments demonstrate that PR-IQA outperforms existing IQA methods, achieving full-reference-level accuracy without ground-truth supervision. Thus, our quality-aware 3DGS approach more effectively filters inconsistencies, producing superior 3D reconstructions and NVS results.The project page is available at https://kakaomacao.github.io/pr-iqa-project-page/.

  </details>



- **Temporal Inversion for Learning Interval Change in Chest X-Rays**  
  Hanbin Ko, Kyeongmin Jeon, Doowoong Choi, Chang Min Park  
  _2026-04-06_ · https://arxiv.org/abs/2604.04563v1  
  <details><summary>Abstract</summary>

  Recent advances in vision--language pretraining have enabled strong medical foundation models, yet most analyze radiographs in isolation, overlooking the key clinical task of comparing prior and current images to assess interval change. For chest radiographs (CXRs), capturing interval change is essential, as radiologists must evaluate not only the static appearance of findings but also how they evolve over time. We introduce TILA (Temporal Inversion-aware Learning and Alignment), a simple yet effective framework that uses temporal inversion, reversing image pairs, as a supervisory signal to enhance the sensitivity of existing temporal vision-language models to directional change. TILA integrates inversion-aware objectives across pretraining, fine-tuning, and inference, complementing conventional appearance modeling with explicit learning of temporal order. We also propose a unified evaluation protocol to assess order sensitivity and consistency under temporal inversion, and introduce MS-CXR-Tretrieval, a retrieval evaluation set constructed through a general protocol that can be applied to any temporal CXR dataset. Experiments on public datasets and real-world hospital cohorts demonstrate that TILA consistently improves progression classification and temporal embedding alignment when applied to multiple existing architectures.

  </details>



- **4C4D: 4 Camera 4D Gaussian Splatting**  
  Junsheng Zhou, Zhifan Yang, Liang Han, Wenyuan Zhang, Kanle Shi, Shenkun Xu, Yu-Shen Liu  
  _2026-04-05_ · https://arxiv.org/abs/2604.04063v1  
  <details><summary>Abstract</summary>

  This paper tackles the challenge of recovering 4D dynamic scenes from videos captured by as few as four portable cameras. Learning to model scene dynamics for temporally consistent novel-view rendering is a foundational task in computer graphics, where previous works often require dense multi-view captures using camera arrays of dozens or even hundreds of views. We propose \textbf{4C4D}, a novel framework that enables high-fidelity 4D Gaussian Splatting from video captures of extremely sparse cameras. Our key insight lies that the geometric learning under sparse settings is substantially more difficult than modeling appearance. Driven by this observation, we introduce a Neural Decaying Function on Gaussian opacities for enhancing the geometric modeling capability of 4D Gaussians. This design mitigates the inherent imbalance between geometry and appearance modeling in 4DGS by encouraging the 4DGS gradients to focus more on geometric learning. Extensive experiments across sparse-view datasets with varying camera overlaps show that 4C4D achieves superior performance over prior art. Project page at: https://junshengzhou.github.io/4C4D.

  </details>



- **M2StyleGS: Multi-Modality 3D Style Transfer with Gaussian Splatting**  
  Xingyu Miao, Xueqi Qiu, Haoran Duan, Yawen Huang, Xian Wu, Jingjing Deng, Yang Long  
  _2026-04-04_ · https://arxiv.org/abs/2604.03773v1  
  <details><summary>Abstract</summary>

  Conventional 3D style transfer methods rely on a fixed reference image to apply artistic patterns to 3D scenes. However, in practical applications such as virtual or augmented reality, users often prefer more flexible inputs, including textual descriptions and diverse imagery. In this work, we introduce a novel real-time styling technique M2StyleGS to generate a sequence of precisely color-mapped views. It utilizes 3D Gaussian Splatting (3DGS) as a 3D presentation and multi-modality knowledge refined by CLIP as a reference style. M2StyleGS resolves the abnormal transformation issue by employing a precise feature alignment, namely subdivisive flow, it strengthens the projection of the mapped CLIP text-visual combination feature to the VGG style feature. In addition, we introduce observation loss, which assists in the stylized scene better matching the reference style during the generation, and suppression loss, which suppresses the offset of reference color information throughout the decoding process. By integrating these approaches, M2StyleGS can employ text or images as references to generate a set of style-enhanced novel views. Our experiments show that M2StyleGS achieves better visual quality and surpasses the previous work by up to 32.92% in terms of consistency.

  </details>


