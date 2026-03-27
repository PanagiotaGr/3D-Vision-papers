# SLAM & Localization

_Updated: 2026-03-27 07:34 UTC_

Total papers shown: **2**


---

- **GaussFusion: Improving 3D Reconstruction in the Wild with A Geometry-Informed Video Generator**  
  Liyuan Zhu, Manjunath Narayana, Michal Stary, Will Hutchcroft, Gordon Wetzstein, Iro Armeni  
  _2026-03-26_ · https://arxiv.org/abs/2603.25053v1  
  <details><summary>Abstract</summary>

  We present GaussFusion, a novel approach for improving 3D Gaussian splatting (3DGS) reconstructions in the wild through geometry-informed video generation. GaussFusion mitigates common 3DGS artifacts, including floaters, flickering, and blur caused by camera pose errors, incomplete coverage, and noisy geometry initialization. Unlike prior RGB-based approaches limited to a single reconstruction pipeline, our method introduces a geometry-informed video-to-video generator that refines 3DGS renderings across both optimization-based and feed-forward methods. Given an existing reconstruction, we render a Gaussian primitive video buffer encoding depth, normals, opacity, and covariance, which the generator refines to produce temporally coherent, artifact-free frames. We further introduce an artifact synthesis pipeline that simulates diverse degradation patterns, ensuring robustness and generalization. GaussFusion achieves state-of-the-art performance on novel-view synthesis benchmarks, and an efficient variant runs in real time at 21 FPS while maintaining similar performance, enabling interactive 3D applications.

  </details>



- **Comparative analysis of dual-form networks for live land monitoring using multi-modal satellite image time series**  
  Iris Dumeur, Jérémy Anger, Gabriele Facciolo  
  _2026-03-25_ · https://arxiv.org/abs/2603.24109v1  
  <details><summary>Abstract</summary>

  Multi-modal Satellite Image Time Series (SITS) analysis faces significant computational challenges for live land monitoring applications. While Transformer architectures excel at capturing temporal dependencies and fusing multi-modal data, their quadratic computational complexity and the need to reprocess entire sequences for each new acquisition limit their deployment for regular, large-area monitoring. This paper studies various dual-form attention mechanisms for efficient multi-modal SITS analysis, that enable parallel training while supporting recurrent inference for incremental processing. We compare linear attention and retention mechanisms within a multi-modal spectro-temporal encoder. To address SITS-specific challenges of temporal irregularity and unalignment, we develop temporal adaptations of dual-form mechanisms that compute token distances based on actual acquisition dates rather than sequence indices. Our approach is evaluated on two tasks using Sentinel-1 and Sentinel-2 data: multi-modal SITS forecasting as a proxy task, and real-world solar panel construction monitoring. Experimental results demonstrate that dual-form mechanisms achieve performance comparable to standard Transformers while enabling efficient recurrent inference. The multimodal framework consistently outperforms mono-modal approaches across both tasks, demonstrating the effectiveness of dual mechanisms for sensor fusion. The results presented in this work open new opportunities for operational land monitoring systems requiring regular updates over large geographic areas.

  </details>


