# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-02-16 07:19 UTC_

Total papers shown: **3**


---

- **MASAR: Motion-Appearance Synergy Refinement for Joint Detection and Trajectory Forecasting**  
  Mohammed Amine Bencheikh Lehocine, Julian Schmidt, Frank Moosmann, Dikshant Gupta, Fabian Flohr  
  _2026-02-13_ · https://arxiv.org/abs/2602.13003v1  
  <details><summary>Abstract</summary>

  Classical autonomous driving systems connect perception and prediction modules via hand-crafted bounding-box interfaces, limiting information flow and propagating errors to downstream tasks. Recent research aims to develop end-to-end models that jointly address perception and prediction; however, they often fail to fully exploit the synergy between appearance and motion cues, relying mainly on short-term visual features. We follow the idea of "looking backward to look forward", and propose MASAR, a novel fully differentiable framework for joint 3D detection and trajectory forecasting compatible with any transformer-based 3D detector. MASAR employs an object-centric spatio-temporal mechanism that jointly encodes appearance and motion features. By predicting past trajectories and refining them using guidance from appearance cues, MASAR captures long-term temporal dependencies that enhance future trajectory forecasting. Experiments conducted on the nuScenes dataset demonstrate MASAR's effectiveness, showing improvements of over 20% in minADE and minFDE while maintaining robust detection performance. Code and models are available at https://github.com/aminmed/MASAR.

  </details>



- **Real-time Rendering with a Neural Irradiance Volume**  
  Arno Coomans, Giacomo Nazzaro, Edoardo A. Dominici, Christian Döring, Floor Verhoeven, Konstantinos Vardis, Markus Steinberger  
  _2026-02-13_ · https://arxiv.org/abs/2602.12949v1  
  <details><summary>Abstract</summary>

  Rendering diffuse global illumination in real-time is often approximated by pre-computing and storing irradiance in a 3D grid of probes. As long as most of the scene remains static, probes approximate irradiance for all surfaces immersed in the irradiance volume, including novel dynamic objects. This approach, however, suffers from aliasing artifacts and high memory consumption. We propose Neural Irradiance Volume (NIV), a neural-based technique that allows accurate real-time rendering of diffuse global illumination via a compact pre-computed model, overcoming the limitations of traditional probe-based methods, such as the expensive memory footprint, aliasing artifacts, and scene-specific heuristics. The key insight is that neural compression creates an adaptive and amortized representation of irradiance, circumventing the cubic scaling of grid-based methods. Our superior memory-scaling improves quality by at least 10x at the same memory budget, and enables a straightforward representation of higher-dimensional irradiance fields, allowing rendering of time-varying or dynamic effects without requiring additional computation at runtime. Unlike other neural rendering techniques, our method works within strict real-time constraints, providing fast inference (around 1 ms per frame on consumer GPUs at full HD resolution), reduced memory usage (1-5 MB for medium-sized scenes), and only requires a G-buffer as input, without expensive ray tracing or denoising.

  </details>



- **X-VORTEX: Spatio-Temporal Contrastive Learning for Wake Vortex Trajectory Forecasting**  
  Zhan Qu, Michael Färber  
  _2026-02-13_ · https://arxiv.org/abs/2602.12869v1  
  <details><summary>Abstract</summary>

  Wake vortices are strong, coherent air turbulences created by aircraft, and they pose a major safety and capacity challenge for air traffic management. Tracking how vortices move, weaken, and dissipate over time from LiDAR measurements is still difficult because scans are sparse, vortex signatures fade as the flow breaks down under atmospheric turbulence and instabilities, and point-wise annotation is prohibitively expensive. Existing approaches largely treat each scan as an independent, fully supervised segmentation problem, which overlooks temporal structure and does not scale to the vast unlabeled archives collected in practice. We present X-VORTEX, a spatio-temporal contrastive learning framework grounded in Augmentation Overlap Theory that learns physics-aware representations from unlabeled LiDAR point cloud sequences. X-VORTEX addresses two core challenges: sensor sparsity and time-varying vortex dynamics. It constructs paired inputs from the same underlying flight event by combining a weakly perturbed sequence with a strongly augmented counterpart produced via temporal subsampling and spatial masking, encouraging the model to align representations across missing frames and partial observations. Architecturally, a time-distributed geometric encoder extracts per-scan features and a sequential aggregator models the evolving vortex state across variable-length sequences. We evaluate on a real-world dataset of over one million LiDAR scans. X-VORTEX achieves superior vortex center localization while using only 1% of the labeled data required by supervised baselines, and the learned representations support accurate trajectory forecasting.

  </details>


