# NeRF & Neural Radiance Fields

_Updated: 2026-02-16 07:19 UTC_

Total papers shown: **1**


---

- **Real-time Rendering with a Neural Irradiance Volume**  
  Arno Coomans, Giacomo Nazzaro, Edoardo A. Dominici, Christian Döring, Floor Verhoeven, Konstantinos Vardis, Markus Steinberger  
  _2026-02-13_ · https://arxiv.org/abs/2602.12949v1  
  <details><summary>Abstract</summary>

  Rendering diffuse global illumination in real-time is often approximated by pre-computing and storing irradiance in a 3D grid of probes. As long as most of the scene remains static, probes approximate irradiance for all surfaces immersed in the irradiance volume, including novel dynamic objects. This approach, however, suffers from aliasing artifacts and high memory consumption. We propose Neural Irradiance Volume (NIV), a neural-based technique that allows accurate real-time rendering of diffuse global illumination via a compact pre-computed model, overcoming the limitations of traditional probe-based methods, such as the expensive memory footprint, aliasing artifacts, and scene-specific heuristics. The key insight is that neural compression creates an adaptive and amortized representation of irradiance, circumventing the cubic scaling of grid-based methods. Our superior memory-scaling improves quality by at least 10x at the same memory budget, and enables a straightforward representation of higher-dimensional irradiance fields, allowing rendering of time-varying or dynamic effects without requiring additional computation at runtime. Unlike other neural rendering techniques, our method works within strict real-time constraints, providing fast inference (around 1 ms per frame on consumer GPUs at full HD resolution), reduced memory usage (1-5 MB for medium-sized scenes), and only requires a G-buffer as input, without expensive ray tracing or denoising.

  </details>


