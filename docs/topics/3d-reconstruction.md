# 3D Reconstruction

_Updated: 2026-02-22 07:04 UTC_

Total papers shown: **1**


---

- **Neural Implicit Representations for 3D Synthetic Aperture Radar Imaging**  
  Nithin Sugavanam, Emre Ertin  
  _2026-02-19_ · https://arxiv.org/abs/2602.17556v1  
  <details><summary>Abstract</summary>

  Synthetic aperture radar (SAR) is a tomographic sensor that measures 2D slices of the 3D spatial Fourier transform of the scene. In many operational scenarios, the measured set of 2D slices does not fill the 3D space in the Fourier domain, resulting in significant artifacts in the reconstructed imagery. Traditionally, simple priors, such as sparsity in the image domain, are used to regularize the inverse problem. In this paper, we review our recent work that achieves state-of-the-art results in 3D SAR imaging employing neural structures to model the surface scattering that dominates SAR returns. These neural structures encode the surface of the objects in the form of a signed distance function learned from the sparse scattering data. Since estimating a smooth surface from a sparse and noisy point cloud is an ill-posed problem, we regularize the surface estimation by sampling points from the implicit surface representation during the training step. We demonstrate the model's ability to represent target scattering using measured and simulated data from single vehicles and a larger scene with a large number of vehicles. We conclude with future research directions calling for methods to learn complex-valued neural representations to enable synthesizing new collections from the volumetric neural implicit representation.

  </details>


