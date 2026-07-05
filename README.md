## Photon Counting EMCCD Module

The **Photon Counting EMCCD** module is useful for converting raw photon-exposed data from an EMCCD camera into probabilistic photon counts. It is directly based on [my work](https://doi.org/10.1364/OPTICAQ.518037) that was published in optica quantum  

1. It performs noise analysis and bias correction, and provides utilities to save the corresponding correction files.

2. These saved correction files can then be used to process photon-exposed data acquired with the same camera settings as the dark-noise data.
3. Run the EMCCD_photon_counting_pipeline.ipynb with a suitable noise file and data file in ".sif" format obtained from an ANDOR EMCCD.
4. Use the "EMCCD_CUDA_documented.py" as the base file to be imported to the notebook.
5.  The EMCCD_pytorch_Documented.py is the version written in pytorch and can run on both an Nvidia PC as well as MAC GPU. It works in the same way, but does not have the modules for correlations yet. Will update soon

## Required Packages

- `sifparser==0.3.6`
- `cupy`
- `pytorch`
- NVIDIA CUDA Toolkit
## Notes
1. When generating the sample probability distribution from either noise data or photon-exposed data, care should be taken in selecting the count range over which the distribution is sampled. The selected range should avoid regions where the distribution tail becomes excessively scattered due to poor statistics.
   In most cases, the fitting algorithm can still identify the correct behavior and produce a reliable fit, provided that the weighting scheme sufficiently suppresses contributions from the noisy tail region. Nevertheless, it is recommended to restrict the sampling range such that the sampled distribution does not become dominated by statistical noise in the tails.
   For example, the range [−50,350] provides a stable sampling region where the distribution remains well behaved.
  <br> <img width="584" height="438" alt="593053285-c2fd4d55-1c9c-4ca9-8ef8-d449cf9c953e" src="https://github.com/user-attachments/assets/bffcfcf3-823a-43e9-83a4-317bc2cef67b" />
   <br> Similarly, extending the range up to 550 can still be handled successfully by the fitting algorithm, although the tail region becomes noticeably noisier.
  <br> <img width="584" height="438" alt="593054362-e39a6a39-d58b-46da-8f44-23705b2d8366" src="https://github.com/user-attachments/assets/7bc437ea-1843-4fdc-af19-00b93fbb3308" />
   <br> However, sampling over excessively large ranges can make the fitting procedure unstable due to the strongly scattered tail statistics.
   <br> <img width="584" height="438" alt="593055402-efb325cc-3a38-4a3b-80f5-2235e2677a75" src="https://github.com/user-attachments/assets/fe1ee1a3-17c0-4073-a286-126b237e99fb" />
   <br> In the example shown, the fitting algorithm was still able to converge only because the weight factor precision was $10^{
−5}$, thereby strongly suppressing the influence of the noisy region.
2. The value of the parameter $p_{\text{ser}}$ is usually substantially small, and for low-sampled noise data, the estimation of $p_{\text{ser}}$ can become somewhat erroneous, effectively pushing its value toward zero. In such cases, one can safely consider $p_{\text{ser}} = 0$ without introducing any significant issues.
## Referencs
-  "Multi-imaging and Bayesian estimation for photon counting with EMCCDs.", 
E. Lantz , et.al., Monthly Notices of the Royal Astronomical Society, Volume 386, Issue 4, June 2008, Pages 2262–2270, [https://doi.org/10.1111/j.1365-2966.2008.13200.x](https://doi.org/10.1111/j.1365-2966.2008.13200.x)
- "Multifold enhancement of quantum SNR by using an EMCCD as a photon number resolving device.", NOMOL lab, Optica Quantum 2, 156-164 (2024), [https://doi.org/10.1364/OPTICAQ.518037](https://doi.org/10.1364/OPTICAQ.518037)
- "Imaging high-dimensional spatial entanglement with a camera.", M. Edgar, et al., Nat Commun 3, 984 (2012), [https://doi.org/10.1038/ncomms1988](https://doi.org/10.1038/ncomms1988)
- "General Model of Photon-Pair Detection with an Image Sensor.", H. Defienne, et.al., Phys. Rev. Lett. 120, 203604 – Published 17 May, 2018, [https://doi.org/10.1103/PhysRevLett.120.203604](https://doi.org/10.1103/PhysRevLett.120.203604)


