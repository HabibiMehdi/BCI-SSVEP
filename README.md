# BCI-SSVEP
SSVEP-CCA-Algorithm

This repository contains an implementation of the Canonical Correlation Analysis (CCA) algorithm for frequency recognition in Steady-State Visual Evoked Potential (SSVEP)-based Brain-Computer Interfaces (BCIs).

**Overview**

Steady-State Visual Evoked Potentials (SSVEPs) are brain signals that occur in response to visual stimuli flickering at specific frequencies. SSVEP-based BCIs use these signals to detect the frequency at which the user is attending, allowing for hands-free communication and control.
The CCA algorithm is a promising approach for frequency recognition in SSVEP-based BCIs, offering advantages such as improved accuracy, robustness to noise, and the ability to utilize harmonic frequencies as stimuli.

**Features**

Frequency Recognition: Implements the CCA algorithm for extracting and recognizing SSVEP frequencies from multi-channel EEG data.
Multi-Channel Support: Supports the use of multiple EEG channels for improved performance and robustness.
Harmonic Frequency Detection: Can recognize SSVEP frequencies that are harmonics of the stimuli frequencies.
Channel Selection: Includes a method for selecting the most relevant EEG channels for SSVEP detection.
Parameter Optimization: Provides tools for optimizing parameters such as window length and the number of harmonics used.

In additionaly this  repository contains a Matlab implementation for distinguish SSVEP Frequency using Mr. Lin's mathematical technique (CCA) features extracted from EEG data and then improving its by machine learning methods such as SVM,KNN

**References**

Lin, Z., Zhang, C., Wu, W., & Gao, X. (2006). Frequency recognition based on canonical correlation analysis for SSVEP-based BCIs. IEEE Transactions on Biomedical Engineering, 53(12), 2610-2614.
Bin, G., Gao, X., Yan, Z., Hong, B., & Gao, S. (2009). An online multi-channel SSVEP-based brain–computer interface using a canonical correlation analysis method. Journal of Neural Engineering, 6(4), 046002.



![image](https://github.com/thehabibimm/BCI-SSVEP/assets/123571190/1cd3e0f2-9fc6-4f89-97b8-73f6c3176da9)

