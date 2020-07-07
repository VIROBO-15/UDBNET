# UDBNET

Although Bhunia et al. [cite] use an unsupervised setup, the major differences are:
### 1) Binarization network from Bhunia et al. is trained from generated noisy image (not real noisy image). It does take into account the performance on real noisy image during training. 
### 2) On the other side, testing is done on real noisy images.  There exists a Domain Gap between generated noisy image vs real noisy image. Thus, it gives rise to a domain-discrepancy between training and testing setup.
### 3) In this framework, we take into account how the model performs on real noisy images during training. Thus the model becomes aware of real noisy image distribution and domain-discrepancy is handled carefully.
