# UDBNET

Although Bhunia et al. [cite] use an unsupervised setup, the major differences are:
1) Binarization network is training from generated degraded image (not real degraded image). It does take into account the performance on real degraded image during training. 
2) On the other side, testing is done on real degraded images. Hence, there exists a Domain Gap between generated noisy image vs real noisy image. 
3) In this framework, we take into account how the model performs on real noisy images during training. Thus the model becomes aware of real noisy image distribution and domain-discrepancy is handled carefully.
