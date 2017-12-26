# Deep Learning Papers Reading Roadmap

>Here is a reading roadmap of Deep Learning papers!

The roadmap is constructed in accordance with the following four guidelines:

- From outline to detail
- From old to state-of-the-art
- from generic to specific areas
- focus on state-of-the-art

You will find many papers that are quite new but really worth reading.

I would continue adding papers to this roadmap.


---------------------------------------

# 1 Deep Learning History and Basics

## 1.0 Book

**[0]** Bengio, Yoshua, Ian J. Goodfellow, and Aaron Courville. "**Deep learning**." An MIT Press book. (2015). [[pdf]](https://github.com/HFTrader/DeepLearningBook/raw/master/DeepLearningBook.pdf) **(Deep Learning Bible, you can read this book while reading following papers.)** :star::star::star::star::star:

## 1.1 Survey

**[1]** LeCun, Yann, Yoshua Bengio, and Geoffrey Hinton. "**Deep learning**." Nature 521.7553 (2015): 436-444. [[pdf]](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf) **(Three Giants' Survey)** :star::star::star::star::star:

## 1.2 Deep Belief Network(DBN)(Milestone of Deep Learning Eve)

**[2]** Hinton, Geoffrey E., Simon Osindero, and Yee-Whye Teh. "**A fast learning algorithm for deep belief nets**." Neural computation 18.7 (2006): 1527-1554. [[pdf]](http://www.cs.toronto.edu/~hinton/absps/ncfast.pdf)**(Deep Learning Eve)** :star::star::star:

**[3]** Hinton, Geoffrey E., and Ruslan R. Salakhutdinov. "**Reducing the dimensionality of data with neural networks**." Science 313.5786 (2006): 504-507. [[pdf]](http://www.cs.toronto.edu/~hinton/science.pdf) **(Milestone, Show the promise of deep learning)** :star::star::star:

## 1.3 ImageNet Evolution（Deep Learning broke out from here）

**[4]** Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "**Imagenet classification with deep convolutional neural networks**." Advances in neural information processing systems. 2012. [[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) **(AlexNet, Deep Learning Breakthrough)** :star::star::star::star::star:

**[5]** Simonyan, Karen, and Andrew Zisserman. "**Very deep convolutional networks for large-scale image recognition**." arXiv preprint arXiv:1409.1556 (2014). [[pdf]](https://arxiv.org/pdf/1409.1556.pdf) **(VGGNet,Neural Networks become very deep!)** :star::star::star:

**[6]** Szegedy, Christian, et al. "**Going deeper with convolutions**." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) **(GoogLeNet)** :star::star::star:

**[7]** He, Kaiming, et al. "**Deep residual learning for image recognition**." arXiv preprint arXiv:1512.03385 (2015). [[pdf]](https://arxiv.org/pdf/1512.03385.pdf) **(ResNet,Very very deep networks, CVPR best paper)** :star::star::star::star::star:

# 2 Deep Learning Method

## 2.1 Model

**[14]** Hinton, Geoffrey E., et al. "**Improving neural networks by preventing co-adaptation of feature detectors**." arXiv preprint arXiv:1207.0580 (2012). [[pdf]](https://arxiv.org/pdf/1207.0580.pdf) **(Dropout)** :star::star::star:

**[15]** Srivastava, Nitish, et al. "**Dropout: a simple way to prevent neural networks from overfitting**." Journal of Machine Learning Research 15.1 (2014): 1929-1958. [[pdf]](http://www.jmlr.org/papers/volume15/srivastava14a.old/source/srivastava14a.pdf) :star::star::star:

**[16]** Ioffe, Sergey, and Christian Szegedy. "**Batch normalization: Accelerating deep network training by reducing internal covariate shift**." arXiv preprint arXiv:1502.03167 (2015). [[pdf]](http://arxiv.org/pdf/1502.03167) **(An outstanding Work in 2015)** :star::star::star::star:

**[17]** Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "**Layer normalization**." arXiv preprint arXiv:1607.06450 (2016). [[pdf]](https://arxiv.org/pdf/1607.06450.pdf?utm_source=sciontist.com&utm_medium=refer&utm_campaign=promote) **(Update of Batch Normalization)** :star::star::star::star:

**[18]** Courbariaux, Matthieu, et al. "**Binarized Neural Networks: Training Neural Networks with Weights and Activations Constrained to+ 1 or−1**." [[pdf]](https://pdfs.semanticscholar.org/f832/b16cb367802609d91d400085eb87d630212a.pdf) **(New Model,Fast)**  :star::star::star:

**[19]** Jaderberg, Max, et al. "**Decoupled neural interfaces using synthetic gradients**." arXiv preprint arXiv:1608.05343 (2016). [[pdf]](https://arxiv.org/pdf/1608.05343) **(Innovation of Training Method,Amazing Work)** :star::star::star::star::star:

**[20]** Chen, Tianqi, Ian Goodfellow, and Jonathon Shlens. "Net2net: Accelerating learning via knowledge transfer." arXiv preprint arXiv:1511.05641 (2015). [[pdf]](https://arxiv.org/abs/1511.05641) **(Modify previously trained network to reduce training epochs)** :star::star::star:

**[21]** Wei, Tao, et al. "Network Morphism." arXiv preprint arXiv:1603.01670 (2016). [[pdf]](https://arxiv.org/abs/1603.01670) **(Modify previously trained network to reduce training epochs)** :star::star::star:

## 2.2 Optimization

**[22]** Sutskever, Ilya, et al. "**On the importance of initialization and momentum in deep learning**." ICML (3) 28 (2013): 1139-1147. [[pdf]](http://www.jmlr.org/proceedings/papers/v28/sutskever13.pdf) **(Momentum optimizer)** :star::star:

**[23]** Kingma, Diederik, and Jimmy Ba. "**Adam: A method for stochastic optimization**." arXiv preprint arXiv:1412.6980 (2014). [[pdf]](http://arxiv.org/pdf/1412.6980) **(Maybe used most often currently)** :star::star::star:

**[24]** Andrychowicz, Marcin, et al. "**Learning to learn by gradient descent by gradient descent**." arXiv preprint arXiv:1606.04474 (2016). [[pdf]](https://arxiv.org/pdf/1606.04474) **(Neural Optimizer,Amazing Work)** :star::star::star::star::star:

**[25]** Han, Song, Huizi Mao, and William J. Dally. "**Deep compression: Compressing deep neural network with pruning, trained quantization and huffman coding**." CoRR, abs/1510.00149 2 (2015). [[pdf]](https://pdfs.semanticscholar.org/5b6c/9dda1d88095fa4aac1507348e498a1f2e863.pdf) **(ICLR best paper, new direction to make NN running fast,DeePhi Tech Startup)** :star::star::star::star::star:

**[26]** Iandola, Forrest N., et al. "**SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 1MB model size**." arXiv preprint arXiv:1602.07360 (2016). [[pdf]](http://arxiv.org/pdf/1602.07360) **(Also a new direction to optimize NN,DeePhi Tech Startup)** :star::star::star::star:

## 2.3 Unsupervised Learning / Deep Generative Model

**[28]** Kingma, Diederik P., and Max Welling. "**Auto-encoding variational bayes**." arXiv preprint arXiv:1312.6114 (2013). [[pdf]](http://arxiv.org/pdf/1312.6114) **(VAE)** :star::star::star::star:

**[29]** Goodfellow, Ian, et al. "**Generative adversarial nets**." Advances in Neural Information Processing Systems. 2014. [[pdf]](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) **(GAN,super cool idea)** :star::star::star::star::star:

**[30]** Radford, Alec, Luke Metz, and Soumith Chintala. "**Unsupervised representation learning with deep convolutional generative adversarial networks**." arXiv preprint arXiv:1511.06434 (2015). [[pdf]](http://arxiv.org/pdf/1511.06434) **(DCGAN)** :star::star::star::star:

**[31]** Gregor, Karol, et al. "**DRAW: A recurrent neural network for image generation**." arXiv preprint arXiv:1502.04623 (2015). [[pdf]](http://jmlr.org/proceedings/papers/v37/gregor15.pdf) **(VAE with attention, outstanding work)** :star::star::star::star::star:

**[32]** Oord, Aaron van den, Nal Kalchbrenner, and Koray Kavukcuoglu. "**Pixel recurrent neural networks**." arXiv preprint arXiv:1601.06759 (2016). [[pdf]](http://arxiv.org/pdf/1601.06759) **(PixelRNN)** :star::star::star::star:

**[33]** Oord, Aaron van den, et al. "Conditional image generation with PixelCNN decoders." arXiv preprint arXiv:1606.05328 (2016). [[pdf]](https://arxiv.org/pdf/1606.05328) **(PixelCNN)** :star::star::star::star:



## 2.8 One Shot Deep Learning

**[59]** Lake, Brenden M., Ruslan Salakhutdinov, and Joshua B. Tenenbaum. "**Human-level concept learning through probabilistic program induction**." Science 350.6266 (2015): 1332-1338. [[pdf]](http://clm.utexas.edu/compjclub/wp-content/uploads/2016/02/lake2015.pdf) **(No Deep Learning,but worth reading)** :star::star::star::star::star:

# 3 Applications

## 3.2 Object Detection

**[1]** Szegedy, Christian, Alexander Toshev, and Dumitru Erhan. "**Deep neural networks for object detection**." Advances in Neural Information Processing Systems. 2013. [[pdf]](http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf) :star::star::star:

**[2]** Girshick, Ross, et al. "**Rich feature hierarchies for accurate object detection and semantic segmentation**." Proceedings of the IEEE conference on computer vision and pattern recognition. 2014. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) **(RCNN)** :star::star::star::star::star:

**[3]** He, Kaiming, et al. "**Spatial pyramid pooling in deep convolutional networks for visual recognition**." European Conference on Computer Vision. Springer International Publishing, 2014. [[pdf]](http://arxiv.org/pdf/1406.4729) **(SPPNet)** :star::star::star::star:

**[4]** Girshick, Ross. "**Fast r-cnn**." Proceedings of the IEEE International Conference on Computer Vision. 2015. [[pdf]](https://pdfs.semanticscholar.org/8f67/64a59f0d17081f2a2a9d06f4ed1cdea1a0ad.pdf) :star::star::star::star:

**[5]** Ren, Shaoqing, et al. "**Faster R-CNN: Towards real-time object detection with region proposal networks**." Advances in neural information processing systems. 2015. [[pdf]](http://papers.nips.cc/paper/5638-analysis-of-variational-bayesian-latent-dirichlet-allocation-weaker-sparsity-than-map.pdf) :star::star::star::star:

**[6]** Redmon, Joseph, et al. "**You only look once: Unified, real-time object detection**." arXiv preprint arXiv:1506.02640 (2015). [[pdf]](http://homes.cs.washington.edu/~ali/papers/YOLO.pdf) **(YOLO,Oustanding Work, really practical)** :star::star::star::star::star:

**[7]** Liu, Wei, et al. "**SSD: Single Shot MultiBox Detector**." arXiv preprint arXiv:1512.02325 (2015). [[pdf]](http://arxiv.org/pdf/1512.02325) :star::star::star:

**[8]** Dai, Jifeng, et al. "**R-FCN: Object Detection via
Region-based Fully Convolutional Networks**." arXiv preprint arXiv:1605.06409 (2016). [[pdf]](https://arxiv.org/abs/1605.06409) :star::star::star::star:

**[9]** He, Gkioxari, et al. "**Mask R-CNN**" arXiv preprint arXiv:1703.06870 (2017). [[pdf]](https://arxiv.org/abs/1703.06870) :star::star::star::star:

# For our own interest 

## Contents

* [Understanding / Generalization / Transfer](#understanding--generalization--transfer)
* [Optimization / Training Techniques](#optimization--training-techniques)
* [Unsupervised / Generative Models](#unsupervised--generative-models)
* [Convolutional Network Models](#convolutional-neural-network-models)
* [Image Segmentation / Object Detection](#image-segmentation--object-detection)
* [Image / Video / Etc](#image--video--etc)
* [Natural Language Processing / RNNs](#natural-language-processing--rnns)
* [Speech / Other Domain](#speech--other-domain)
* [Reinforcement Learning / Robotics](#reinforcement-learning--robotics)
* [More Papers from 2016](#more-papers-from-2016)

*(More than Top 100)*

* [New Papers](#new-papers) : Less than 6 months
* [Old Papers](#old-papers) : Before 2012
* [HW / SW / Dataset](#hw--sw--dataset) : Technical reports
* [Book / Survey / Review](#book--survey--review)
* [Video Lectures / Tutorials / Blogs](#video-lectures--tutorials--blogs)
* [Appendix: More than Top 100](#appendix-more-than-top-100) : More papers not in the list

* * *


### Unsupervised / Generative Models
- **Pixel recurrent neural networks** (2016), A. Oord et al. [[pdf]](http://arxiv.org/pdf/1601.06759v2.pdf)
- **Improved techniques for training GANs** (2016), T. Salimans et al. [[pdf]](http://papers.nips.cc/paper/6125-improved-techniques-for-training-gans.pdf)
- **Unsupervised representation learning with deep convolutional generative adversarial networks** (2015), A. Radford et al. [[pdf]](https://arxiv.org/pdf/1511.06434v2)
- **DRAW: A recurrent neural network for image generation** (2015), K. Gregor et al. [[pdf]](http://arxiv.org/pdf/1502.04623)
- **Generative adversarial nets** (2014), I. Goodfellow et al. [[pdf]](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
- **Auto-encoding variational Bayes** (2013), D. Kingma and M. Welling [[pdf]](http://arxiv.org/pdf/1312.6114)
- **Building high-level features using large scale unsupervised learning** (2013), Q. Le et al. [[pdf]](http://arxiv.org/pdf/1112.6209)

<!---[Key researchers] [Yoshua Bengio](https://scholar.google.ca/citations?user=kukA0LcAAAAJ), [Ian Goodfellow](https://scholar.google.ca/citations?user=iYN86KEAAAAJ), [Alex Graves](https://scholar.google.ca/citations?user=DaFHynwAAAAJ)-->
### Convolutional Neural Network Models
- **Rethinking the inception architecture for computer vision** (2016), C. Szegedy et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)
- **Inception-v4, inception-resnet and the impact of residual connections on learning** (2016), C. Szegedy et al. [[pdf]](http://arxiv.org/pdf/1602.07261)
- **Identity Mappings in Deep Residual Networks** (2016), K. He et al. [[pdf]](https://arxiv.org/pdf/1603.05027v2.pdf)
- **Deep residual learning for image recognition** (2016), K. He et al. [[pdf]](http://arxiv.org/pdf/1512.03385)
- **Spatial transformer network** (2015), M. Jaderberg et al., [[pdf]](http://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf)
- **Going deeper with convolutions** (2015), C. Szegedy et al.  [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)
- **Very deep convolutional networks for large-scale image recognition** (2014), K. Simonyan and A. Zisserman [[pdf]](http://arxiv.org/pdf/1409.1556)
- **Return of the devil in the details: delving deep into convolutional nets** (2014), K. Chatfield et al. [[pdf]](http://arxiv.org/pdf/1405.3531)
- **OverFeat: Integrated recognition, localization and detection using convolutional networks** (2013), P. Sermanet et al. [[pdf]](http://arxiv.org/pdf/1312.6229)
- **Maxout networks** (2013), I. Goodfellow et al. [[pdf]](http://arxiv.org/pdf/1302.4389v4)
- **Network in network** (2013), M. Lin et al. [[pdf]](http://arxiv.org/pdf/1312.4400)
- **ImageNet classification with deep convolutional neural networks** (2012), A. Krizhevsky et al. [[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

<!---[Key researchers]  [Christian Szegedy](https://scholar.google.ca/citations?hl=en&user=3QeF7mAAAAAJ), [Kaming He](https://scholar.google.ca/citations?hl=en&user=DhtAFkwAAAAJ), [Shaoqing Ren](https://scholar.google.ca/citations?hl=en&user=AUhj438AAAAJ), [Jian Sun](https://scholar.google.ca/citations?hl=en&user=ALVSZAYAAAAJ), [Geoffrey Hinton](https://scholar.google.ca/citations?user=JicYPdAAAAAJ), [Yoshua Bengio](https://scholar.google.ca/citations?user=kukA0LcAAAAJ), [Yann LeCun](https://scholar.google.ca/citations?hl=en&user=WLN3QrAAAAAJ)-->




### New papers
*Newly published papers (< 6 months) which are worth reading*
- Convolutional Sequence to Sequence Learning (2017), Jonas Gehring et al. [[pdf]](https://arxiv.org/pdf/1705.03122)
- Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour (2017), Priya Goyal et al. [[pdf]](https://research.fb.com/wp-content/uploads/2017/06/imagenet1kin1h3.pdf)
- TACOTRON: Towards end-to-end speech synthesis (2017), Y. Wang et al. [[pdf]](https://arxiv.org/pdf/1703.10135.pdf)
- Deep Photo Style Transfer (2017), F. Luan et al. [[pdf]](http://arxiv.org/pdf/1703.07511v1.pdf)
- Evolution Strategies as a Scalable Alternative to Reinforcement Learning (2017), T. Salimans et al. [[pdf]](http://arxiv.org/pdf/1703.03864v1.pdf)
- Deformable Convolutional Networks (2017), J. Dai et al. [[pdf]](http://arxiv.org/pdf/1703.06211v2.pdf)
- Mask R-CNN (2017), K. He et al. [[pdf]](https://128.84.21.199/pdf/1703.06870)
- Learning to discover cross-domain relations with generative adversarial networks (2017), T. Kim et al. [[pdf]](http://arxiv.org/pdf/1703.05192v1.pdf) 
- Deep voice: Real-time neural text-to-speech (2017), S. Arik et al., [[pdf]](http://arxiv.org/pdf/1702.07825v2.pdf)
- PixelNet: Representation of the pixels, by the pixels, and for the pixels (2017), A. Bansal et al. [[pdf]](http://arxiv.org/pdf/1702.06506v1.pdf)
- Batch renormalization: Towards reducing minibatch dependence in batch-normalized models (2017), S. Ioffe. [[pdf]](https://arxiv.org/abs/1702.03275)
- Wasserstein GAN (2017), M. Arjovsky et al. [[pdf]](https://arxiv.org/pdf/1701.07875v1)
- Understanding deep learning requires rethinking generalization (2017), C. Zhang et al. [[pdf]](https://arxiv.org/pdf/1611.03530)
- Least squares generative adversarial networks (2016), X. Mao et al. [[pdf]](https://arxiv.org/abs/1611.04076v2)


### Old Papers
*Classic papers published before 2012*
- An analysis of single-layer networks in unsupervised feature learning (2011), A. Coates et al. [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2011_CoatesNL11.pdf)
- Deep sparse rectifier neural networks (2011), X. Glorot et al. [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2011_GlorotBB11.pdf)
- Natural language processing (almost) from scratch (2011), R. Collobert et al. [[pdf]](http://arxiv.org/pdf/1103.0398)
- Recurrent neural network based language model (2010), T. Mikolov et al. [[pdf]](http://www.fit.vutbr.cz/research/groups/speech/servite/2010/rnnlm_mikolov.pdf)
- Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion (2010), P. Vincent et al. [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.297.3484&rep=rep1&type=pdf)
- Learning mid-level features for recognition (2010), Y. Boureau [[pdf]](http://ece.duke.edu/~lcarin/boureau-cvpr-10.pdf)
- A practical guide to training restricted boltzmann machines (2010), G. Hinton [[pdf]](http://www.csri.utoronto.ca/~hinton/absps/guideTR.pdf)
- Understanding the difficulty of training deep feedforward neural networks (2010), X. Glorot and Y. Bengio [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_GlorotB10.pdf)
- Why does unsupervised pre-training help deep learning (2010), D. Erhan et al. [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_ErhanCBV10.pdf)
- Learning deep architectures for AI (2009), Y. Bengio. [[pdf]](http://sanghv.com/download/soft/machine%20learning,%20artificial%20intelligence,%20mathematics%20ebooks/ML/learning%20deep%20architectures%20for%20AI%20(2009).pdf)
- Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations (2009), H. Lee et al. [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.149.802&rep=rep1&type=pdf)
- Greedy layer-wise training of deep networks (2007), Y. Bengio et al. [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/NIPS2006_739.pdf)
- Reducing the dimensionality of data with neural networks, G. Hinton and R. Salakhutdinov. [[pdf]](http://homes.mpimf-heidelberg.mpg.de/~mhelmsta/pdf/2006%20Hinton%20Salakhudtkinov%20Science.pdf)
- A fast learning algorithm for deep belief nets (2006), G. Hinton et al. [[pdf]](http://nuyoo.utm.mx/~jjf/rna/A8%20A%20fast%20learning%20algorithm%20for%20deep%20belief%20nets.pdf)
- Gradient-based learning applied to document recognition (1998), Y. LeCun et al. [[pdf]](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- Long short-term memory (1997), S. Hochreiter and J. Schmidhuber. [[pdf]](http://www.mitpressjournals.org/doi/pdfplus/10.1162/neco.1997.9.8.1735)


### HW / SW / Dataset
- OpenAI gym (2016), G. Brockman et al. [[pdf]](https://arxiv.org/pdf/1606.01540)
- TensorFlow: Large-scale machine learning on heterogeneous distributed systems (2016), M. Abadi et al. [[pdf]](http://arxiv.org/pdf/1603.04467)
- Theano: A Python framework for fast computation of mathematical expressions, R. Al-Rfou et al.
- Torch7: A matlab-like environment for machine learning, R. Collobert et al. [[pdf]](https://ronan.collobert.com/pub/matos/2011_torch7_nipsw.pdf)
- MatConvNet: Convolutional neural networks for matlab (2015), A. Vedaldi and K. Lenc [[pdf]](http://arxiv.org/pdf/1412.4564)
- Imagenet large scale visual recognition challenge (2015), O. Russakovsky et al. [[pdf]](http://arxiv.org/pdf/1409.0575)
- Caffe: Convolutional architecture for fast feature embedding (2014), Y. Jia et al. [[pdf]](http://arxiv.org/pdf/1408.5093)


### Book / Survey / Review
- On the Origin of Deep Learning (2017), H. Wang and Bhiksha Raj. [[pdf]](https://arxiv.org/pdf/1702.07800)
- Deep Reinforcement Learning: An Overview (2017), Y. Li, [[pdf]](http://arxiv.org/pdf/1701.07274v2.pdf)
- Neural Machine Translation and Sequence-to-sequence Models(2017): A Tutorial, G. Neubig. [[pdf]](http://arxiv.org/pdf/1703.01619v1.pdf)
- Neural Network and Deep Learning (Book, Jan 2017), Michael Nielsen. [[html]](http://neuralnetworksanddeeplearning.com/index.html)
- Deep learning (Book, 2016), Goodfellow et al. [[html]](http://www.deeplearningbook.org/)
- LSTM: A search space odyssey (2016), K. Greff et al. [[pdf]](https://arxiv.org/pdf/1503.04069.pdf?utm_content=buffereddc5&utm_medium=social&utm_source=plus.google.com&utm_campaign=buffer)
- Tutorial on Variational Autoencoders (2016), C. Doersch. [[pdf]](https://arxiv.org/pdf/1606.05908)
- Deep learning (2015), Y. LeCun, Y. Bengio and G. Hinton [[pdf]](https://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf)
- Deep learning in neural networks: An overview (2015), J. Schmidhuber [[pdf]](http://arxiv.org/pdf/1404.7828)
- Representation learning: A review and new perspectives (2013), Y. Bengio et al. [[pdf]](http://arxiv.org/pdf/1206.5538)

### Appendix: More than Top 100
*(2016)*
- A character-level decoder without explicit segmentation for neural machine translation (2016), J. Chung et al. [[pdf]](https://arxiv.org/pdf/1603.06147)
- Dermatologist-level classification of skin cancer with deep neural networks (2017), A. Esteva et al. [[html]](http://www.nature.com/nature/journal/v542/n7639/full/nature21056.html)
- Weakly supervised object localization with multi-fold multiple instance learning (2017), R. Gokberk et al. [[pdf]](https://arxiv.org/pdf/1503.00949)
- Brain tumor segmentation with deep neural networks (2017), M. Havaei et al. [[pdf]](https://arxiv.org/pdf/1505.03540)
- Professor Forcing: A New Algorithm for Training Recurrent Networks (2016), A. Lamb et al. [[pdf]](https://arxiv.org/pdf/1610.09038)
- Adversarially learned inference (2016), V. Dumoulin et al. [[web]](https://ishmaelbelghazi.github.io/ALI/)[[pdf]](https://arxiv.org/pdf/1606.00704v1)
- Understanding convolutional neural networks (2016), J. Koushik [[pdf]](https://arxiv.org/pdf/1605.09081v1)
- Taking the human out of the loop: A review of bayesian optimization (2016), B. Shahriari et al. [[pdf]](https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf)
- Adaptive computation time for recurrent neural networks (2016), A. Graves [[pdf]](http://arxiv.org/pdf/1603.08983)
- Densely connected convolutional networks (2016), G. Huang et al. [[pdf]](https://arxiv.org/pdf/1608.06993v1)
- Region-based convolutional networks for accurate object detection and segmentation (2016), R. Girshick et al. 
- Continuous deep q-learning with model-based acceleration (2016), S. Gu et al. [[pdf]](http://www.jmlr.org/proceedings/papers/v48/gu16.pdf)
- A thorough examination of the cnn/daily mail reading comprehension task (2016), D. Chen et al. [[pdf]](https://arxiv.org/pdf/1606.02858)
- Achieving open vocabulary neural machine translation with hybrid word-character models, M. Luong and C. Manning. [[pdf]](https://arxiv.org/pdf/1604.00788)
- Very Deep Convolutional Networks for Natural Language Processing (2016), A. Conneau et al. [[pdf]](https://arxiv.org/pdf/1606.01781)
- Bag of tricks for efficient text classification (2016), A. Joulin et al. [[pdf]](https://arxiv.org/pdf/1607.01759)
- Efficient piecewise training of deep structured models for semantic segmentation (2016), G. Lin et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Lin_Efficient_Piecewise_Training_CVPR_2016_paper.pdf)
- Learning to compose neural networks for question answering (2016), J. Andreas et al. [[pdf]](https://arxiv.org/pdf/1601.01705)
- Perceptual losses for real-time style transfer and super-resolution (2016), J. Johnson et al. [[pdf]](https://arxiv.org/pdf/1603.08155)
- Reading text in the wild with convolutional neural networks (2016), M. Jaderberg et al. [[pdf]](http://arxiv.org/pdf/1412.1842)
- What makes for effective detection proposals? (2016), J. Hosang et al. [[pdf]](https://arxiv.org/pdf/1502.05082)
- Inside-outside net: Detecting objects in context with skip pooling and recurrent neural networks (2016), S. Bell et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Bell_Inside-Outside_Net_Detecting_CVPR_2016_paper.pdf).
- Instance-aware semantic segmentation via multi-task network cascades (2016), J. Dai et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Dai_Instance-Aware_Semantic_Segmentation_CVPR_2016_paper.pdf)
- Conditional image generation with pixelcnn decoders (2016), A. van den Oord et al. [[pdf]](http://papers.nips.cc/paper/6527-tree-structured-reinforcement-learning-for-sequential-object-localization.pdf)
- Deep networks with stochastic depth (2016), G. Huang et al., [[pdf]](https://arxiv.org/pdf/1603.09382)
- Consistency and Fluctuations For Stochastic Gradient Langevin Dynamics (2016), Yee Whye Teh et al. [[pdf]](http://www.jmlr.org/papers/volume17/teh16a/teh16a.pdf)

*(2015)*
- Ask your neurons: A neural-based approach to answering questions about images (2015), M. Malinowski et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Malinowski_Ask_Your_Neurons_ICCV_2015_paper.pdf)
- Exploring models and data for image question answering (2015), M. Ren et al. [[pdf]](http://papers.nips.cc/paper/5640-stochastic-variational-inference-for-hidden-markov-models.pdf)
- Are you talking to a machine? dataset and methods for multilingual image question (2015), H. Gao et al. [[pdf]](http://papers.nips.cc/paper/5641-are-you-talking-to-a-machine-dataset-and-methods-for-multilingual-image-question.pdf)
- Mind's eye: A recurrent visual representation for image caption generation (2015), X. Chen and C. Zitnick. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Chen_Minds_Eye_A_2015_CVPR_paper.pdf)
- From captions to visual concepts and back (2015), H. Fang et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Fang_From_Captions_to_2015_CVPR_paper.pdf).
- Towards AI-complete question answering: A set of prerequisite toy tasks (2015), J. Weston et al. [[pdf]](http://arxiv.org/pdf/1502.05698)
- Ask me anything: Dynamic memory networks for natural language processing (2015), A. Kumar et al. [[pdf]](http://arxiv.org/pdf/1506.07285)
- Unsupervised learning of video representations using LSTMs (2015), N. Srivastava et al. [[pdf]](http://www.jmlr.org/proceedings/papers/v37/srivastava15.pdf)
- Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding (2015), S. Han et al. [[pdf]](https://arxiv.org/pdf/1510.00149)
- Improved semantic representations from tree-structured long short-term memory networks (2015), K. Tai et al. [[pdf]](https://arxiv.org/pdf/1503.00075)
- Character-aware neural language models (2015), Y. Kim et al. [[pdf]](https://arxiv.org/pdf/1508.06615)
- Grammar as a foreign language (2015), O. Vinyals et al. [[pdf]](http://papers.nips.cc/paper/5635-grammar-as-a-foreign-language.pdf)
- Trust Region Policy Optimization (2015), J. Schulman et al. [[pdf]](http://www.jmlr.org/proceedings/papers/v37/schulman15.pdf)
- Beyond short snippents: Deep networks for video classification (2015) [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ng_Beyond_Short_Snippets_2015_CVPR_paper.pdf)
- Learning Deconvolution Network for Semantic Segmentation (2015), H. Noh et al. [[pdf]](https://arxiv.org/pdf/1505.04366v1)
- Learning spatiotemporal features with 3d convolutional networks (2015), D. Tran et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf)
- Understanding neural networks through deep visualization (2015), J. Yosinski et al. [[pdf]](https://arxiv.org/pdf/1506.06579)
- An Empirical Exploration of Recurrent Network Architectures (2015), R. Jozefowicz et al.  [[pdf]](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
- Deep generative image models using a￼ laplacian pyramid of adversarial networks (2015), E.Denton et al. [[pdf]](http://papers.nips.cc/paper/5773-deep-generative-image-models-using-a-laplacian-pyramid-of-adversarial-networks.pdf)
- Gated Feedback Recurrent Neural Networks (2015), J. Chung et al. [[pdf]](http://www.jmlr.org/proceedings/papers/v37/chung15.pdf)
- Fast and accurate deep network learning by exponential linear units (ELUS) (2015), D. Clevert et al. [[pdf]](https://arxiv.org/pdf/1511.07289.pdf%5Cnhttp://arxiv.org/abs/1511.07289%5Cnhttp://arxiv.org/abs/1511.07289)
- Pointer networks (2015), O. Vinyals et al. [[pdf]](http://papers.nips.cc/paper/5866-pointer-networks.pdf)
- Visualizing and Understanding Recurrent Networks (2015), A. Karpathy et al. [[pdf]](https://arxiv.org/pdf/1506.02078)
- Attention-based models for speech recognition (2015), J. Chorowski et al. [[pdf]](http://papers.nips.cc/paper/5847-attention-based-models-for-speech-recognition.pdf)
- End-to-end memory networks (2015), S. Sukbaatar et al. [[pdf]](http://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf)
- Describing videos by exploiting temporal structure (2015), L. Yao et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Yao_Describing_Videos_by_ICCV_2015_paper.pdf)
- A neural conversational model (2015), O. Vinyals and Q. Le. [[pdf]](https://arxiv.org/pdf/1506.05869.pdf)
- Improving distributional similarity with lessons learned from word embeddings, O. Levy et al. [[pdf]] (https://www.transacl.org/ojs/index.php/tacl/article/download/570/124)
- Transition-Based Dependency Parsing with Stack Long Short-Term Memory (2015), C. Dyer et al. [[pdf]](http://aclweb.org/anthology/P/P15/P15-1033.pdf)
- Improved Transition-Based Parsing by Modeling Characters instead of Words with LSTMs (2015), M. Ballesteros et al. [[pdf]](http://aclweb.org/anthology/D/D15/D15-1041.pdf)
- Finding function in form: Compositional character models for open vocabulary word representation (2015), W. Ling et al. [[pdf]](http://aclweb.org/anthology/D/D15/D15-1176.pdf)


*(~2014)*
- DeepPose: Human pose estimation via deep neural networks (2014), A. Toshev and C. Szegedy [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Toshev_DeepPose_Human_Pose_2014_CVPR_paper.pdf)
- Learning a Deep Convolutional Network for Image Super-Resolution (2014, C. Dong et al. [[pdf]](https://www.researchgate.net/profile/Chen_Change_Loy/publication/264552416_Lecture_Notes_in_Computer_Science/links/53e583e50cf25d674e9c280e.pdf)
- Recurrent models of visual attention (2014), V. Mnih et al. [[pdf]](http://arxiv.org/pdf/1406.6247.pdf)
- Empirical evaluation of gated recurrent neural networks on sequence modeling (2014), J. Chung et al. [[pdf]](https://arxiv.org/pdf/1412.3555)
- Addressing the rare word problem in neural machine translation (2014), M. Luong et al. [[pdf]](https://arxiv.org/pdf/1410.8206)
- On the properties of neural machine translation: Encoder-decoder approaches (2014), K. Cho et. al.
- Recurrent neural network regularization (2014), W. Zaremba et al. [[pdf]](http://arxiv.org/pdf/1409.2329)
- Intriguing properties of neural networks (2014), C. Szegedy et al. [[pdf]](https://arxiv.org/pdf/1312.6199.pdf)
- Towards end-to-end speech recognition with recurrent neural networks (2014), A. Graves and N. Jaitly. [[pdf]](http://www.jmlr.org/proceedings/papers/v32/graves14.pdf)
- Scalable object detection using deep neural networks (2014), D. Erhan et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Erhan_Scalable_Object_Detection_2014_CVPR_paper.pdf)
- On the importance of initialization and momentum in deep learning (2013), I. Sutskever et al. [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2013_sutskever13.pdf)
- Regularization of neural networks using dropconnect (2013), L. Wan et al. [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2013_wan13.pdf)
- Learning Hierarchical Features for Scene Labeling (2013), C. Farabet et al. [[pdf]](https://hal-enpc.archives-ouvertes.fr/docs/00/74/20/77/PDF/farabet-pami-13.pdf)
- Linguistic Regularities in Continuous Space Word Representations (2013), T. Mikolov et al. [[pdf]](http://www.aclweb.org/anthology/N13-1#page=784)
- Large scale distributed deep networks (2012), J. Dean et al. [[pdf]](http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf)
- A Fast and Accurate Dependency Parser using Neural Networks. Chen and Manning. [[pdf]](http://cs.stanford.edu/people/danqi/papers/emnlp2014.pdf)



## Acknowledgement

Thank you for all your contributions. Please make sure to read the [contributing guide](https://github.com/terryum/awesome-deep-learning-papers/blob/master/Contributing.md) before you make a pull request.
