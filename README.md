# Deep Learning Papers Reading Roadmap

>If you are a newcomer to the Deep Learning area, the first question you may have is "Which paper should I start reading from?"

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

**[27]** Le, Quoc V. "**Building high-level features using large scale unsupervised learning**." 2013 IEEE international conference on acoustics, speech and signal processing. IEEE, 2013. [[pdf]](http://arxiv.org/pdf/1112.6209.pdf&embed) **(Milestone, Andrew Ng, Google Brain Project, Cat)** :star::star::star::star:


**[28]** Kingma, Diederik P., and Max Welling. "**Auto-encoding variational bayes**." arXiv preprint arXiv:1312.6114 (2013). [[pdf]](http://arxiv.org/pdf/1312.6114) **(VAE)** :star::star::star::star:

**[29]** Goodfellow, Ian, et al. "**Generative adversarial nets**." Advances in Neural Information Processing Systems. 2014. [[pdf]](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) **(GAN,super cool idea)** :star::star::star::star::star:

**[30]** Radford, Alec, Luke Metz, and Soumith Chintala. "**Unsupervised representation learning with deep convolutional generative adversarial networks**." arXiv preprint arXiv:1511.06434 (2015). [[pdf]](http://arxiv.org/pdf/1511.06434) **(DCGAN)** :star::star::star::star:

**[31]** Gregor, Karol, et al. "**DRAW: A recurrent neural network for image generation**." arXiv preprint arXiv:1502.04623 (2015). [[pdf]](http://jmlr.org/proceedings/papers/v37/gregor15.pdf) **(VAE with attention, outstanding work)** :star::star::star::star::star:

**[32]** Oord, Aaron van den, Nal Kalchbrenner, and Koray Kavukcuoglu. "**Pixel recurrent neural networks**." arXiv preprint arXiv:1601.06759 (2016). [[pdf]](http://arxiv.org/pdf/1601.06759) **(PixelRNN)** :star::star::star::star:

**[33]** Oord, Aaron van den, et al. "Conditional image generation with PixelCNN decoders." arXiv preprint arXiv:1606.05328 (2016). [[pdf]](https://arxiv.org/pdf/1606.05328) **(PixelCNN)** :star::star::star::star:

## 2.4 RNN / Sequence-to-Sequence Model

**[34]** Graves, Alex. "**Generating sequences with recurrent neural networks**." arXiv preprint arXiv:1308.0850 (2013). [[pdf]](http://arxiv.org/pdf/1308.0850) **(LSTM, very nice generating result, show the power of RNN)** :star::star::star::star:

**[35]** Cho, Kyunghyun, et al. "**Learning phrase representations using RNN encoder-decoder for statistical machine translation**." arXiv preprint arXiv:1406.1078 (2014). [[pdf]](http://arxiv.org/pdf/1406.1078) **(First Seq-to-Seq Paper)** :star::star::star::star:

**[36]** Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "**Sequence to sequence learning with neural networks**." Advances in neural information processing systems. 2014. [[pdf]](http://papers.nips.cc/paper/5346-information-based-learning-by-agents-in-unbounded-state-spaces.pdf) **(Outstanding Work)** :star::star::star::star::star:

**[37]** Bahdanau, Dzmitry, KyungHyun Cho, and Yoshua Bengio. "**Neural Machine Translation by Jointly Learning to Align and Translate**." arXiv preprint arXiv:1409.0473 (2014). [[pdf]](https://arxiv.org/pdf/1409.0473v7.pdf) :star::star::star::star:

**[38]** Vinyals, Oriol, and Quoc Le. "**A neural conversational model**." arXiv preprint arXiv:1506.05869 (2015). [[pdf]](http://arxiv.org/pdf/1506.05869.pdf%20(http://arxiv.org/pdf/1506.05869.pdf)) **(Seq-to-Seq on Chatbot)** :star::star::star:


## 2.6 Deep Reinforcement Learning

**[45]** Mnih, Volodymyr, et al. "**Playing atari with deep reinforcement learning**." arXiv preprint arXiv:1312.5602 (2013). [[pdf]](http://arxiv.org/pdf/1312.5602.pdf)) **(First Paper named deep reinforcement learning)** :star::star::star::star:

**[46]** Mnih, Volodymyr, et al. "**Human-level control through deep reinforcement learning**." Nature 518.7540 (2015): 529-533. [[pdf]](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf) **(Milestone)** :star::star::star::star::star:

**[47]** Wang, Ziyu, Nando de Freitas, and Marc Lanctot. "**Dueling network architectures for deep reinforcement learning**." arXiv preprint arXiv:1511.06581 (2015). [[pdf]](http://arxiv.org/pdf/1511.06581) **(ICLR best paper,great idea)**  :star::star::star::star:

**[48]** Mnih, Volodymyr, et al. "**Asynchronous methods for deep reinforcement learning**." arXiv preprint arXiv:1602.01783 (2016). [[pdf]](http://arxiv.org/pdf/1602.01783) **(State-of-the-art method)** :star::star::star::star::star:

**[49]** Lillicrap, Timothy P., et al. "**Continuous control with deep reinforcement learning**." arXiv preprint arXiv:1509.02971 (2015). [[pdf]](http://arxiv.org/pdf/1509.02971) **(DDPG)** :star::star::star::star:

**[50]** Gu, Shixiang, et al. "**Continuous Deep Q-Learning with Model-based Acceleration**." arXiv preprint arXiv:1603.00748 (2016). [[pdf]](http://arxiv.org/pdf/1603.00748) **(NAF)** :star::star::star::star:

**[51]** Schulman, John, et al. "**Trust region policy optimization**." CoRR, abs/1502.05477 (2015). [[pdf]](http://www.jmlr.org/proceedings/papers/v37/schulman15.pdf) **(TRPO)** :star::star::star::star:

**[52]** Silver, David, et al. "**Mastering the game of Go with deep neural networks and tree search**." Nature 529.7587 (2016): 484-489. [[pdf]](http://willamette.edu/~levenick/cs448/goNature.pdf) **(AlphaGo)** :star::star::star::star::star:


## 2.8 One Shot Deep Learning

**[59]** Lake, Brenden M., Ruslan Salakhutdinov, and Joshua B. Tenenbaum. "**Human-level concept learning through probabilistic program induction**." Science 350.6266 (2015): 1332-1338. [[pdf]](http://clm.utexas.edu/compjclub/wp-content/uploads/2016/02/lake2015.pdf) **(No Deep Learning,but worth reading)** :star::star::star::star::star:

**[60]** Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov. "**Siamese Neural Networks for One-shot Image Recognition**."(2015) [[pdf]](http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf) :star::star::star:

**[61]** Santoro, Adam, et al. "**One-shot Learning with Memory-Augmented Neural Networks**." arXiv preprint arXiv:1605.06065 (2016). [[pdf]](http://arxiv.org/pdf/1605.06065) **(A basic step to one shot learning)** :star::star::star::star:

**[62]** Vinyals, Oriol, et al. "**Matching Networks for One Shot Learning**." arXiv preprint arXiv:1606.04080 (2016). [[pdf]](https://arxiv.org/pdf/1606.04080) :star::star::star:

**[63]** Hariharan, Bharath, and Ross Girshick. "**Low-shot visual object recognition**." arXiv preprint arXiv:1606.02819 (2016). [[pdf]](http://arxiv.org/pdf/1606.02819) **(A step to large data)** :star::star::star::star:


# 3 Applications

## 3.1 NLP(Natural Language Processing)

**[1]** Antoine Bordes, et al. "**Joint Learning of Words and Meaning Representations for Open-Text Semantic Parsing**." AISTATS(2012) [[pdf]](https://www.hds.utc.fr/~bordesan/dokuwiki/lib/exe/fetch.php?id=en%3Apubli&cache=cache&media=en:bordes12aistats.pdf) :star::star::star::star:

**[2]** Mikolov, et al. "**Distributed representations of words and phrases and their compositionality**." ANIPS(2013): 3111-3119 [[pdf]](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) **(word2vec)** :star::star::star:

**[3]** Sutskever, et al. "**“Sequence to sequence learning with neural networks**." ANIPS(2014) [[pdf]](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) :star::star::star:

**[4]** Ankit Kumar, et al. "**“Ask Me Anything: Dynamic Memory Networks for Natural Language Processing**." arXiv preprint arXiv:1506.07285(2015) [[pdf]](https://arxiv.org/abs/1506.07285) :star::star::star::star:

**[5]** Yoon Kim, et al. "**Character-Aware Neural Language Models**." NIPS(2015) arXiv preprint arXiv:1508.06615(2015) [[pdf]](https://arxiv.org/abs/1508.06615) :star::star::star::star:

**[6]** Jason Weston, et al. "**Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks**." arXiv preprint arXiv:1502.05698(2015) [[pdf]](https://arxiv.org/abs/1502.05698) **(bAbI tasks)** :star::star::star:

**[7]** Karl Moritz Hermann, et al. "**Teaching Machines to Read and Comprehend**." arXiv preprint arXiv:1506.03340(2015) [[pdf]](https://arxiv.org/abs/1506.03340) **(CNN/DailyMail cloze style questions)** :star::star:

**[8]** Alexis Conneau, et al. "**Very Deep Convolutional Networks for Natural Language Processing**." arXiv preprint arXiv:1606.01781(2016) [[pdf]](https://arxiv.org/abs/1606.01781) **(state-of-the-art in text classification)** :star::star::star:

**[9]** Armand Joulin, et al. "**Bag of Tricks for Efficient Text Classification**." arXiv preprint arXiv:1607.01759(2016) [[pdf]](https://arxiv.org/abs/1607.01759) **(slightly worse than state-of-the-art, but a lot faster)** :star::star::star:

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

## 3.4 Image Caption
**[1]** Farhadi,Ali,etal. "**Every picture tells a story: Generating sentences from images**". In Computer VisionECCV 2010. Springer Berlin Heidelberg:15-29, 2010. [[pdf]](https://www.cs.cmu.edu/~afarhadi/papers/sentence.pdf) :star::star::star:

**[2]** Kulkarni, Girish, et al. "**Baby talk: Understanding and generating image descriptions**". In Proceedings of the 24th CVPR, 2011. [[pdf]](http://tamaraberg.com/papers/generation_cvpr11.pdf):star::star::star::star:

**[3]** Vinyals, Oriol, et al. "**Show and tell: A neural image caption generator**". In arXiv preprint arXiv:1411.4555, 2014. [[pdf]](https://arxiv.org/pdf/1411.4555.pdf):star::star::star:

**[4]** Donahue, Jeff, et al. "**Long-term recurrent convolutional networks for visual recognition and description**". In arXiv preprint arXiv:1411.4389 ,2014. [[pdf]](https://arxiv.org/pdf/1411.4389.pdf)

**[5]** Karpathy, Andrej, and Li Fei-Fei. "**Deep visual-semantic alignments for generating image descriptions**". In arXiv preprint arXiv:1412.2306, 2014. [[pdf]](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf):star::star::star::star::star:

**[6]** Karpathy, Andrej, Armand Joulin, and Fei Fei F. Li. "**Deep fragment embeddings for bidirectional image sentence mapping**". In Advances in neural information processing systems, 2014. [[pdf]](https://arxiv.org/pdf/1406.5679v1.pdf):star::star::star::star:

**[7]** Fang, Hao, et al. "**From captions to visual concepts and back**". In arXiv preprint arXiv:1411.4952, 2014. [[pdf]](https://arxiv.org/pdf/1411.4952v3.pdf):star::star::star::star::star:

**[8]** Chen, Xinlei, and C. Lawrence Zitnick. "**Learning a recurrent visual representation for image caption generation**". In arXiv preprint arXiv:1411.5654, 2014. [[pdf]](https://arxiv.org/pdf/1411.5654v1.pdf):star::star::star::star:

**[9]** Mao, Junhua, et al. "**Deep captioning with multimodal recurrent neural networks (m-rnn)**". In arXiv preprint arXiv:1412.6632, 2014. [[pdf]](https://arxiv.org/pdf/1412.6632v5.pdf):star::star::star:

**[10]** Xu, Kelvin, et al. "**Show, attend and tell: Neural image caption generation with visual attention**". In arXiv preprint arXiv:1502.03044, 2015. [[pdf]](https://arxiv.org/pdf/1502.03044v3.pdf):star::star::star::star::star:




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

### Natural Language Processing / RNNs
- **Neural Architectures for Named Entity Recognition** (2016), G. Lample et al. [[pdf]](http://aclweb.org/anthology/N/N16/N16-1030.pdf)
- **Exploring the limits of language modeling** (2016), R. Jozefowicz et al. [[pdf]](http://arxiv.org/pdf/1602.02410)
- **Teaching machines to read and comprehend** (2015), K. Hermann et al. [[pdf]](http://papers.nips.cc/paper/5945-teaching-machines-to-read-and-comprehend.pdf)
- **Effective approaches to attention-based neural machine translation** (2015), M. Luong et al. [[pdf]](https://arxiv.org/pdf/1508.04025)
- **Conditional random fields as recurrent neural networks** (2015), S. Zheng and S. Jayasumana. [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Conditional_Random_Fields_ICCV_2015_paper.pdf)
- **Memory networks** (2014), J. Weston et al. [[pdf]](https://arxiv.org/pdf/1410.3916)
- **Neural turing machines** (2014), A. Graves et al. [[pdf]](https://arxiv.org/pdf/1410.5401)
- **Neural machine translation by jointly learning to align and translate** (2014), D. Bahdanau et al. [[pdf]](http://arxiv.org/pdf/1409.0473)
- **Sequence to sequence learning with neural networks** (2014), I. Sutskever et al. [[pdf]](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
- **Learning phrase representations using RNN encoder-decoder for statistical machine translation** (2014), K. Cho et al. [[pdf]](http://arxiv.org/pdf/1406.1078)
- **A convolutional neural network for modeling sentences** (2014), N. Kalchbrenner et al. [[pdf]](http://arxiv.org/pdf/1404.2188v1)
- **Convolutional neural networks for sentence classification** (2014), Y. Kim [[pdf]](http://arxiv.org/pdf/1408.5882)
- **Glove: Global vectors for word representation** (2014), J. Pennington et al. [[pdf]](http://anthology.aclweb.org/D/D14/D14-1162.pdf)
- **Distributed representations of sentences and documents** (2014), Q. Le and T. Mikolov [[pdf]](http://arxiv.org/pdf/1405.4053)
- **Distributed representations of words and phrases and their compositionality** (2013), T. Mikolov et al. [[pdf]](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- **Efficient estimation of word representations in vector space** (2013), T. Mikolov et al.  [[pdf]](http://arxiv.org/pdf/1301.3781)
- **Recursive deep models for semantic compositionality over a sentiment treebank** (2013), R. Socher et al. [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.383.1327&rep=rep1&type=pdf)
- **Generating sequences with recurrent neural networks** (2013), A. Graves. [[pdf]](https://arxiv.org/pdf/1308.0850)

<!---[Key researchers]  [Kyunghyun Cho](https://scholar.google.ca/citations?user=0RAmmIAAAAAJ), [Oriol Vinyals](https://scholar.google.ca/citations?user=NkzyCvUAAAAJ), [Richard Socher](https://scholar.google.ca/citations?hl=en&user=FaOcyfMAAAAJ), [Tomas Mikolov](https://scholar.google.ca/citations?user=oBu8kMMAAAAJ), [Christopher D. Manning](https://scholar.google.ca/citations?user=1zmDOdwAAAAJ), [Yoshua Bengio](https://scholar.google.ca/citations?user=kukA0LcAAAAJ)-->


### Image: Segmentation / Object Detection
- **Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks** (2015), S. Ren et al. [[pdf]](http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf)
- **Fast R-CNN** (2015), R. Girshick [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)


### More Papers from 2016
- **Layer Normalization** (2016), J. Ba et al. [[pdf]](https://arxiv.org/pdf/1607.06450v1.pdf)
- **Learning to learn by gradient descent by gradient descent** (2016), M. Andrychowicz et al. [[pdf]](http://arxiv.org/pdf/1606.04474v1)
- **Domain-adversarial training of neural networks** (2016), Y. Ganin et al. [[pdf]](http://www.jmlr.org/papers/volume17/15-239/source/15-239.pdf)
- **WaveNet: A Generative Model for Raw Audio** (2016), A. Oord et al. [[pdf]](https://arxiv.org/pdf/1609.03499v2) [[web]](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)
- **Colorful image colorization** (2016), R. Zhang et al. [[pdf]](https://arxiv.org/pdf/1603.08511)
- **Generative visual manipulation on the natural image manifold** (2016), J. Zhu et al. [[pdf]](https://arxiv.org/pdf/1609.03552)
- **Texture networks: Feed-forward synthesis of textures and stylized images** (2016), D Ulyanov et al. [[pdf]](http://www.jmlr.org/proceedings/papers/v48/ulyanov16.pdf)
- **SSD: Single shot multibox detector** (2016), W. Liu et al. [[pdf]](https://arxiv.org/pdf/1512.02325)
- **SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 1MB model size** (2016), F. Iandola et al. [[pdf]](http://arxiv.org/pdf/1602.07360)
- **Eie: Efficient inference engine on compressed deep neural network** (2016), S. Han et al. [[pdf]](http://arxiv.org/pdf/1602.01528)
- **Binarized neural networks: Training deep neural networks with weights and activations constrained to+ 1 or-1** (2016), M. Courbariaux et al. [[pdf]](https://arxiv.org/pdf/1602.02830)
- **Dynamic memory networks for visual and textual question answering** (2016), C. Xiong et al. [[pdf]](http://www.jmlr.org/proceedings/papers/v48/xiong16.pdf)
- **Stacked attention networks for image question answering** (2016), Z. Yang et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Yang_Stacked_Attention_Networks_CVPR_2016_paper.pdf)
- **Hybrid computing using a neural network with dynamic external memory** (2016), A. Graves et al. [[pdf]](https://www.gwern.net/docs/2016-graves.pdf)
- **Google's neural machine translation system: Bridging the gap between human and machine translation** (2016), Y. Wu et al. [[pdf]](https://arxiv.org/pdf/1609.08144)

* * *


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

### Video Lectures / Tutorials / Blogs

*(Lectures)*
- CS231n, Convolutional Neural Networks for Visual Recognition, Stanford University [[web]](http://cs231n.stanford.edu/)
- CS224d, Deep Learning for Natural Language Processing, Stanford University [[web]](http://cs224d.stanford.edu/)
- Oxford Deep NLP 2017, Deep Learning for Natural Language Processing, University of Oxford [[web]](https://github.com/oxford-cs-deepnlp-2017/lectures)

*(Tutorials)*
- NIPS 2016 Tutorials, Long Beach [[web]](https://nips.cc/Conferences/2016/Schedule?type=Tutorial)
- ICML 2016 Tutorials, New York City [[web]](http://techtalks.tv/icml/2016/tutorials/)
- ICLR 2016 Videos, San Juan [[web]](http://videolectures.net/iclr2016_san_juan/)
- Deep Learning Summer School 2016, Montreal [[web]](http://videolectures.net/deeplearning2016_montreal/)
- Bay Area Deep Learning School 2016, Stanford [[web]](https://www.bayareadlschool.org/)

*(Blogs)*
- OpenAI [[web]](https://www.openai.com/)
- Distill [[web]](http://distill.pub/)
- Andrej Karpathy Blog [[web]](http://karpathy.github.io/)
- Colah's Blog [[Web]](http://colah.github.io/)
- WildML [[Web]](http://www.wildml.com/)
- FastML [[web]](http://www.fastml.com/)
- TheMorningPaper [[web]](https://blog.acolyer.org)

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
