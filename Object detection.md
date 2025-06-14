# 📚 Awesome Object detection Research

> 🗓️ 自动生成的论文列表 | 最后更新: 2025-06-14

## 🔍 目录
- [3D Object Detection](#3d-object-detection)
- [Contextual Object Detection](#contextual-object-detection)
- [General Object Detection](#general-object-detection)
- [Object Attribute Prediction](#object-attribute-prediction)
- [Open World Object Detection](#open-world-object-detection)
- [Small Object Detection](#small-object-detection)
- [Specialized Object Detection](#specialized-object-detection)
- [Video Object Detection](#video-object-detection)

---

## 📁 3D Object Detection

### 📄 [Out-of-Distribution Detection for LiDAR-based 3D Object Detection](http://arxiv.org/pdf/2209.14435v1)

👤 **Authors**: Chengjie Huang, Van Duong Nguyen, Vahdat Abdelzad, Christopher Gus Mannes, Luke Rowe, Benjamin Therien, Rick Salay, Krzysztof Czarnecki

📅 **Published**: `Sep 28, 2022` | 🔄 **Updated**: `Sep 28, 2022` | 🆔 **Arxiv ID**: [2209.14435v1](http://arxiv.org/pdf/2209.14435v1)

💡 **Abstract**:
> 3D object detection is an essential part of automated driving, and deep
> neural networks (DNNs) have achieved state-of-the-art performance for this
> task. However, deep models are notorious for assigning high confidence scores
> to out-of-distribution (OOD) inputs, that is, inputs that are not drawn from
> the training distribution. Detecting OOD inputs is challenging and essential
> for the safe deployment of models. OOD detection has been studied extensively
> for the classification task, but it has not received enough attention for the
> object detection task, specifically LiDAR-based 3D object detection. In this
> paper, we focus on the detection of OOD inputs for LiDAR-based 3D object
> detection. We formulate what OOD inputs mean for object detection and propose
> to adapt several OOD detection methods for object detection. We accomplish this
> by our proposed feature extraction method. To evaluate OOD detection methods,
> we develop a simple but effective technique of generating OOD objects for a
> given object detection model. Our evaluation based on the KITTI dataset shows
> that different OOD detection methods have biases toward detecting specific OOD
> objects. It emphasizes the importance of combined OOD detection methods and
> more research in this direction.

---

## 📁 Contextual Object Detection

### 📄 [Context in object detection: a systematic literature review](http://arxiv.org/pdf/2503.23249v1)

👤 **Authors**: Mahtab Jamali, Paul Davidsson, Reza Khoshkangini, Martin Georg Ljungqvist, Radu-Casian Mihailescu

📅 **Published**: `Mar 29, 2025` | 🔄 **Updated**: `Mar 29, 2025` | 🆔 **Arxiv ID**: [2503.23249v1](http://arxiv.org/pdf/2503.23249v1)

💡 **Abstract**:
> Context is an important factor in computer vision as it offers valuable
> information to clarify and analyze visual data. Utilizing the contextual
> information inherent in an image or a video can improve the precision and
> effectiveness of object detectors. For example, where recognizing an isolated
> object might be challenging, context information can improve comprehension of
> the scene. This study explores the impact of various context-based approaches
> to object detection. Initially, we investigate the role of context in object
> detection and survey it from several perspectives. We then review and discuss
> the most recent context-based object detection approaches and compare them.
> Finally, we conclude by addressing research questions and identifying gaps for
> further studies. More than 265 publications are included in this survey,
> covering different aspects of context in different categories of object
> detection, including general object detection, video object detection, small
> object detection, camouflaged object detection, zero-shot, one-shot, and
> few-shot object detection. This literature review presents a comprehensive
> overview of the latest advancements in context-based object detection,
> providing valuable contributions such as a thorough understanding of contextual
> information and effective methods for integrating various context types into
> object detection, thus benefiting researchers.

---

### 📄 [Detecting out-of-context objects using contextual cues](http://arxiv.org/pdf/2202.05930v1)

👤 **Authors**: Manoj Acharya, Anirban Roy, Kaushik Koneripalli, Susmit Jha, Christopher Kanan, Ajay Divakaran

📅 **Published**: `Feb 11, 2022` | 🔄 **Updated**: `Feb 11, 2022` | 🆔 **Arxiv ID**: [2202.05930v1](http://arxiv.org/pdf/2202.05930v1)

💡 **Abstract**:
> This paper presents an approach to detect out-of-context (OOC) objects in an
> image. Given an image with a set of objects, our goal is to determine if an
> object is inconsistent with the scene context and detect the OOC object with a
> bounding box. In this work, we consider commonly explored contextual relations
> such as co-occurrence relations, the relative size of an object with respect to
> other objects, and the position of the object in the scene. We posit that
> contextual cues are useful to determine object labels for in-context objects
> and inconsistent context cues are detrimental to determining object labels for
> out-of-context objects. To realize this hypothesis, we propose a graph
> contextual reasoning network (GCRN) to detect OOC objects. GCRN consists of two
> separate graphs to predict object labels based on the contextual cues in the
> image: 1) a representation graph to learn object features based on the
> neighboring objects and 2) a context graph to explicitly capture contextual
> cues from the neighboring objects. GCRN explicitly captures the contextual cues
> to improve the detection of in-context objects and identify objects that
> violate contextual relations. In order to evaluate our approach, we create a
> large-scale dataset by adding OOC object instances to the COCO images. We also
> evaluate on recent OCD benchmark. Our results show that GCRN outperforms
> competitive baselines in detecting OOC objects and correctly detecting
> in-context objects.

---

## 📁 General Object Detection

### 📄 [Recent Advances in Deep Learning for Object Detection](http://arxiv.org/pdf/1908.03673v1)

👤 **Authors**: Xiongwei Wu, Doyen Sahoo, Steven C. H. Hoi

📅 **Published**: `Aug 10, 2019` | 🔄 **Updated**: `Aug 10, 2019` | 🆔 **Arxiv ID**: [1908.03673v1](http://arxiv.org/pdf/1908.03673v1)

💡 **Abstract**:
> Object detection is a fundamental visual recognition problem in computer
> vision and has been widely studied in the past decades. Visual object detection
> aims to find objects of certain target classes with precise localization in a
> given image and assign each object instance a corresponding class label. Due to
> the tremendous successes of deep learning based image classification, object
> detection techniques using deep learning have been actively studied in recent
> years. In this paper, we give a comprehensive survey of recent advances in
> visual object detection with deep learning. By reviewing a large body of recent
> related work in literature, we systematically analyze the existing object
> detection frameworks and organize the survey into three major parts: (i)
> detection components, (ii) learning strategies, and (iii) applications &
> benchmarks. In the survey, we cover a variety of factors affecting the
> detection performance in detail, such as detector architectures, feature
> learning, proposal generation, sampling strategies, etc. Finally, we discuss
> several future directions to facilitate and spur future research for visual
> object detection with deep learning. Keywords: Object Detection, Deep Learning,
> Deep Convolutional Neural Networks

---

## 📁 Object Attribute Prediction

### 📄 [Detect-and-describe: Joint learning framework for detection and description of objects](http://arxiv.org/pdf/2204.08828v1)

👤 **Authors**: Addel Zafar, Umar Khalid

📅 **Published**: `Apr 19, 2022` | 🔄 **Updated**: `Apr 19, 2022` | 🆔 **Arxiv ID**: [2204.08828v1](http://arxiv.org/pdf/2204.08828v1)

💡 **Abstract**:
> Traditional object detection answers two questions; "what" (what the object
> is?) and "where" (where the object is?). "what" part of the object detection
> can be fine-grained further i.e. "what type", "what shape" and "what material"
> etc. This results in the shifting of the object detection tasks to the object
> description paradigm. Describing an object provides additional detail that
> enables us to understand the characteristics and attributes of the object
> ("plastic boat" not just boat, "glass bottle" not just bottle). This additional
> information can implicitly be used to gain insight into unseen objects (e.g.
> unknown object is "metallic", "has wheels"), which is not possible in
> traditional object detection. In this paper, we present a new approach to
> simultaneously detect objects and infer their attributes, we call it Detect and
> Describe (DaD) framework. DaD is a deep learning-based approach that extends
> object detection to object attribute prediction as well. We train our model on
> aPascal train set and evaluate our approach on aPascal test set. We achieve
> 97.0% in Area Under the Receiver Operating Characteristic Curve (AUC) for
> object attributes prediction on aPascal test set. We also show qualitative
> results for object attribute prediction on unseen objects, which demonstrate
> the effectiveness of our approach for describing unknown objects.

---

## 📁 Open World Object Detection

### 📄 [PROB: Probabilistic Objectness for Open World Object Detection](http://arxiv.org/pdf/2212.01424v1)

👤 **Authors**: Orr Zohar, Kuan-Chieh Wang, Serena Yeung

📅 **Published**: `Dec 02, 2022` | 🔄 **Updated**: `Dec 02, 2022` | 🆔 **Arxiv ID**: [2212.01424v1](http://arxiv.org/pdf/2212.01424v1)

💻 **Code**: [Link](https://github.com/orrzohar/PROB)

💡 **Abstract**:
> Open World Object Detection (OWOD) is a new and challenging computer vision
> task that bridges the gap between classic object detection (OD) benchmarks and
> object detection in the real world. In addition to detecting and classifying
> seen/labeled objects, OWOD algorithms are expected to detect novel/unknown
> objects - which can be classified and incrementally learned. In standard OD,
> object proposals not overlapping with a labeled object are automatically
> classified as background. Therefore, simply applying OD methods to OWOD fails
> as unknown objects would be predicted as background. The challenge of detecting
> unknown objects stems from the lack of supervision in distinguishing unknown
> objects and background object proposals. Previous OWOD methods have attempted
> to overcome this issue by generating supervision using pseudo-labeling -
> however, unknown object detection has remained low. Probabilistic/generative
> models may provide a solution for this challenge. Herein, we introduce a novel
> probabilistic framework for objectness estimation, where we alternate between
> probability distribution estimation and objectness likelihood maximization of
> known objects in the embedded feature space - ultimately allowing us to
> estimate the objectness probability of different proposals. The resulting
> Probabilistic Objectness transformer-based open-world detector, PROB,
> integrates our framework into traditional object detection models, adapting
> them for the open-world setting. Comprehensive experiments on OWOD benchmarks
> show that PROB outperforms all existing OWOD methods in both unknown object
> detection ($\sim 2\times$ unknown recall) and known object detection ($\sim
> 10\%$ mAP). Our code will be made available upon publication at
> https://github.com/orrzohar/PROB.

---

## 📁 Small Object Detection

### 📄 [A Coarse to Fine Framework for Object Detection in High Resolution Image](http://arxiv.org/pdf/2303.01219v1)

👤 **Authors**: Jinyan Liu, Jie Chen

📅 **Published**: `Mar 02, 2023` | 🔄 **Updated**: `Mar 02, 2023` | 🆔 **Arxiv ID**: [2303.01219v1](http://arxiv.org/pdf/2303.01219v1)

💡 **Abstract**:
> Object detection is a fundamental problem in computer vision, aiming at
> locating and classifying objects in image. Although current devices can easily
> take very high-resolution images, current approaches of object detection seldom
> consider detecting tiny object or the large scale variance problem in high
> resolution images. In this paper, we introduce a simple yet efficient approach
> that improves accuracy of object detection especially for small objects and
> large scale variance scene while reducing the computational cost in high
> resolution image. Inspired by observing that overall detection accuracy is
> reduced if the image is properly down-sampled but the recall rate is not
> significantly reduced. Besides, small objects can be better detected by
> inputting high-resolution images even if using lightweight detector. We propose
> a cluster-based coarse-to-fine object detection framework to enhance the
> performance for detecting small objects while ensure the accuracy of large
> objects in high-resolution images. For the first stage, we perform coarse
> detection on the down-sampled image and center localization of small objects by
> lightweight detector on high-resolution image, and then obtains image chips
> based on cluster region generation method by coarse detection and center
> localization results, and further sends chips to the second stage detector for
> fine detection. Finally, we merge the coarse detection and fine detection
> results. Our approach can make good use of the sparsity of the objects and the
> information in high-resolution image, thereby making the detection more
> efficient. Experiment results show that our proposed approach achieves
> promising performance compared with other state-of-the-art detectors.

---

## 📁 Specialized Object Detection

### 📄 [Towards Reflected Object Detection: A Benchmark](http://arxiv.org/pdf/2407.05575v1)

👤 **Authors**: Zhongtian Wang, You Wu, Hui Zhou, Shuiwang Li

📅 **Published**: `Jul 08, 2024` | 🔄 **Updated**: `Jul 08, 2024` | 🆔 **Arxiv ID**: [2407.05575v1](http://arxiv.org/pdf/2407.05575v1)

💻 **Code**: [Link](https://github.com/Tqybu-hans/RODD)

💡 **Abstract**:
> Object detection has greatly improved over the past decade thanks to advances
> in deep learning and large-scale datasets. However, detecting objects reflected
> in surfaces remains an underexplored area. Reflective surfaces are ubiquitous
> in daily life, appearing in homes, offices, public spaces, and natural
> environments. Accurate detection and interpretation of reflected objects are
> essential for various applications. This paper addresses this gap by
> introducing a extensive benchmark specifically designed for Reflected Object
> Detection. Our Reflected Object Detection Dataset (RODD) features a diverse
> collection of images showcasing reflected objects in various contexts,
> providing standard annotations for both real and reflected objects. This
> distinguishes it from traditional object detection benchmarks. RODD encompasses
> 10 categories and includes 21,059 images of real and reflected objects across
> different backgrounds, complete with standard bounding box annotations and the
> classification of objects as real or reflected. Additionally, we present
> baseline results by adapting five state-of-the-art object detection models to
> address this challenging task. Experimental results underscore the limitations
> of existing methods when applied to reflected object detection, highlighting
> the need for specialized approaches. By releasing RODD, we aim to support and
> advance future research on detecting reflected objects. Dataset and code are
> available at: https: //github.com/Tqybu-hans/RODD.

---

## 📁 Video Object Detection

### 📄 [Plug & Play Convolutional Regression Tracker for Video Object Detection](http://arxiv.org/pdf/2003.00981v1)

👤 **Authors**: Ye Lyu, Michael Ying Yang, George Vosselman, Gui-Song Xia

📅 **Published**: `Mar 02, 2020` | 🔄 **Updated**: `Mar 02, 2020` | 🆔 **Arxiv ID**: [2003.00981v1](http://arxiv.org/pdf/2003.00981v1)

💡 **Abstract**:
> Video object detection targets to simultaneously localize the bounding boxes
> of the objects and identify their classes in a given video. One challenge for
> video object detection is to consistently detect all objects across the whole
> video. As the appearance of objects may deteriorate in some frames, features or
> detections from the other frames are commonly used to enhance the prediction.
> In this paper, we propose a Plug & Play scale-adaptive convolutional regression
> tracker for the video object detection task, which could be easily and
> compatibly implanted into the current state-of-the-art detection networks. As
> the tracker reuses the features from the detector, it is a very light-weighted
> increment to the detection network. The whole network performs at the speed
> close to a standard object detector. With our new video object detection
> pipeline design, image object detectors can be easily turned into efficient
> video object detectors without modifying any parameters. The performance is
> evaluated on the large-scale ImageNet VID dataset. Our Plug & Play design
> improves mAP score for the image detector by around 5% with only little speed
> drop.

---

### 📄 [TrackNet: Simultaneous Object Detection and Tracking and Its Application in Traffic Video Analysis](http://arxiv.org/pdf/1902.01466v1)

👤 **Authors**: Chenge Li, Gregory Dobler, Xin Feng, Yao Wang

📅 **Published**: `Feb 04, 2019` | 🔄 **Updated**: `Feb 04, 2019` | 🆔 **Arxiv ID**: [1902.01466v1](http://arxiv.org/pdf/1902.01466v1)

💡 **Abstract**:
> Object detection and object tracking are usually treated as two separate
> processes. Significant progress has been made for object detection in 2D images
> using deep learning networks. The usual tracking-by-detection pipeline for
> object tracking requires that the object is successfully detected in the first
> frame and all subsequent frames, and tracking is done by associating detection
> results. Performing object detection and object tracking through a single
> network remains a challenging open question. We propose a novel network
> structure named trackNet that can directly detect a 3D tube enclosing a moving
> object in a video segment by extending the faster R-CNN framework. A Tube
> Proposal Network (TPN) inside the trackNet is proposed to predict the
> objectness of each candidate tube and location parameters specifying the
> bounding tube. The proposed framework is applicable for detecting and tracking
> any object and in this paper, we focus on its application for traffic video
> analysis. The proposed model is trained and tested on UA-DETRAC, a large
> traffic video dataset available for multi-vehicle detection and tracking, and
> obtained very promising results.

---

