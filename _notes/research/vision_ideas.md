[Implicit 3D, David Fouhey] -- These days, 3D vision from a single image is usually a single, explicit representation like a depthmap. We know that this is not how it works in the case of humans; I think we can do something much more sensible and useful instead.


[Data-Driven Affordances/Grasping, David Fouhey] -- Affordance works tends to be either in contrived lab settings and interesting, or, in-the-wild and boring. We have a new dataset of people naturally interacting with objects and this presents an opportunity to study how people interact with the world.


[Visual explanations, Anja Rohrbach] -- A number of visual explanation approaches have been proposed to gain insights on decision making in deep models (e.g. visualizing attention maps). We want to make a systematic analysis of various visual explanations to see if they actually help us understand the models’ reasoning and shed light on the sources of errors.


[Human-like automatic video description, Anja Rohrbach] -- Generating natural language descriptions for video has many issues, e.g. most methods fail to capture the diversity and large vocabulary of human-generated descriptions. We want to study how to achieve higher diversity as well as specificity (i.e. produce more nuanced and relevant descriptions) in context of movie description.


[Learning Efficient Pixel-Level Representation, Fisher Yu] Learning image representation is a fundamental problem in computer vision. Research has shown that better representation can directly improve performance of computer vision applications. Instead of image level, we study pixel-level representation, which has direct impact on other basic tasks such as semantic segmentation and object detection. We also focus on computation efficiency of the representation so that it can be applied on real-world applications such as autonomous driving and mobile app.


[Large Scale Visual Database, Fisher Yu] Modern Computer Vision research is established on learning from real-world data. We aim to build visual database with billions of images and connect them in one single graph. The record-breaking number of images and dense connections among them will shed light on new ideas in image retrieval, weakly supervised learning and one/few shot learning.


[Autonomous Driving System, Fisher Yu] Self-driving is poised to revolutionize our society. We aim to use computer vision to solve perception, mapping and planing problems in self-driving. Our goal is to explore the boundaries of computer vision system in real-world application and establish reproducible research directions in self-driving.


[Meta self-supervised learning, Andrew Owens] --  Discovering new unsupervised learning tasks today is largely a matter of trial and error: you might propose a new task -- e.g. predicting sound from video -- train a model to solve that task, and then afterwards look at how well the learned image features perform on object recognition tasks.  Instead, we propose to do both of these things jointly: given a labeling task, we’d like to find an “unsupervised” learning task that -- when solved -- produces features that are useful for the given labeling task.


[Do we really need tactile sensing?, Andrew Owens] -- Touch sensing is useful for many robotics tasks, but it is also a difficult modality to use and to collect data for. We ask under what circumstances we can replace touch sensing with vision and sound.  We’ll train a microphone-equipped robot to reproduce the information provided by a touch sensor by poking, scratching, and manipulating an object and analyzing the resulting images and sounds. 


[Continuous Adaptation, Judy Hoffman] -- The world is an ever changing place. A model trained on yesterday’s data may perform poorly today. We seek to produce models that may be adapted continuously to the evolving world in an automatic way. 


[Lifelong Learning, Judy Hoffman] -- Humans tend to learn new concepts sequentially (and are always learning!), but we focus mainly on training our algorithms to learn a set of concepts in one large batch. Our goal is to learn to expand our visual knowledge in an online fashion using structure readily available in the data (ex: self-supervision).




[Physics meets Pixels, Angjoo] There's been a lot of progress in the physics simulation world like interactive character control in Graphics or learning locomotion in RL, but connection to the real world images are often missing. Can we marry the two, so both learning from simulation and from images can benefit each other? Ex: Simulated dynamics as a prior for image inference/Learn physical structure so natural motions like seen in videos emerge/TidyMan.


[Task-driven 3D inference - Angjoo] The world is 3D so we want to be able to understand the world in 3D, but that in itself is not the end, but the means for doing something else in the world. For example having the 3D geometry of the person can be used to infer his/her intention, can we also improve the 3D inference procedure such that we can do our task better without reducing it to a black box.