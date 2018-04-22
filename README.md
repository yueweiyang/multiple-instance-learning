# multiple-instance-learning
### When does multiple instance learning fail on popular deep learning problems?

*Yuewei Yang^1^, Daniel Reichman^1^, Leslie M. Collins^1^, and Jordan M.
Malof^1\
^*^\
1^Department of Electrical & Computer Engineering, Duke University,
Durham, NC 27708

*Abstract*— **Within the field of machine learning, multiple instance
learning (MIL) problems have received substantial attention because of
their applicability to many real-world problems, and their difficulty
for conventional classifiers. algorithms (e.g., neural networks). A
variety of techniques have been developed to adapt conventional
classification classifiers to MIL problems, yielding substantial
performance improvements. Recently, these MIL techniques have been
applied to deep learning (DL) algorithms for object recognition in
imagery. DL algorithms require substantial amounts of labeled training
data, in the form of costly object-level annotations. For example, if we
are looking for dogs in images, then each dog in each image must be
manually annotated with a bounding box. MIL methods have allowed DL
algorithms to train effectively with image-level labels (i.e., does this
image contain a dog, or not?), dramatically lowering effort for
labeling. Despite this success, it is unclear what limitations MIL
methods have, if any, for solving this problem. Specifically, are their
limits on the size of the images, and how rare the objects of interest
are within the imagery? In this work, we investigate these limits using
carefully designed synthetic data, and a non-linear support vector
machines, as a representative problem. The results suggest that MIL can
fail under some circumstances, but that these failures can be avoided by
making simple adjustments to the complexity (i.e., learning capacity) of
the support vector machine. **

Keywords—multiple instance learning, deep learning

INTRODUCTION 
=============

Within the field of machine learning, the problem of supervised binary
classification has received substantial attention \[1\]\[2\], \[3\]. The
goal of this problem is to build a parameterized function that can
distinguish between instances of data (e.g., represented as vectors)
from each of two classes: a positive class and a negative class. The
positive class often (but not always) refers to something that we wish
to find, among many other types of data which are aggregated into the
negative class. For example, within the field of computer vision, a
popular problem is to classify between images that contain a face (the
positive class), and that do not contain a face (the negative class). In
supervised binary classification, we assume some labeled examples of
data from each class is available so that the supervised learning
algorithm can infer parameters that result in good classification
accuracy. Due to the importance of this problem, and the attention it
has received, a vast number of classification models have been proposed,
with notable examples including neural network\[4\], support vector
machines\[5\]，and the random forest \[6\].

In this work, we consider a problem that is closely related to
supervised binary classification, called Multiple instance learning
(MIL). The main distinction of this problem is that the instances are
organized into groups, called bags. Negative bags are generally assumed
to include only negative instances; however, positive bags are only
guaranteed to include one positive instance and potentially many
negative instances. This structure of the data in an MIL problem is
illustrated in Fig. 1. MIL has received substantial attention in the
research literature because it is often applicable to many real-world
problems, and standard binary classification algorithms can perform
quite poorly them. As a result, a variety of techniques have been
developed to adapt standard classification algorithms to MIL problems,
often yielding substantial improvements in performance\[7\], \[8\]. Some
notable specific problems on which this has been successful include
object detection and classification\[2\], object localization \[9\], and
object tracking\[10\].

![](media/image1.png){width="3.379861111111111in"
height="2.6134962817147858in"}

Recently MIL has been used to improve popular deep learning algorithms
computer vision tasks, such as the recognition of objects in imagery
\[11\]\[12\]\[13\], \[14\]. A common goal of these algorithms is to (i)
detect the presence of an object within an image or (ii) localize the
object in the image by placing a bounding-box around the object. Whether
we wish to solve problem (i) or (ii), DL algorithms (and most supervised
models) require labeled training data in the form of tight bounding
boxes around each instance of the desired object in the imagery. For
example, if we are looking for faces, then each face in each image
requires a bounding box. Obtaining these annotations is costly and
time-consuming, which is exacerbated because DL algorithms usually
require relatively large quantities of training data compared to other
supervised classifiers.

MIL methods have allowed DL algorithms to train effectively with
image-level labels (i.e., does this image contain a dog, or not?),
dramatically lowering effort for labeling. MIL can be applied to these
problems by treating each full image as a bag of potential bounding box
locations (i.e., instances). If the image contains the object of
interest, then it is a positive bag, and otherwise it is a negative bag.
This is illustrated in Fig. 2. Despite the success of MIL techniques on
this problem, it is unclear what limitations MIL methods may have, if
any, for solving this problem. Specifically, are their limits on the
size of the large images, and how rare the objects of interest are
within the imagery? For example, in a positive bag, is there any limit
on the magnitude of the ratio of total instances to true positive
instances? It seems plausible that MIL algorithms should fail if this
ratio grows too large.

In this work, we apply MIL to a simple binary classification problem in
a deep learning setting. We explore the conditions that MIL will fail on
a support vector machine model. Further, if MIL fail, we investigate
when and why it happens. To answer these questions, we conduct
experiments with controlled synthetic data and a non-linear support
vector machine as a proxy to a DL model. We discovered that in a
fine-tuned model, MIL can perform as well as or even better than a
supervised model. But MIL does break under certain conditions. More
interestingly, changing the complexity of the model would change the
conditions when MIL breaks.

The structure of this paper is as follows. Section 2 lists some previous
work that is related to multiple instances learning and MIL in deep
learning. Section 3 revisits on the fundamental theory of support vector
machine and how MIL based SVM (mi-SVM) works. In section 4, we employ
the methods described in Section 3, design experiment parameters, and
specify the scoring metric to compare the performance. Sections 5 and 6
demonstrate the results and offer our understanding on the behavior of
MIL models. The paper is summarized and concluded in Section 7.

![](media/image2.png){width="3.379861111111111in"
height="2.7607360017497813in"}

related work
============

**Multiple instance learning** (MIL) was first introduced in \[15\]. The
study was motivated by a drug activity problem that each molecule
represents a learning sample that has different descriptions. After
that, researchers have developed many algorithms such as learning
axis-parallel concepts \[15\], diverse density \[16\], and extended
Citation kNN \[7\]. Axis-parallel concept is to find an axis-parallel
hyper-rectangle (APR) is the feature plane that contains at least one
instance from each positive bag and excludes all instances in a negative
bag. Diverse density is to locate an optimal concept point is feature
space that is close to at least one instance from each positive bag and
far away from all instances in negative bags. An improvement of diverse
density was proposed by Zhang and Goldman \[17\] that adapted
expectation maximization to diverse density so that estimates a sets of
hidden variables that further determine the label of a bag. The k
nearest neighbor was adapted to MIL problems to label a bag based on the
minimum Hausdorff distance to a labeled bag and citation approach
combines the information from bags that count the unlabeled bag as the
nearest neighbor. **Support vector machine for MIL** \[18\] was an
modification to SVM that iteratively improves the hyperplane that is
similar to APR, mi-SVM for instance-level classification and MI-SVM for
bag-level classification. A detailed description of mi-SVM will be
included in the next section. Studies that explore MIL problems using
traditional neural network were also inspired \[8\]\[19\].

**Deep learning** algorithms have received increasing interest for their
superior performance on fully supervised learning tasks, and many
researchers address a weekly supervised deep learning problem using MIL
approaches. \[20\] used CNN features to localize an object using weekly
supervised dataset. \[11\] constructs a deep learning framework in a
multiple-instance-learning setting of image classification and
annotation. \[12\]proposes an algorithm with a minimum of manual
annotations and good feature representations to perform classification
and segmentation in medical images and MIL serves as a boost algorithm
to a linear SVM classifier.

All of these studies indicate that in a weakly supervised learning
problem, MIL can achieve improved performance. However, there are few
studies focusing on the limitations of applying MIL on deep learning
problems. Previous studies applied MIL algorithms to the deep learning
model as an assistant to the existing model in situations in which the
training samples are bags of instances and only the labels of bags are
known. The proportion of correct information in a bag is not considered
and the complexity of the deep learning model is not investigated in
relation to the performance.

In this paper, we conduct an investigation on the conditions of a weakly
supervised deep learning problem and the parameters of a learning model
when MIL does work and does not work.

methods
=======

The support vector machine (SVM)
--------------------------------

In fully supervised learning settings, SVM is a learning model that used
for classification and regression. It constructs a hyperplane or a set
of hyperplanes that separate each class in feature space. The optimal
hyperplane has the greatest distances to any class. Maximizing the
margin for each class is the objective for SVM. The hyperplane can have
different complexity depending on the parameters of SVM. A **linear
SVM** assembles a hyperplane that linearly separates dataset. The
example in a 2D dataset is a line separating each class. If training
dataset of the form $\left( \overset{}{x_{n}},y_{n} \right)$, where
$\overset{}{x_{n}}$ is data point and $y_{n}$ is the class the point
belongs to,$\ $is trained with a linear SVM, the hyperplane is expressed
as $\overset{}{w} \bullet \overset{}{x} - b = 0$, where $\overset{}{w}$
is vector normal to the hyperplane. Many datasets cannot be linearly
separated without high error in classification, so hyperplane can be
amended to have different shapes to better separate each class. The dot
product of linear SVM is replaced by a nonlinear kernel function:
$k(\overset{}{x_{i}},\overset{}{x_{j}})$. Two common kernels used for
classification and regression are
**polynomial**:$k\left( \overset{}{x_{i}},\overset{}{x_{j}} \right) = {(\overset{}{x_{i}} \bullet \overset{}{x_{j}})\ }^{d}$and
**Gaussian**:
$k\left( \overset{}{x_{i}},\overset{}{x_{j}} \right) = exp( - \frac{1}{2\sigma^{2}}{||\overset{}{x_{i}} - \overset{}{x_{j}}||}^{2})$.
In this study, the Gaussian SVM is chosen as the classification model
because the complexity of model can be controlled by changing $\sigma$
and the computation of such kernel function is cheap. A smaller $\sigma$
yields to a less linear SVM and a larger $\sigma$ corresponds to a more
linear SVM.

The multiple instance support vector machine (mi-SVM)
-----------------------------------------------------

In MIL criteria, training samples are **bags** of **instances**. A bag
is negative if all instances in the bag are negative. A positive bag
contains at least one positive instance. mi-SVM modifies the labels of
instances in all positive bags according to an initial hyperplane and
then modifies the hyperplane accordingly. The process is repeated until
the hyperplane converge. The classification of mi-SVM is for instance
level classification. \[18\] proposed the algorithm for mi-SVM and steps
are as following:

1.  Initialize labels of all instances as the label of the bag they
    belong to.

2.  Construct an optimal hyperplane using SVM with initialized
    instances.

3.  Relabel all instances in positive bags according to the
    classification result from step 2.

4.  Label the most positive instance to be positive in a positive bag if
    all instances in that positive bag are classified as negative.

5.  Repeat steps 2,3,4 until no change in label after each
    classification.

The final hyperplane is used to classify new instances. It has the
largest margin in each class, and at least one positive instance in each
positive bag is on the positive plane of the hyperplane. In our
experiment, each bag has the same bag size, and the proportion of
positive instances in a positive bag (denote as $\alpha$) is controlled.

Experimental design
===================

The objective of the experiment is to examine the conditions when mi-SVM
has a better performance and when mi-SVM fails in a weakly supervised
deep learning setting. The experiment aims to investigate the
performance of the mi-SVM classification model in relation to the
performance of a traditional SVM when varying kernel size ($\sigma$) of
a Gaussian SVM and percentage of correctness ($\alpha$) in a positive
bag compared to a traditional Gaussian SVM.

Designing a synthetic dataset that mimics deep learning problems
----------------------------------------------------------------

In this experiment, the dataset is synthesized as data points in 2
dimensions and in a spiral shape as illustrated in Fig. 3. The shape of
spiral datasets is complex to simulate datasets that are commonly
encountered in a deep learning problem. And the iterative process of
mi-SVM can be easily demonstrated using synthesized dataset. The
addition of noise is avoided in the synthesized dataset since datasets
in deep learning problems are not noisy. So, there is no overlap between
two classes. To generate bags for MIL datasets, instances in positive
bags are randomly sampled from positive class and negative class and
instances in negative bags are randomly sampled from negative class. In
the experiment, 150 positive and 150 negative bags are generated, and
each bag has 50 instances.

![](media/image3.png){width="3.379861111111111in"
height="4.862069116360455in"}

Training and testing of algorithms
----------------------------------

100 spiral datasets are generated. Each spiral dataset is sampled 300
bags of MIL datasets, 150 positive bags and 150 negative bags. In the
experiment, for each kernel size ($\sigma$) and percentage of
correctness in positive bags ($\alpha$), classification models are
trained with 100 positive bags and 100 negative bags. Then tested with
instances in the rest 100 bags. As a result, the classification result
is averaged over 100 spiral datasets.

Scoring metrics 
----------------

> To characterize the performance of a classifier, a **receiver
> operating characteristic curve** (**ROC curve**) is constructed by
> plotting the true positive rate against false positive rate at each
> decision threshold. In other words, the curve indicates the percentage
> of instances from each class that is classified correctly. The **area
> under curve** (**AUC**) is measured as the area under a ROC curve of a
> classifier. AUC represents the probability that the classifier
> correctly labels positive and negative instances. In more general, AUC
> is a measure of the accuracy of a classification model, with AUC=1
> being a perfect classifier and AUC=0.5 being random guessing. In our
> study, the performances of SVM and mi-SVM are all quantified as AUC
> values.

 {#section .ListParagraph}

using the svm and mi-svm to solve multiple instance problems
============================================================

To investigate the performance of SVM and mi-SVM on a MIL setting, a
range of kernel sizes are applied to both SVM and mi-SVM at each
percentage of correctness in a positive bag. As illustrated in Fig. 4A
more nonlinear SVM performs at a moderate level when there are fewer
correctly labeled instances in a positive bag. Then the SVM achieves
perfect performance even when there are few correctly labeled instances
in a positive bag as the SVM becomes more linear. However, performance
drops once the SVM is too linear ($\sigma > 3$). The reason for this is
that due to the complexity of the synthesized dataset, a linear SVM
cannot create a proper hyperplane that accurately separates two classes.
mi-SVM behaves in the same way, excepts that a nonlinear mi-SVM displays
a worse result when $\alpha$ is low, and the overall performance is
higher as mi-SVM is more linear compared with SVM. Fig. 5 is a figure of
AUC of mi-SVM.

![](media/image5.png){width="3.379861111111111in"
height="2.967361111111111in"}

![](media/image6.png){width="3.379861111111111in"
height="2.9415583989501313in"}

when does the mi-svm fail?
==========================

The next thing to examine is under which conditions mi-SVM starts to
perform worse than SVM. The conditions are carefully considered, and
results show the limitations of mi-SVM when learning a MIL problem.

When does a naïve multiple instance conversion of an algorithm fail? 
---------------------------------------------------------------------

> ![](media/image7.png){width="3.379861111111111in"
> height="3.103448162729659in"}
>
> mi-SVM is superior to SVM in most cases when the model is nonlinear.
> mi-SVM starts to fail when kernel size tends to make the model linear.
> This could be explained that when the hyperplane cannot accurately
> separate classes, through number of iterations, mi-SVM will make
> considerable mistakes every iteration. This leads to a portion of true
> positive instances misclassified as negative instances and pushes the
> hyperplane further into positive class hence includes more positive
> instances in the negative plane. It is clear that a linear model takes
> more iterations than a nonlinear one. So, the mi-SVM fails to work as
> good as SVM when kernel size is large. Surprisingly, mi-SVM also fails
> even if the model is nonlinear when the percentage of correctness is
> low (top left corner in Fig. 6). In the presence of few positive
> instances in a positive bag, model with small kernel size tends to
> overfit each class. The extensive overlap between two classes causes
> the confusion of the model. When the model is linear enough, even when
> there is only 1 positive instance in a positive bag, mi-SVM performs
> better than SVM.

If we pick good parameters, when does the miSVM fail? 
------------------------------------------------------

An optimal kernel size of a model is picked by considering the best
accuracy the model can achieve at a specific kernel size. Then the
performances of mi-SVM and SVM at corresponding optimal kernel size are
compared. In this case both models achieved near perfect (AUC\~0.99).
When there is less information about positive instances, $\alpha < 0.4$,
mi-SVM is superior to SVM at optimal kernel size. However, the advantage
of mi-SVM is more obvious when $\alpha$ is smaller. mi-SVM at optimal
conditions will correct the decision boundary until it correctly labels
the true positive instance as many as possible. The negative instances
in positive bags will be relabeled as negative as the hyperplane becomes
optimal. The higher difference in performances when $\alpha$ is small is
due to SVM makes more errors as the overlap between positive and
negative bags significantly at these conditions. But as proportion of
positive instances increases in positive bags, the overlap decreases and
hence at optimal conditions mi-SVM and SVM tend to perform equally well.
Therefore, mi-SVM always perform better than SVM when the model is
optimal.

![](media/image8.png){width="3.379861111111111in"
height="3.045138888888889in"}

Analysis: how and why do we need to adjust model complexity for multiple instance approaches to work?
-----------------------------------------------------------------------------------------------------

![](media/image9.png){width="3.379861111111111in"
height="3.045138888888889in"}

The optimal kernel size is relatively stable across different $\alpha$.
mi-SVM adjusts the hyperplane through iterations and then converge to a
stable hyperplane. Through iterations, mi-SVM corrects mislabeled
instances in the positive bag and reduces the overlap between classes.
Fig. 9 illustrates the process of correction. Therefore, a mi-SVM is
always optimal because with the same kernel size the hyperplane will
converge to the optimal hyperplane. SVM, however, has a high variant
optimal kernel size. This is due to the fact SVM only takes one
iteration to perform and with the overlap between two classes decreases
a more linear SVM can separate two classes. Nevertheless, optimal SVM
tends to be more nonlinear when $\alpha$ exceeds certain threshold
(&gt;0.3). This indicates that the model produces a more nonlinear
boundary to fit the positive class better as positive class becomes more
complicated. When $\alpha$ is very small, SVM tends to overfit data
because the model is too nonlinear, and mi-SVM can perform better than
SVM using a more linear model.

![](media/image10.png){width="3.379861111111111in"
height="5.8045975503062115in"}

conclusions
===========

In this work, we investigated conditions under which a multiple instance
learning approach can be harmful in classifying performance. We compared
the performances of a MIL SVM model, mi-SVM, with a supervised SVM model
when classifying synthesized spiral dataset. The complexity of the model
is taken into account when examining its relationship to when MIL fails.
mi-SVM works equally well as SVM when the complexity of the model is
nonlinear enough and works better when there are few positive instances
in a positive bag. However, when the model is too linear or too
nonlinear mi-SVM starts to fail and is more obvious under the same
condition. The different behaviors at low percentage of positive
instances is because mi-SVM takes more iterations to converge and
mistakes are made every iteration. With an optimally tuned kernel size,
mi-SVM demonstrates advantage over SVM over all conditions and the it is
more obvious when there are few correctly labeled instances in a
positive bag. The optimal performance of mi-SVM is more consistent to
the complexity of the model as an optimal mi-SVM converges to a similar
optimal hyperplane to the same dataset.

references {#references .ListParagraph}
==========

\[1\] X. Liu, J. Wang, M. Yin, B. Edwards, and P. Xu, “Supervised
learning of sparse context reconstruction coefficients for data
representation and classification,” *Neural Comput. Appl.*, vol. 28, no.
1, pp. 135–143, 2017.

\[2\] F. Shahi and A. T. Rezakhani, “Binary classification of quantum
states: Supervised and unsupervised learning,” pp. 1–4, 2017.

\[3\] F. Garrido, W. Verbeke, and C. Bravo, “A Robust profit measure for
binary classification model evaluation,” *Expert Syst. Appl.*, vol. 92,
pp. 154–160, 2018.

\[4\] M. Jefferson, N. Pendleton, S. Lucas, M. Horan, and L. Tarassenko,
“Neural networks,” *Lancet*, vol. 346, no. 8991–8992, p. 1712, 1995.

\[5\] J. A. K. Suykens and J. Vandewalle, “Least Squares Support Vector
Machine Classifiers,” *Neural Process. Lett.*, vol. 9, no. 3, pp.
293–300, 1999.

\[6\] L. Breiman, “Random forests,” *Mach. Learn.*, vol. 45, no. 1, pp.
5–32, 2001.

\[7\] J. Wang and J.-D. Zucker, “Solving Multiple-Instance Problem: A
Lazy Learning Approach,” *Proc. 17th Int. Conf. Mach. Learn.*, no. 1994,
pp. 1119--1125, 2000.

\[8\] J. Ramon and L. De Raedt, “Multi instance neural networks,”
*ICML-2000 Work. Attrib. Relational Learn.*, pp. 53–60, 2000.

\[9\] K. Sikka, A. Dhall, and M. Bartlett, “Weakly supervised pain
localization using multiple instance learning,” *Autom. Face Gesture
Recognit. (FG), 2013 10th IEEE Int. Conf. Work.*, pp. 1–8, 2013.

\[10\] B. Babenko, M.-H. Yang, and S. Belongie, “Visual Tracking with
Online Multiple Instance Learning.,” *Cvpr*, pp. 983–990, 2009.

\[11\] J. Wu, Yinan Yu, Chang Huang, and Kai Yu, “Deep multiple instance
learning for image classification and auto-annotation,” *2015 IEEE Conf.
Comput. Vis. Pattern Recognit.*, pp. 3460–3469, 2015.

\[12\] Y. Xu, T. Mo, Q. Feng, P. Zhong, M. Lai, and E. I.-C. Chang,
“Deep learning of feature representation with multiple instance learning
for medical image analysis,” *IEEE Int. Conf. Acoust. Speech Signal
Process. ICASSP Italy*, no. 1, pp. 1626–1630, 2014.

\[13\] X. Wang, Y. Yan, P. Tang, X. Bai, and W. Liu, “Revisiting
multiple instance neural networks,” *Pattern Recognit.*, vol. 74, pp.
15–24, 2018.

\[14\] S. Tulsiani, A. A. Efros, and J. Malik, “Multi-view Consistency
as Supervisory Signal for Learning Shape and Pose Prediction,” 2018.

\[15\] T. G. Dietterich, R. H. Lathrop, and T. Lozano-Pérez, “Solving
the multiple instance problem with axis-parallel rectangles,” *Artif.
Intell.*, vol. 89, no. 1–2, pp. 31–71, 1997.

\[16\] O. Maron and T. Lozano-Pérez, “A Framework for Multiple-Instance
Learning,” *NIPS ’97 Proc. 1997 Conf. Adv. neural Inf. Process. Syst.
10*, pp. 570–576, 1997.

\[17\] Q. Zhang and S. a. Goldman, “EM-DD: An Improved Multiple-Instance
Learning Technique,” *Nips*, vol. 14, pp. 1073–1080, 2002.

\[18\] S. Andrews, I. Tsochantaridis, and T. Hofmann, “Support Vector
Machines for Multi ple-Instance Learning,” *Adv. Neural Inf. Process.
Syst. 15*, pp. 561--568, 2003.

\[19\] Z.-H. Zhou and M.-L. Zhang, “Neural Networks for Multi-Instance
Learning,” 2002.

\[20\] H. O. Song, Y. J. Lee, S. Jegelka, and T. Darrell,
“Weakly-supervised Discovery of Visual Pattern Configurations,” pp.
1–10, 2014.

\[21\] F. ul A. Afsar Minhas, E. D. Ross, and A. Ben-Hur, “Amino acid
composition predicts prion activity,” *PLoS Comput. Biol.*, vol. 13, no.
4, 2017.

 {#section-1 .ListParagraph}

