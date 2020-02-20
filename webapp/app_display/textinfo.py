# -*- coding: utf-8 -*-

intro_text_0 = """
I went through a tutorial on programming neural networks in PyTorch, but I did
not know how to pick from the various options available when designing a
network from scratch afterwards. I wanted to see how different design choices
mattered, so I generated 1001 networks and ran them on the MNIST dataset.

Every network takes input of the same size (a vector of 784 values), produces
output of the same size (a vector of 10 values), and passes it through a
LogSoftmax layer to get prediction scores. The loss function (Negative Log
Likelihood loss) and optimizer (Adam) are the same throughout, with the
learning rate of the optimizer being the only change. I tested these networks
over three different runs:

* run t1 was for 4 epochs, learning rate  = 0.002, tested every epoch
* run t2 was for 8 epochs, learning rate  = 0.001, tested every 2 epochs
* run t3 was for 16 epochs, learning rate  = 0.0005, tested every 4 epochs
"""

intro_text_1 = """
The generated networks fall into nine different classes:

* Basic - the simplest network. A single hidden Linear layer with bias.
* Linear - a sequence of Linear layers
* Conv1d - a sequence of Conv1d layers
* Conv2d - a sequence of Conv2d layers
* Conv3d - a sequence of Conv3d layers
* Conv1dThenLinear - a Conv1d sequence followed by a Linear sequence
* Conv2dThenLinear - a Conv2d sequence followed by a Linear sequence
* Conv3dThenLinear - a Conv3d sequence followed by a Linear sequence
* ResNetStyle - contains a sequence of ResNet BasicBlocks

The networks vary in the activation layers used (ReLU, SeLU, Tanh, Sigmoid, or no activation).
Some layers in the networks have bias, others don't.
"""

ov_text_0 = """

I measured the network performance in terms of raw accuracy and AUC, per digit
and overall.  I also measured the training time, number of parameters,
(approximate) memory usage per pass, and number of operations for each network.

The below graph shows the performance of the generated networks. You can:

* Vary the X-Axis, Y-Axis metrics via the dropdowns, and their range via the sliders just below
* Vary the grouping of the data points
* Select the run/epoch(s) (selecting none is same as selecting all)
* Select the classe(s) of networks (selecting none is same as selecting all)
* Click on a point to go to a separate page and view its details

"""

top10_text_0 = """
The below table contains the top ten network snapshots for the selected
metrics.  It changes along with the above settings, and is sorted with respect
to the Y-Axis in descending order, with ties resolved by the X-Axis. Green
indicates best, bold indicates highest, italics indicates least.

Select a row to go to a separate page and view the network details.
"""

loss_text_0 = """
The graph on the left shows the loss function's value over the training and
test sets, and the graphs on the right show the accuracy/AUC for each digit.

Click on any point showing the loss over the test set to see the comparison
of metrics between epochs.
"""

heatmap_text_0 = """
The network produces an output of ten values in the range [-\u221e, 0) for
each input.  The prediction scores are computed as the anti-log (base *e*) of
these outputs. The prediction of a network is the class with the *best* (i.e.
the maximum) prediction score .

The accuracy metrics are computed using only the network's *best* prediction
score.  These metrics alone might not be sufficient to describe the network's
capability: a confusion matrix has been created from the network's predictions
on the left.  The pie chart on the right shows the distribution of the
prediction scores of the network for a given truth/prediction class.  The
scores are averaged over all elements in the class.

Click on any square in the confusion matrix to see how the distribution of
prediction scores varies.

"""


prediction_text_0 = """
One can look at the gradients produced when a particular prediction is made on
a given input, to get some idea of how the network responds to a particular
class of images exemplified by that input.

Select the class of images using the dropdowns, or by clicking on the confusion
matrix. Click on the button to compute gradients.
"""

prediction_text_1 = """
The images are in grayscale, with higher values in darker shades.
The gradients are in a red-blue colorscale, with higher values in red shades,
lower values in blue shades and values close to zero are transparent.

Click on a prediction on the left to see the gradients appear on the image
in the right.
"""

prediction_text_2 = """
The gradients somewhat answer the question of: "Where/how should this input
change in order for the selected prediction to occur?"

For example, if the input image is a **0**, and has been correctly predicted as
a **0**, the gradients would (should) all be negligible, because the input
image does not need to change much anywhere in order for the prediction of
**0** to occur.
"""

################################
# Text Requiring Substitutions #
################################

ranking_text_0 = """
Finally, we have the structure of the network and its rankings across the metrics.

Some metrics are structural, the same across all runs of a network (i.e. number of
parameters), others depend on the run/epoch. There are four types of rankings:

* Global Rank 	- across all data points
* Group Rank  	- across data points of the same group (in this case {groupname})
* Form Rank   	- across data points of the network form (in this case {formname})
* Run Rank 	- across data points in run {run}, epoch {epoch}

The bracketed value in the headers shows the maximum rank.
"""
