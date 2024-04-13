r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
1. According to the given values:
 * `in_features=1024` (input features)
 * `out_features=2048` (output features)
 * `N=128` (batch of N samples) 

We defined a Fully Connected Linear Layer based on these values. \
Therefore, the total Jacobian size for these parameters will be `[1024, 2048, 128]` with an overall number of `1024 * 2048 * 128 (=268435456)`.

2. When using a single-precision floating-point (32 bits) to represent our tensors, we'd get that the required number of bits to store the above Jacobian will be `268435456 * 32 bits (=8589934592)`. \
In terms of Gigabytes, we'd get:
`8589934592/8 = 1073741824 bytes`, which is approximately `1 GB`.
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.1, 0.05, 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0.1, 0.05, 0.005, 0.0001, 0.001
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # ====== YOUR CODE: ======
    wstd = 1e-1
    lr = 1e-3
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
1. Yes. At the end, some dropout may improve generalization for the test set and to get less
overfit on train set. from the plots we can see that this is the case. Moreover, we got a clear overfitting
between on train set.

2. Low dropout allowed us to both improve generalization for the test set and reduce train-set overfitting.
When applying high dropout, we lose crucial information that would allow the network
to perform better. Moreover, applying high dropout causes the network to not learn 
as much as we wanted it to learn (dropped to many neurons) and it shows.
"""

part2_q2 = r"""
Yes, it might be possible. In categorical cross-entropy case, accuracy measures true positive i.e accuracy is discrete values, while the log loss of softmax loss so to speak is a continuous variable that measures the models' performance against false negatives. A wrong prediction affects accuracy slightly but penalizes the loss disproportionately. Assuming you have a balanced dataset.

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
1. In general, for a conv layer, num params is: $ K \times (C_{in}\times F^2 + 1)$. So for a regular 2 3x3 convs, num params is:\n
<ul><li> $layer_1 = 64\times (256 \times 3^2 + 1)  = 147520</li> <li> $layer_2 = 64 \times (64*9 + 1) =36928</li></ul>
So a 2 3x3 conv layers will have: $147520 + 36928 = 184448$
on the other hand a bottleneck block will have:
<ul> <li>$layer_1 => 64 \times (256 \times 1^2 + 1) = 16448$</li>
<li>$layer_2 => 64 \times (64 \times 3^2 + 1) = 36928$</li>
<li>$layer_3 = 256 \times (64 \times 1^2 + 1) = 16640$</li>
</ul>
$16448 + 36928 + 16640 = 70016$ total params
So the regular conv has more parameters to learn.
2. number of floating point operations is dependant on the elementwise operations, namely: $2WH\times(C_{in} \times F^2 + 1) \times C_{out}$ (W,H are input height and width)
note that $(C_{in} \times F^2 + 1) \times C_{out}$ is the number of params for each layer so we just need to multiply it by 2WH
The output size of the first layer will be: $(H+2*0-1(3-1)-1)/1+1=H-3$ (same for W) so for the regular block we get: $2HW(147520) + 2(H-3)(W-3)*36928$
For the bottleneck the output sizes will be  (W,H), (H-3, W-3), (H, W) and therefore we get that the amount of FLOP is:
$2HW16448+2(H-3)(W-3)36928 + 2WH16640$
3. If we compare the ability to combine the input - the regular block has a better ability to do so. This, is because a regular layer has
a bigger filter size than the other one. therefore, It can actually help us understand each feature map individually.
In the case of understanding across feature maps - the bottleneck artchitecture actually combines all the feature maps into a projected space
and therefore gets a better understanding on these spatial projections. 

"""

part3_q2 = r"""
1. When we had deeper networks, the accuracy went down severely as we were not training the model at all.
The best result was obtained when L=2 and even L=4 (results were close), using this model. The reason might be that the gradient is vanishing when we have too many convolutional layers.

2. As we mentioned, the higher values of L's dropped the accuracy severely and in practice those networks were not trainable (L=8, L=16). The reason is probably the vanishing gradient in deep convolutional networks, which can be solved using Residual Block networks (like the one we implemented).
We can also adjust the learning rate to delay a bit the process of vanishing gradient, which might hold for bigger than 4 L's but will fail again if we take a too big L value (means it is a specific solution, not a general case solution).
"""

part3_q3 = r"""
Comparing the results to 1.1, we see that our conclusions from 1.1 are consistent and higher L values (L=8) produce very poor accuracy, which indicates the network is not training at all, no matter the kernel size used. Moreover, when using 
L=8 we see that the training process isn't converging and is stopped after 10 epochs thanks to early_stopping.
However, we see the higher kernel values using low L values (L=2, L=4) in particular produce better accuracies compared to the smaller kernal values used in 1.1 .
using Large kernel sizes at the beginning allows the network to understand lower resolutions and non-fine structures - and this may bet
the reason for the better performance. 
"""

part3_q4 = r"""
Same as 1.2 we see that increasing kernel but making more complex and deepere network size actually caused the network not to be trainable or to have 
lower accuracy and a higher loss.
This time, L=3 and L=4 which gave good results in previous experiments, give poor results. The explantion for this must be the same as stated on the previous question - higher kernel sizes at the beginning help identify low resolution structures
and therefore in general classification. Moreover, vanishing gradient is a major problem in the current architecture as we go deep.
 

"""

part3_q5 = r"""
In both experiments we ran in 1.4, we see that ResNet solves the issues we presented in the models above. The reason for that is the implementation of the residual blocks which let the network skip the calculations at points, therefore avoiding problems such as the vanishing gradient we saw in 1.1 (allowing higher L values with no drops in the accuracy of the model) or the over parameterization using higher kernels produced in 1.3 (allowing bigger kernel sizes with no drops in the accuracy of the model).

"""

part3_q6 = r"""
**Your answer:**
1. After understanding the ResNet actually assists in preventing vanishing gradient problem and recognize patterns in the picture,
We wanted to add the bottleneck to the end of the resnet in order to help the network to get better learning of the across-feature maps combinations.
We added to the resnet a bottleneck at the end. Moreover, we applied dropout to prevent overfitting and batch normalization as we learned. Another addition
was adding a res-net after every pooling in order not to loose parameters the we get as a network output.
2. We see that all models finished their training. Moroever, using this architecture the use of $L=6$ was the best of all models in terms of loss and 
accuracy both on test and train sets. We see the even when increasing L the network was able to generalize from the trainset to the test set.
Generally - except for L=3 all other L values were able to cause the network to generalize and perform well on the test set.
"""
# ==============
