# %% [markdown] tags=[]
# # Exercise 8: Knowledge Extraction from a Pre-trained Neural Network
#
# The goal of this exercise is to learn how to probe what a pre-trained classifier has learned about the data it was trained on.
#
# We will be working with a simple example which is a fun derivation on the MNIST dataset that you will have seen in previous exercises in this course.
# Unlike regular MNIST, our dataset is classified not by number, but by color!
#
# We will:
# 1. Load a pre-trained classifier and try applying conventional attribution methods
# 2. Train a GAN to create counterfactual images - translating images from one class to another
# 3. Evaluate the GAN - see how good it is at fooling the classifier
# 4. Create attributions from the counterfactual, and learn the differences between the classes.
#
# If time permits, we will try to apply this all over again as a bonus exercise to a much more complex and more biologically relevant problem.
# ### Acknowledgments
#
# This notebook was written by Diane Adjavon, from a previous version written by Jan Funke and modified by Tri Nguyen, using code from Nils Eckstein.
#
# %% [markdown]
# <div class="alert alert-danger">
# Set your python kernel to <code>08_knowledge_extraction</code>
# </div>
# %% [markdown]
#
# # Part 1: Setup
#
# In this part of the notebook, we will load the same dataset as in the previous exercise.
# We will also learn to load one of our trained classifiers from a checkpoint.

# %%
# loading the data
from classifier.data import ColoredMNIST

mnist = ColoredMNIST("extras/data", download=True)
# %% [markdown]
# Some information about the dataset:
# - The dataset is a colored version of the MNIST dataset.
# - Instead of using the digits as classes, we use the colors.
# - There are four classes - the goal of the exercise is to find out what these are.
#
# Let's plot some examples
# %%
import matplotlib.pyplot as plt

# Show some examples
fig, axs = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axs.flatten()):
    x, y = mnist[i]
    x = x.permute((1, 2, 0))  # make channels last
    ax.imshow(x)
    ax.set_title(f"Class {y}")
    ax.axis("off")

# %% [markdown]
# We have pre-traiend a classifier for you on this dataset. It is the same architecture classifier as you used in the Failure Modes exercise: a `DenseModel`.
# Let's load that classifier now!
# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task 1.1: Load the classifier</h3>
# We have written a slightly more general version of the `DenseModel` that you used in the previous exercise. Ours requires two inputs:
# - `input_shape`: the shape of the input images, as a tuple
# - `num_classes`: the number of classes in the dataset
#
# Create a dense model with the right inputs and load the weights from the checkpoint.
# <div>
# %% tags=["task"]
import torch
from classifier.model import DenseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO Load the model with the correct input shape
model = DenseModel(input_shape=(...), num_classes=4)

# TODO modify this with the location of your classifier checkpoint
checkpoint = torch.load(...)
model.load_state_dict(checkpoint)
model = model.to(device)
# %% tags=["solution"]
import torch
from classifier.model import DenseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the model
model = DenseModel(input_shape=(3, 28, 28), num_classes=4)
# Load the checkpoint
checkpoint = torch.load("extras/checkpoints/model.pth")
model.load_state_dict(checkpoint)
model = model.to(device)

# %% [markdown]
# Don't take my word for it! Let's see how well the classifier does on the test set.
# %%
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns

test_mnist = ColoredMNIST("extras/data", download=True, train=False)
dataloader = DataLoader(test_mnist, batch_size=32, shuffle=False)

labels = []
predictions = []
for x, y in dataloader:
    pred = model(x.to(device))
    labels.extend(y.cpu().numpy())
    predictions.extend(pred.argmax(dim=1).cpu().numpy())

cm = confusion_matrix(labels, predictions, normalize="true")
sns.heatmap(cm, annot=True, fmt=".2f")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.show()

# %% [markdown]
# # Part 2: Using Integrated Gradients to find what the classifier knows
#
# In this section we will make a first attempt at highlighting differences between the "real" and "fake" images that are most important to change the decision of the classifier.
#

# %% [markdown]
# ## Attributions through integrated gradients
#
# Attribution is the process of finding out, based on the output of a neural network, which pixels in the input are (most) responsible for the output. Another way of thinking about it is: which pixels would need to change in order for the network's output to change.
#
# Here we will look at an example of an attribution method called [Integrated Gradients](https://captum.ai/docs/extension/integrated_gradients). If you have a bit of time, have a look at this [super fun exploration of attribution methods](https://distill.pub/2020/attribution-baselines/), especially the explanations on Integrated Gradients.

# %% tags=[]
batch_size = 4
batch = []
for i in range(4):
    batch.append(next(image for image in mnist if image[1] == i))
x = torch.stack([b[0] for b in batch])
y = torch.tensor([b[1] for b in batch])
x = x.to(device)
y = y.to(device)

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h3>Task 2.1 Get an attribution</h3>
#
# In this next part, we will get attributions on single batch. We use a library called [captum](https://captum.ai), and focus on the `IntegratedGradients` method.
# Create an `IntegratedGradients` object and run attribution on `x,y` obtained above.
#
# </div>

# %% tags=["task"]
from captum.attr import IntegratedGradients

############### Task 2.1 TODO ############
# Create an integrated gradients object.
integrated_gradients = ...

# Generated attributions on integrated gradients
attributions = ...

# %% tags=["solution"]
#########################
# Solution for Task 2.1 #
#########################

from captum.attr import IntegratedGradients

# Create an integrated gradients object.
integrated_gradients = IntegratedGradients(model)

# Generated attributions on integrated gradients
attributions = integrated_gradients.attribute(x, target=y)

# %% tags=[]
attributions = (
    attributions.cpu().numpy()
)  # Move the attributions from the GPU to the CPU, and turn then into numpy arrays for future processing

# %% [markdown] tags=[]
# Here is an example for an image, and its corresponding attribution.


# %% tags=[]
from captum.attr import visualization as viz
import numpy as np


def visualize_attribution(attribution, original_image):
    attribution = np.transpose(attribution, (1, 2, 0))
    original_image = np.transpose(original_image, (1, 2, 0))

    viz.visualize_image_attr_multiple(
        attribution,
        original_image,
        methods=["original_image", "heat_map"],
        signs=["all", "absolute_value"],
        show_colorbar=True,
        titles=["Image", "Attribution"],
        use_pyplot=True,
    )


# %% tags=[]
for attr, im, lbl in zip(attributions, x.cpu().numpy(), y.cpu().numpy()):
    print(f"Class {lbl}")
    visualize_attribution(attr, im)

# %% [markdown]
#
# The attributions are shown as a heatmap. The brighter the pixel, the more important this attribution method thinks that it is.
# As you can see, it is pretty good at recognizing the number within the image.
# As we know, however, it is not the digit itself that is important for the classification, it is the color!
# Although the method is picking up really well on the region of interest, it would be difficult to conclude from this that it is the color that matters.


# %% [markdown]
# Something is slightly unfair about this visualization though.
# We are visualizing as if it were grayscale, but both our images and our attributions are in color!
# Can we learn more from the attributions if we visualize them in color?
# %%
def visualize_color_attribution(attribution, original_image):
    attribution = np.transpose(attribution, (1, 2, 0))
    original_image = np.transpose(original_image, (1, 2, 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(original_image)
    ax1.set_title("Image")
    ax1.axis("off")
    ax2.imshow(np.abs(attribution))
    ax2.set_title("Attribution")
    ax2.axis("off")
    plt.show()


for attr, im, lbl in zip(attributions, x.cpu().numpy(), y.cpu().numpy()):
    print(f"Class {lbl}")
    visualize_color_attribution(attr, im)

# %% [markdown]
# We get some better clues when looking at the attributions in color.
# The highlighting doesn't just happen in the region with number, but also seems to hapen in a channel that matches the color of the image.
# Just based on this, however, we don't get much more information than we got from the images themselves.
#
# If we didn't know in advance, it is unclear whether the color or the number is the most important feature for the classifier.
# %% [markdown]
#
# ### Changing the baseline
#
# Many existing attribution algorithms are comparative: they show which pixels of the input are responsible for a network output *compared to a baseline*.
# The baseline is often set to an all 0 tensor, but the choice of the baseline affects the output.
# (For an interactive illustration of how the baseline affects the output, see [this Distill paper](https://distill.pub/2020/attribution-baselines/))
#
# You can change the baseline used by the `integrated_gradients` object.
#
# Use the command:
# ```
# ?integrated_gradients.attribute
# ```
# To get more details about how to include the baseline.
#
# Try using the code below to change the baseline and see how this affects the output.
#
# 1. Random noise as a baseline
# 2. A blurred/noisy version of the original image as a baseline.

# %% [markdown]
# <div class="alert alert-block alert-info"><h4>Task 2.3: Use random noise as a baseline</h4>
#
# Hint: `torch.rand_like`
# </div>

# %% tags=["task"]
# Baseline
random_baselines = ...  # TODO Change
# Generate the attributions
attributions_random = integrated_gradients.attribute(...)  # TODO Change

# Plotting
for attr, im, lbl in zip(attributions, x.cpu().numpy(), y.cpu().numpy()):
    print(f"Class {lbl}")
    visualize_attribution(attr, im)

# %% tags=["solution"]
#########################
# Solution for task 2.3 #
#########################
# Baseline
random_baselines = torch.rand_like(x)
# Generate the attributions
attributions_random = integrated_gradients.attribute(
    x, target=y, baselines=random_baselines
)

# Plotting
for attr, im, lbl in zip(attributions, x.cpu().numpy(), y.cpu().numpy()):
    print(f"Class {lbl}")
    visualize_color_attribution(attr, im)

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h4>Task 2.4: Use a blurred image a baseline</h4>
#
# Hint: `torchvision.transforms.functional` has a useful function for this ;)
# </div>

# %% tags=["task"]
# TODO Import required function

# Baseline
blurred_baselines = ...  # TODO Create blurred version of the images
# Generate the attributions
attributions_blurred = integrated_gradients.attribute(...)  # TODO Fill

# Plotting
for attr, im, lbl in zip(attributions, x.cpu().numpy(), y.cpu().numpy()):
    print(f"Class {lbl}")
    visualize_color_attribution(attr, im)

# %% tags=["solution"]
#########################
# Solution for task 2.4 #
#########################
from torchvision.transforms.functional import gaussian_blur

# Baseline
blurred_baselines = gaussian_blur(x, kernel_size=(5, 5))
# Generate the attributions
attributions_blurred = integrated_gradients.attribute(
    x, target=y, baselines=blurred_baselines
)

# Plotting
for attr, im, lbl in zip(attributions, x.cpu().numpy(), y.cpu().numpy()):
    print(f"Class {lbl}")
    visualize_color_attribution(attr, im)

# %% [markdown] tags=[]
# <div class="altert alert-block alert-warning"><h4> Questions </h4>
# <ul>
# <li>What baseline do you like best so far? Why?</li>
# <li>Why do you think some baselines work better than others?</li>
# <li>If you were to design an ideal baseline, what would you choose?</li>
# </ul>
# </div>

# %% [markdown]
# <div class="alert alert-block alert-info"><h2>BONUS Task: Using different attributions.</h2>
#
#
# [`captum`](https://captum.ai/tutorials/Resnet_TorchVision_Interpret) has access to various different attribution algorithms.
#
# Replace `IntegratedGradients` with different attribution methods. Are they consistent with each other?
# </div>

# %% [markdown]
# <div class="alert alert-block alert-success"><h2>Checkpoint 2</h2>
# Let us know on the exercise chat when you've reached this point!
#
# At this point we have:
#
# - Loaded a classifier that classifies MNIST-like images by color, but we don't know how!
# - Tried applying Integrated Gradients to find out what the classifier is looking at - with little success.
# - Discovered the effect of changing the baseline on the output of integrated gradients.
#
# Coming up in the next section, we will learn how to create counterfactual images.
# These images will change *only what is necessary* in order to change the classification of the image.
# We'll see that using counterfactuals we will be able to disambiguate between color and number as an important feature.
# </div>

# %% [markdown]
# # Part 3: Train a GAN to Translate Images
#
# To gain insight into how the trained network classifies images, we will use [Discriminative Attribution from Counterfactuals](https://arxiv.org/abs/2109.13412), a feature attribution with counterfactual explanations methodology.
# This method employs a StarGAN to translate images from one class to another to make counterfactual explanations.
#
# **What is a counterfactual?**
#
# You've learned about adversarial examples in the lecture on failure modes. These are the imperceptible or noisy changes to an image that drastically changes a classifier's opinion.
# Counterfactual explanations are the useful cousins of adversarial examples. They are *perceptible* and *informative* changes to an image that changes a classifier's opinion.
#
# In the image below you can see the difference between the two. In the first column are MNIST images along with their classifictaions, and in the second column are counterfactual explanations to *change* that class. You can see that in both cases a human being would (hopefully) agree with the new classification. By comparing the two columns, we can therefore begin to define what makes each digit special.
#
# In contrast, the third and fourth columns show an MNIST image and a corresponding adversarial example. Here the network returns a prediction that most human beings (who aren't being facetious) would strongly disagree with.
#
# <img src="assets/ce_vs_ae.png" width=50% />
#
# **Counterfactual synapses**
#
# In this example, we will train a StarGAN network that is able to take any of our special MNIST images and change its class.
# %% [markdown] tags=[]
# ### The model
# ![stargan.png](assets/stargan.png)
#
# In the following, we create a [StarGAN model](https://arxiv.org/abs/1711.09020).
# It is a Generative Adversarial model that is trained to turn one class of images X into a different class of images Y.
#
# We will not be using the random latent code (green, in the figure), so the model we use is made up of three networks:
# - The generator - this will be the bulk of the model, and will be responsible for transforming the images: we're going to use a `UNet`
# - The discriminator - this will be responsible for telling the difference between real and fake images: we're going to use a `DenseModel`
# - The style encoder - this will be responsible for encoding the style of the image: we're going to use a `DenseModel`
#
# Let's start by creating these!
# %%
from dlmbl_unet import UNet
from torch import nn


class Generator(nn.Module):

    def __init__(self, generator, style_encoder):
        super().__init__()
        self.generator = generator
        self.style_encoder = style_encoder

    def forward(self, x, y):
        """
        x: torch.Tensor
            The source image
        y: torch.Tensor
            The style image
        """
        style = self.style_encoder(y)
        # Concatenate the style vector with the input image
        style = style.unsqueeze(-1).unsqueeze(-1)
        style = style.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, style], dim=1)
        return self.generator(x)


# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task 3.1: Create the models</h3>
#
# We are going to create the models for the generator, discriminator, and style mapping.
#
# Given the Generator structure above, fill in the missing parts for the unet and the style mapping.
# %% tags=["task"]
style_size = ...  # TODO choose a size for the style space
unet_depth = ...  # TODO Choose a depth for the UNet
style_encoder = DenseModel(
    input_shape=..., num_classes=...  # How big is the style space?
)
unet = UNet(depth=..., in_channels=..., out_channels=..., final_activation=nn.Sigmoid())

generator = Generator(unet, style_encoder=style_encoder)
# %% tags=["solution"]
# Here is an example of a working setup! Note that you can change the hyperparameters as you experiment.
# Choose your own setup to see what works for you.
style_encoder = DenseModel(input_shape=(3, 28, 28), num_classes=3)
unet = UNet(depth=2, in_channels=6, out_channels=3, final_activation=nn.Sigmoid())
generator = Generator(unet, style_encoder=style_encoder)

# %% [markdown] tags=[]
# <div class="alert alert-block alert-warning"><h3>Hyper-parameter choices</h3>
# <ul>
# <li>Are any of the hyperparameters you choose above constrained in some way?</li>
# <li>What would happen if you chose a depth of 10 for the UNet?</li>
# <li>Is there a minimum size for the style space? Why or why not?</li>
# </ul>

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h3>Task 3.2: Create the discriminator</h3>
#
# We want the discriminator to be like a classifier, so it is able to look at an image and tell not only whether it is real, but also which class it came from.
# The discriminator will take as input either a real image or a fake image.
# Fill in the following code to create a discriminator that can classify the images into the correct number of classes.
# </div>
# %% tags=["task"]
discriminator = DenseModel(input_shape=..., num_classes=...)
# %% tags=["solution"]
discriminator = DenseModel(input_shape=(3, 28, 28), num_classes=4)
# %% [markdown]
# Let's move all models onto the GPU
# %%
generator = generator.to(device)
discriminator = discriminator.to(device)

# %% [markdown] tags=[]
# ## Training a GAN
#
# Training an adversarial network is a bit more complicated than training a classifier.
# For starters, we are simultaneously training two different networks that work against each other.
# As such, we need to be careful about how and when we update the weights of each network.
#
# We will have two different optimizers, one for the Generator and one for the Discriminator.
#
# %%
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-5)
optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-4)
# %% [markdown] tags=[]
#
# There are also two different types of losses that we will need.
# **Adversarial loss**
# This loss describes how well the discriminator can tell the difference between real and generated images.
# In our case, this will be a sort of classification loss - we will use Cross Entropy.
# <div class="alert alert-block alert-warning">
# The adversarial loss will be applied differently to the generator and the discriminator! Be very careful!
# </div>
# %%
adversarial_loss_fn = nn.CrossEntropyLoss()

# %% [markdown] tags=[]
#
# **Cycle/reconstruction loss**
# The cycle loss is there to make sure that the generator doesn't output an image that looks nothing like the input!
# Indeed, by training the generator to be able to cycle back to the original image, we are making sure that it makes a minimum number of changes.
# The cycle loss is applied only to the generator.
#
# %%
cycle_loss_fn = nn.L1Loss()

# %% [markdown] tags=[]
# To load the data as batches, with shuffling and other useful features, we will use a `DataLoader`.
# %%
from torch.utils.data import DataLoader

dataloader = DataLoader(
    mnist, batch_size=32, drop_last=True, shuffle=True
)  # We will use the same dataset as before


# %% [markdown] tags=[]
# As we stated earlier, it is important to make sure when each network is being trained when working with a GAN.
# Indeed, if we update the weights at the same time, we may lose the adversarial aspect of the training altogether, with information leaking into the generator or discriminator causing them to collaborate when they should be competing!
# `set_requires_grad` is a function that allows us to determine when the weights of a network are trainable (if it is `True`) or not (if it is `False`).
# %%
def set_requires_grad(module, value=True):
    """Sets `requires_grad` on a `module`'s parameters to `value`"""
    for param in module.parameters():
        param.requires_grad = value


# %% [markdown] tags=[]
# Another consequence of adversarial training is that it is very unstable.
# While this instability is what leads to finding the best possible solution (which in the case of GANs is on a saddle point), it can also make it difficult to train the model.
# To force some stability back into the training, we will use Exponential Moving Averages (EMA).
#
# In essence, each time we update the generator's weights, we will also update the EMA model's weights as an average of all the generator's previous weights as well as the current update.
# A certain weight is given to the previous weights, which is what ensures that the EMA update remains rather smooth over the training period.
# Each epoch, we will then copy the EMA model's weights back to the generator.
# This is a common technique used in GAN training to stabilize the training process.
# Pay attention to what this does to the loss during the training process!
# %%
from copy import deepcopy


def exponential_moving_average(model, ema_model, beta=0.999):
    """Update the EMA model's parameters with an exponential moving average"""
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(beta).add_((1 - beta) * param.data)


def copy_parameters(source_model, target_model):
    """Copy the parameters of a model to another model"""
    for param, target_param in zip(
        source_model.parameters(), target_model.parameters()
    ):
        target_param.data.copy_(param.data)


# %%
generator_ema = Generator(deepcopy(unet), style_encoder=deepcopy(style_encoder))
generator_ema = generator_ema.to(device)

# %% [markdown] tags=[]
# <div class="alert alert-banner alert-info"><h4>Task 3.3: Training!</h4>
# You were given several different options in the training code below. In each case, one of the options will work, and the other will not.
# Comment out the option that you think will not work.
# <ul>
#   <li>Choose the values for `set_requires_grad`. Hint: which part of the code is training the generator? Which part is training the discriminator</li>
#   <li>Choose the values of `set_requires_grad`, again. Hint: you may want to switch</li>
#   <li>Choose the sign of the discriminator loss. Hint: what does the discriminator want to do?</li>
# . <li>Apply the EMA update. Hint: which model do you want to update? You can look again at the code we wrote above.</li>
# </ul>
# Let's train the StarGAN one batch a time.
# While you watch the model train, consider whether you think it will be successful at generating counterfactuals in the number of steps we give it. What is the minimum number of iterations you think are needed for this to work, and how much time do yo uthink it will take?
# </div>
# %% [markdown] tags=[]
# Once you're happy with your choices, run the training loop! &#x1F682; &#x1F68B; &#x1F68B; &#x1F68B;
# %% tags=["task"]
from tqdm import tqdm  # This is a nice library for showing progress bars


losses = {"cycle": [], "adv": [], "disc": []}

for epoch in range(15):
    for x, y in tqdm(dataloader, desc=f"Epoch {epoch}"):
        x = x.to(device)
        y = y.to(device)
        # get the target y by shuffling the classes
        # get the style sources by random sampling
        random_index = torch.randperm(len(y))
        x_style = x[random_index].clone()
        y_target = y[random_index].clone()

        # TODO - Choose an option by commenting out what you don't want
        ############
        # Option 1 #
        ############
        set_requires_grad(generator, True)
        set_requires_grad(discriminator, False)
        ############
        # Option 2 #
        ############
        set_requires_grad(generator, False)
        set_requires_grad(discriminator, True)

        optimizer_g.zero_grad()
        # Get the fake image
        x_fake = generator(x, x_style)
        # Try to cycle back
        x_cycled = generator(x_fake, x)
        # Discriminate
        discriminator_x_fake = discriminator(x_fake)
        # Losses to  train the generator

        # 1. make sure the image can be reconstructed
        cycle_loss = cycle_loss_fn(x, x_cycled)
        # 2. make sure the discriminator is fooled
        adv_loss = adversarial_loss_fn(discriminator_x_fake, y_target)

        # Optimize the generator
        (cycle_loss + adv_loss).backward()
        optimizer_g.step()

        # TODO - Choose an option by commenting out what you don't want
        ############
        # Option 1 #
        ############
        set_requires_grad(generator, True)
        set_requires_grad(discriminator, False)
        ############
        # Option 2 #
        ############
        set_requires_grad(generator, False)
        set_requires_grad(discriminator, True)
        #
        optimizer_d.zero_grad()
        #
        discriminator_x = discriminator(x)
        discriminator_x_fake = discriminator(x_fake.detach())

        # TODO - Choose an option by commenting out what you don't want
        # Losses to train the discriminator
        # 1. make sure the discriminator can tell real is real
        # 2. make sure the discriminator can tell fake is fake
        ############
        # Option 1 #
        ############
        real_loss = adversarial_loss_fn(discriminator_x, y)
        fake_loss = -adversarial_loss_fn(discriminator_x_fake, y_target)
        ############
        # Option 2 #
        ############
        real_loss = adversarial_loss_fn(discriminator_x, y)
        fake_loss = adversarial_loss_fn(discriminator_x_fake, y_target)
        #
        disc_loss = (real_loss + fake_loss) * 0.5
        disc_loss.backward()
        # Optimize the discriminator
        optimizer_d.step()

        losses["cycle"].append(cycle_loss.item())
        losses["adv"].append(adv_loss.item())
        losses["disc"].append(disc_loss.item())

        # EMA update
        # TODO - perform the EMA update
        ############
        # Option 1 #
        ############
        exponential_moving_average(generator, generator_ema)
        ############
        # Option 2 #
        ############
        exponential_moving_average(generator_ema, generator)
    # Copy the EMA model's parameters to the generator
    copy_parameters(generator_ema, generator)
# %% tags=["solution"]
from tqdm import tqdm  # This is a nice library for showing progress bars


losses = {"cycle": [], "adv": [], "disc": []}
for epoch in range(15):
    for x, y in tqdm(dataloader, desc=f"Epoch {epoch}"):
        x = x.to(device)
        y = y.to(device)
        # get the target y by shuffling the classes
        # get the style sources by random sampling
        random_index = torch.randperm(len(y))
        x_style = x[random_index].clone()
        y_target = y[random_index].clone()

        set_requires_grad(generator, True)
        set_requires_grad(discriminator, False)
        optimizer_g.zero_grad()
        # Get the fake image
        x_fake = generator(x, x_style)
        # Try to cycle back
        x_cycled = generator(x_fake, x)
        # Discriminate
        discriminator_x_fake = discriminator(x_fake)
        # Losses to  train the generator

        # 1. make sure the image can be reconstructed
        cycle_loss = cycle_loss_fn(x, x_cycled)
        # 2. make sure the discriminator is fooled
        adv_loss = adversarial_loss_fn(discriminator_x_fake, y_target)

        # Optimize the generator
        (cycle_loss + adv_loss).backward()
        optimizer_g.step()

        set_requires_grad(generator, False)
        set_requires_grad(discriminator, True)
        optimizer_d.zero_grad()
        #
        discriminator_x = discriminator(x)
        discriminator_x_fake = discriminator(x_fake.detach())
        # Losses to train the discriminator
        # 1. make sure the discriminator can tell real is real
        real_loss = adversarial_loss_fn(discriminator_x, y)
        # 2. make sure the discriminator can tell fake is fake
        fake_loss = -adversarial_loss_fn(discriminator_x_fake, y_target)
        #
        disc_loss = (real_loss + fake_loss) * 0.5
        disc_loss.backward()
        # Optimize the discriminator
        optimizer_d.step()

        losses["cycle"].append(cycle_loss.item())
        losses["adv"].append(adv_loss.item())
        losses["disc"].append(disc_loss.item())
        exponential_moving_average(generator, generator_ema)
    # Copy the EMA model's parameters to the generator
    copy_parameters(generator_ema, generator)


# %% [markdown] tags=[]
# Once training is complete, we can plot the losses to see how well the model is doing.
# %%
plt.plot(losses["cycle"], label="Cycle loss")
plt.plot(losses["adv"], label="Adversarial loss")
plt.plot(losses["disc"], label="Discriminator loss")
plt.legend()
plt.show()

# %% [markdown] tags=[]
# <div class="alert alert-block alert-warning"><h3>Questions</h3>
# <ul>
# <li> Do the losses look like what you expected? </li>
# <li> How do these losses differ from the losses you would expect from a classifier? </li>
# <li> Based only on the losses, do you think the model is doing well? </li>
# </ul>

# %% [markdown] tags=[]
# We can also look at some examples of the images that the generator is creating.
# %%
idx = 0
fig, axs = plt.subplots(1, 4, figsize=(12, 4))
axs[0].imshow(x[idx].cpu().permute(1, 2, 0).detach().numpy())
axs[0].set_title("Input image")
axs[1].imshow(x_style[idx].cpu().permute(1, 2, 0).detach().numpy())
axs[1].set_title("Style image")
axs[2].imshow(x_fake[idx].cpu().permute(1, 2, 0).detach().numpy())
axs[2].set_title("Generated image")
axs[3].imshow(x_cycled[idx].cpu().permute(1, 2, 0).detach().numpy())
axs[3].set_title("Cycled image")

for ax in axs:
    ax.axis("off")
plt.show()

# %%
# %% [markdown] tags=[]
# <div class="alert alert-block alert-success"><h2>Checkpoint 3</h2>
# You've now learned the basics of what makes up a StarGAN, and details on how to perform adversarial training.
# The same method can be used to create a StarGAN with different basic elements.
# For example, you can change the archictecture of the generators, or of the discriminator to better fit your data in the future.
#
# You know the drill... let us know on the exercise chat when you have arrived here!
# </div>

# %% [markdown] tags=[]
# # Part 4: Evaluating the GAN and creating Counterfactuals

# %% [markdown] tags=[]
# ## Creating counterfactuals
#
# The first thing that we want to do is make sure that our GAN is able to create counterfactual images.
# To do this, we have to create them, and then pass them through the classifier to see if they are classified correctly.
#
# First, let's get the test dataset, so we can evaluate the GAN on unseen data.
# Then, let's get four prototypical images from the dataset as style sources.

# %% Loading the test dataset
test_mnist = ColoredMNIST("extras/data", download=True, train=False)
prototypes = {}


for i in range(4):
    options = np.where(test_mnist.conditions == i)[0]
    # Note that you can change the image index if you want to use a different prototype.
    image_index = 0
    x, y = test_mnist[options[image_index]]
    prototypes[i] = x

# %% [markdown] tags=[]
# Let's have a look at the prototypes.
# %%
fig, axs = plt.subplots(1, 4, figsize=(12, 4))
for i, ax in enumerate(axs):
    ax.imshow(prototypes[i].permute(1, 2, 0))
    ax.axis("off")
    ax.set_title(f"Prototype {i}")

# %% [markdown]
# Now we need to use these prototypes to create counterfactual images!
# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task 4: Create counterfactuals</h3>
# In the below, we will store the counterfactual images in the `counterfactuals` array.
#
# <ul>
# <li> Create a counterfactual image for each of the prototypes. </li>
# <li> Classify the counterfactual image using the classifier. </li>
# <li> Store the source and target labels; which is which?</li>
# </ul>
# %% tags=["task"]
num_images = 1000
random_test_mnist = torch.utils.data.Subset(
    test_mnist, np.random.choice(len(test_mnist), num_images, replace=False)
)
counterfactuals = np.zeros((4, num_images, 3, 28, 28))

predictions = []
source_labels = []
target_labels = []

for i, (x, y) in tqdm(enumerate(random_test_mnist), total=num_images):
    for lbl in range(4):
        # TODO Create the counterfactual
        x_fake = generator(x.unsqueeze(0).to(device), ...)
        # TODO Predict the class of the counterfactual image
        pred = model(...)

        # TODO Store the source and target labels
        source_labels.append(...)  # The original label of the image
        target_labels.append(...)  # The desired label of the counterfactual image
        # Store the counterfactual image and prediction
        counterfactuals[lbl][i] = x_fake.cpu().detach().numpy()
        predictions.append(pred.argmax().item())
# %% tags=["solution"]
num_images = 1000
random_test_mnist = torch.utils.data.Subset(
    test_mnist, np.random.choice(len(test_mnist), num_images, replace=False)
)
counterfactuals = np.zeros((4, num_images, 3, 28, 28))

predictions = []
source_labels = []
target_labels = []

for i, (x, y) in tqdm(enumerate(random_test_mnist), total=num_images):
    for lbl in range(4):
        # Create the counterfactual
        x_fake = generator(
            x.unsqueeze(0).to(device), prototypes[lbl].unsqueeze(0).to(device)
        )
        # Predict the class of the counterfactual image
        pred = model(x_fake)

        # Store the source and target labels
        source_labels.append(y)  # The original label of the image
        target_labels.append(lbl)  # The desired label of the counterfactual image
        # Store the counterfactual image and prediction
        counterfactuals[lbl][i] = x_fake.cpu().detach().numpy()
        predictions.append(pred.argmax().item())

# %% [markdown] tags=[]
# Let's plot the confusion matrix for the counterfactual images.
# %%
cf_cm = confusion_matrix(target_labels, predictions, normalize="true")
sns.heatmap(cf_cm, annot=True, fmt=".2f")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.show()

# %% [markdown] tags=[]
# <div class="alert alert-block alert-warning"><h3>Questions</h3>
# <ul>
# <li> How well is our GAN doing at creating counterfactual images? </li>
# <li> Does your choice of prototypes matter? Why or why not? </li>
# </ul>
# </div>

# %% [markdown] tags=[]
# Let's also plot some examples of the counterfactual images.

# %%
for i in np.random.choice(range(num_images), 4):
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    for j, ax in enumerate(axs):
        ax.imshow(counterfactuals[j][i].transpose(1, 2, 0))
        ax.axis("off")
        ax.set_title(f"Class {j}")

# %% [markdown] tags=[]
# <div class="alert alert-block alert-warning"><h3>Questions</h3>
# <ul>
# <li>Can you easily tell which of these images is the original, and which ones are the counterfactuals?</li>
# <li>What is your hypothesis for the features that define each class?</li>
# </ul>
# </div>

# %% [markdown]
# At this point we have:
# - A classifier that can differentiate between image of different classes
# - A GAN that has correctly figured out how to change the class of an image
#
# Let's try putting the two together to see if we can figure out what exactly makes a class.
#
# %%
batch_size = 4
batch = [random_test_mnist[i] for i in range(batch_size)]
x = torch.stack([b[0] for b in batch])
y = torch.tensor([b[1] for b in batch])
x_fake = torch.tensor(counterfactuals[0, :batch_size])
x = x.to(device).float()
y = y.to(device)
x_fake = x_fake.to(device).float()

# Generated attributions on integrated gradients
attributions = integrated_gradients.attribute(x, baselines=x_fake, target=y)


# %% Another visualization function
def visualize_color_attribution_and_counterfactual(
    attribution, original_image, counterfactual_image
):
    attribution = np.transpose(attribution, (1, 2, 0))
    original_image = np.transpose(original_image, (1, 2, 0))
    counterfactual_image = np.transpose(counterfactual_image, (1, 2, 0))

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))
    ax0.imshow(original_image)
    ax0.set_title("Image")
    ax0.axis("off")
    ax1.imshow(counterfactual_image)
    ax1.set_title("Counterfactual")
    ax1.axis("off")
    ax2.imshow(np.abs(attribution))
    ax2.set_title("Attribution")
    ax2.axis("off")
    plt.show()


# %%
for idx in range(batch_size):
    print("Source class:", y[idx].item())
    print("Target class:", 0)
    visualize_color_attribution_and_counterfactual(
        attributions[idx].cpu().numpy(), x[idx].cpu().numpy(), x_fake[idx].cpu().numpy()
    )
# %% [markdown]
# <div class="alert alert-block alert-warning"><h3>Questions</h3>
# <ul>
# <li> Do the attributions explain the differences between the images and their counterfactuals? </li>
# <li> What happens when the "counterfactual" and the original image are of the same class? Why do you think this is? </li>
# <li> Do you have a more refined hypothesis for what makes each class unique? </li>
# </ul>
# </div>
# %% [markdown]
# <div class="alert alert-block alert-success"><h2>Checkpoint 4</h2>
# At this point you have:
# - Created a StarGAN that can change the class of an image
# - Evaluated the StarGAN on unseen data
# - Used the StarGAN to create counterfactual images
# - Used the counterfactual images to highlight the differences between classes
#
# %% [markdown]
# # Part 5: Exploring the Style Space, finding the answer
# By now you will have hopefully noticed that it isn't the exact color of the image that determines its class, but that two images with a very similar color can be of different classes!
#
# Here is an example of two images that are very similar in color, but are of different classes.
# ![same_color_diff_class](assets/same_color_diff_class.png)
# While both of the images are yellow, the attribution tells us (if you squint!) that one of the yellows has slightly more blue in it!
#
# Conversely, here is an example of two images with very different colors, but that are of the same class:
# ![same_class_diff_color](assets/same_class_diff_color.png)
# Here the attribution is empty! Using the discriminative attribution we can see that the significant color change doesn't matter at all!
#
#
# So color is important... but not always? What's going on!?
# There is a final piece of information that we can use to solve the puzzle: the style space.
# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task 5.1: Explore the style space</h3>
# Let's take a look at the style space.
# We will use the style encoder to encode the style of the images and then use PCA to visualize it.
# </div>

# %%
from sklearn.decomposition import PCA


styles = []
labels = []
for img, label in random_test_mnist:
    styles.append(
        style_encoder(img.unsqueeze(0).to(device)).cpu().detach().numpy().squeeze()
    )
    labels.append(label)

# PCA
pca = PCA(n_components=2)
styles_pca = pca.fit_transform(styles)

# Plot the PCA
markers = ["o", "s", "P", "^"]
plt.figure(figsize=(10, 10))
for i in range(4):
    plt.scatter(
        styles_pca[np.array(labels) == i, 0],
        styles_pca[np.array(labels) == i, 1],
        marker=markers[i],
        label=f"Class {i}",
    )
plt.legend()
plt.show()

# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task 5.1: Adding color to the style space</h3>
# We know that color is important. Does interpreting the style space as colors help us understand better?
#
# Let's use the style space to color the PCA plot.
# (Note: there is no code to write here, just run the cell and answer the questions below)
# </div>
# %%
styles = np.array(styles)
normalized_styles = (styles - np.min(styles, axis=1, keepdims=True)) / np.ptp(
    styles, axis=1, keepdims=True
)

# Plot the PCA again!
plt.figure(figsize=(10, 10))
for i in range(4):
    plt.scatter(
        styles_pca[np.array(labels) == i, 0],
        styles_pca[np.array(labels) == i, 1],
        c=normalized_styles[np.array(labels) == i],
        marker=markers[i],
        label=f"Class {i}",
    )
plt.legend()
plt.show()
# %% [markdown]
# <div class="alert alert-block alert-warning"><h3>Questions</h3>
# <ul>
# <li> Do the colors match those that you have seen in the data?</li>
# <li> Can you see any patterns in the colors? Is the space smooth, for example?</li>
# </ul>
# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task 5.2: Using the images to color the style space</h3>
# Finally, let's just use the colors from the images themselves!
# The maximum value in the image (since they are "black-and-color") can be used as a color!
#
# Let's get that color, then plot the style space again.
# (Note: once again, no coding needed here, just run the cell and think about the results with the questions below)
# </div>
# %%
colors = np.array([np.max(x.numpy(), axis=(1, 2)) for x, _ in random_test_mnist])

# Plot the PCA again!
plt.figure(figsize=(10, 10))
for i in range(4):
    plt.scatter(
        styles_pca[np.array(labels) == i, 0],
        styles_pca[np.array(labels) == i, 1],
        c=colors[np.array(labels) == i],
        marker=markers[i],
        label=f"Class {i}",
    )
plt.legend()
plt.show()

# %%
# %% [markdown]
# <div class="alert alert-block alert-warning"><h3>Questions</h3>
# <ul>
# <li> Do the colors match those that you have seen in the data?</li>
# <li> Can you see any patterns in the colors?</li>
# <li> Can you guess what the classes correspond to?</li>

# %% [markdown]
# <div class="alert alert-block alert-success"><h2>Checkpoint 5</h2>
# Congratulations! You have made it to the end of the exercise!
# You have:
# - Created a StarGAN that can change the class of an image
# - Evaluated the StarGAN on unseen data
# - Used the StarGAN to create counterfactual images
# - Used the counterfactual images to highlight the differences between classes
# - Used the style space to understand the differences between classes
#
# If you have any questions, feel free to ask them in the chat!
# And check the Solutions exercise for a definite answer to how these classes are defined!

# %% [markdown] tags=["solution"]
# The colors for the classes are sampled from matplotlib colormaps! They are the four seasons: spring, summer, autumn, and winter.
# Check your style space again to see if you can see the patterns now!
# %% tags=["solution"]
# Let's plot the colormaps
import matplotlib as mpl
import numpy as np


def plot_color_gradients(cmap_list):
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh, left=0.2, right=0.99)

    for ax, name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect="auto", cmap=mpl.colormaps[name])
        ax.text(
            -0.01,
            0.5,
            name,
            va="center",
            ha="right",
            fontsize=10,
            transform=ax.transAxes,
        )

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()


plot_color_gradients(["spring", "summer", "autumn", "winter"])
