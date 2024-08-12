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

mnist = ColoredMNIST("data", download=True)
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

test_mnist = ColoredMNIST("data", download=True, train=False)
dataloader = DataLoader(test_mnist, batch_size=32, shuffle=False)

labels = []
predictions = []
for x, y in dataloader:
    pred = model(x.to(device))
    labels.extend(y.cpu().numpy())
    predictions.extend(pred.argmax(dim=1).cpu().numpy())

cm = confusion_matrix(labels, predictions, normalize="true")
sns.heatmap(cm, annot=True, fmt=".2f")


# %% [markdown]
# # Part 2: Using Integrated Gradients to find what the classifier knows
#
# In this section we will make a first attempt at highlight differences between the "real" and "fake" images that are most important to change the decision of the classifier.
#

# %% [markdown]
# ## Attributions through integrated gradients
#
# Attribution is the process of finding out, based on the output of a neural network, which pixels in the input are (most) responsible. Another way of thinking about it is: which pixels would need to change in order for the network's output to change.
#
# Here we will look at an example of an attribution method called [Integrated Gradients](https://captum.ai/docs/extension/integrated_gradients). If you have a bit of time, have a look at this [super fun exploration of attribution methods](https://distill.pub/2020/attribution-baselines/), especially the explanations on Integrated Gradients.

# %% tags=[]
batch_size = 4
batch = [mnist[i] for i in range(batch_size)]
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
for attr, im in zip(attributions, x.cpu().numpy()):
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


for attr, im in zip(attributions, x.cpu().numpy()):
    visualize_color_attribution(attr, im)

# %% [markdown]
# We get some better clues when looking at the attributions in color.
# The highlighting doesn't just happen in the region with number, but also seems to hapen in a channel that matches the color of the image.
# Just based on this, however, we don't get much more information than we got from the images themselves.
#
# If we didn't know in advance, it is unclear whether the color or the number is the most important feature for the classifier.
# %% [markdown]
#
# ### Changing the basline
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
# Try using the code above to change the baseline and see how this affects the output.
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
for attr, im in zip(attributions_random.cpu().numpy(), x.cpu().numpy()):
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
for attr, im in zip(attributions_random.cpu().numpy(), x.cpu().numpy()):
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
for attr, im in zip(attributions_blurred.cpu().numpy(), x.cpu().numpy()):
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
for attr, im in zip(attributions_blurred.cpu().numpy(), x.cpu().numpy()):
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
# To gain insight into how the trained network classify images, we will use [Discriminative Attribution from Counterfactuals](https://arxiv.org/abs/2109.13412), a feature attribution with counterfactual explanations methodology.
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
style_mapping = DenseModel(
    input_shape=..., num_classes=...  # How big is the style space?
)
unet = UNet(depth=..., in_channels=..., out_channels=..., final_activation=nn.Sigmoid())

generator = Generator(unet, style_mapping=style_mapping)
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
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
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
# Stuff about the dataloader

# %%
from torch.utils.data import DataLoader

dataloader = DataLoader(
    mnist, batch_size=32, drop_last=True, shuffle=True
)  # We will use the same dataset as before

# %% [markdown] tags=[]
# TODO - Describe set_requires_grad


# %%
def set_requires_grad(module, value=True):
    """Sets `requires_grad` on a `module`'s parameters to `value`"""
    for param in module.parameters():
        param.requires_grad = value


# %% [markdown] tags=[]
# <div class="alert alert-banner alert-info"><h4>Task 3.2: Training!</h4>
#
# TODO - the task is to choose where to apply set_requires_grad
# <ul>
#   <li>Choose the values for `set_requires_grad`. Hint: which part of the code is training the generator? Which part is training the discriminator</li>
#   <li>Choose the values of `set_requires_grad`, again. Hint: you may want to switch</li>
#   <li>Choose the sign of the discriminator loss. Hint: what does the discriminator want to do?</li>
# </ul>
# Let's train the StarGAN one batch a time.
# While you watch the model train, consider whether you think it will be successful at generating counterfactuals in the number of steps we give it. What is the minimum number of iterations you think are needed for this to work, and how much time do yo uthink it will take?
# </div>
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
        # TODO Do I need to re-do the forward pass?
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


# %% [markdown] tags=[]
# ...this time again. &#x1F682; &#x1F68B; &#x1F68B; &#x1F68B;
#
# Once training is complete, we can plot the losses to see how well the model is doing.
# %%
plt.plot(losses["cycle"], label="Cycle loss")
plt.plot(losses["adv"], label="Adversarial loss")
plt.plot(losses["disc"], label="Discriminator loss")
plt.legend()
plt.show()

# %% [markdown] tags=[]
# We can also look at some examples of the images that the generator is creating.
# %%
idx = 0
fig, axs = plt.subplots(1, 4, figsize=(12, 4))
axs[0].imshow(x[idx].cpu().permute(1, 2, 0).detach().numpy())
axs[1].imshow(x_style[idx].cpu().permute(1, 2, 0).detach().numpy())
axs[2].imshow(x_fake[idx].cpu().permute(1, 2, 0).detach().numpy())
axs[3].imshow(x_cycled[idx].cpu().permute(1, 2, 0).detach().numpy())

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
# # Part 4: Evaluating the GAN

# %% [markdown] tags=[]
# ## Creating counterfactuals
#
# The first thing that we want to do is make sure that our GAN is able to create counterfactual images.
# To do this, we have to create them, and then pass them through the classifier to see if they are classified correctly.
#
# First, let's get the test dataset, so we can evaluate the GAN on unseen data.
# Then, let's get four prototypical images from the dataset as style sources.

# %% Loading the test dataset
test_mnist = ColoredMNIST("data", download=True, train=False)
prototypes = {}


for i in range(4):
    options = np.where(test_mnist.targets == i)[0]
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
# TODO make a task here!
# %%
num_images = len(test_mnist)
counterfactuals = np.zeros((4, num_images, 3, 28, 28))

predictions = []
source_labels = []
target_labels = []

for x, y in test_mnist:
    for i in range(4):
        if i == y:
            # Store the image as is.
            counterfactuals[i] = ...
        # Create the counterfactual from the image and prototype
        x_fake = generator(x.unsqueeze(0).to(device), ...)
        counterfactuals[i] = x_fake.cpu().detach().numpy()
        pred = model(...)

        source_labels.append(y)
        target_labels.append(i)
        predictions.append(pred.argmax().item())

# %% tags=["solution"]
num_images = len(test_mnist)
counterfactuals = np.zeros((4, num_images, 3, 28, 28))

predictions = []
source_labels = []
target_labels = []

for x, y in test_mnist:
    for i in range(4):
        if i == y:
            # Store the image as is.
            counterfactuals[i] = x
        # Create the counterfactual
        x_fake = generator(
            x.unsqueeze(0).to(device), prototypes[i].unsqueeze(0).to(device)
        )
        counterfactuals[i] = x_fake.cpu().detach().numpy()
        pred = model(x_fake)

        source_labels.append(y)
        target_labels.append(i)
        predictions.append(pred.argmax().item())

# %% [markdown] tags=[]
# Let's plot the confusion matrix for the counterfactual images.
# %%
cf_cm = confusion_matrix(target_labels, predictions, normalize="true")
sns.heatmap(cf_cm, annot=True, fmt=".2f")

# %% [markdown] tags=[]
# <div class="alert alert-block alert-warning"><h3>Questions</h3>
# <ul>
# <li> How well is our GAN doing at creating counterfactual images? </li>
# <li> Do you think that the prototypes used matter? Why or why not? </li>
# </ul>
# </div>

# %% [markdown] tags=[]
# Let's also plot some examples of the counterfactual images.

for i in np.random.choice(range(num_images), 4):
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    for j, ax in enumerate(axs):
        ax.imshow(counterfactuals[j][i].transpose(1, 2, 0))
        ax.axis("off")
        ax.set_title(f"Class {j}")

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h3>Questions</h3>
# <ul>
# <li>Can you easily tell which of these images is the original, and which ones are the counterfactuals?</li>
# <li>What is your hypothesis for the features that define each class?</li>
# </ul>
# </div>

# TODO wip here
# %% [markdown]
# # Part 5: Highlighting Class-Relevant Differences

# %% [markdown]
# At this point we have:
# - A classifier that can differentiate between neurotransmitters from EM images of synapses
# - A vague idea of which parts of the images it thinks are important for this classification
# - A CycleGAN that is sometimes able to trick the classifier with barely perceptible changes
#
# What we don't know, is *how* the CycleGAN is modifying the images to change their class.
#
# To start to answer this question, we will use a [Discriminative Attribution from Counterfactuals](https://arxiv.org/abs/2109.13412) method to highlight differences between the "real" and "fake" images that are most important to change the decision of the classifier.

# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task 5.1 Get sucessfully converted samples</h3>
# The CycleGAN is able to convert some, but not all images into their target types.
# In order to observe and highlight useful differences, we want to observe our attribution method at work only on those examples of synapses:
# <ol>
#     <li> That were correctly classified originally</li>
#     <li>Whose counterfactuals were also correctly classified</li>
# </ol>
#
# TODO
# - Get a boolean description of the `real` samples that were correctly predicted
# - Get the target class for the `counterfactual` images (Hint: It isn't `cf_gt`!)
# - Get a boolean description of the `cf` samples that have the target class
# </div>

# %% tags=[]
####### Task 5.1 TODO #######

# Get the samples where the real is correct
correct_real = ...

# HINT GABA is class 1 and ACh is class 0
target = ...

# Get the samples where the counterfactual has reached the target
correct_cf = ...

# Successful conversions
success = np.where(np.logical_and(correct_real, correct_cf))[0]

# Create datasets with only the successes
cf_success_ds = Subset(ds_counterfactual, success)
real_success_ds = Subset(ds_real, success)


# %% tags=["solution"]
########################
# Solution to Task 5.1 #
########################

# Get the samples where the real is correct
correct_real = real_pred == real_gt

# HINT GABA is class 1 and ACh is class 0
target = 1 - real_gt

# Get the samples where the counterfactual has reached the target
correct_cf = cf_pred == target

# Successful conversions
success = np.where(np.logical_and(correct_real, correct_cf))[0]

# Create datasets with only the successes
cf_success_ds = Subset(ds_counterfactual, success)
real_success_ds = Subset(ds_real, success)


# %% [markdown] tags=[]
# To check that we have got it right, let us get the accuracy on the best 100 vs the worst 100 samples:

# %% tags=[]
model = model.to("cuda")

# %% tags=[]
real_true, real_pred = predict(real_success_ds, "Real")
cf_true, cf_pred = predict(cf_success_ds, "Counterfactuals")

print(
    "Accuracy of the classifier on successful real images",
    accuracy_score(real_true, real_pred),
)
print(
    "Accuracy of the classifier on successful counterfactual images",
    accuracy_score(cf_true, cf_pred),
)

# %% [markdown] tags=[]
# ### Creating hybrids from attributions
#
# Now that we have a set of successfully translated counterfactuals, we can use them as a baseline for our attribution.
# If you remember from earlier, `IntegratedGradients` does a interpolation between the model gradients at the baseline and the model gradients at the sample. Here, we're also going to be doing an interpolation between the baseline image and the sample image, creating a hybrid!
#
# To do this, we will take the sample image and mask out all of the pixels in the attribution. We will then replace these masked out pixels by the equivalent values in the counterfactual. So we'll have a hybrid image that is like the original everywhere except in the areas that matter for classification.

# %% tags=[]
dataloader_real = DataLoader(real_success_ds, batch_size=10)
dataloader_counter = DataLoader(cf_success_ds, batch_size=10)

# %% tags=[]
# %%time
with torch.no_grad():
    model.to(device)
    # Create an integrated gradients object.
    # integrated_gradients = IntegratedGradients(model)
    # Generated attributions on integrated gradients
    attributions = np.vstack(
        [
            integrated_gradients.attribute(
                real.to(device),
                target=target.to(device),
                baselines=counterfactual.to(device),
            )
            .cpu()
            .numpy()
            for (real, target), (counterfactual, _) in zip(
                dataloader_real, dataloader_counter
            )
        ]
    )

# %%

# %% tags=[]
# Functions for creating an interactive visualization of our attributions
model.cpu()

import matplotlib

cmap = matplotlib.cm.get_cmap("viridis")
colors = cmap([0, 255])


@torch.no_grad()
def get_classifications(image, counter, hybrid):
    model.eval()
    class_idx = [full_dataset.classes.index(c) for c in classes]
    tensor = torch.from_numpy(np.stack([image, counter, hybrid])).float()
    with torch.no_grad():
        logits = model(tensor)[:, class_idx]
        probs = torch.nn.Softmax(dim=1)(logits)
        pred, counter_pred, hybrid_pred = probs
    return pred.numpy(), counter_pred.numpy(), hybrid_pred.numpy()


def visualize_counterfactuals(idx, threshold=0.1):
    image = real_success_ds[idx][0].numpy()
    counter = cf_success_ds[idx][0].numpy()
    mask = get_mask(attributions[idx], threshold)
    hybrid = (1 - mask) * image + mask * counter
    nan_mask = copy.deepcopy(mask)
    nan_mask[nan_mask != 0] = 1
    nan_mask[nan_mask == 0] = np.nan
    # PLOT
    fig, axes = plt.subplot_mosaic(
        """
                                   mmm.ooo.ccc.hhh
                                   mmm.ooo.ccc.hhh
                                   mmm.ooo.ccc.hhh
                                   ....ggg.fff.ppp
                                   """,
        figsize=(20, 5),
    )
    # Original
    viz.visualize_image_attr(
        np.transpose(mask, (1, 2, 0)),
        np.transpose(image, (1, 2, 0)),
        method="blended_heat_map",
        sign="absolute_value",
        show_colorbar=True,
        title="Mask",
        use_pyplot=False,
        plt_fig_axis=(fig, axes["m"]),
    )
    # Original
    axes["o"].imshow(image.squeeze(), cmap="gray")
    axes["o"].set_title("Original", fontsize=24)
    # Counterfactual
    axes["c"].imshow(counter.squeeze(), cmap="gray")
    axes["c"].set_title("Counterfactual", fontsize=24)
    # Hybrid
    axes["h"].imshow(hybrid.squeeze(), cmap="gray")
    axes["h"].set_title("Hybrid", fontsize=24)
    # Mask
    pred, counter_pred, hybrid_pred = get_classifications(image, counter, hybrid)
    axes["g"].barh(classes, pred, color=colors)
    axes["f"].barh(classes, counter_pred, color=colors)
    axes["p"].barh(classes, hybrid_pred, color=colors)
    for ix in ["m", "o", "c", "h"]:
        axes[ix].axis("off")

    for ix in ["g", "f", "p"]:
        for tick in axes[ix].get_xticklabels():
            tick.set_rotation(90)
        axes[ix].set_xlim(0, 1)


# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h3>Task 5.2: Observing the effect of the changes on the classifier</h3>
# Below is a small widget to interact with the above analysis. As you change the `threshold`, see how the prediction of the hybrid changes.
# At what point does it swap over?
#
# If you want to see different samples, slide through the `idx`.
# </div>

# %% tags=[]
interact(visualize_counterfactuals, idx=(0, 99), threshold=(0.0, 1.0, 0.05))

# %% [markdown]
# HELP!!! Interactive (still!) doesn't work. No worries... uncomment the following cell and choose your index and threshold by typing them out.

# %% tags=[]
# Choose your own adventure
# idx = 0
# threshold = 0.1

# # Plotting :)
# visualize_counterfactuals(idx, threshold)

# %% [markdown] tags=[]
# <div class="alert alert-warning">
# <h4>Questions</h4>
#
# - Can you find features that define either of the two classes?
# -  How consistent are they across the samples?
# -  Is there a range of thresholds where most of the hybrids swap over to the target class? (If you want to see that area, try to change the range of thresholds in the slider by setting `threshold=(minimum_value, maximum_value, step_size)`
#
# Feel free to discuss your answers on the exercise chat!
# </div>

# %% [markdown] tags=[]
# <div class="alert alert-block alert-success">
#     <h1>The End.</h1>
#     Go forth and train some GANs!
# </div>

# %% [markdown] tags=[]
# ## Going Further
#
# Here are some ideas for how to continue with this notebook:
#
# 1. Improve the classifier. This code uses a VGG network for the classification. On the synapse dataset, we will get a validation accuracy of around 80%. Try to see if you can improve the classifier accuracy.
#     * (easy) Data augmentation: The training code for the classifier is quite simple in this example. Enlarge the amount of available training data by adding augmentations (transpose and mirror the images, add noise, change the intensity, etc.).
#     * (easy) Network architecture: The VGG network has a few parameters that one can tune. Try a few to see what difference it makes.
#     * (easy) Inspect the classifier predictions: Take random samples from the test dataset and classify them. Show the images together with their predicted and actual labels.
#     * (medium) Other networks:  Try different architectures (e.g., a [ResNet](https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/#resnet-from-scratch)) and see if the accuracy can be improved.
#
# 2. Explore the CycleGAN.
#     * (easy) The example code below shows how to translate between GABA and acetylcholine. Try different combinations. Can you start to see differences between some pairs of classes? Which are the ones where the differences are the most or the least obvious? Can you see any differences that aren't well described by the mask? How would you describe these?
#
# 3. Try on your own data!
#     * Have a look at how the synapse images are organized in `data/raw/synapses`. Copy the directory structure and use your own images. Depending on your data, you might have to adjust the image size (128x128 for the synapses) and number of channels in the VGG network and CycleGAN code.
