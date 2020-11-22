import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

from bindsnet.encoding import RepeatEncoder
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.monitors import NetworkMonitor

from bindsnet.analysis.plotting import plot_spikes, plot_voltages, plot_input, plot_weights

from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.learning import Hebbian

from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.utils import get_square_weights, get_square_assignments

### Input Data Parameters ###

# number of training samples
training_samples = 2
testing_samples = 20

# set number of classes
n_classes = 2

### Network Configuration Parameters ###

# configure number of input neurons
input_layer_name = "Input Layer"
input_neurons = 15

# configure number of hidden neurons
# hidden_layer_name = "Hidden Layer"
# hidden_neurons = 11

# configure the number of output neurons
output_layer_name = "Output Layer"
output_neurons = 2

### Simulation Parameters ###

# simulation time
time = 2000
dt = 0.2
timesteps = int(time/dt)

# number of training iterations
epochs = 100

# ratio of neurons to classes
per_class = int(output_neurons / n_classes)

### Construct Input Dataset ###

# store unique images in a list
imgs = []

# Class 0 Image
img0 = {"Label" : 0, "Image" : torch.FloatTensor([[1,1,1],[1,0,1],[1,0,1],[1,0,1],[1,1,1]])}
imgs.append(img0)

# Class 1 Image
img1 = {"Label" : 1, "Image" : torch.FloatTensor([[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]])}
imgs.append(img1)

# initialize list of inputs for training
training_dataset = []

# for the number of specified training samples
for i in range(training_samples):

    # randomly select a training sample
    # rand_sample = random.randint(0,n_classes-1)
    
    # provide an even number of training samples
    rand_sample = i % n_classes

    # add the sample to the list of training samples
    training_dataset.append(imgs[rand_sample])

# initialize the encoder
encoder = RepeatEncoder(time=time, dt=dt)

# list of encoded images for random selection during training
encoded_train_inputs = []

# loop through encode each image type and store into a list of encoded images
for sample in training_dataset:

    # encode the image 
    encoded_img = encoder(torch.flatten(sample["Image"]))

    # encoded image input for the network
    encoded_img_input = {input_layer_name: encoded_img}

    # encoded image label
    encoded_img_label = sample["Label"]

    # add to the encoded input list along with the input layer name
    encoded_train_inputs.append({"Label" : encoded_img_label, "Inputs" : encoded_img_input})

# initialize list of inputs for testing
testing_dataset = []

# for the number of specified testing samples
for i in range(testing_samples):

    # randomly select a training sample
    rand_sample = random.randint(0,n_classes-1)

    # add the sample to the list of training samples
    testing_dataset.append(imgs[rand_sample])

# list of encoded images for random selection during training
encoded_test_inputs = []

# loop through encode each image type and store into a list of encoded images
for sample in testing_dataset:

    # encode the image 
    encoded_img = encoder(torch.flatten(sample["Image"]))

    # encoded image input for the network
    encoded_img_input = {input_layer_name: encoded_img}

    # encoded image label
    encoded_img_label = sample["Label"]

    # add to the encoded input list along with the input layer name
    encoded_test_inputs.append({"Label" : encoded_img_label, "Inputs" : encoded_img_input})

### NETWORK CONFIGURATION ###

# initialize network
network = Network(dt=dt)

# configure weights for the synapses between the input layer and hidden layer
#w = torch.round(torch.abs(2 * torch.randn(input_neurons, lif_neurons)))
#w_1_2 = torch.randn(input_neurons,hidden_neurons)

# configure weights for the synapses between the hidden layer and output layer
#w_2_3 = torch.randn(hidden_neurons,output_neurons)

# initialize input and LIF layers
# spike traces must be recorded (why?)

# initialize input layer
input_layer = Input(n=input_neurons,traces=True)

# initialize hidden layer
# hidden_layer = LIFNodes(n=hidden_neurons,traces=True, rest=0, reset=0, thresh=0.5)

# initialize the output layer
output_layer = LIFNodes(
    n=output_neurons,
    traces=True, 
    rest=0, 
    reset=-1, 
    thresh=10,
    tc_decay=30
    )

# initialize connection between the input layer and the hidden layer
# specify the learning (update) rule and learning rate (nu)
# input_hidden_connection = Connection(
#     #source=input_layer, target=lif_layer, w=w, update_rule=PostPre, nu=(1e-4, 1e-2)
#     source=input_layer, target=hidden_layer, update_rule=Hebbian, nu=(1, 1), norm=1
# )

# initialize connection between the hidden layer and the output layer
input_output_connection = Connection(
    #source=input_layer, target=lif_layer, w=w, update_rule=PostPre, nu=(1e-4, 1e-2)
    source=input_layer, 
    target=output_layer, 
    update_rule=Hebbian, 
    nu=(0.1, 0.1), 
    norm=1
)


# add input layer to the network
network.add_layer(
    layer=input_layer, name=input_layer_name
)

# add hidden neuron layer to the network
# network.add_layer(
#     layer=hidden_layer, name=hidden_layer_name
# )

# add output neuron layer to the network
network.add_layer(
    layer=output_layer, name=output_layer_name
)

# # add connection to network
# network.add_connection(
#     connection=input_hidden_connection, source=input_layer_name, target=hidden_layer_name
# )

# add connection to network
network.add_connection(
    connection=input_output_connection, source=input_layer_name, target=output_layer_name
)

### SIMULATION VARIABLES ###

# record the spike times of each neuron during the simulation.
spike_record = torch.zeros(1, timesteps, output_neurons)

# record the mapping of each neuron to its corresponding label
assignments = -torch.ones_like(torch.Tensor(output_neurons))

# how frequently each neuron fires for each input class
rates = torch.zeros_like(torch.Tensor(output_neurons, n_classes))

# the likelihood of each neuron firing for each input class
proportions = torch.zeros_like(torch.Tensor(output_neurons, n_classes))


# label(s) of the input(s) being processed
labels = torch.empty(1,dtype=torch.int)

# create a spike monitor for each layer in the network
# this allows us to read the spikes in order to assign labels to neurons and determine the predicted class 
layer_monitors = {}
for layer in set(network.layers):

    # initialize spike monitor at the layer
    # do not record the voltage if at the input layer
    state_vars = ["s","v"] if (layer != input_layer_name) else ["s"]
    layer_monitors[layer] = Monitor(network.layers[layer], state_vars=state_vars, time=timesteps)

    # connect the monitor to the network
    network.add_monitor(layer_monitors[layer], name="%s_spikes" % layer)

weight_history = None
num_correct = 0.0

### DEBUG ###
### can be used to force the network to learn the inputs in a specific way
supervised = False
### used to determine if status messages are printed out at each sample
log_messages = False
### used to show weight changes
graph_weights = False
###
epochs = 5
###############

# show current weights
#print("Current Weights:")
#print(network.connections[("Input Layer", "LIF Layer")].w)

# iterate for epochs
for step in range(epochs):

    # index of the sample in the list of encoded trainining inputs
    sample_num = 0

    for sample in encoded_train_inputs:
        
        # print sample number
        if(sample_num % (training_samples / 10) == 0):
            print("Epoch:",str(step+1) + "/" + str(epochs),"Training Sample:", str(sample_num+1) + "/" + str(training_samples))

        sample_num += 1
        
        # get the label for the current image
        labels[0] = sample["Label"]

        # randomly decide which output neuron should spike if more than one neuron corresponds to the class
        # choice will always be 0 if there is one neuron per output class
        choice = np.random.choice(per_class, size=1, replace=False)

        # clamp on the output layer forces the node corresponding to the label's class to spike
        # this is necessary in order for the network to learn which neurons correspond to which classes
        # clamp: Mapping of layer names to boolean masks if neurons should be clamped to spiking. 
        # The ``Tensor``s have shape ``[n_neurons]`` or ``[time, n_neurons]``.
        clamp = {output_layer_name: per_class * labels[0] + torch.Tensor(choice).long()} if supervised else {}

        #print(sample["Inputs"])

        ### Step 1: Run the network with the provided inputs ###
        network.run(inputs=sample["Inputs"], time=time, clamp=clamp)

        ### Step 2: Get the spikes produced at the output layer ###
        spike_record[0] = layer_monitors[output_layer_name].get("s").squeeze()
        
        ### Step 3: ###

        # Assign labels to the neurons based on highest average spiking activity.
        # Returns a Tuple of class assignments, per-class spike proportions, and per-class firing rates 
        # Return Type: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        assignments, proportions, rates = assign_labels( spike_record, labels, n_classes, rates )

        ### Step 4: Classify data based on the neuron (label) with the highest average spiking activity ###

        # Classify data with the label with highest average spiking activity over all neurons.
        all_activity_pred = all_activity(spike_record, assignments, n_classes)

        ### Step 5: Classify data based on the neuron (label) with the highest average spiking activity
        ###         weighted by class-wise proportion ###
        proportion_pred = proportion_weighting(spike_record, assignments, proportions, n_classes)

        ### Update Accuracy
        num_correct += 1 if (labels.numpy()[0] == all_activity_pred.numpy()[0]) else 0

        ######## Display Information ########
        if log_messages:
            print("Actual Label:",labels.numpy(),"|","Predicted Label:",all_activity_pred.numpy(),"|","Proportionally Predicted Label:",proportion_pred.numpy())
            
            print("Neuron Label Assignments:")
            for idx in range(assignments.numel()):
                print(
                    "\t Output Neuron[",idx,"]:",assignments[idx],
                    "Proportions:",proportions[idx],
                    "Rates:",rates[idx]
                    )
            print("\n")
        
            print("Input:")
            print(sample["Inputs"])
            print("Output Spikes:")
            print(spike_record)
        #####################################
        
        
    #print("MONTOR OUTPUT SPIKES(2):")
    #print(layer_monitors[output_layer_name].get("s"))

    ### For Weight Plotting ###
    if graph_weights:
        weights = network.connections[("Input Layer", "LIF Layer")].w[:,0].numpy().reshape((1,input_neurons))
        weight_history = weights.copy() if step == 0 else np.concatenate((weight_history,weights),axis=0)
        print("Neuron 0 Weights:\n",network.connections[("Input Layer", "LIF Layer")].w[:,0])
        print("Neuron 1 Weights:\n",network.connections[("Input Layer", "LIF Layer")].w[:,1])
        print("====================")
    #############################

    if log_messages:
        print("Epoch #",step,"\tAccuracy:", num_correct / ((step + 1) * len(encoded_train_inputs)) )
        print("===========================\n\n")
        
    ## Print Final Class Assignments and Proportions ###
    print("Neuron Label Assignments:")
    for idx in range(assignments.numel()):
        print(
            "\t Output Neuron[",idx,"]:",assignments[idx],
            "Proportions:",proportions[idx],
            "Rates:",rates[idx]
            )
            

### For Weight Plotting ###
# Plot Weight Changes
if graph_weights:
    [plt.plot(weight_history[:,idx]) for idx in range(weight_history.shape[1])]
    plt.show()
    
#############################

### Print Final Class Assignments and Proportions ###
# print("Neuron Label Assignments:")
# for idx in range(assignments.numel()):
#     print(
#         "\t Output Neuron[",idx,"]:",assignments[idx],
#         "Proportions:",proportions[idx],
#         "Rates:",rates[idx]
#         )

print("Final Weights:")
print(network.connections[(input_layer_name, output_layer_name)].w)



#### Test Data ####

num_correct = 0

log_messages = False

# disable training mode
network.train(False)

# loop through each test example and record performance
for sample in encoded_test_inputs:

    # get the label for the current image
    labels[0] = sample["Label"]

    ### Step 1: Run the network with the provided inputs ###
    network.run(inputs=sample["Inputs"], time=time)

    ### Step 2: Get the spikes produced at the output layer ###
    spike_record[0] = layer_monitors[output_layer_name].get("s").squeeze()

    ### Step 3: ###

    # Assign labels to the neurons based on highest average spiking activity.
    # Returns a Tuple of class assignments, per-class spike proportions, and per-class firing rates 
    # Return Type: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    assignments, proportions, rates = assign_labels( spike_record, labels, n_classes, rates )

    ### Step 4: Classify data based on the neuron (label) with the highest average spiking activity ###

    # Classify data with the label with highest average spiking activity over all neurons.
    all_activity_pred = all_activity(spike_record, assignments, n_classes)

    ### Step 5: Classify data based on the neuron (label) with the highest average spiking activity
    ###         weighted by class-wise proportion ###
    proportion_pred = proportion_weighting(spike_record, assignments, proportions, n_classes)

    ### Update Accuracy
    num_correct += 1 if (labels.numpy()[0] == all_activity_pred.numpy()[0]) else 0

    ######## Display Information ########
    if log_messages:
        print("Actual Label:",labels.numpy(),"|","Predicted Label:",all_activity_pred.numpy(),"|","Proportionally Predicted Label:",proportion_pred.numpy())
        
        print("Neuron Label Assignments:")
        for idx in range(assignments.numel()):
            print(
                "\t Output Neuron[",idx,"]:",assignments[idx],
                "Proportions:",proportions[idx],
                "Rates:",rates[idx]
                )
        print("\n")
    #####################################
print("Accuracy:", num_correct / len(encoded_test_inputs) )