# Attention dimension
ATT_DIM = 16

#Backbone
BACKBONE = "resnet18"  

# Layers to be hooked 
CONV_LAYERS = {
    "layer1.0.conv1": 64,
    "layer2.0.conv1": 128,
    "layer3.0.conv1": 256,
    "layer4.0.conv1": 512
}

# Alpha init (gain control)
ALPHA_INIT = 0.0
