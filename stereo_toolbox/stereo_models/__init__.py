from .PSMNet.stackhourglass import PSMNet
from .GwcNet.gwcnet import GwcNet_G, GwcNet_GC
from .PCWNet.pcwnet import PWCNet_GC as PCWNet

Stereo_Models = {
    "PSMNet": PSMNet,
    "GwcNet_G": GwcNet_G,
    "GwcNet_GC": GwcNet_GC,
    "PCWNet": PCWNet,
}