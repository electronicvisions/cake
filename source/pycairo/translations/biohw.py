## @package biohw
# Documentation for bioHW

#Imports
from . import bioscaled as bioS
from . import scaledhw as SHW

## Definition of the class bioHW
# 
# This class defines the translation from the biological parameters to hardware parameters
class bioHW:

    ## The constructor that initialize the bio to scaled and scaled to hardware interfaces
    def __init__(self,use_db=True):

        # Initialize interfaces
        self.bios = bioS.bioScaled()
        self.shw = SHW.scaledHW(use_db)

    ## Generated hardware parameters from biological model parameters
    # @param hicann_index The list of hicann chips
    # @param neuron_index The list of neurons, sorted by hicann numbers
    # @param neuronparams The biological model parameters to be converted
    # @param option Option for the scaled to hardware translation
    def gen_hw_parameters(self,hicann_index,neuron_index,neuronparams,option='LIF'):

        # Generate scaled parameters
        scaled_p = self.bios.bio_to_scaled(neuronparams,autoscale=True)

        # Convert into hardware parameters
        hw_p = []
        for h, hicann in enumerate(hicann_index):
            hw_p_hicann = self.shw.scaled_to_hw(hicann,neuron_index[h],scaled_p,option)
            hw_p.append(hw_p_hicann)

        return hw_p
