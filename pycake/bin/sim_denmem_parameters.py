from Coordinate import NeuronOnHICANN, FGBlockOnHICANN, Enum

from fastcalibration_parameters import parameters

sim_denmem_parameters = {
        # Use only two neurons, MUAAHHAHAHAHAHAHHAHA
        "neurons": [NeuronOnHICANN(Enum(i)) for i in range(2)],
        "blocks":  [FGBlockOnHICANN(Enum(i)) for i in range(1)],

        "sim_denmem" : "vtitan:8123",
}

parameters.update(sim_denmem_parameters)
