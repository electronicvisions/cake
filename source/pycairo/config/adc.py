import pyhalbe.ADC

INPUT_CHANNEL = pyhalbe.ADC.INPUT_CHANNEL_3 # which pin on is connected on the ADC board?

class SampleTime(object):
    '''Define constant values for calibration sample times [us].
    
    These values were stripped out of the calibration ADC interface.'''

    min = 100
    max = 100
    mean = 100
    spikes = 10000
