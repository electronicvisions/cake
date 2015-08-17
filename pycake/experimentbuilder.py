import pylogging
import Coordinate
import pyhalbe
import pysthal
import copy
from itertools import product
import numpy

import pycake.helpers.TraceAverager as TraceAverager
from pycake.helpers.sthal import StHALContainer
from pycake.helpers.sthal import SimStHALContainer
from pycake.helpers.sthal import UpdateParameterUp
from pycake.measure import ADCMeasurement
from pycake.measure import ADCMeasurementWithSpikes
from pycake.measure import SpikeMeasurement
from pycake.measure import I_gl_Measurement
from pycake.experiment import SequentialExperiment, I_pl_Experiment
from pycake.experiment import IncrementalExperiment
import pycake.analyzer
from pycake.helpers.units import Unit, Ampere, Volt, DAC
from pycake.helpers.trafos import DACtoHW
from pycake.measure import ADCFreq_Measurement
from pycake.analyzer import ADCFreq_Analyzer

# shorter names
Enum = Coordinate.Enum
neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter


class BaseExperimentBuilder(object):
    """ Builds a list of measurements from a config object.

        Args:
            config: pycake.config.Config object
            test:   (True|False) whether this should be a test measurement or not.
                    If it is a test measurement, the target parameter is also calibrated.
                    Otherwise, it stays uncalibrated even if there is calibration data for it.
    """
    logger = pylogging.get("pycake.experimentbuilder")

    def __init__(self, config, test, calibtic_helper):
        self.config = config

        path, name = self.config.get_calibtic_backend()
        wafer, hicann = self.config.get_coordinates()
        self.calibtic = calibtic_helper

        self.neurons = self.config.get_neurons()
        self.blocks = self.config.get_blocks()
        self.test = test

    def get_sthal(self, analog=Coordinate.AnalogOnHICANN(0)):
        """
        returns sthal helper depending on config (e.g. hw vs. sim)
        """

        coord_wafer, coord_hicann = self.config.get_coordinates()
        wafer_cfg = self.config.get_wafer_cfg()
        PLL = self.config.get_PLL()

        sim_denmem_cfg = self.config.get_sim_denmem()
        if sim_denmem_cfg:
            sthal = SimStHALContainer(
                coord_wafer, coord_hicann, wafer_cfg=wafer_cfg, PLL=PLL,
                config=self.config)
        else:
            sthal = StHALContainer(
                coord_wafer, coord_hicann, wafer_cfg=wafer_cfg, PLL=PLL)

        sthal.set_bigcap(self.config.get_bigcap())
        # Get the SpeedUp type from string
        speedup = pysthal.SpeedUp.names[self.config.get_speedup().upper()]
        sthal.set_speedup(speedup)

        return sthal

    def generate_measurements(self):
        self.logger.INFO("Building experiment {}".format(self.config.get_target()))
        measurements = []
        coord_wafer, coord_hicann = self.config.get_coordinates()
        steps = self.config.get_steps()
        repetitions = self.config.get_repetitions()

        # Get readout shifts
        readout_shifts = self.get_readout_shifts(self.neurons)

        wafer_cfg = self.config.get_wafer_cfg()

        # Create one sthal container for each step
        # order is step 1, step 2, step 3, ..., step 1, step 2, step 3, ...
        for rep, step in product(range(repetitions), steps):
            self.logger.INFO(
                "Building step {}, repetition {}".format(step, rep))

            sthal = self.get_sthal()

            # TODO maybe find better solution
            if self.test:
                step = step.copy()
                for value in step.itervalues():
                    if isinstance(value, Unit):
                        value.apply_calibration = True

            step_parameters = self.get_step_parameters(step)

            if not wafer_cfg:
                self.prepare_hicann(sthal.hicann, step_parameters)
                sthal.hicann.floating_gates = self.prepare_parameters(step_parameters)
            sthal = self.prepare_specific_config(sthal)

            measurement = self.make_measurement(sthal, self.neurons, readout_shifts)
            measurements.append(measurement)

        return measurements, repetitions

    def get_step_parameters(self, step):
        """ Get parameters for this step.
        """
        return self.config.get_step_parameters(step)

    def prepare_hicann(self, hicann, parameters):
        """Write HICANN configuration from parameters
        to sthal.hicann."""

        speedup_gl = parameters.get("speedup_gl", None)
        if speedup_gl is not None:
            hicann.set_speed_up_gl(speedup_gl)

        speedup_gladapt = parameters.get("speedup_gladapt", None)
        if speedup_gladapt is not None:
            hicann.set_speed_up_gladapt(speedup_gladapt)

        speedup_radapt = parameters.get("speedup_radapt", None)
        if speedup_radapt is not None:
            hicann.set_speed_up_radapt(speedup_radapt)

    def prepare_parameters(self, parameters):
        """ Writes floating gate parameters into a
        sthal FloatingGates container.
            This includes calibration and transformation from V or nA to DAC values.

        Returns:
            pysthal.FloatingGates with given parameters
        """

        floating_gates = pysthal.FloatingGates()

        self.calibtic.set_calibrated_parameters(
            parameters, self.neurons, self.blocks, floating_gates)
        return floating_gates

    def prepare_specific_config(self, sthal):
        """ Hook function to specify additional stuff, e.g. current injection, spike generation, ...
        """
        return sthal

    def get_readout_shifts(self, neurons):
        """ Get readout shifts (in V) for a list of neurons.
            If no readout shift is saved, a shift of 0 is returned.

            Args:
                neurons: a list of NeuronOnHICANN coordinates
            Returns:
                shifts: a dictionary {neuron: shift (in V)}
        """
        return dict((n, self.calibtic.get_readout_shift(n)) for n in neurons)

    def get_experiment(self):
        """
        """
        measurements, repetitions = self.generate_measurements()
        analyzer = self.get_analyzer()
        experiment = SequentialExperiment(measurements, analyzer, repetitions)
        experiment = self.add_additional_measurements(experiment)
        return experiment

    def get_analyzer(self):
        """
        Get the appropriate analyzer for a specific parameter.
        Overwritte in subclass, when required.
        """
        return pycake.analyzer.MeanOfTraceAnalyzer()

    def make_measurement(self, sthal, neurons, readout_shifts):
        """
        Create a measurement object.
        Overwrite in subclass, when required.
        """

        return ADCMeasurement(sthal, neurons, readout_shifts)

    def add_additional_measurements(self, experiment):
        """ Dummy for adding additional initial measurements if needed.
        """
        return experiment


class InputSpike_Experimentbuilder(BaseExperimentBuilder):
    """Send input spikes to neurons"""
    def prepare_specific_config(self, sthal):
        spikes = self.config.get_input_spikes()
        sthal.set_input_spikes(spikes)
        sthal.stimulateNeurons(100000., 4)
        return sthal

    def make_measurement(self, sthal, neurons, readout_shifts):
        return ADCMeasurementWithSpikes(sthal, neurons, readout_shifts)


"""
Custom calibration experiment builders can be imported below,
after the base classes were defined.
"""
from pycake.calibration.I_gl_charging import I_gl_charging_Experimentbuilder


class E_l_Experimentbuilder(BaseExperimentBuilder):
    pass


class Spikes_Experimentbuilder(BaseExperimentBuilder):

    def get_analyzer(self):
        "get analyzer"
        return pycake.analyzer.Spikes_Analyzer()

    def make_measurement(self, sthal, neurons, readout_shifts):
        """
        Create a measurement object.
        Overwrite in subclass, when required.
        """

        return SpikeMeasurement(sthal, neurons)


class V_reset_Experimentbuilder(BaseExperimentBuilder):
    def get_analyzer(self):
        "get analyzer"
        return pycake.analyzer.V_reset_Analyzer()

    def prepare_specific_config(self, sthal):
        sthal.set_recording_time(25.0e-6, 4)
        sthal.maximum_spikes = 3
        return sthal


class V_t_Experimentbuilder(BaseExperimentBuilder):
    def get_analyzer(self):
        "get analyzer"
        return pycake.analyzer.V_t_Analyzer()

    def prepare_specific_config(self, sthal):
        sthal.set_recording_time(25.0e-6, 4)
        sthal.maximum_spikes = 10
        return sthal


class I_gladapt_Experimentbuilder(BaseExperimentBuilder):
    def get_analyzer(self):
        """ISI std is a hint whether adaptation takes place.
        Should be replaced by a better analysis before real usage.
        """
        return pycake.analyzer.ISI_Analyzer()


class I_pl_Experimentbuilder(BaseExperimentBuilder):
    """Longer recording time than parent class.

    To be used with hardware."""
    def prepare_specific_config(self, sthal):
        I_pl_DAC = sthal.hicann.floating_gates.getNeuron(Coordinate.NeuronOnHICANN(Enum(0)), neuron_parameter.I_pl)
        expected_ISI = DACtoHW(I_pl_DAC, neuron_parameter.I_pl) + 1e-8 # estimate 1e-8 as minimum ISI
        # Record at least 50 but not more than 5000 microseconds
        # If errors occur during calibration, these values need to be tuned
        sthal.recording_time = max(50e-6, min([expected_ISI * 70, 5e-4]))
        sthal.maximum_spikes = 10
        return sthal

    def get_analyzer(self):
        "get analyzer"
        return pycake.analyzer.I_pl_Analyzer()

    def get_experiment(self):
        """
        """
        measurements, repetitions = self.generate_measurements()
        analyzer = self.get_analyzer()
        experiment = I_pl_Experiment(measurements, analyzer, repetitions)
        experiment = self.add_additional_measurements(experiment)
        return experiment

    def add_additional_measurements(self, experiment):
        """ Add the initial measurement to I_gl experiment.
            This measurement determines the ADC frequency needed for the TraceAverager
        """
        coord_wafer, coord_hicann = self.config.get_coordinates()
        steps = self.config.get_steps()
        readout_shifts = self.get_readout_shifts(self.neurons)
        wafer_cfg = self.config.get_wafer_cfg()

        sthal = self.get_sthal()

        # get a step with I_pl = 1023 DAC
        parameters = self.get_step_parameters({neuron_parameter.I_pl: DAC(1023, apply_calibration=False)})

        if not wafer_cfg:
            self.prepare_hicann(sthal.hicann, parameters)
            sthal.hicann.floating_gates = self.prepare_parameters(parameters)
        sthal = self.prepare_specific_config(sthal)

        measurement = self.make_measurement(sthal, self.neurons, readout_shifts)
        analyzer = pycake.analyzer.ISI_Analyzer()
        experiment.add_initial_measurement(measurement, analyzer)
        return experiment

class E_syn_Experimentbuilder(BaseExperimentBuilder):
    EXCITATORY = None

    def prepare_specific_config(self, sthal):
        sthal.simulation_init_time = 100.0e-6
        sthal.set_recording_time(10e-6, 10)
        sthal.stimulateNeurons(5.0e6, 1, excitatory=self.EXCITATORY)
        sthal.maximum_spikes = 1
        sthal.spike_counter_offset = 0.0
        return sthal


class E_syni_Experimentbuilder(E_syn_Experimentbuilder):
    EXCITATORY = False


class E_synx_Experimentbuilder(E_syn_Experimentbuilder):
    EXCITATORY = True


class V_convoff_MeasurementGenerator(object):
    def __init__(self, parameters, floating_gates,
                 neurons, readout_shifts):
        self.floating_gates = floating_gates
        self.neurons = neurons
        self.readout_shifts = readout_shifts
        self.parameters = parameters

    def __len__(self):
        return len(self.floating_gates)

    def __call__(self, sthal):
        for fg_params in self.floating_gates:
            sthal.hicann.floating_gates = fg_params
            configurator = UpdateParameterUp(self.parameters)
            measurement = ADCMeasurement(sthal,
                                         self.neurons,
                                         self.readout_shifts)
            yield configurator, measurement


class V_convoff_Experimentbuilder(BaseExperimentBuilder):
    EXCITATORY = None
    WITH_SPIKES = False

    def __init__(self, *args, **kwargs):
        super(V_convoff_Experimentbuilder, self).__init__(*args, **kwargs)
        self.init_time = 400.0e-6
        # self.recording_time = 1e-4
        self.recording_time = 80.0e-6
        self.no_spikes = 100 if self.WITH_SPIKES else 1

    def prepare_specific_config(self, sthal):
        sthal.simulation_init_time = self.init_time
        sthal.set_recording_time(self.recording_time, self.no_spikes)
        if self.no_spikes > 1:
            sthal.stimulateNeurons(1.0/self.recording_time, 1,
                                   excitatory=self.EXCITATORY)
        return sthal

    def make_measurement(self, sthal, neurons, readout_shifts):
        return ADCMeasurement(sthal, neurons, readout_shifts)

    def generate_measurements(self):
        self.logger.INFO("Building experiment {}".format(self.config.get_target()))
        coord_wafer, coord_hicann = self.config.get_coordinates()
        steps = self.config.get_steps()
        if not steps:
            raise RuntimeError("Missing steps for {}".format(type(self)))

        readout_shifts = self.get_readout_shifts(self.neurons)
        wafer_cfg = self.config.get_wafer_cfg()

        parameters = set(sum((s.keys() for s in steps), []))
        floating_gates = [self.prepare_parameters(self.get_step_parameters(s))
                          for s in steps]

        sthal = self.get_sthal()
        sthal = self.prepare_specific_config(sthal)
        sthal.hicann.floating_gates = copy.copy(floating_gates[0])

        readout_shifts = self.get_readout_shifts(self.neurons)
        generator = V_convoff_MeasurementGenerator(
                parameters, floating_gates, self.neurons, readout_shifts)
        return sthal, generator

    def get_experiment(self):
        """
        """
        sthal, generator = self.generate_measurements()
        analyzer = self.get_analyzer()
        experiment = IncrementalExperiment(sthal, generator, analyzer)

        coord_wafer, coord_hicann = self.config.get_coordinates()
        wafer_cfg = self.config.get_wafer_cfg()
        PLL = self.config.get_PLL()
        sthal = self.get_sthal(Coordinate.AnalogOnHICANN(1))
        if sthal.is_hardware():
            experiment.add_initial_measurement(
                ADCFreq_Measurement(sthal, self.neurons, bg_rate=100e3),
                ADCFreq_Analyzer())
        else:
            experiment.initial_data.update(
                {ADCFreq_Analyzer.KEY: 96e6})
        return experiment


class V_convoffi_Experimentbuilder(V_convoff_Experimentbuilder):
    EXCITATORY = False
    ANALYZER = pycake.analyzer.V_convoffi_Analyzer


class V_convoffi_S_Experimentbuilder(V_convoff_Experimentbuilder):
    EXCITATORY = False
    ANALYZER = pycake.analyzer.V_convoffi_Analyzer
    WITH_SPIKES = True


class V_convoffx_Experimentbuilder(V_convoff_Experimentbuilder):
    EXCITATORY = True
    ANALYZER = pycake.analyzer.V_convoffx_Analyzer


class V_convoffx_S_Experimentbuilder(V_convoff_Experimentbuilder):
    EXCITATORY = True
    ANALYZER = pycake.analyzer.V_convoffx_Analyzer
    WITH_SPIKES = True

#FIXME: #1615
#from calibration.vsyntc import SimplePSPAnalyzer

class V_convoff_test_Experimentbuilder(BaseExperimentBuilder):
    ANALYZER = pycake.analyzer.MeanOfTraceAnalyzer

    def prepare_specific_config(self, sthal):
        sthal.init_time = 400.0e-6
        sthal.recording_time = 1e-4
        sthal.maximum_spikes = 1
        return sthal


class V_syntc_Experimentbuilder(BaseExperimentBuilder):
    EXCITATORY = None
    ANALYZER = None

    def __init__(self, *args, **kwargs):
        super(V_syntc_Experimentbuilder, self).__init__(*args, **kwargs)
        self.init_time = 300.0e-6
        self.recording_time = 400.0e-6
        #self.spikes = numpy.array([20e-6])
        self.gmax_div = 10

    def prepare_specific_config(self, sthal):
        sthal.simulation_init_time = self.init_time
        sthal.set_recording_time(self.recording_time, 100)
        sthal.stimulateNeurons(1.0/self.recording_time, 1,
                               excitatory=self.EXCITATORY)
        # sthal.send_spikes_to_all_neurons(
        #     self.spikes, excitatory=self.EXCITATORY, gmax_div=self.gmax_div)
        # sthal.maximum_spikes = 2
        # sthal.spike_counter_offset = 0.0
        return sthal

    def make_measurement(self, sthal, neurons, readout_shifts):
        return ADCMeasurement(sthal, neurons, readout_shifts)

    def generate_measurements(self):
        self.logger.INFO("Building experiment {}".format(self.config.get_target()))
        coord_wafer, coord_hicann = self.config.get_coordinates()
        steps = self.config.get_steps()
        if not steps:
            raise RuntimeError("Missing steps for {}".format(type(self)))

        readout_shifts = self.get_readout_shifts(self.neurons)
        wafer_cfg = self.config.get_wafer_cfg()

        parameters = set(sum((s.keys() for s in steps), []))
        floating_gates = [self.prepare_parameters(self.get_step_parameters(s))
                          for s in steps]

        sthal = self.get_sthal()
        sthal = self.prepare_specific_config(sthal)
        sthal.hicann.floating_gates = copy.copy(floating_gates[0])

        readout_shifts = self.get_readout_shifts(self.neurons)
        generator = V_convoff_MeasurementGenerator(
                parameters, floating_gates, self.neurons, readout_shifts)
        return sthal, generator

    def get_experiment(self):
        """
        """
        sthal, generator = self.generate_measurements()
        analyzer = self.get_analyzer()
        experiment = IncrementalExperiment(sthal, generator, analyzer)

        coord_wafer, coord_hicann = self.config.get_coordinates()
        wafer_cfg = self.config.get_wafer_cfg()
        PLL = self.config.get_PLL()
        sthal = self.get_sthal(Coordinate.AnalogOnHICANN(1))
        if sthal.is_hardware():
            experiment.add_initial_measurement(
                ADCFreq_Measurement(sthal, self.neurons, bg_rate=100e3),
                ADCFreq_Analyzer())
        else:
            experiment.initial_data.update(
                {ADCFreq_Analyzer.KEY: 96e6})


        return experiment

    def get_analyzer(self):
        "get analyzer"
        return self.ANALYZER(self.spikes)

class V_syntcx_Experimentbuilder(V_syntc_Experimentbuilder):
    EXCITATORY = True
    ANALYZER = None#SimplePSPAnalyzer #FIXME: #1615


class V_syntci_Experimentbuilder(V_syntc_Experimentbuilder):
    EXCITATORY = False
    ANALYZER = None#SimplePSPAnalyzer #FIXME: #1615

class I_gl_Experimentbuilder(BaseExperimentBuilder):
    def __init__(self, *args, **kwargs):
        super(I_gl_Experimentbuilder, self).__init__(*args, **kwargs)

    def prepare_specific_config(self, sthal):
        sthal.recording_time = 5e-3
        return sthal

    def make_measurement(self, sthal, neurons, readout_shifts):
        return I_gl_Measurement(sthal, neurons, readout_shifts)

    def get_analyzer(self):
        """ Create an I_gl analyzer WITHOUT setting the ADC frequency.
            This frequency needs to be measured before using the analyzer!
            This is done by the SequentialExperimentWithAveraging or manually by
                calling analyzer.measure_adc_frequency()
        """
        pll = self.config.get_PLL()
        return pycake.analyzer.I_gl_Analyzer(pll)

    def add_additional_measurements(self, experiment):
        """ Add the initial measurement to I_gl experiment.
            This measurement determines the ADC frequency needed for the TraceAverager
        """
        coord_wafer, coord_hicann = self.config.get_coordinates()
        wafer_cfg = self.config.get_wafer_cfg()
        PLL = self.config.get_PLL()
        sthal = self.get_sthal(Coordinate.AnalogOnHICANN(1))
        measurement = ADCFreq_Measurement(sthal, self.neurons, bg_rate=100e3)
        analyzer = ADCFreq_Analyzer()
        experiment.add_initial_measurement(measurement, analyzer)
        return experiment


class I_gl_sim_Experimentbuilder(I_gl_Experimentbuilder):
    class FakeTraceAverager(TraceAverager.TraceAverager):
        def get_average(self, v, period):
            """
            Instead of averaging, just return the full trace.
            In simulation, trace is smooth anyways. This assumes
            that the trace is EXACTLY one period long.
            """
            return v, numpy.zeros(len(v)), 1

        def get_chunks(self, trace, dt):
            # original function does not work with non-equidistant times
            raise NotImplemented

    def prepare_specific_config(self, sthal):
        # set recording time to one stimulation period
        sthal.recording_time = self.get_analyzer().dt
        return sthal

    def get_analyzer(self):
        coord_wafer, coord_hicann = self.config.get_coordinates()
        save_traces = self.config.get_save_traces()
        pll = self.config.get_PLL()

        # this method is hacky, but works using existing functionality
        # scipy.signal.resample assumes a periodic signal
        # resample voltage trace at this frequency
        # for trace averager
        sampling_freq = 90.e6
        trace_averager = TraceAverager(sampling_freq)
        return pycake.analyzer.I_gl_sim_Analyzer(trace_averager, save_traces, pll)


class E_l_I_gl_fixed_Experimentbuilder(E_l_Experimentbuilder):
    def get_analyzer(self):
        """ Get the appropriate analyzer for a specific parameter.
        """
        return pycake.analyzer.MeanOfTraceAnalyzer()


#from calibration.vsyntc import V_syntci_Experimentbuilder
#from calibration.vsyntc import V_syntcx_Experimentbuilder


class V_syntci_psp_max_Experimentbuilder(BaseExperimentBuilder):
    def prepare_specific_config(self, sthal):
        sthal.stimulateNeurons(0.1e6, 1, excitatory=False)
        return sthal


class V_syntcx_psp_max_Experimentbuilder(BaseExperimentBuilder):
    def prepare_specific_config(self, sthal):
        sthal.stimulateNeurons(0.1e6, 1, excitatory=True)
        return sthal


class readout_shift_Experimentbuilder(BaseExperimentBuilder):
    def prepare_specific_config(self, sthal):
        sthal.set_neuron_size(64)
        return sthal

    def get_readout_shifts(self, neurons):
        """ Get readout shifts (in V) for a list of neurons.
            If no readout shift is saved, a shift of 0 is returned.

            Args:
                neurons: a list of NeuronOnHICANN coordinates
            Returns:
                shifts: a dictionary {neuron: shift (in V)}
        """
        return dict((n, 0) for n in neurons)


class I_bexp_Experimentbuilder(BaseExperimentBuilder):
    def get_analyzer(self):
        return pycake.analyzer.ISI_Analyzer()
