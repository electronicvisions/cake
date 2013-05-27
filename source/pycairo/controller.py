"""Controller for the calibration of the BrainScaleS hardware"""

import time
import numpy as np

import pycairo.simulator

import pycairo.interfaces.database  # will be replaced by pycalibtic
import pycairo.translations.scaledhw  # will be replaced by pycalibtic
#import pycalibtic

import pycairo.interfaces.halbe

from pycairo.config.coordinates import get_fpga_ip
import pycairo.config.controller as config


class CalibrationController(object):
    """Main calibration class."""

    def __init__(self, hardware, neurons_range, fpga_id):
        """Creation of the different interfaces to devices used for calibration.

        Args:
            hardware: The type of hardware to use. Possible choices are USB, WSS.
            neurons_range: The range of neurons to calibrate per chip
            fpga_id: The logical number of the FPGA board to use
        """

        assert hardware == 'USB'  # TODO remove hardware parameter

        self.neurons_range = neurons_range
        self.fpga_id = fpga_id

        self.dbi = pycairo.interfaces.database.DatabaseInterface()
        if self.dbi.is_empty():
            # create DB if not already existing
            self.dbi.create_db(hardware, fpga_id)
            self.dbi.activate_fpga(fpga_id)
            self.dbi.activate_dnc(0, fpga_id)
            self.dbi.activate_hicann(0, 0, fpga_id)

        # hardware interface for single HICANN
        self.hwi = pycairo.interfaces.halbe.HalbeInterface(0, get_fpga_ip(fpga_id))

        # Scaled to hardware module
        self.scali = pycairo.translations.scaledhw.scaledHW()

        self.simi = pycairo.simulator.Simulator()

    def calibrate(self, model='LIF'):
        """Main calibration function

        Args:
            model: The model can be chosen between "LIF", "ALIF", "AdEx" and single_param, or directly the name of the parameter like "EL"
        """

        if config.verbose:
            print "### Calibration started ###"
            print ""
            calib_start = time.time()

        # Parameters per model
        LIF_parameters = ['EL', 'Vreset', 'Vt', 'gL', 'tauref', 'tausynx', 'tausyni']
        ALIF_parameters = LIF_parameters + ['a', 'tw', 'b']
        AdEx_parameters = ALIF_parameters + ['dT', 'Vexp']

        parameter_set = {'single_param': ['EL'],
                         'LIF': LIF_parameters,
                         'ALIF': ALIF_parameters,
                         'AdEx': AdEx_parameters
                         }

        if isinstance(model, basestring):
            parameters = parameter_set[model]
        else:
            parameters = model

        # Creation of neuron index
        for param in parameters:
            fpga_index, neuron_index = self.dbi.create_neuron_index(param, self.neurons_range, self.fpga_id)
            if config.verbose:
                print "Resetting Hardware"
            for ii, fpga_id in enumerate(fpga_index):
                self.calibrate_parameter(param, fpga_id, neuron_index[ii])

        if config.verbose:
            print "### Calibration completed in " + str(time.time() - calib_start) + " s ###"

    def calibrate_parameter(self, parameter, fpga_id, neurons):
        """Launch calibration for one parameter

        Args:
            parameter: The parameter to be calibrated, for example "EL"
            fpga_id: FPGA that controls HICANN to be calibrated
            neurons: list of neurons to be calibrated
        """

        ## Init phase ##
        start_time = time.time()
        if neurons:
            if config.verbose:
                print "## Starting calibration of parameter {} on HICANN {}".format(parameter, fpga_id)
                print "Calibrating neurons " + ", ".join([str(jj) for jj in neurons])
        else:
            print "## Parameter {} on HICANN {} already calibrated for neurons".format(parameter, fpga_id)
            return

        if config.verbose:
            print "Init phase started"

        parameters = self.get_parameters(parameter)

        # Create input_array
        input_array = self.get_steps(parameter)
        repetitions = config.parameter_ranges[parameter]['reps']

        output_array_mean = []
        output_array_err = []

        if config.verbose:
            print "Init phase finished in " + str(time.time() - start_time) + "s"

        ####### Measure #######
        start_measure = time.time()
        if config.verbose:
            print "Measurement phase started"

        # Main measurement loop
        for index, value in enumerate(input_array):
            if config.verbose:
                print "Measure for value " + str(value) + " [" + str(len(input_array) - index) + " values remaining]"

            # Set value to be calibrated
            parameters[parameter] = value

            # Convert parameters for one HICANN
            try:
                # Use calibration database
                calibrated_parameters = self.scali.scaled_to_hw(fpga_id, neurons, parameters, parameters=parameter+'_calibration')
            except:
                # If no calibration available, use direct translation
                calibrated_parameters = self.scali.scaled_to_hw(fpga_id, neurons, parameters, parameters='None')

            meas_array = []
            for run in range(repetitions):  # Do a given number of repetitions
                if config.verbose:
                    print "Starting repetition {} of {}".format(run + 1, repetitions)
                result = self.measure(neurons, parameter, parameters, value, calibrated_parameters)
                meas_array.append(result)

            meas_array = np.array(meas_array)
            meas_array_mean = np.mean(meas_array.T, axis=1)
            meas_array_err = np.std(meas_array.T, axis=1)

            output_array_mean.append(meas_array_mean)
            output_array_err.append(meas_array_err)

        if config.verbose:
            print "Measurement phase completed in " + str(time.time() - start_measure) + "s"

        ####### Process #######
        if config.verbose:
            print "Process phase started"
        process_start = time.time()

        # Sort measurements
        sorted_array_mean = np.array(output_array_mean).T
        sorted_array_err = np.array(output_array_err).T

        if config.debug_print_results:
            print "Sorted array, mean :", sorted_array_mean
            print ""
            print "Sorted array, error :", sorted_array_err
            print ""

        processed_array_mean, processed_array_err = self.process_result(parameter, parameters, sorted_array_mean, sorted_array_err)
        processed_array_mean = np.array(processed_array_mean)
        processed_array_err = np.array(processed_array_err)

        if config.debug_print_results:
            print "Processed array, mean :", processed_array_mean
            print ""
            print "Processed array, error :", processed_array_err

        if config.verbose:
            print "Process phase completed in " + str(time.time() - process_start) + "s"

        ####### Store #######
        if config.verbose:
            print "Store phase started"
        store_start = time.time()

        db_input_array = input_array
        if parameter in ('tauref', 'tw'):
            db_input_array = 1.0 / input_array

        for n, neuron in enumerate(neurons):
            # Compute calibration function
            try:
                parameter_fit = self.compute_calib_function(processed_array_mean[n], db_input_array)

                # Set the parameter as calibrated
                self.dbi.change_parameter_neuron(fpga_id, neuron, parameter+"_calibrated", True)
            except Exception as e:  # TODO catch specific exception
                # If fail, don't store anything, mark as non calibrated
                parameter_fit = []
                self.dbi.change_parameter_neuron(fpga_id, neuron, parameter+"_calibrated", False)
                raise e

            # Store function in DB
            self.dbi.change_parameter_neuron(fpga_id, neuron, parameter+"_fit", parameter_fit)

            # Store standard deviation in DB
            self.dbi.change_parameter_neuron(fpga_id, neuron, parameter+"_dev", processed_array_err[n].tolist())

            # Store data in DB
            self.dbi.change_parameter_neuron(fpga_id, neuron, parameter, [processed_array_mean[n].tolist(), db_input_array.tolist()])

        # Evaluate calibration
        # self.dbi.evaluate_db(fpga_id,neuron_index[h],parameter)

        # Set HICANN as calibrated if all parameters are calibrated
        self.dbi.check_hicann_calibration(fpga_id)

        if config.verbose:
            print "Store phase completed in " + str(time.time() - store_start) + "s"
            print ""
            print "## Calibration of parameter " + parameter + " completed in " + str(time.time() - start_time) + "s ##"
            print "## Calibration took " + str((time.time() - start_time) / len(neurons)) + "s per neuron ##"
            print ""

    def get_parameters(self, parameter):
        """Loads parameter values needed for the requested parameter calibration from configuration.

        Args:
            parameter: the parameter which is calibrated.

        Returns:
            dictionary containing values for all parameters.
        """

        parameters = {}
        # Parameter specific initialization
        if parameter == 'EL':
            parameters.update({'Vt': config.parameter_max['Vt'], 'gL': config.parameter_default['gL']})
        elif parameter in ('Vreset', 'Vt'):
            parameters.update({'EL': config.parameter_max['EL'], 'gL': config.parameter_default['gL']})
        elif parameter == 'gL':
            parameters.update({'Vreset': config.parameter_IF['Vreset'], 'EL': config.parameter_IF['EL'],
                               'Vt': config.parameter_IF['Vt']})
        elif parameter == 'tauref':
            parameters.update(config.parameter_IF)
        elif parameter == 'a':
            parameters.update(config.parameter_IF)
            parameters['tauw'] = config.parameter_default['tauw']
        elif parameter == 'tw':
            parameters.update(config.parameter_IF)
            parameters['a'] = config.parameter_default['a']
        elif parameter == 'b':
            parameters.update(config.parameter_IF)
            parameters.update({'a': config.parameter_default['a'], 'tauw': config.parameter_default['tauw']})
        elif parameter in ('tausynx', 'tausyni'):
            parameters.update(config.parameter_IF)
            parameters['EL'] = config.parameter_default['EL']
            parameters['Vt'] = config.parameter_max['Vt']
            parameters['Esynx'] = config.parameter_default['Esynx']
            parameters['Esyni'] = config.parameter_default['Esyni']
        elif parameter == 'dT':
            parameters.update(config.parameter_IF)
            parameters['expAct'] = config.parameter_special['expAct']
            parameters['Vexp'] = self.Vexp_default
        elif parameter == 'Vexp':
            parameters.update(config.parameter_IF)
            parameters['expAct'] = config.parameter_special['expAct']
            parameters['dT'] = self.dT_default
        return parameters

    def process_result(self, parameter, parameters, sorted_array_mean, sorted_array_err):
        if parameter in ('EL', 'Vt', 'Vreset', 'tw', 'dT'):
            return sorted_array_mean, sorted_array_err

        elif parameter == 'gL':
            # Calculate relation between frequency and gL
            gL_coefs = self.simi.get_gl_freq(500, 1200, parameters['EL'], parameters['Vreset'], parameters['Vt'])
            mean = [self.poly_process(a, gL_coefs) for a in sorted_array_mean]
            err = [self.poly_process(a, gL_coefs) for a in sorted_array_err]
            return mean, err

        elif parameter == 'tauref':
            # Compute base frequency (tau_ref -> 0 )
            base_freq = self.simi.compute_freq(30e-6, 0, parameters)
            calc_freq = lambda x: (1.0 / x - 1.0 / base_freq) * 1e6

            m = sorted_array_mean[sorted_array_mean != 0.0]
            e = sorted_array_err[sorted_array_err != 0.0]
            return calc_freq(m), calc_freq(e)

        elif parameter in ('tausynx', 'tausyni'):
            # Fitting is already done in HW interface, just transmit results
            processed_array_err = sorted_array_err
            processed_array_mean = sorted_array_mean

            return sorted_array_mean, sorted_array_err

        assert False, "TODO, fix stuff for other parameters"  # FIXME
        #elif
        if (parameter == 'a'):
            # Calculate relation between frequency and gL
            a_coefs = self.simi.get_a_freq(200, 600, parameters['EL'], parameters['Vreset'], parameters['Vt'], parameters['gL'])

            for h, hicann in enumerate(hicann_index):
                hicann_array_mean = []
                hicann_array_err = []
                for n, neuron in enumerate(neuron_index[h]):
                    # Calculate a and error
                    hicann_array_mean.append(self.poly_process(sorted_array_mean[h][n], a_coefs))
                    hicann_array_err.append(self.poly_process(sorted_array_err[h][n], a_coefs))

                processed_array_mean.append(hicann_array_mean)
                processed_array_err.append(hicann_array_err)
        elif (parameter == 'b'):
            # Calculate relation between frequency and b
            b_coefs = self.simi.get_b_freq(10, 100, parameters['EL'], parameters['Vreset'], parameters['Vt'], parameters['gL'], parameters['a'], parameters['tauw'])

            for h, hicann in enumerate(hicann_index):
                hicann_array_mean = []
                hicann_array_err = []

                for n, neuron in enumerate(neuron_index[h]):
                    # Calcultate b and error
                    hicann_array_mean.append(self.poly_process(sorted_array_mean[h][n], b_coefs))
                    hicann_array_err.append(self.poly_process(sorted_array_err[h][n], b_coefs))

                processed_array_mean.append(hicann_array_mean)
                processed_array_err.append(hicann_array_err)

        elif (parameter == 'Vexp'):
            # Calculate relation between frequency and gL
            Vexp_coefs = self.simi.get_vexp_freq(800, 1000, parameters['EL'], parameters['Vreset'], parameters['Vt'], parameters['gL'], parameters['dT'])

            for h, hicann in enumerate(hicann_index):
                hicann_array_mean = []
                hicann_array_err = []

                for n, neuron in enumerate(neuron_index[h]):
                    # Calcultate Vexp and error
                    hicann_array_mean.append(self.poly_process(sorted_array_mean[h][n], Vexp_coefs))
                    hicann_array_err.append(self.poly_process(sorted_array_err[h][n], Vexp_coefs))

                processed_array_mean.append(hicann_array_mean)
                processed_array_err.append(hicann_array_err)
        return processed_array_mean, processed_array_err

    def measure(self, neurons, parameter, parameters, value, calibrated_parameters):
        if config.verbose:
            print "Configuring the hardware ..."
            config_start = time.time()

        # If no debug mode, configure the hardware. If debug, do nothing
        if not config.debug_mode:
            self.hwi.send_fg_configure(neurons, calibrated_parameters)

        if config.verbose:
            print "Hardware configuration completed in {:.2}s".format(time.time() - config_start)
            print "Measuring the hardware ..."
            meas_start = time.time()

        # If no debug mode, measure the hardware. If debug, return 0
        if not config.debug_mode:
            measurement = self.hwi.measure(neurons, parameter, parameters, 0, value)
        else:
            measurement = neuron_index  # FIXME this looks wrong

        if config.verbose:
            print "Measurement completed in " + str(time.time() - meas_start) + " s"

        return measurement

    def poly_process(self, array, coefficients):
        """Process an array with a polynomial function.

        Args:
            array: The array of data to be processed.
            coefficients: The polynomial coefficients.
        """

        poly_array = []
        for i in array:
            poly_array.append(float(np.polyval(coefficients, i)))
        return poly_array

    def get_steps(self, p):
        # Create input_array
        start, stop = config.parameter_ranges[p]['min'], config.parameter_ranges[p]['max']
        pts = config.parameter_ranges[p]['pts']
        return np.linspace(start, stop, pts)

    def compute_calib_function(self, measuredValues, values):
        """Compute calibration function.

        Args:
            measuredValues: The measured values.
            values: The input values.
        """

        calibCoeff = np.polyfit(measuredValues, values, 2)
        a = float(calibCoeff[0])
        b = float(calibCoeff[1])
        c = float(calibCoeff[2])
        fit = (a, b, c)
        return fit
