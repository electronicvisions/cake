all: v_syntcx_averaged.dat v_syntci_averaged.dat i_pl_fitted.dat i_gl_slow0_fast0_bigcap1_fitted.dat e_syni_averaged.dat i_gl_slow0_fast0_bigcap0_domain.dat i_gl_slow0_fast0_bigcap1_domain.dat i_gl_slow0_fast1_bigcap0_domain.dat i_gl_slow0_fast1_bigcap1_domain.dat i_gl_slow1_fast0_bigcap0_domain.dat i_gl_slow1_fast0_bigcap1_domain.dat
.PHONY: all

clean:
	rm *dat

V_syntci.dat V_syntcx.dat I_pl.dat I_gl_slow0_fast0_bigcap1.dat: dc.py
	./dc.py /wang/data/calibration/brainscales/default-2017-09-26-1 /wang/data/calibration/brainscales/default-2017-09-26-1 33 --hicann `seq 0 383` --neuron `seq 0 64 511`

v_syntci_averaged.dat: V_syntci.dat generate_scaled_and_clipped_lookup.py
# smaller DAC values can lead to "infinity" time constants
# scale factor is heuristic
	./generate_scaled_and_clipped_lookup.py $< --scale_factor 1.08 --min_max_dac 226 1023 --outputfilename $@

v_syntcx_averaged.dat: V_syntci.dat generate_scaled_and_clipped_lookup.py
# smaller DAC values can lead to "infinity" time constants
# scale factor is heuristic
	./generate_scaled_and_clipped_lookup.py $< --scale_factor 1.08 --min_max_dac 226 1023 --outputfilename $@

i_pl_fitted.dat: I_pl.dat i_pl_fit.py
	./i_pl_fit.py $< --outputfilename $@

i_gl_slow0_fast0_bigcap1_fitted.dat: I_gl_slow0_fast0_bigcap1.dat i_gl_fit.py
	./i_gl_fit.py $< --paramfilename $@

i_gl_slow0_fast0_bigcap0_domain.dat i_gl_slow0_fast0_bigcap1_domain.dat i_gl_slow0_fast1_bigcap0_domain.dat i_gl_slow0_fast1_bigcap1_domain.dat i_gl_slow1_fast0_bigcap0_domain.dat i_gl_slow1_fast0_bigcap1_domain.dat: i_gl_slow0_fast0_bigcap1_fitted.dat i_gl_domains.py
	./i_gl_domains.py $< --tofile

e_syni_averaged.dat: E_syni.dat e_syni_avg.py
	./e_syni_avg.py $< --outputfilename $@
