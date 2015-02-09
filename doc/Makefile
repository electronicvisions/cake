all: calibration.pdf sd3t.pdf

%.pdf: %.tex
	pdflatex $*
	pdflatex $*
	pdflatex $*

clean:
	rm *.pdf *.log *.aux *.toc

.PHONY: clean
