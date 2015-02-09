all: calibration.pdf

%.pdf: %.tex
	pdflatex $*
	pdflatex $*
	pdflatex $*

clean:
	rm *.pdf *.log *.aux *.toc

.PHONY: clean
