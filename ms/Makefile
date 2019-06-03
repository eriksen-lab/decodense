.DEFAULT_GOAL := achemso

all: achemso

achemso:
	pdflatex ms
	bibtex ms
	pdflatex ms
	pdflatex ms

clean:
	rm ms.aux
	rm ms.bbl
	rm ms.log
	rm ms.blg

