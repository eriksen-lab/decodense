.DEFAULT_GOAL := achemso

all: achemso

achemso:
	pdflatex notes
	bibtex notes
	pdflatex notes
	pdflatex notes

clean:
	rm notes.aux
	rm notes.bbl
	rm notes.log
	rm notes.blg

