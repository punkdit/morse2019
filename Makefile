
all: amt morse


amt: amt.tex refs.bib
	pdflatex amt.tex
	bibtex amt
	pdflatex amt.tex
	pdflatex amt.tex



morse: morse.tex refs.bib
	pdflatex morse.tex
	bibtex morse
	pdflatex morse.tex
	pdflatex morse.tex



