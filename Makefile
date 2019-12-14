

morse: morse.tex refs.bib
	pdflatex morse.tex
	bibtex morse
	pdflatex morse.tex
	pdflatex morse.tex



