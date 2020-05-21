.PHONY: clean

notes.pdf: notes.tex
	pdflatex notes.tex

clean:
	rm *.log *.aux *.pdf
