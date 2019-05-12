all: dir
	@pdflatex report
	@pdflatex report
	@bibtex report
	@pdflatex report
	@pdflatex report
	@mv `ls report* | grep -v tex | grep -v pdf` aux

dir:
	@mkdir -p aux

clean:
	@rm -f `ls report* | grep -v tex | grep -v pdf`
	@rm -rf aux

mrproper: clean
	@rm -f report.pdf

