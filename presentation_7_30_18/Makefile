all: dir
	@pdflatex presentation
	@biber presentation
	@pdflatex presentation
	@pdflatex presentation
	@mv `ls presentation* | grep -v tex | grep -v pdf` aux

dir:
	@mkdir -p aux

clean:
	@rm -f `ls presentation* | grep -v tex | grep -v pdf`
	@rm -rf aux

mrproper: clean
	@rm -f presentation.pdf

