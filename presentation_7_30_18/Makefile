all: dir
	@pdflatex main
	@biber main
	@pdflatex main
	@pdflatex main
	@mv main.pdf presentation.pdf
	@mv `ls main* | grep -v tex` aux

dir:
	@mkdir -p aux

clean:
	@rm -f `ls main* | grep -v tex`
	@rm -rf aux

mrproper: clean
	@rm -f presentation.pdf

