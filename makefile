# $@ = target, $^ = all dep, $< = first dep

# making the pdf script
makepdf = pdflatex -shell-escape -interaction=nonstopmode -file-line-error

# making plots function script
code = ph20_set3-2.py
makeplotz = python $(code)


# generate pdf
results.pdf : ph20_set3-2.tex $(code) png 
	$(makepdf) $<

# produce plots:
all : results.pdf

# look at pdf
view : 
	open results.pdf

# grouping the plots to check for so it's easier to write in a dependency statement
plotz1 : plot1.png plot2.png plot3.png plot4.png plot5.png

plotz2 : plot6.png plot7.png plot8.png plot9.png plot10.png

plotz3 : plot11.png plot12.png plot13.png plot14.png plot15.png plot16.png

plotz : plotz1 plotz2 plotz3


clean :
	rm -f results.pdf *.png

#remaking the plots if they're not all there/aren't made
.PHONY png : $(code)
	$(makeplotz)

