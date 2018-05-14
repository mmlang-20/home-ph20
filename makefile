# $@ = target, $^ = all dep, $< = first dep

# insertcodehere script
code_var = insertcodehere.py
code_exe = python $(code_var)

# pdf function script
pdf_var = pdf_function.py
pdf_exe = python $(pdf_var)



# generate pdf
results.pdf : pdf_var program_1.py program_2.py
	python $< *.dat > $@

# produce plots:
.PHONY : dats

dats : program_1.dat program_2.dat

%.dat : home/ph20/ph20_set3.ipynb insertcodehere.py
	code_exe $<  $*.dat


.PHONY : clean

clean :
	rm -f *.dat
	rm -f results.pdf