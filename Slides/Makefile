.PHONY: all clean

all: main.pdf

main.pdf: main.tex assets/l1_start.png assets/l2_end.png
	tectonic --keep-logs --synctex main.tex

clean:
	rm -f main.pdf main.log main.synctex.gz
