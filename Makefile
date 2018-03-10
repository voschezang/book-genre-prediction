start:
	jupyter notebook src/

install:
	pip3 install -r requirements.txt

install2:
	pip install -r requirements.txt

clean:
	find . -name \*.pyc -delete

cprofile:
	python3 -m cProfile -o test/program.prof src/main.py

snakeviz_:
	snakeviz test/program.prof

