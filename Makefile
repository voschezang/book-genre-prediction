LOG_DIR := /tmp/ml_model_books

start:
	jupyter notebook src/

logs:
	rm -Rf $(LOG_DIR)/*
	tensorboard --logdir=$(LOG_DIR)

clear:
	rm -r $(LOG_DIR)/*

ls:
	ls $(LOG_DIR)/

deps:
	pip3 install -r requirements.txt
	python3 src/setup.py

install:
	pip3 install -r requirements.txt
	python3 src/setup.py

deps2:
	pip install -r requirements.txt
	python src/setup.py

predict:
	python3 src/main.py $(book)

clean:
	find . -name \*.pyc -delete
