# Specify the default target
all: run

# Specify the target to run the main script
run: script.py
	python script.py

# Specify a target to clean up generated files
clean:
	rm -f *.pyc

# Specify a target to run tests
test:
	python -m unittest discover tests

# Specify a target to install dependencies
install:
	pip install -r requirements.txt
