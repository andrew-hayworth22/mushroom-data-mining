FLASK_APP = ./server/app.py
PORT = 5001

.PHONY: server
server:
	FLASK_APP=$(FLASK_APP) flask run --port $(PORT) --debug

preprocess:
	python3 internal/data_processing.py

explore:
	python3 internal/data_exploration.py