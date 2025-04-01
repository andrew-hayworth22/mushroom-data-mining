FLASK_APP = ./server/app.py
PORT = 5001

.PHONY: server
server:
	FLASK_APP=$(FLASK_APP) flask run --port $(PORT) --debug