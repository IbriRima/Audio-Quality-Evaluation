# Specify the target to run the main script with "mp3" to dowload mp3 as default action
run:
	python main.py mp3

# Specify a target to run the main script with the "audio_viz" action
audio_viz:
	python main.py audio_viz

# Specify a target to install dependencies
install:
	pip install -r requirements.txt
