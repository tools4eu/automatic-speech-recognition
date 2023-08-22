# Automatic speech recognition

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Automatic speech recognition uses [faster-whisper](https://github.com/guillaumekln/faster-whisper) to transcribe audio files and [pyannote-audio](https://github.com/pyannote/pyannote-audio) to add speaker diarization.

> :warning: **Always review transcriptions.** Transcriptions are done using AI models which are never 100% accurate.

The repo contains (will contain) code to run the software

- as a command-line tool
- as graphical interface
- as an inference API

## Installation

### Docker (recommended)

Build the Docker image

`docker build -t asr .` (make sure Docker is running on your system)

Run the Docker image, forward port 7860 (Gradio) and pass your GPU(s) to the container

`docker run -p 7860:7860 --gpus all asr`

Or in detached mode (in background)

`docker run -d -p 7860:7860 --gpus all asr`

You can check whether it is running with

`docker ps`

If you want to follow terminal output of a detached container, you can use

`docker logs -f <first n digits of the container id>`

The first time a transcription is requested, it will download the model.
To avoid this happening each time, make sure you stop and start the same container, instead of using

`docker run ...` again

use `docker start <first n digits of container>`

You can find the list of all containers, also stopped ones by using

`docker ps -a`

## License

GNU General Public License v3.0 or later

See [COPYING](COPYING) to see the full text.
