
<h1 align="center">
  <br>
  <a href="https://openpecha.org"><img src="https://avatars.githubusercontent.com/u/82142807?s=400&u=19e108a15566f3a1449bafb03b8dd706a72aebcd&v=4" alt="OpenPecha" width="150"></a>
  <br>
</h1>

<!-- Replace with 1-sentence description about what this tool is or does.-->

<h3 align="center">Use this repo template for all new Python projects.</h3>

## Description

Project description goes here.

## Project owner(s)

<!-- Link to the repo owners' github profiles -->

- [@10zinten](https://github.com/10zinten)
- [@evanyerburgh](https://github.com/evanyerburgh)

## Integrations

<!-- Add any intregrations here or delete `- []()` and write None-->

None
## Docs

1. To run the API locally 
    `uvicorn src.MonlamOCR.API:app --reload`
2. To login to the docker from the terminal and push the docker image build
    `docker login
     docker tag fastapi-ocr-app ta4tsering/fastapi-ocr-app:latest
     docker push ta4tsering/fastapi-ocr-app:latest
    `
3. To create a new build instance
    `docker buildx create --name mybuilder --use
     docker buildx inspect --bootstrap

4. Build a Multi-Architecture Docker image.
    `docker buildx build --platform linux/amd64,linux/arm64 -t ocr_pipeline_api --push .
    `
5. Install Docker on the server
    `sudo apt-get update
     sudo apt-get install -y docker.io
     sudo systemctl start docker
     sudo systemctl enable docker
     sudo systemctl start docker
     sudo systemctl enable docker
`
6. Pull your docker image from from docker hub
    `docker login
     docker pull ta4tsering/fastapi-ocr-app:latest
    `