# Plant detection (Webapp)

Web application of plant-detection repository

Check [licenses](./LICENSE.md)!

# How to run

## Web version

Clic [here](https://vegetation-detector.herokuapp.com/) to see the demo of plant-detection-web in action for images and video (<5 min).

## Docker version

To downloand the image and run the contaider in detach mode, run the code below.

```bash
docker container run -p 8501:8501 --rm -d pablogod/plantdetection:latest
```
To shutdown the docker type this:

```bash
docker ps -aq # Check which id was assigned for the plant-detection-web instance
docker stop <weird id of plant-detection-web> # Type the id
```

## Local computer

Run this code locally on Linux based distros:
```bash
# Clone and install requirements
git clone https://github.com/DZDL/plant-detection-web
cd plant-detection-web
pip3 install -r requirements.txt
# Run streamlit
streamlit run app.py
# Then a webapp will open, check console output.
```

## Deploy docker on Heroku

Only maintainers of the repository can do this.
```bash
heroku login
docker ps
heroku container:login
heroku container:push web -a vegetation-detector
heroku container:release web -a vegetation-detector
```

<!-- - Automatic deploy comming (working)

https://www.r-bloggers.com/2020/12/creating-a-streamlit-web-app-building-with-docker-github-actions-and-hosting-on-heroku/ -->

