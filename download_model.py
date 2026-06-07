import requests

# Download emotion model prototxt
url1 = "https://raw.githubusercontent.com/spmallick/learnopencv/master/EmotionRecognitionUsingFacialLandmarks/deploy.prototxt"
response1 = requests.get(url1)
with open("deploy.prototxt", "wb") as f:
    f.write(response1.content)
print("Emotion model prototxt downloaded successfully.")

# Download emotion model caffemodel
url2 = "https://github.com/spmallick/learnopencv/raw/master/EmotionRecognitionUsingFacialLandmarks/deploy.caffemodel"
response2 = requests.get(url2)
with open("deploy.caffemodel", "wb") as f:
    f.write(response2.content)
print("Emotion model caffemodel downloaded successfully.")

# Download face cascade
url3 = "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml"
response3 = requests.get(url3)
with open("haarcascade_frontalface_default.xml", "wb") as f:
    f.write(response3.content)
print("Face cascade downloaded successfully.")