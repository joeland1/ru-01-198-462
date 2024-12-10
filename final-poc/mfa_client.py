import requests
import torchvision 
import pickle

FILENAME='../auto-save/p1-0.png'
url = 'http://127.0.0.1:5000/login'
my_img = {'image': open(FILENAME,'rb')}
data = {"uname": 'p1',"pw":"password1"}
r = requests.post(url, files=my_img,data=data)

# convert server response into JSON format.
print(r.json())