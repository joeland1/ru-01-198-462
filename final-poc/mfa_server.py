from flask import Flask, request, jsonify
import torch

import torchvision.transforms as transforms
from PIL import Image
import pickle
import torchvision
import PIL
# Define a transform to convert PIL 
# image to a Torch tensor
transform = transforms.Compose([
    transforms.PILToTensor()
])

# transform = transforms.PILToTensor()
# Convert the PIL image to Torch tensor

app = Flask(__name__)

model_ft = torch.load('../train_model_full.pt',weights_only=False)
model_ft.eval()

PERSON_DB ={
    "p1": {
        "pw": "password1",
        "index": 0
    },
    "p2": {
        "pw": "password2",
        "index": 1
    }
}
# obviously not train
THRESH=0.95

@app.route("/login", methods=["POST"])
def process_image():
    uname = request.form.get('uname')
    pw = request.form.get('pw')
    
    file = request.files['image']
    file.save('something.png')                                    # hacky fix, im not sure of the right way to do this...
    img_tensor = torchvision.io.read_image('something.png')
    #img_tensor = pickle.load(file.stream)
    # Read the image via file.stream
    #img = Image.open(file.stream)
    #img_tensor = transform(img)

    outputs = model_ft(torch.stack( [img_tensor.to('mps').float() / 255.0], dim=0) )
    probs = torch.nn.functional.softmax(outputs, dim=1)
    
    if uname not in PERSON_DB:
        return jsonify({"error":"bad uname"})
    
    person = PERSON_DB[uname]
    print(person)
    if person["pw"] != pw:
        return jsonify({"error":"bad pw"})
    
    valid_adv_image = (probs.argmax(dim=-1) == person["index"])
    all_adv_image_valid = torch.all(valid_adv_image)
    
    all_adv_image_gt_thresh = torch.all(probs[:,person["index"]] > THRESH)
    if all_adv_image_valid and all_adv_image_gt_thresh:
        return jsonify({"cookie or token": "random number 123"})
    
    probs.argmax(dim=-1)
    return jsonify({"error": "didnt pass mfa"})


if __name__ == "__main__":
    app.run(debug=True)
