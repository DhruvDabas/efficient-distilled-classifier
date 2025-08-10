import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for


UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# student model 
class MobileNetV2Student(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV2Student, self).__init__()
        self.model = mobilenet_v2(weights=None)  # will use my best_student.pth weights
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# load trained model
student_model = MobileNetV2Student(num_classes=NUM_CLASSES).to(DEVICE)
checkpoint = torch.load("best_student.pth", map_location=DEVICE)
student_model.load_state_dict(checkpoint['model_state_dict'])
student_model.eval()


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    ),
])



## Flask app 
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = student_model(img_t)
        _, predicted = outputs.max(1)
        class_id = predicted.item()
    
    return CLASS_NAMES[class_id]

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            prediction = predict_image(filepath)
            return render_template("result.html", filename=file.filename, prediction=prediction)
    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return redirect(url_for("static", filename="uploads/" + filename))

if __name__ == "__main__":
    app.run(debug=True)