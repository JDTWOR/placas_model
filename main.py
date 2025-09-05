from roboflow import Roboflow

rf = Roboflow(api_key="rf_iGsNbgEKxYaly45mJLSkx5WAl6T2")
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
model = project.version(1).model

# Inferencia en una imagen
prediction = model.predict("placa.jpg").json()
print(prediction)
