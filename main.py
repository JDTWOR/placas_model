from roboflow import Roboflow

rf = Roboflow(api_key="stpB56ARIMQO2pA5pclc")
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
model = project.version(11).model

# Inferencia en una imagen
prediction = model.predict("placa.png").json()
print(prediction)
model.predict("placa.png").save("resultado.png")
