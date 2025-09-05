from roboflow import Roboflow
import cv2
import pytesseract

# Conectar Roboflow
rf = Roboflow(api_key="apikey de roboflow")
project = rf.project("license-plate-recognition-rxg4e")
model = project.version(11).model  # ajusta la versión según Roboflow

# Inferencia en imagen
predictions = model.predict("placa.png").json()

if len(predictions['predictions']) > 0:
    pred = predictions['predictions'][0]  # tomar primera predicción
    x_center, y_center, width, height = pred['x'], pred['y'], pred['width'], pred['height']

    img = cv2.imread("placa.png")

    # Convertir a coordenadas de esquina
    x1 = int(x_center - width/2)
    y1 = int(y_center - height/2)
    x2 = int(x_center + width/2)
    y2 = int(y_center + height/2)

    placa_crop = img[y1:y2, x1:x2]
    cv2.imwrite("placa_recortada.png", placa_crop)
    model.predict("placa.png").save("resultado.png")  # imagen con bounding box

    # OCR
    gray = cv2.cvtColor(placa_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    texto = pytesseract.image_to_string(thresh, config='--psm 7')
    print("Placa detectada:", texto.strip())

    # Guardar en archivo
    with open("placas_detectadas.txt", "a") as f:
        f.write(texto.strip() + "\n")
else:
    print("No se detectó ninguna placa")
