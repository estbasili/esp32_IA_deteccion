import cv2
import numpy as np
import requests
from ultralytics import YOLO
import time

# ==============================
# CONFIGURACION
# ==============================
URL = "http://192.168.0.17/capture"

# Cargar modelo liviano
model = YOLO("yolov8n.pt")

print("✅ Sistema iniciado")
print("📡 Conectando a ESP32-CAM...")

cv2.namedWindow("ESP32 IA", cv2.WINDOW_NORMAL)

# Control de FPS
prev_time = 0

while True:
    try:
        # ==============================
        # CAPTURA DE IMAGEN
        # ==============================
        response = requests.get(URL, timeout=2)

        img_np = np.frombuffer(response.content, dtype=np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        # ==============================
        # IA DETECCION
        # ==============================
        results = model(frame, verbose=False)

        personas = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                # Clase 0 = persona
                if cls == 0:
                    personas += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, "Persona",
                                (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0,255,0),
                                2)

        # ==============================
        # INFO EN PANTALLA
        # ==============================
        # FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        cv2.putText(frame, f"Personas: {personas}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

        # ==============================
        # MOSTRAR VIDEO
        # ==============================
        cv2.imshow("ESP32 IA", frame)

        # Cerrar ventana
        if cv2.getWindowProperty("ESP32 IA", cv2.WND_PROP_VISIBLE) < 1:
            print("🛑 Ventana cerrada")
            break

        # ESC para salir
        if cv2.waitKey(1) == 27:
            break

    except Exception as e:
        print("⚠ Error:", e)

cv2.destroyAllWindows()