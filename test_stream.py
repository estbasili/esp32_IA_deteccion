import cv2
import numpy as np
import requests

URL = "http://192.168.0.17/capture"

print("✅ Iniciando captura por snapshots")

cv2.namedWindow("ESP32-CAM SNAPSHOT", cv2.WINDOW_NORMAL)

while True:
    try:
        response = requests.get(URL, timeout=2)

        img_np = np.frombuffer(response.content, dtype=np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        cv2.imshow("ESP32-CAM SNAPSHOT", frame)

        # 🔴 Detectar cierre de ventana
        if cv2.getWindowProperty("ESP32-CAM SNAPSHOT", cv2.WND_PROP_VISIBLE) < 1:
            print("🛑 Ventana cerrada")
            break

        # Salir con ESC
        if cv2.waitKey(1) == 27:
            break

    except Exception as e:
        print("⚠ Error:", e)

cv2.destroyAllWindows()
