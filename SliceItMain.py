import cv2
import numpy as np
import math
import streamlit as st

st.title("SliceIt - Reparto de comida")

# Subir imagen desde el móvil
uploaded_file = st.file_uploader("Sube una foto de tu plato", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    # Convertir a imagen OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    if img is None:
        st.error("No se pudo abrir la imagen. Sube un PNG/JPG válido.")
        st.stop()
    
    # Convertir a HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Sliders para ajustar rango de segmentación
    st.sidebar.header("Ajustes de segmentación")
    low_s = st.sidebar.slider("Saturación mínima", 0, 255, 40)
    low_v = st.sidebar.slider("Valor mínimo (brillo)", 0, 255, 40)
    
    lower = np.array([0, low_s, low_v])
    upper = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Suavizar y limpiar bordes
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # Detectar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        st.error("No se detectó ninguna comida con los parámetros actuales. Ajusta los sliders.")
        st.stop()
    
    main_contour = max(contours, key=cv2.contourArea)

    # Centroide
    M = cv2.moments(main_contour)
    if M["m00"] == 0:
        st.error("No se pudo calcular el centroide.")
        st.stop()
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Número de cortes
    n = st.slider("Número de personas", 2, 10, 4)
    height, width = mask.shape
    for i in range(n):
        angle = 2*math.pi*i/n
        for r in range(0, max(width, height)):
            x = int(cx + r*math.cos(angle))
            y = int(cy + r*math.sin(angle))
            if x < 0 or x >= width or y < 0 or y >= height:
                break
            if mask[y, x] == 0:
                break
        cv2.line(img, (cx, cy), (x, y), (255,0,0), 2)

    # Dibujar contorno principal y centroide
    cv2.drawContours(img, [main_contour], -1, (0,255,0), 3)
    cv2.circle(img, (cx, cy), 5, (0,0,255), -1)

    # Mostrar resultado
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Cortes de comida", use_column_width=True)


