import cv2
import numpy as np

img_path = "/home/martin/repos/SegmentacionEsferoides/outputs/predictions/default/20 - 4d 3T3_10x.tiff"
img = cv2.imread(img_path)

if img is None:
    print("No se pudo cargar la imagen.")
else:
    # Verde es C1, Rojo (o Azul dependiendo de RGB/BGR) es C2
    # cv2.imread carga en BGR.
    # utils.py dice: color_img[mask == 1] = (0, 255, 0) # Verde
    # color_img[mask == 2] = (255, 0, 0) # Azul en OpenCV BGR! Wait...
    # cv2.imwrite(out_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
    # En RGB, Verde es (0, 255, 0) y Rojo es (255, 0, 0).
    # Al convertir a BGR: Verde sigue en [:,:,1], Rojo pasa a [:,:,2].

    # Vamos a recrear la mascara de clases
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Verde (C1)
    c1_pixels = (img[:,:,1] > 127) & (img[:,:,0] < 127) & (img[:,:,2] < 127)
    mask[c1_pixels] = 1

    # Rojo/Azul (C2) - vamos a buscar cualquier cosa que no sea verde ni negro
    c2_pixels = (img[:,:,2] > 127) & (img[:,:,1] < 127)
    mask[c2_pixels] = 2

    c3_pixels = (img[:,:,0] > 127) & (img[:,:,1] > 127) & (img[:,:,2] > 127)
    mask[c3_pixels] = 3

    print(f"C1 pixels: {np.sum(mask==1)}")
    print(f"C2 pixels: {np.sum(mask==2)}")
    print(f"C3 pixels: {np.sum(mask==3)}")

    # Encontrar C1 dentro de C2/C3
    # Podemos hacer esto encontrando las manchas C1, y viendo si están rodeadas por C2/C3
    # OpenCV hierarchy:
    # 1. Crear una mascara binaria de "Todo lo que es C1, C2 o C3"
    foreground = (mask > 0).astype(np.uint8)

    contours, hierarchy = cv2.findContours(foreground, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    holes = 0
    c1_in_holes = 0
    if hierarchy is not None:
        for i, h in enumerate(hierarchy[0]):
            # h = [Next, Previous, First_Child, Parent]
            # Si Parent != -1, este contorno es un agujero (hijo) dentro de un objeto
            if h[3] != -1:
                holes += 1
                # Crear mascara de este agujero
                hole_mask = np.zeros_like(mask)
                cv2.drawContours(hole_mask, [contours[i]], -1, 1, thickness=cv2.FILLED)

                # Chequear que clases hay dentro del agujero
                pixels_in_hole = mask[hole_mask == 1]
                if np.any(pixels_in_hole == 1):
                    c1_in_holes += 1

    print(f"Total de 'agujeros' encontrados en las masas: {holes}")
    print(f"Agujeros que contienen píxeles C1 (Verdes): {c1_in_holes}")
