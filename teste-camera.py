import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort
import pyautogui

# ============================================================
# Caminhos dos modelos 
# ============================================================
modelo_pose = "/home/inovia/Documentos/rep-app-pose/yolo11n-pose.onnx"
modelo_alerta = "/home/inovia/Documentos/rep-app-pose/pose_alert_box.onnx"

# ============================================================
# Utilitários
# ============================================================
def scalar_f32(x):
    """Garante ESCALAR float32 rank-0 para ONNX (shape: ())."""
    return np.array(x, dtype=np.float32).reshape(())

def ensure_kps_n173_float32(kps_list_px, W, H):
    """
    Recebe lista de (17,3) em pixels (x,y,conf) e:
      - empilha para (N,17,3)
      - normaliza x por W e y por H
      - retorna float32 contíguo
    """
    kps_array = np.array(kps_list_px, dtype=np.float32)  # (N,17,3) em pixels
    kps_array[:, :, 0] /= float(W)
    kps_array[:, :, 1] /= float(H)
    return np.ascontiguousarray(kps_array, dtype=np.float32)

def ensure_bboxes_n4_float32(bboxes_list_px):
    """
    Recebe lista de bboxes em pixels [x1,y1,x2,y2] e empilha em (N,4) float32 contíguo.
    """
    if not bboxes_list_px:
        return np.zeros((0, 4), dtype=np.float32)
    arr = np.array(bboxes_list_px, dtype=np.float32)  # (N,4)
    return np.ascontiguousarray(arr, dtype=np.float32)

# ============================================================
# Carrega modelos
# ============================================================
print(f"Loading {modelo_pose} for ONNX Runtime inference...")
model = YOLO(modelo_pose)

sess = ort.InferenceSession(modelo_alerta, providers=["CPUExecutionProvider"])
print("Using ONNX Runtime CPUExecutionProvider")

# (Opcional) Inspecionar I/O do ONNX para confirmar shapes/nomes
# print("== ONNX PoseAlert I/O ==")
# for i in sess.get_inputs():
#     print("IN :", i.name, "shape:", i.shape, "type:", i.type)
# for o in sess.get_outputs():
#     print("OUT:", o.name, "shape:", o.shape, "type:", o.type)

# ============================================================
# Tela / janela
# ============================================================
screen_width, screen_height = pyautogui.size()

modo = input("Selecione o modo de pose ('front', 'back', 'right' ou 'left'): ").strip().lower()
if modo not in ["front", "back", "right", "left"]:
    raise ValueError("Modo inválido. Escolha entre 'front', 'back', 'right' ou 'left'.")

cap = cv2.VideoCapture(1)  # troque o índice se necessário (0, 1, ...)
if not cap.isOpened():
    raise RuntimeError("Erro ao abrir a câmera!")

cv2.namedWindow("Alerta de Postura", cv2.WINDOW_NORMAL)

# ============================================================
# Parâmetros fixos de exibição 
# ============================================================
target_height = 768 
ret, test_frame = cap.read()
if not ret:
    raise RuntimeError("Não foi possível capturar o primeiro frame para calcular proporção.")
orig_h, orig_w = test_frame.shape[:2]
aspect_ratio = orig_w / orig_h
target_width = int(target_height * aspect_ratio)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('teste_postura.mp4', fourcc, 20.0, (target_width, target_height))

# ============================================================
# Conexões dos keypoints (COCO-like)
# ============================================================
CONEXOES = [
    (0, 1), (1, 3), (0, 2), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# ============================================================
# Loop principal
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_height, image_width = frame.shape[:2]

    # Inferência YOLO Pose
    results = model.predict(frame, imgsz=640, verbose=False)
    alertas = np.zeros(14, dtype=np.float32)
    pessoas_detectadas = False
    num_pessoas = 0

    if results:
        result = results[0]
        keypoints_tensor = result.keypoints
        boxes = result.boxes

        all_kps_px = []   # lista de (17,3) em pixels (x,y,conf)
        all_bboxes = []   # lista de (4) em pixels [x1,y1,x2,y2]

        # Garante que temos keypoints e boxes (e que suas contagens casam)
        if (keypoints_tensor is not None and len(keypoints_tensor.xy) > 0
                and boxes is not None and len(boxes) > 0):
            xyxy = boxes.xyxy.cpu().numpy()  # (M,4) em pixels

            # IMPORTANTE: p_idx dos keypoints corresponde à mesma pessoa em xyxy[p_idx]
            for p_idx in range(len(keypoints_tensor.xy)):
                kps_xy = keypoints_tensor.xy[p_idx].cpu().numpy()            # (17,2) pixels
                confs  = keypoints_tensor.data[p_idx, :, 2].cpu().numpy()    # (17,)
                # filtro: pelo menos 5 keypoints confiáveis
                if np.sum(confs > 0.5) < 5:
                    continue

                # empilha (x,y,conf)
                kps_conf = np.concatenate([kps_xy, confs[:, None]], axis=1)  # (17,3)
                all_kps_px.append(kps_conf)

                # bbox correspondente ao mesmo índice p_idx
                all_bboxes.append(xyxy[p_idx])  # [x1,y1,x2,y2]

        if all_kps_px:
            pessoas_detectadas = True
            num_pessoas = len(all_kps_px)

            # Normaliza keypoints para [0,1] e empilha
            kps_array = ensure_kps_n173_float32(all_kps_px, image_width, image_height)  # (N,17,3)
            # Empilha bboxes (N,4) float32
            bboxes_array = ensure_bboxes_n4_float32(all_bboxes)

            # ---------- Desenho (debug visual) ----------
            # desenha skeleton
            for person in kps_array:
                for x, y, conf in person:
                    if conf > 0.5:
                        cx, cy = int(x * image_width), int(y * image_height)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                for idx1, idx2 in CONEXOES:
                    x1, y1, c1 = person[idx1]
                    x2, y2, c2 = person[idx2]
                    if c1 > 0.5 and c2 > 0.5:
                        p1 = (int(x1 * image_width), int(y1 * image_height))
                        p2 = (int(x2 * image_width), int(y2 * image_height))
                        cv2.line(frame, p1, p2, (255, 0, 0), 2)

            # desenha as bboxes do YOLO e linhas de margem relativas à ALTURA da bbox (debug)
            for bb in all_bboxes:
                x1, y1, x2, y2 = bb.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                h_bbox = max(1, y2 - y1)
                margem_px = max(8, int(0.10 * h_bbox))  # igual ao usado no modelo (10% da altura, min 8px)
                # linhas de margem "globais" (para referência)
                cv2.line(frame, (margem_px, 0), (margem_px, image_height), (0, 255, 255), 1)
                cv2.line(frame, (image_width - margem_px, 0), (image_width - margem_px, image_height), (0, 255, 255), 1)
                cv2.line(frame, (0, margem_px), (image_width, margem_px), (0, 255, 255), 1)
                cv2.line(frame, (0, image_height - margem_px), (image_width, image_height - margem_px), (0, 255, 255), 1)

            # ---------- Monta inputs ONNX ----------
            onnx_inputs = {
                "keypoints": kps_array,                         # (N,17,3) normalizado
                "image_width":  scalar_f32(image_width),        # escalar
                "image_height": scalar_f32(image_height),       # escalar
                "num_pessoas_tensor": scalar_f32(num_pessoas),  # escalar
                "yolo_bboxes": bboxes_array,                    # (N,4) pixels
            }

            # Execução
            try:
                alertas_out = sess.run(None, onnx_inputs)[0]  # (14,) ou (B,14)
            except Exception as e:
                print("\n[ERRO ONNX] Falha no sess.run:", e)
                raise e

            if alertas_out.ndim == 1:
                alertas_out = alertas_out.reshape(1, -1)
            # agregação por pessoa (OR saturado 0/1)
            alertas = np.clip(alertas_out.sum(axis=0), 0, 1).astype(np.float32)

    # ------------------------------------------------------------
    # Nomes dos alertas — 14 saídas (0..13)
    # ------------------------------------------------------------
    alertas_nome = [
        ("Afaste as pernas", alertas[0]),
        ("Mais de uma pessoa detectada", alertas[1]),
        ("Abaixe os braços", alertas[2]),
        ("Abra os braços", alertas[3]),
        ("Afaste-se da camera (margem YOLO bbox)", alertas[4]),  #bbox do YOLO
        ("Aproxime-se da camera", alertas[5]),
        ("Vire de costas", alertas[6]),
        ("Estique bracos na altura do ombro", alertas[7]),  # placeholders
        ("Vire para direita", alertas[8]),                  # placeholders
        ("Estique bracos na altura do ombro", alertas[9]),  # placeholders
        ("Vire à esquerda", alertas[10]),                   # placeholders
        ("Afaste-se (pontos virtuais)", alertas[11]),
        ("Fique de frente com a camera", alertas[12]),
        ("Aproxime-se (bbox pequena YOLO)", alertas[13]),   #bbox do YOLO
    ]

    # Seleção por modo (ajuste conforme preferir)
    if modo == "back":
        alertas_visiveis = [alertas_nome[i] for i in [0, 1, 2, 3, 4, 6]]
    elif modo == "right":
        alertas_visiveis = [alertas_nome[i] for i in [1, 7, 8, 4, 13]]
    elif modo == "left":
        alertas_visiveis = [alertas_nome[i] for i in [1, 9, 10, 4]]
    elif modo == "front":
        alertas_visiveis = [alertas_nome[i] for i in [0, 1, 2, 3, 4, 12, 13]]

    alertas_ativos = [txt for txt, val in alertas_visiveis if val > 0.0]

    # ------------------------------------------------------------
    # Redimensiona para altura padrão (768) mantendo proporção
    # ------------------------------------------------------------
    frame_resized = cv2.resize(frame, (target_width, target_height))
    out.write(frame_resized)

    if not pessoas_detectadas:
        texto_exibir = "Nenhuma pessoa encontrada"
        cor_texto = (0, 0, 255)
    elif alertas_ativos:
        texto_exibir = None
        for i, txt in enumerate(alertas_ativos):
            cv2.putText(frame_resized, f"Alerta: {txt}", (40, 120 + i * 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5, cv2.LINE_AA)
    else:
        texto_exibir = "Postura OK"
        cor_texto = (0, 255, 0)

    if texto_exibir:
        cv2.putText(frame_resized, texto_exibir, (40, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, cor_texto, 6, cv2.LINE_AA)

    cv2.imshow("Alerta de Postura", frame_resized)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

# ============================================================
# Libera recursos
# ============================================================
cap.release()
out.release()
cv2.destroyAllWindows()