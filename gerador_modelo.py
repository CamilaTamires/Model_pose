import torch
import torch.nn as nn
import math

class PoseAlertLegsAndArms(nn.Module):
    """
    Gera 14 alertas de postura. Agora aceita também a bbox do YOLO (yolo_bboxes: (N,4) em pixels)
    e usa ESSA bbox para:
      - 4 = afaste (bbox perto de QUALQUER borda)
      - 13 = aproximar_lateral (bbox pequena)
    Os demais alertas seguem baseados em keypoints.
    """

    def __init__(self,
                 angulo_max_fechadas: float = 7.0,
                 angulo_braco_lateral: float = 90.0,
                 angulo_braco_fechado: float = 60.0,
                 limite_aproximar: float = 0.12,
                 # limites do alerta 13 com bbox do YOLO (percentuais da imagem)
                 limite_largura_bbox_img: float = 0.18,
                 limite_altura_bbox_img: float  = 0.45,
                 # margem do alerta 4 com bbox do YOLO
                 margem_pct_altura_bbox: float = 0.12,  # 12% da ALTURA da bbox
                 pix_margem_min: float = 8.0):
        super().__init__()

        # Hiperparâmetros
        self.angulo_max_fechadas  = math.radians(angulo_max_fechadas)
        self.angulo_braco_lateral = math.radians(angulo_braco_lateral)
        self.angulo_braco_fechado = math.radians(angulo_braco_fechado)
        self.limite_aproximar     = limite_aproximar

        # Para alertas 4 e 13 via YOLO bbox
        self.limite_largura_bbox_img = limite_largura_bbox_img
        self.limite_altura_bbox_img  = limite_altura_bbox_img
        self.margem_pct_altura_bbox  = margem_pct_altura_bbox
        self.pix_margem_min          = pix_margem_min

        # Índices COCO-like
        self.points = {
            "nariz": 0, "olho_esq": 1, "olho_dir": 2,
            "orelha_esq": 3, "orelha_dir": 4,
            "ombro_esq": 5, "ombro_dir": 6,
            "cotovelo_esq": 7, "cotovelo_dir": 8,
            "pulso_esq": 9, "pulso_dir": 10,
            "quadril_esq": 11, "quadril_dir": 12,
            "joelho_esq": 13, "joelho_dir": 14,
            "tornozelo_esq": 15, "tornozelo_dir": 16
        }

        # Buffers (constantes)
        self.register_buffer("eixo_vertical", torch.tensor([0.0, 1.0]))
        self.register_buffer("offset_pes",    torch.tensor([0.0, 2.0]))

    def forward(self, keypoints, image_width, image_height, num_pessoas_tensor, yolo_bboxes):
        """
        Inputs:
          keypoints:         (N,17,3) em [0,1]  (x_norm,y_norm,conf)
          image_width:        escalar (float)
          image_height:       escalar (float)
          num_pessoas_tensor: escalar (float)
          yolo_bboxes:       (N,4) em pixels [x1,y1,x2,y2] (mesma ordem de keypoints!)
        Saída:
          Tensor (14,)
        """
        EPS = keypoints.new_tensor(1e-6)
        ACOS_EPS = keypoints.new_tensor(1e-6)

        N = keypoints.shape[0]
        key = keypoints[0]  # usando a 1ª pessoa para os cálculos internos

        # Converte keypoints normalizados -> pixels
        scale = torch.stack([image_width, image_height]).view(1, 2).to(keypoints)
        key_px = key.clone()
        key_px[:, 0:2] *= scale

        conf  = key[:, 2]
        valid = (conf > 0.5).float()

        # === ALERTA 1: mais de uma pessoa ===
        alerta_mais_de_uma_pessoa = (num_pessoas_tensor > 1.0).float()

        # Pontos para alguns cálculos
        nariz    = key_px[self.points["nariz"], :2]
        olho_esq = key_px[self.points["olho_esq"], :2]
        olho_dir = key_px[self.points["olho_dir"], :2]

        dist_olhos_horizontal = torch.abs(olho_esq[0] - olho_dir[0])

        # Pés "virtuais" (para alerta 11)
        tdir = key_px[self.points["tornozelo_dir"], :2]
        tesq = key_px[self.points["tornozelo_esq"], :2]
        pes_dir = tdir + self.offset_pes * dist_olhos_horizontal
        pes_esq = tesq + self.offset_pes * dist_olhos_horizontal

        def ponto_dentro(p):
            return ((p[0] >= 0) & (p[0] <= image_width) &
                    (p[1] >= 0) & (p[1] <= image_height)).bool()

        # === ALERTA 0: pernas_fechadas (ângulo c/ eixo vertical)
        qesq = key_px[self.points["quadril_esq"], :2]
        qdir = key_px[self.points["quadril_dir"], :2]
        qctr = (qesq + qdir) / 2.0

        def angulo_vetor(ponto):
            v = ponto - qctr
            v = v / (torch.norm(v) + EPS)
            cosang = torch.clamp((v * self.eixo_vertical).sum(), -1.0 + ACOS_EPS, 1.0 - ACOS_EPS)
            return torch.acos(cosang)

        ang_esq = angulo_vetor(tesq)
        ang_dir = angulo_vetor(tdir)
        pernas_fechadas = ((ang_esq < self.angulo_max_fechadas) |
                           (ang_dir < self.angulo_max_fechadas)).float()

        # === ALERTAS 2 e 3: braços
        def angulo_lateral(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            v1 = v1 / (torch.norm(v1) + EPS)
            v2 = v2 / (torch.norm(v2) + EPS)
            cosang = torch.clamp((v1 * v2).sum(), -1.0 + ACOS_EPS, 1.0 - ACOS_EPS)
            return torch.acos(cosang)

        def alerta_braco(lado):
            ang = angulo_lateral(
                key_px[self.points[f"quadril_{lado}"], :2],
                key_px[self.points[f"cotovelo_{lado}"], :2],
                key_px[self.points[f"pulso_{lado}"], :2]
            )
            lat = (ang > self.angulo_braco_lateral).float()
            fech = (ang < self.angulo_braco_fechado).float()
            return lat, fech

        lat_esq, fech_esq = alerta_braco("esq")
        lat_dir, fech_dir = alerta_braco("dir")
        bracos_laterais = (lat_esq + lat_dir > 0).float()
        bracos_fechados = (fech_esq + fech_dir > 0).float()

        # === ALERTA 5: aproximar (ombros muito próximos na normalização)
        ombro_esq = key[self.points["ombro_esq"], 0]
        ombro_dir = key[self.points["ombro_dir"], 0]
        aproximar = (torch.abs(ombro_dir - ombro_esq) < self.limite_aproximar).float()

        # === ALERTA 6 / 12 (placeholders simples)
        vire_costa = (valid[self.points["olho_esq"]] * valid[self.points["olho_dir"]]).float()
        frente = 1.0 - vire_costa

        # === 7..10: inicialização
        right_bracoalto = keypoints.new_tensor(0.0)
        right           = keypoints.new_tensor(0.0)
        left_bracoalto  = keypoints.new_tensor(0.0)
        left            = keypoints.new_tensor(0.0)

        # ------------------------------------------------------------------
        # >>> ALTERAÇÃO SOLICITADA: ALERTA 8 (right) com lógica "legada"
        #     (mantive right_bracoalto inalterado, como você pediu)
        # ------------------------------------------------------------------
        def alerta_bracos_fora_direita_legacy(key_px_, valid_, points_, image_height_):
            altura_permitida = 0.1 * image_height_
            def lado(lado_tag):
                ombro = key_px_[points_[f"ombro_{lado_tag}"], :2]
                pulso = key_px_[points_[f"pulso_{lado_tag}"], :2]
                mask  = valid_[points_[f"ombro_{lado_tag}"]] * valid_[points_[f"pulso_{lado_tag}"]]
                altura_ok = (torch.abs(pulso[1] - ombro[1]) < altura_permitida).float()
                no_lado   = (pulso[0] > ombro[0]).float()  # direita: pulso.x > ombro.x
                return altura_ok * mask, no_lado * mask
            altura_esq, lado_esq = lado("esq")
            altura_dir, lado_dir = lado("dir")
            # retorna (right_bracoalto, right)
            return 1.0 - (altura_esq * altura_dir), 1.0 - (lado_esq * lado_dir)

        # calcula apenas o 'right' (alerta 8) com a regra antiga
        _, right = alerta_bracos_fora_direita_legacy(key_px, valid, self.points, image_height)
        # ------------------------------------------------------------------

        # === ALERTA 11: distancia_lateral (cabeça virtual OU pés fora)
        orelha_esq = key_px[self.points["orelha_esq"], :2]
        orelha_dir = key_px[self.points["orelha_dir"], :2]
        desloc_esq = 5.0 * torch.abs(orelha_esq[1] - olho_esq[1])
        desloc_dir = 5.0 * torch.abs(orelha_dir[1] - olho_dir[1])
        pto_esq = torch.stack([orelha_esq[0], orelha_esq[1] - desloc_esq])
        pto_dir = torch.stack([orelha_dir[0], orelha_dir[1] - desloc_dir])

        dentro_cabeca_esq = ponto_dentro(pto_esq)
        dentro_cabeca_dir = ponto_dentro(pto_dir)
        dentro_pes_esq    = ponto_dentro(pes_esq)
        dentro_pes_dir    = ponto_dentro(pes_dir)
        distancia_lateral = (~((dentro_cabeca_esq | dentro_cabeca_dir) &
                               dentro_pes_esq & dentro_pes_dir)).float()

        # ==========================================================
        # USO DA BBOX DO YOLO PARA ALERTAS 4 e 13
        # ==========================================================
        if yolo_bboxes is None or yolo_bboxes.numel() == 0:
            x1 = y1 = x2 = y2 = None
        else:
            bb = yolo_bboxes[0]
            x1, y1, x2, y2 = bb[0], bb[1], bb[2], bb[3]

        afaste = keypoints.new_tensor(0.0)
        aproximar_lateral = keypoints.new_tensor(0.0)

        if x1 is not None:
            w_bbox = torch.clamp(x2 - x1, min=1.0)
            h_bbox = torch.clamp(y2 - y1, min=1.0)

            margem_limite = torch.maximum(
                keypoints.new_tensor(self.pix_margem_min),
                keypoints.new_tensor(self.margem_pct_altura_bbox) * h_bbox
            )
            dist_esq  = x1
            dist_dir  = image_width  - x2
            dist_top  = y1
            dist_base = image_height - y2
            menor_dist = torch.min(torch.stack([dist_esq, dist_dir, dist_top, dist_base]))
            afaste = (menor_dist < margem_limite).float()

            cond_w = (w_bbox < self.limite_largura_bbox_img * image_width)
            cond_h = (h_bbox < self.limite_altura_bbox_img  * image_height)
            aproximar_lateral = (cond_w | cond_h).float()
        else:
            xs = key_px[:, 0]
            ys = key_px[:, 1]
            mask = (valid > 0).bool()
            INF = key_px.new_full((), float("inf"))
            NINF = key_px.new_full((), float("-inf"))
            x_min = torch.min(torch.where(mask, xs, INF))
            x_max = torch.max(torch.where(mask, xs, NINF))
            y_min = torch.min(torch.where(mask, ys, INF))
            y_max = torch.max(torch.where(mask, ys, NINF))
            w_bbox = torch.clamp(x_max - x_min, min=1.0)
            h_bbox = torch.clamp(y_max - y_min, min=1.0)

            margem_limite = torch.maximum(
                keypoints.new_tensor(self.pix_margem_min),
                keypoints.new_tensor(self.margem_pct_altura_bbox) * h_bbox
            )
            menor_dist = torch.min(torch.stack([x_min, image_width - x_max, y_min, image_height - y_max]))
            afaste = (menor_dist < margem_limite).float()

            cond_w = (w_bbox < self.limite_largura_bbox_img * image_width)
            cond_h = (h_bbox < self.limite_altura_bbox_img  * image_height)
            aproximar_lateral = (cond_w | cond_h).float()

        # === Empilha todos os 14 alertas
        return torch.stack([
            pernas_fechadas,           # 0
            alerta_mais_de_uma_pessoa, # 1
            bracos_laterais,           # 2
            bracos_fechados,           # 3
            afaste,                    # 4  (YOLO bbox)
            aproximar,                 # 5
            vire_costa,                # 6
            right_bracoalto,           # 7  (mantido)
            right,                     # 8  <<< alterado (lógica LEGADA)
            left_bracoalto,            # 9
            left,                      # 10
            distancia_lateral,         # 11
            frente,                    # 12
            aproximar_lateral          # 13 (YOLO bbox)
        ], dim=0)


if __name__ == "__main__":
    model = PoseAlertLegsAndArms().eval()

    # Dummy export (N=2 pessoas)
    inputs = {
        "keypoints": torch.rand(2, 17, 3),
        "image_width": torch.tensor(640.0),
        "image_height": torch.tensor(480.0),
        "num_pessoas_tensor": torch.tensor(2.0),
        "yolo_bboxes": torch.tensor([[50.0, 60.0, 250.0, 420.0],
                                     [120.0, 80.0, 300.0, 460.0]])
    }

    with torch.no_grad():
        torch.onnx.export(
            model,
            (inputs["keypoints"], inputs["image_width"], inputs["image_height"],
             inputs["num_pessoas_tensor"], inputs["yolo_bboxes"]),
            "pose_alert_box.onnx",
            input_names=list(inputs.keys()),
            output_names=["alertas"],
            dynamic_axes={
                "keypoints":   {0: "num_pessoas"},
                "yolo_bboxes": {0: "num_pessoas"},
            },
            opset_version=17,
        )
    print("✅ Modelo exportado com bbox do YOLO embutida (pose_alert_box.onnx)")
