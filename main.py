import cv2

cap = cv2.VideoCapture("traffic.mp4")

#Detecção de objetos em movimento
object_detector = cv2.createBackgroundSubtractorMOG2(history=200,varThreshold=100)

#É preciso que seja true porque vai capturar uma frame de cada vez
while True:
    #inserindo uma moldura
    ret, frame = cap.read()

    height, width,_ = frame.shape
    print(height, width)


    recorte = frame[100:800,250:1000]





    #para extrair os objetos em movimento é preciso criar uma máscara, não da tela inteira mas sim do recorte
    #mascara = object_detector.apply(recorte)
    mascara = object_detector.apply(recorte)


    #vamos mostrar o que a mascara está fazendo,o objetivo da mascara é tornar tudo preto aqui que não precisamos
    #enquanto os objetos que queremos detectar em branco
    cv2.imshow("mascara", mascara)

    #Será preciso remover elementos pequenos da imagem para não causar falsa detecção como arvores,postes etc
    #Vamos encontrar os contornos
    contorno, _ = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    #vamos ver os contornos
    for cnt in contorno:
        #Filtrando os elementos maiores
        area = cv2.contourArea(cnt)
        if area > 300:

            #drawcontours desenha contorno no frame,cnt = todos os elementos com a cor verde com espessura 2
            #cv2.drawContours(recorte, [cnt], -1, (0, 255, 0), 2)
            x,y,w,z = cv2.boundingRect(cnt)
            cv2.rectangle(recorte, (x, y), (x + w, y + z), (0,255,0),3)

    #vamos mostrar o ponto cv2 em tempo real
    #cv2.imshow("Frame", frame)

    #vendo recorte
    cv2.imshow("recorte",recorte)

    key = cv2.waitKey(30)

    #se pressionar a tecla 's' a operação para
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

























