leer_datos_partidos = function(){

    datos = read.csv("./datos/partidos.csv", header = TRUE)
    datos = datos[,-c(1,2,3)]
    etiquetas = datos[,1]
    list(datos=datos[,-1],etiquetas=etiquetas)
    
}

datos = leer_datos_partidos()
etiquetas = datos$etiquetas
datos = datos$datos
