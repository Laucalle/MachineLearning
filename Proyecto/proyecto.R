set.seed(3)

leer_datos_partidos = function(){

    datos = read.csv("./datos/partidos.csv", header = TRUE)
    datos = datos[,-c(1,2,3)]
    etiquetas = datos[,1]
    list(datos=datos[,-1],etiquetas=etiquetas)
    
}

# Leemos los datos y los separamos de las etiquetas
datos = leer_datos_partidos()
etiquetas = datos$etiquetas
datos = datos$datos

# Buscamos las columnas que tienen valores perdidos
colnames(datos)[colSums(is.na(datos))>0]
# Se asume que son 0 (sets no jugados, errores no cometidos, subidas a la red no intentadas...)
datos[is.na(datos)] = 0
# Determinar quién ha ganado viendo los sets es trivial, así que quitamos dicha información
datos = subset(datos,select=-c(FNL1,FNL2)) # Número de sets ganados por cada uno
datos = subset(datos,select=-c(ST1.1,ST2.1,ST3.1,ST4.1,ST5.1,ST1.2,ST2.2,ST3.2,ST4.2,ST5.2)) # Juegos de cada set