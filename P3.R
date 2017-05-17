set.seed(3)

clasificar_recta = function(punto, recta){
    
    sign(punto[2] - recta[1]*punto[1] - recta[2])
    
}

data_error = function(clasificados,reales){
    
    squared_error = function(pair_h_y){ # (h(x) - y_n)^2
        (pair_h_y[1]-pair_h_y[2])^2
    }
    
    pos_errores = which(clasificados != reales) # Qué etiquetas ha clasificado mal.
    pares_errores = cbind(clasificados[pos_errores],reales[pos_errores],deparse.level=0)
    # Sumamos los errores al cuadrado y hacemos la media.
    sum(apply(pares_errores,1,squared_error))/length(clasificados)
}

# Clasificación: Email Spam

leer_datos_spam = function(){
    
    datos = read.table("./datos/spam.data")
    etiquetas = datos[,length(datos)]
    list(datos=datos[,-length(datos)],etiquetas=etiquetas)
    
}

spam = leer_datos_spam()
spam$etiquetas[spam$etiquetas == 0] = -1
train_indexes = sample(nrow(spam$datos))