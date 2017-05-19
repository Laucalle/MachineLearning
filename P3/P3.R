set.seed(3)

clasificar_recta = function(punto, recta){
    
    sign(punto[2] - recta[1]*punto[1] - recta[2])
    
}

error_cuadratico = function(clasificados,reales){
    
    squared_error = function(pair_h_y){ # (h(x) - y_n)^2
        (pair_h_y[1]-pair_h_y[2])^2
    }
    
    if(min(reales) == 0){
        
        reales[reales == 0] = -1
        clasificados[clasificados < 0.5] = -1
        clasificados[clasificados >= 0.5] = 1
        
    }
    
    pos_errores = which(clasificados != reales) # Qué etiquetas ha clasificado mal.
    pares_errores = cbind(clasificados[pos_errores],reales[pos_errores],deparse.level=0)
    # Sumamos los errores al cuadrado y hacemos la media.
    sum(apply(pares_errores,1,squared_error))/length(clasificados)
}

porcentaje_error = function(clasificados,reales,umbral=0.5){
    
    if(min(reales) == 0){
        
        reales[reales == 0] = -1
        clasificados[clasificados < umbral] = -1
        clasificados[clasificados >= umbral] = 1
        
    }
    
    t = table(clasificados,reales)
    100*(1-sum(diag(t))/sum(t))
    
}

# Clasificación: Email Spam

leer_datos_spam = function(){
    
    datos = read.table("./datos/spam.data")
    conjuntos = read.table("./datos/spam.traintest")
    etiquetas = datos[,ncol(datos)]
    list(datos=datos[,-ncol(datos)],etiquetas=etiquetas,conjuntos=conjuntos)
    
}

# Preprocesamiento (centrado, escalado, análisis de componentes principales...)
preprocesar_datos = function(datos,metodos,umbral_varianza){
    
    preprocess_obj = preProcess(datos,method=metodos,umbral_varianza)
    nuevosDatos = predict(preprocess_obj,datos)
    
}

# Evalúa la regresión para unos datos
evaluar_regresion = function(regresion,datos){
    
    predict(regresion,datos) # Los datos no deben incluir las etiquetas
    
}

# Evalúa una regresión lineal dada una fórmula y unos datos de entrenamiento 
evalua_lm = function(formula,datos,subconjunto){
    reg_lin = do.call("lm", list(formula=formula, data=substitute(datos), subset=substitute(subconjunto)))
    prediccion_test = evaluar_regresion(reg_lin,datos[-subconjunto,-ncol(datos)])
    porc_error = porcentaje_error(prediccion_test,datos[-subconjunto,ncol(datos)])
    list(formula=formula, error = porc_error)
}

evalua_glm = function(formula,datos,subconjunto){
    reg_lin = do.call("glm", list(formula=formula, data=substitute(datos), subset=substitute(subconjunto),family=binomial()))
    prediccion_test = evaluar_regresion(reg_lin,datos[-subconjunto,-ncol(datos)])
    porc_error = porcentaje_error(prediccion_test,datos[-subconjunto,ncol(datos)])
    list(formula=formula, error = porc_error)
}

spam = leer_datos_spam()
# Preprocesar los datos 
spam_procesado = preprocesar_datos(spam$datos,c("YeoJohnson","center","scale","pca"),0.8)
# Obtener el conjunto de entrenamiento
indices_train = which(spam$conjuntos == 0)
# Añadir las etiquetas para la regresión lineal
spam_procesado = cbind(spam_procesado,spam$etiquetas)
colnames(spam_procesado)[ncol(spam_procesado)] = "etiquetas"
# Hacer regresión lineal de las etiquetas según las otras características
reg_lin_spam = lm(etiquetas~.,data=spam_procesado,subset=indices_train)
# Obtener predicciones de la regresión sobre los datos de test
prediccion_test = evaluar_regresion(reg_lin_spam,spam_procesado[-indices_train,-ncol(spam_procesado)])
# Porcentaje de error en el conjunto de test
porc_error = porcentaje_error(prediccion_test,spam_procesado[-indices_train,ncol(spam_procesado)])
# Buscamos exhaustivamente conjuntos de características que usar
max_caracteristicas = ncol(spam_procesado)-1
subsets_spam = regsubsets(etiquetas~.,data=spam_procesado[indices_train,],method="exhaustive",nvmax=max_caracteristicas)
# Obtenemos la matriz de características seleccionadas por grupos de tamaño desde 1 hasta nvmax
matriz_subconjuntos = summary(subsets_spam)$which[,-1]
# Guardamos, para cada fila, las columnas cuyas variables han sido seleccionadas.
seleccionados = apply(matriz_subconjuntos,1,which)
# Obtenemos los nombres de esas columnas (para utilizarlos en la regresión)
seleccionados = lapply(seleccionados,names)
# Construimos la suma de las variables que usaremos en la regresión lineal
seleccionados = mapply(paste,seleccionados,MoreArgs=list(collapse="+"))
# Construimos strings equivalentes a las fórmulas que usaremos en la regresión lineal
formulas = mapply(paste,rep("etiquetas~",max_caracteristicas),seleccionados,USE.NAMES = FALSE)
# Construimos objetos fórmula
formulas = apply(matrix(formulas,nrow=length(formulas)), 1, as.formula)
# Obtenemos los resultados de evaluar todos los modelos
ajustes = mapply(evalua_lm, formulas, MoreArgs = list(datos = spam_procesado, subconjunto = indices_train))