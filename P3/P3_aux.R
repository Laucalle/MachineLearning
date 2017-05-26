set.seed(3)

error_cuadratico_medio = function(clasificados,reales){

    mean((clasificados-reales)^2)

}

leer_datos_ozono = function(){
    
    datos = read.table("./datos/LAozone.data",sep=",",head=T)
    etiquetas = datos[,1]
    list(datos=datos[,-1],etiquetas=etiquetas)
    
}

# Preprocesamiento (centrado, escalado, análisis de componentes principales...)
preprocesar_datos = function(datos,indices_train,metodos,umbral_varianza){
    
    preprocess_obj = preProcess(datos[indices_train,],method=metodos,umbral_varianza)
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
    error = error_cuadratico_medio(prediccion_test,datos[-subconjunto,ncol(datos)])
    list(formula=formula, error = error, reg = reg_lin)
}

# PRE: la columna de etiquetas se llama exactamente "etiquetas"
subconjuntos_formulas = function(datos,max_tam,metodo="exhaustive"){
    
    # Obtenemos los subconjuntos de variables
    subsets = regsubsets(etiquetas~.,data=datos,method=metodo,nvmax=max_tam)
    # Obtenemos la matriz de características seleccionadas por grupos de tamaño desde 1 hasta nvmax
    matriz_subsets = summary(subsets)$which[,-1]
    # Guardamos, para cada fila, las columnas cuyas variables han sido seleccionadas.
    seleccionados = apply(matriz_subsets,1,which)
    # Obtenemos los nombres de esas columnas (para utilizarlos en la regresión)
    seleccionados = lapply(seleccionados,names)
    # Construimos la suma de las variables que usaremos en la regresión lineal
    seleccionados = mapply(paste,seleccionados,MoreArgs=list(collapse="+"))
    # Construimos strings equivalentes a las fórmulas que usaremos en la regresión lineal
    formulas = mapply(paste,rep("etiquetas~",max_tam),seleccionados,USE.NAMES = FALSE)
    # Construimos objetos fórmula
    formulas = apply(matrix(formulas,nrow=length(formulas)), 1, as.formula)
    list(formulas=formulas,cp=summary(subsets)$cp,bic=summary(subsets)$bic)
    
}

ozono = leer_datos_ozono()
o_indexes_train = sample(nrow(ozono$datos),round(0.7*nrow(ozono$datos)))
o_labels = ozono$etiquetas

# Preprocesar con y sin PCA
o_procesados = preprocesar_datos(ozono$datos,o_indexes_train,c("YeoJohnson","center","scale","pca"),0.85)
o_procesados_sin_pca = preprocesar_datos(ozono$datos,o_indexes_train,c("YeoJohnson","center","scale"),0.85)
o_procesados = as.data.frame(poly(x=as.matrix(o_procesados[,-ncol(o_procesados)]),degree=2))
o_procesados_sin_pca = as.data.frame(poly(x=as.matrix(o_procesados_sin_pca[,-ncol(o_procesados_sin_pca)]),degree=2))

# Añadir las etiquetas para la regresión lineal
o_procesados = cbind(o_procesados,ozono$etiquetas)
o_procesados_sin_pca = cbind(o_procesados_sin_pca,ozono$etiquetas)
colnames(o_procesados)[ncol(o_procesados)] = "etiquetas"
colnames(o_procesados_sin_pca)[ncol(o_procesados_sin_pca)] = "etiquetas"

# Tamaño máximo de los conjuntos de características de regsubsets
o_max_caracteristicas = ncol(o_procesados)-1
o_max_caracteristicas_sin_pca = ncol(o_procesados_sin_pca)-1

# Calculamos los objetos fórmula para cada subconjunto de variables
o_seleccion_caracteristicas = subconjuntos_formulas(o_procesados[o_indexes_train,],o_max_caracteristicas)
o_formulas = o_seleccion_caracteristicas$formulas
o_seleccion_caracteristicas_sin_pca = subconjuntos_formulas(o_procesados_sin_pca[o_indexes_train,],o_max_caracteristicas_sin_pca,metodo="forward")
o_formulas_sin_pca = o_seleccion_caracteristicas_sin_pca$formulas

# Obtenemos los resultados de evaluar todos los modelos
ajustes = mapply(evalua_lm, o_formulas, MoreArgs = list(datos = o_procesados, subconjunto = o_indexes_train))
ajustes_sin_pca = mapply(evalua_lm, o_formulas_sin_pca, MoreArgs = list(datos = o_procesados_sin_pca, subconjunto = o_indexes_train))
plot(ajustes_sin_pca[3,11]$reg,which=c(1),pch=20,col="blue")