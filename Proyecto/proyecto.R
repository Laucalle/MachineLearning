set.seed(3)

leer_datos_partidos = function(){

    datos = read.csv("./datos/partidos.csv", header = TRUE)
    datos = datos[,-c(1,2,3)]
    # Añadimos tipo de pista como característica
    pista_dura = c(rep(1,253),rep(0,252),rep(1,202),rep(0,236))
    tierra_batida = c(rep(0,253),rep(1,252),rep(0,438))
    #hierba = c(rep(0,707),rep(1,236))
    datos = cbind(datos,pista_dura,tierra_batida)
    etiquetas = datos[,1]
    indices = sample(nrow(datos),round(0.7*nrow(datos)))
    list(datos=datos[,-1],etiquetas=etiquetas,indices_train=indices)
    
}

# Preprocesamiento (centrado, escalado, análisis de componentes principales...)
preprocesar_datos = function(datos,indices_train,metodos,umbral_varianza=0.9){
    
    preprocess_obj = preProcess(datos[indices_train,],method=metodos,umbral_varianza)
    nuevosDatos = predict(preprocess_obj,datos)
    
}

# Evalúa la regresión para unos datos
evaluar_regresion = function(regresion,datos){
    
    predict(regresion,datos) # Los datos no deben incluir las etiquetas
    
}

porcentaje_error = function(clasificados,reales,fp=1,fn=1){
    
    reales[reales == 0] = -1
    t = table(clasificados,reales)
    total_predicciones = sum(t)
    t[1,2] = t[1,2]*fn
    t[2,1] = t[2,1]*fp
    100*(1-sum(diag(t))/total_predicciones)
    
}

categorizar = function(clasificados,umbral=0.5){
    
    clasificados[clasificados < umbral] = -1
    clasificados[clasificados >= umbral] = 1
    clasificados
    
}

# Evalúa una regresión lineal dada una fórmula y unos datos de entrenamiento 
evalua_lm = function(formula,datos,subconjunto,fp=1,fn=1){
    reg_lin = do.call("lm", list(formula=formula, data=substitute(datos), subset=substitute(subconjunto)))
    prediccion_test = evaluar_regresion(reg_lin,datos[-subconjunto,-ncol(datos)])
    porc_error = porcentaje_error(categorizar(prediccion_test),datos[-subconjunto,ncol(datos)],fp,fn)
    list(formula=formula, error = porc_error)
}

evalua_glm = function(formula,datos,subconjunto,fp=1,fn=1,familia=binomial()){
    reg_lin = do.call("glm", list(formula=formula, data=substitute(datos), subset=substitute(subconjunto),familia))
    prediccion_test = evaluar_regresion(reg_lin,datos[-subconjunto,-ncol(datos)])
    porc_error = porcentaje_error(categorizar(prediccion_test),datos[-subconjunto,ncol(datos)],fp,fn)
    list(formula=formula, error=porc_error,reg=reg_lin)
}

# Calcula objetos fórmula a partir de subconjuntos de variables
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

###############################################################################

# Leemos los datos y los separamos de las etiquetas
datos = leer_datos_partidos()
etiquetas = datos$etiquetas
indices_train = datos$indices_train
datos = datos$datos

# Buscamos las columnas que tienen valores perdidos
colnames(datos)[colSums(is.na(datos))>0]
# Se asume que son 0 (sets no jugados, errores no cometidos, subidas a la red no intentadas...)
datos[is.na(datos)] = 0
# Determinar quién ha ganado viendo los sets es trivial, así que quitamos dicha información
datos = subset(datos,select=-c(FNL1,FNL2)) # Número de sets ganados por cada uno
datos = subset(datos,select=-c(ST1.1,ST2.1,ST3.1,ST4.1,ST5.1,ST1.2,ST2.2,ST3.2,ST4.2,ST5.2)) # Juegos de cada set

# Preprocesamos los datos
datos_procesados = preprocesar_datos(datos,indices_train,c("YeoJohnson","center","scale","pca"),0.85)
datos_procesados_sin_pca = preprocesar_datos(datos,indices_train,c("YeoJohnson","center","scale"))

# Añadimos las etiquetas al final
datos_procesados = cbind(datos_procesados,etiquetas)
datos_procesados_sin_pca = cbind(datos_procesados_sin_pca,etiquetas)
colnames(datos_procesados)[ncol(datos_procesados)] = "etiquetas"
colnames(datos_procesados_sin_pca)[ncol(datos_procesados_sin_pca)] = "etiquetas"

reg_lin = evalua_glm(etiquetas~.,datos_procesados,indices_train)
reg_lin_sin_pca = evalua_glm(etiquetas~.,datos_procesados_sin_pca,indices_train)

# Seleccionamos subconjuntos de características para los datos con PCA
max_caracteristicas = ncol(datos_procesados)-1
seleccion_caracteristicas = subconjuntos_formulas(datos_procesados[indices_train,],max_caracteristicas,metodo="exhaustive")
formulas = seleccion_caracteristicas$formulas

# Seleccionamos subconjuntos de características para los datos sin PCA
max_caracteristicas_sin_pca = ncol(datos_procesados_sin_pca)-1
seleccion_caracteristicas_sin_pca = subconjuntos_formulas(datos_procesados_sin_pca[indices_train,],max_caracteristicas_sin_pca,metodo="exhaustive")
formulas_sin_pca = seleccion_caracteristicas_sin_pca$formulas

# Ajustamos todas las fórmulas
ajustes_glm = mapply(evalua_glm, formulas, MoreArgs = list(datos = datos_procesados, subconjunto = indices_train))
ajustes_glm_sin_pca = mapply(evalua_glm, formulas_sin_pca, MoreArgs = list(datos = datos_procesados_sin_pca, subconjunto = indices_train))

# Representamos la variación del error en función de las fórmulas
glm_min_error_index = which.min(unlist(ajustes_glm[2,]))
glm_sin_pca_min_error_index = which.min(unlist(ajustes_glm_sin_pca[2,]))
plot(x=1:ncol(ajustes_glm_sin_pca),y=ajustes_glm_sin_pca[2,],pch=20,ylim=c(4,30),type="o",col="blue",xlab="Tamaño del conjunto",ylab="Error porcentual", main = "Comparativa de regresión logística")
points(x=glm_sin_pca_min_error_index, y=ajustes_glm_sin_pca[2,glm_sin_pca_min_error_index], pch=19, col="orange")
points(x=1:ncol(ajustes_glm),y=ajustes_glm[2,],type="o",pch=20,col="red")
points(x=glm_min_error_index, y=ajustes_glm[2,glm_min_error_index],pch=19,col="green")
legend(16.5,29,c("Sin PCA","Con PCA"),lty=c(1,1),lwd=c(2.5,2.5),col=c("blue","red"))
