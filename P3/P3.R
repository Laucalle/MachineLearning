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
evalua_lm = function(formula,datos,subconjunto,fp=1,fn=1){
    reg_lin = do.call("lm", list(formula=formula, data=substitute(datos), subset=substitute(subconjunto)))
    prediccion_test = evaluar_regresion(reg_lin,datos[-subconjunto,-ncol(datos)])
    porc_error = porcentaje_error(categorizar(prediccion_test),datos[-subconjunto,ncol(datos)],fp,fn)
    list(formula=formula, error = porc_error)
}

# Evalúa una regresión lineal dada una fórmula, una familia y unos datos de entrenamiento 
evalua_glm = function(formula,datos,subconjunto,fp=1,fn=1,familia=binomial()){
    reg_lin = do.call("glm", list(formula=formula, data=substitute(datos), subset=substitute(subconjunto),familia))
    prediccion_test = evaluar_regresion(reg_lin,datos[-subconjunto,-ncol(datos)])
    porc_error = porcentaje_error(categorizar(prediccion_test),datos[-subconjunto,ncol(datos)],fp,fn)
    list(formula=formula, error = porc_error)
}

# Calcula una cota para E_out en función del error E_test
calcular_cota_eout = function(N,delta) sqrt((log(delta/2))/(-2*N))

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
    
}

# Clasificación: Email Spam

spam = leer_datos_spam()
# Preprocesar los datos 
spam_procesado = preprocesar_datos(spam$datos,c("YeoJohnson","center","scale","pca"),0.85)
spam_procesado_sin_pca = preprocesar_datos(spam$datos,c("YeoJohnson","center","scale"),0.85)
# Obtener el conjunto de entrenamiento
indices_train = which(spam$conjuntos == 0)
# Añadir las etiquetas para la regresión lineal
spam_procesado = cbind(spam_procesado,spam$etiquetas)
spam_procesado_sin_pca = cbind(spam_procesado_sin_pca,spam$etiquetas)
colnames(spam_procesado)[ncol(spam_procesado)] = "etiquetas"
colnames(spam_procesado_sin_pca)[ncol(spam_procesado_sin_pca)] = "etiquetas"
# Hacer regresión lineal de las etiquetas según las otras características
reg_lin_spam = glm(etiquetas~.,data=spam_procesado,subset=indices_train,family=binomial())
# Error residual:       plot(reg_lin_spam,which=c(1))
# Obtener predicciones de la regresión sobre los datos de test
prediccion_test = evaluar_regresion(reg_lin_spam,spam_procesado[-indices_train,-ncol(spam_procesado)])
# Porcentaje de error en el conjunto de test
porc_error = porcentaje_error(categorizar(prediccion_test),spam_procesado[-indices_train,ncol(spam_procesado)],fp=1)
# Buscamos exhaustivamente conjuntos de características que usar
max_caracteristicas = ncol(spam_procesado)-1
# Calculamos los objetos fórmula para cada subconjunto de variables
formulas = subconjuntos_formulas(spam_procesado[indices_train,],max_caracteristicas)
formulas_sin_pca = subconjuntos_formulas(spam_procesado_sin_pca[indices_train,],max_caracteristicas,metodo="forward")
# Obtenemos los resultados de evaluar todos los modelos
ajustes_lm = mapply(evalua_lm, formulas, MoreArgs = list(datos = spam_procesado, subconjunto = indices_train))
ajustes_lm_sin_pca = mapply(evalua_lm, formulas_sin_pca, MoreArgs = list(datos = spam_procesado_sin_pca, subconjunto = indices_train))
ajustes_glm = mapply(evalua_glm, formulas, MoreArgs = list(datos = spam_procesado, subconjunto = indices_train))
ajustes_glm_sin_pca = mapply(evalua_glm, formulas_sin_pca, MoreArgs = list(datos = spam_procesado_sin_pca, subconjunto = indices_train))
# Representamos cómo varía el error con los distintos conjuntos de fórmulas
plot(x=1:ncol(ajustes_lm),y=ajustes_lm[2,],pch=20,ylim=c(6,12),col="blue")
# Creamos la matriz de datos en el formato que necesita glmnet
x = model.matrix(etiquetas~.,spam_procesado_sin_pca)[,-ncol(spam_procesado_sin_pca)]
y = spam_procesado_sin_pca$etiquetas
# Obtenemos los errores de validación cruzada en el conjunto
cv.out = cv.glmnet(x[indices_train,],y[indices_train],alpha=0)
plot(cv.out)
# Guardamos el lambda que ha dado menor error de validación cruzada
bestlambda = cv.out$lambda.min
# Obtenemos un modelo de Ridge
grid = 10^seq(5,0,length=100)
modelo_ridge = glmnet(x,y,alpha=0,lambda=grid)
# Calculamos las predicciones y el error asociado a ellas
modelo_ridge.pred = predict(modelo_ridge,s=bestlambda,newx=x[-indices_train,])
error_ridge = porcentaje_error(categorizar(modelo_ridge.pred),spam_procesado[-indices_train,ncol(spam_procesado)],fp=1)
# Calculamos una cota para E_out basada en E_test
delta = 0.05 # Tolerancia
N = nrow(spam_procesado)-length(indices_train) # N datos de test
cota_test = calcular_cota_eout(N,delta) # E_test +/- esta cota * 100

# Obtiene la curva de ROC o el área bajo la curva (AUC)
roc_curve = function(pred,truth,area=F,...){
    predob = prediction(pred,truth)
    if(area)
        perf = performance(predob,"auc")
    else{
        perf = performance(predob,"tpr","fpr")
        par(pty="s")
        plot(perf,...)
        par(pty="m")
    }
    perf
}

curva = roc_curve(prediccion_test,spam_procesado[-indices_train,ncol(spam_procesado)])
area = roc_curve(prediccion_test,spam_procesado[-indices_train,ncol(spam_procesado)],area=T)