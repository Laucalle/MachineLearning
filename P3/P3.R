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
preprocesar_datos = function(datos,indices_train,metodos,umbral_varianza){
    
    preprocess_obj = preProcess(datos[indices_train,],method=metodos,umbral_varianza)
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
    list(formula=formula, error=porc_error,reg=reg_lin)
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
    list(formulas=formulas,cp=summary(subsets)$cp,bic=summary(subsets)$bic)
    
}

# Clasificación: Email Spam

spam = leer_datos_spam()
# Obtener el conjunto de entrenamiento
indices_train = which(spam$conjuntos == 0)
# Preprocesar los datos 
spam_procesado = preprocesar_datos(spam$datos,indices_train,c("YeoJohnson","center","scale","pca"),0.85)
spam_procesado_sin_pca = preprocesar_datos(spam$datos,indices_train,c("YeoJohnson","center","scale"),0.85)
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
max_caracteristicas_sin_pca = ncol(spam_procesado_sin_pca)-1
# Calculamos los objetos fórmula para cada subconjunto de variables
seleccion_caracteristicas = subconjuntos_formulas(spam_procesado[indices_train,],max_caracteristicas)
formulas = seleccion_caracteristicas$formulas
seleccion_caracteristicas_sin_pca = subconjuntos_formulas(spam_procesado_sin_pca[indices_train,],max_caracteristicas_sin_pca,metodo="forward")
formulas_sin_pca = seleccion_caracteristicas_sin_pca$formulas
# Obtenemos los resultados de evaluar todos los modelos
ajustes_lm = mapply(evalua_lm, formulas, MoreArgs = list(datos = spam_procesado, subconjunto = indices_train))
ajustes_lm_sin_pca = mapply(evalua_lm, formulas_sin_pca, MoreArgs = list(datos = spam_procesado_sin_pca, subconjunto = indices_train))
ajustes_glm = mapply(evalua_glm, formulas, MoreArgs = list(datos = spam_procesado, subconjunto = indices_train))
ajustes_glm_sin_pca = mapply(evalua_glm, formulas_sin_pca, MoreArgs = list(datos = spam_procesado_sin_pca, subconjunto = indices_train))
# Representamos cómo varía el error con los distintos conjuntos de fórmulas
min_cp_index = which.min(seleccion_caracteristicas$cp)
min_bic_index = which.min(seleccion_caracteristicas$bic)
sp_min_cp_index = which.min(seleccion_caracteristicas_sin_pca$cp)
sp_min_bic_index = which.min(seleccion_caracteristicas_sin_pca$bic)

lm_min_error_index = which.min(unlist(ajustes_lm[2,]))
plot(x=1:ncol(ajustes_lm),y=ajustes_lm[2,],pch=20,ylim=c(6,13),type="o",col="blue",xlab="Tamaño del conjunto",ylab="Error porcentual", main = "Regresión lineal con PCA")
points(x=lm_min_error_index, y=ajustes_lm[2,lm_min_error_index], pch=10, col="red")
points(x=min_cp_index, y=ajustes_lm[2,min_cp_index], pch=10, col="green")
points(x=min_bic_index, y=ajustes_lm[2,min_bic_index], pch=10, col="orange")

glm_min_error_index = which.min(unlist(ajustes_glm[2,]))
plot(x=1:ncol(ajustes_glm),y=ajustes_glm[2,],pch=20,ylim=c(6,13),type="o",col="blue",xlab="Tamaño del conjunto",ylab="Error porcentual", main = "Regresión logística con PCA")
points(x=glm_min_error_index, y=ajustes_glm[2,glm_min_error_index], pch=10, col="red")
points(x=min_cp_index, y=ajustes_glm[2,min_cp_index], pch=10, col="green")
points(x=min_bic_index, y=ajustes_glm[2,min_bic_index], pch=10, col="orange")

lm_sp_min_error_index = which.min(unlist(ajustes_lm_sin_pca[2,]))
plot(x=1:ncol(ajustes_lm_sin_pca),y=ajustes_lm_sin_pca[2,],pch=20,ylim=c(6,21),type="o",col="blue",xlab="Tamaño del conjunto",ylab="Error porcentual", main = "Regresión lineal sin PCA")
points(x=lm_sp_min_error_index, y=ajustes_lm_sin_pca[2,lm_sp_min_error_index], pch=10, col="red")
points(x=sp_min_cp_index, y=ajustes_lm_sin_pca[2,sp_min_cp_index], pch=10, col="green")
points(x=sp_min_bic_index, y=ajustes_lm_sin_pca[2,sp_min_bic_index], pch=10, col="orange")

glm_sp_min_error_index = which.min(unlist(ajustes_glm_sin_pca[2,]))
plot(x=1:ncol(ajustes_glm_sin_pca),y=ajustes_glm_sin_pca[2,],pch=20,ylim=c(6,21),type="o",col="blue",xlab="Tamaño del conjunto",ylab="Error porcentual", main = "Regresión logística sin PCA")
points(x=glm_sp_min_error_index, y=ajustes_glm_sin_pca[2,glm_sp_min_error_index], pch=10, col="red")
points(x=sp_min_cp_index, y=ajustes_glm_sin_pca[2,sp_min_cp_index], pch=10, col="green")
points(x=sp_min_bic_index, y=ajustes_glm_sin_pca[2,sp_min_bic_index], pch=10, col="orange")

# Comparativa de evolución de errores en los 4 tipos de modelos
plot(x=10:ncol(ajustes_lm),y=ajustes_lm[2,-(1:9)],type="o",pch=20,xlim=c(10,57),ylim=c(5,10),col="blue",xlab="Tamaño del conjunto",ylab="Error porcentual", main = "Comparativa")
points(x=10:ncol(ajustes_glm),y=ajustes_glm[2,-(1:9)],type="o",pch=20,col="red")
points(x=10:ncol(ajustes_lm_sin_pca),y=ajustes_lm_sin_pca[2,-(1:9)],type="o",pch=20,col="green")
points(x=10:ncol(ajustes_glm_sin_pca),y=ajustes_glm_sin_pca[2,-(1:9)],type="o",pch=20,col="orange")
legend(36.5,10,c("R. Lineal","R. Logística", "R. Lineal sin PCA", "R. Logística sin PCA"),lty=c(1,1),lwd=c(2.5,2.5),col=c("blue","red","green","orange"))

# Creamos la matriz de datos en el formato que necesita glmnet
x = model.matrix(etiquetas~.,spam_procesado_sin_pca)[,-ncol(spam_procesado_sin_pca)]
y = spam_procesado_sin_pca$etiquetas
# Obtenemos los errores de validación cruzada en el conjunto
cv.out = cv.glmnet(x[indices_train,],y[indices_train],alpha=0)
plot(cv.out)
# Guardamos el lambda que ha dado menor error de validación cruzada
bestlambda = cv.out$lambda.min
# Obtenemos un modelo de regresión ridge
modelo_ridge = glmnet(x,y,alpha=0,lambda=bestlambda)
# Calculamos las predicciones y el error asociado a ellas
modelo_ridge.pred = predict(modelo_ridge,s=bestlambda,newx=x[-indices_train,]) 
error_ridge = porcentaje_error(categorizar(modelo_ridge.pred),spam_procesado[-indices_train,ncol(spam_procesado)],fp=1)
# Calculamos una cota para E_out basada en E_test
delta = 0.05 # Tolerancia
N = nrow(spam_procesado)-length(indices_train) # N datos de test
cota_test = calcular_cota_eout(N,delta) # E_test +/- esta cota * 100

# Obtiene la curva de ROC y el área bajo la curva (AUC)
calcula_curva_roc = function(pred,truth){
    predob = prediction(pred,truth)
    area = performance(predob,"auc")
    curva = performance(predob,"tpr","fpr")
    list(curva=curva,area=area)
}

# Calculamos curvas ROC para los 4 mejores modelos (regresiones logísticas, 3 de ellas sin PCA)

roc_glm_min_err = calcula_curva_roc(evaluar_regresion(ajustes_glm[3,glm_min_error_index],spam_procesado[-indices_train,-ncol(spam_procesado)]),spam_procesado[-indices_train,ncol(spam_procesado)])
roc_glm_sin_pca_min_err = calcula_curva_roc(evaluar_regresion(ajustes_glm_sin_pca[3,glm_min_error_index],spam_procesado_sin_pca[-indices_train,-ncol(spam_procesado_sin_pca)]),spam_procesado_sin_pca[-indices_train,ncol(spam_procesado_sin_pca)])
roc_glm_sin_pca_40 = calcula_curva_roc(evaluar_regresion(ajustes_glm_sin_pca[3,40],spam_procesado_sin_pca[-indices_train,-ncol(spam_procesado_sin_pca)]),spam_procesado_sin_pca[-indices_train,ncol(spam_procesado_sin_pca)])
roc_glm_sin_pca_54 = calcula_curva_roc(evaluar_regresion(ajustes_glm_sin_pca[3,54],spam_procesado_sin_pca[-indices_train,-ncol(spam_procesado_sin_pca)]),spam_procesado_sin_pca[-indices_train,ncol(spam_procesado_sin_pca)])

auc_min = roc_glm_min_err$area@y.values
auc_sin_pca_min = roc_glm_sin_pca_min_err$area@y.values
auc_sin_pca_40 = roc_glm_sin_pca_40$area@y.values
auc_sin_pca_54 = roc_glm_sin_pca_54$area@y.values

par(mfrow=c(2,2))
par(pty="s")
plot(roc_glm_min_err$curva,main="R. Logística, 23 variables, PCA",col="blue",lwd=1.5)
plot(roc_glm_sin_pca_min_err$curva,main="R. Logística, 56 variables, sin PCA",col="blue",lwd=1.5)
plot(roc_glm_sin_pca_40$curva,main="R. Logística, 40 variables, sin PCA",col="blue",lwd=1.5)
plot(roc_glm_sin_pca_54$curva,main="R. Logística, 54 variables, sin PCA",col="blue",lwd=1.5)
par(pty="m")
par(mfrow=c(1,1))
