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

porcentaje_error = function(clasificados,reales,umbral=0.5,fp=1,fn=1){
    
    if(min(reales) == 0){
        
        reales[reales == 0] = -1
        clasificados[clasificados < umbral] = -1
        clasificados[clasificados >= umbral] = 1
        
    }
    
    t = table(clasificados,reales)
    total_predicciones = sum(t)
    t[1,2] = t[1,2]*fn
    t[2,1] = t[2,1]*fp
    100*(1-sum(diag(t))/total_predicciones)
    
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
    porc_error = porcentaje_error(prediccion_test,datos[-subconjunto,ncol(datos)],fp,fn)
    list(formula=formula, error = porc_error)
}

# Evalúa una regresión lineal dada una fórmula, una familia y unos datos de entrenamiento 
evalua_glm = function(formula,datos,subconjunto,fp=1,fn=1,familia=binomial()){
    reg_lin = do.call("glm", list(formula=formula, data=substitute(datos), subset=substitute(subconjunto),familia))
    prediccion_test = evaluar_regresion(reg_lin,datos[-subconjunto,-ncol(datos)])
    porc_error = porcentaje_error(prediccion_test,datos[-subconjunto,ncol(datos)],fp,fn)
    list(formula=formula, error = porc_error)
}

# Calcula una cota para E_out en función del error E_test
calcular_cota_eout = function(N,delta) sqrt((log(delta/2))/(-2*N))

# Clasificación: Email Spam

spam = leer_datos_spam()
# Preprocesar los datos 
spam_procesado = preprocesar_datos(spam$datos,c("YeoJohnson","center","scale","pca"),0.85)
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
porc_error = porcentaje_error(prediccion_test,spam_procesado[-indices_train,ncol(spam_procesado)],fp=1)
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
ajustes_lm = mapply(evalua_lm, formulas, MoreArgs = list(datos = spam_procesado, subconjunto = indices_train))
ajustes_glm = mapply(evalua_glm, formulas, MoreArgs = list(datos = spam_procesado, subconjunto = indices_train))
# Creamos la matriz de datos en el formato que necesita glmnet
x = model.matrix(as.formula(ajustes_glm[1,34]),spam_procesado)[,-ncol(spam_procesado)]
y = spam_procesado$etiquetas
# Obtenemos los errores de validación cruzada en el conjunto
cv.out = cv.glmnet(x[indices_train,],y[indices_train],alpha=0)
plot(cv.out)
# Guardamos el lambda que ha dado menor error de validación cruzada
bestlambda = cv.out$lambda.min
# Obtenemos un modelo de Ridge
grid = 10^seq(0,-5,length=100)
modelo_ridge = glmnet(x,y,alpha=0,lambda=grid)
# Calculamos las predicciones y el error asociado a ellas
modelo_ridge.pred = predict(modelo_ridge,s=bestlambda,newx=x[-indices_train,])
error_ridge = porcentaje_error(modelo_ridge.pred,spam_procesado[-indices_train,ncol(spam_procesado)],fp=1)
# Calculamos una cota para E_out basada en E_test
delta = 0.05 # Tolerancia
N = nrow(spam_procesado)-length(indices_train) # N datos de test
cota_test = calcular_cota_eout(N,delta) # E_test +/- esta cota * 100

# Pinta la curva ROC
rocplot = function(pred,truth,...){
    predob = prediction(pred,truth)
    perf = performance(predob,"tpr","fpr")
    par(pty="s")
    plot(perf,...)
    par(pty="m")
    perf
}
