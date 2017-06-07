set.seed(3)

leer_datos_partidos = function(){

    datos = read.csv("./datos/partidos.csv", header = TRUE)
    datos = datos[,-c(1,2,3)]
    # Añadimos tipo de pista como característica
    pista_dura = c(rep(1,253),rep(0,252),rep(1,202),rep(0,236))
    tierra_batida = c(rep(0,253),rep(1,252),rep(0,438))
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
evaluar_modelo = function(regresion,datos){
    
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

# Evalúan un cierto modelo dada una fórmula y unos datos de entrenamiento 
evalua_lm = function(formula,datos,subconjunto,fp=1,fn=1){
    reg_lin = do.call("lm", list(formula=formula, data=substitute(datos), subset=substitute(subconjunto)))
    prediccion_test = evaluar_modelo(reg_lin,datos[-subconjunto,-ncol(datos)])
    porc_error = porcentaje_error(categorizar(prediccion_test),datos[-subconjunto,ncol(datos)],fp,fn)
    list(formula=formula, error = porc_error)
}

evalua_glm = function(formula,datos,subconjunto,fp=1,fn=1,familia=binomial()){
    reg_lin = do.call("glm", list(formula=formula, data=substitute(datos), subset=substitute(subconjunto),familia))
    prediccion_test = evaluar_modelo(reg_lin,datos[-subconjunto,-ncol(datos)])
    porc_error = porcentaje_error(categorizar(prediccion_test),datos[-subconjunto,ncol(datos)],fp,fn)
    list(formula=formula, error=porc_error,reg=reg_lin)
}

evalua_random_forest = function(datos,subconjunto,arboles=100){
    rf = do.call("randomForest",list(formula=etiquetas~.,data=substitute(datos),subset=substitute(subconjunto),ntree=arboles,mtry=sqrt(ncol(datos)-1)))
    rf_pred = evaluar_modelo(rf,datos[-subconjunto,-ncol(datos)])
    rf_error = porcentaje_error(categorizar(rf_pred),datos[-subconjunto,ncol(datos)])
    list(arboles=arboles,error=rf_error,rf=rf)
}

# Evalúa un Random Forest con k-fold cv repetida i veces 
evalua_random_forest_cv = function(datos,etiquetas,control,arboles=100){
    mtry = sqrt(ncol(datos))
    rf_train = train(datos,as.factor(etiquetas),method="rf",ntree=arboles,preProcess=c("YeoJohnson","center","scale"),trControl=control,tuneGrid=expand.grid(.mtry=mtry))
    rf_error = (1-rf_train$results$Accuracy)*100
    list(arboles=arboles,error=rf_error,rf=rf_train)
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

#######################################
# Regresión lineal / logística

reg_log = evalua_glm(etiquetas~.,datos_procesados,indices_train)
reg_log_sin_pca = evalua_glm(etiquetas~.,datos_procesados_sin_pca,indices_train)

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

error_glm = ajustes_glm_sin_pca[2,glm_sin_pca_min_error_index]

###############################################################################
# Modelos no lineales

control = trainControl(method = "cv", number = 10)

#######################################
# Random Forest

set.seed(111)
num_arboles = seq(10,100,10)

# Mediante validación cruzada obtenemos el mejor hiperparámetro número de árboles
ajustes_rf_cv = mapply(evalua_random_forest_cv,num_arboles,MoreArgs = list(datos=datos[indices_train,],etiquetas=etiquetas[indices_train],control=control))
rf_cv_min_error_index = which.min(unlist(ajustes_rf_cv[2,]))

plot(x=num_arboles,y=ajustes_rf_cv[2,],pch=20,ylim=c(4,19),type="o",col="blue",xlab="Número de árboles",ylab="% Error de validación cruzada", main = "Comparativa de número de árboles")
points(x=ajustes_rf_cv[1,rf_cv_min_error_index], y=ajustes_rf_cv[2,rf_cv_min_error_index], pch=19, col="orange")

rf_pred_test = evaluar_modelo(ajustes_rf_cv[3,rf_cv_min_error_index]$rf,datos[-indices_train,])
error_rf = porcentaje_error(as.numeric(rf_pred_test),etiquetas[-indices_train])
# rf_pca = evalua_random_forest(datos_procesados,indices_train)
# rf_sin_pca = evalua_random_forest(datos_procesados_sin_pca,indices_train)
# 
# ajustes_rf = mapply(evalua_random_forest,num_arboles,MoreArgs = list(datos=datos_procesados,subconjunto=indices_train))
# ajustes_rf_sin_pca = mapply(evalua_random_forest,num_arboles,MoreArgs = list(datos=datos_procesados_sin_pca,subconjunto=indices_train))
# 
# # Representamos la variación del error en función del número de árboles
# rf_min_error_index = which.min(unlist(ajustes_rf[2,]))
# rf_sin_pca_min_error_index = which.min(unlist(ajustes_rf_sin_pca[2,]))
# plot(x=num_arboles,y=ajustes_rf_sin_pca[2,],pch=20,ylim=c(4,19),type="o",col="blue",xlab="Número de árboles",ylab="Error porcentual", main = "Comparativa de Random Forests")
# points(x=ajustes_rf_sin_pca[1,rf_sin_pca_min_error_index], y=ajustes_rf_sin_pca[2,rf_sin_pca_min_error_index], pch=19, col="orange")
# points(x=num_arboles,y=ajustes_rf[2,],type="o",pch=20,col="red")
# points(x=ajustes_rf[1,rf_min_error_index], y=ajustes_rf[2,rf_min_error_index],pch=19,col="green")
# legend(67,19,c("Sin PCA","Con PCA"),lty=c(1,1),lwd=c(2.5,2.5),col=c("blue","red"))

# Con tune
# tuned_rf = tune.randomForest(datos[indices_train,],as.factor(etiquetas[indices_train]),mtry=sqrt(ncol(datos)),ntree=seq(10,100,10))
# rf_pred_test = evaluar_modelo(tuned_rf$best.model,datos[-indices_train,])
# error_rf = porcentaje_error(as.numeric(rf_pred_test),etiquetas[-indices_train])

#######################################
# Adaboost

set.seed(111)
grid = expand.grid(maxdepth=1, iter=c(20,30,40,50,60,70,80,90,100), nu=c(0.15,0.2,0.25,0.3))
rcontrol = rpart.control(maxdepth=1,cp=-1,minsplit=0)
ada_fit = train(x = datos[indices_train,], y = as.factor(etiquetas[indices_train]),method = "ada", trControl = control, preProcess = c("YeoJohnson","center","scale"),tuneGrid = grid, control = rcontrol)
ada_pred = predict(ada_fit, datos[-indices_train,])
error_ada = porcentaje_error(as.numeric(ada_pred), etiquetas[-indices_train])

arboles_nu = ada_fit$result[which(ada_fit$results[,1] == 0.15)[which.max(ada_fit$results[ada_fit$results[,1] == 0.15,4])],3]
arboles_nu = c(arboles_nu,ada_fit$result[which(ada_fit$results[,1] == 0.2)[which.max(ada_fit$results[ada_fit$results[,1] == 0.2,4])],3])
arboles_nu = c(arboles_nu,ada_fit$result[which(ada_fit$results[,1] == 0.25)[which.max(ada_fit$results[ada_fit$results[,1] == 0.25,4])],3])
arboles_nu = c(arboles_nu,ada_fit$result[which(ada_fit$results[,1] == 0.3)[which.max(ada_fit$results[ada_fit$results[,1] == 0.3,4])],3])
datos_nu = c(0.15, max(ada_fit$results[ada_fit$results[,1] == 0.15,4]))
datos_nu = rbind(datos_nu,c(0.2, max(ada_fit$results[ada_fit$results[,1] == 0.2,4])))
datos_nu = rbind(datos_nu,c(0.25, max(ada_fit$results[ada_fit$results[,1] == 0.25,4])))
datos_nu = rbind(datos_nu,c(0.3, max(ada_fit$results[ada_fit$results[,1] == 0.3,4])))

plot(datos_nu, pch = 20, ylab = "Precisión", xlab = "Coeficiente de aprendizaje", xlim = c(0.13,0.32),col = "blue", type = "o")
text(x = datos_nu[,1], y = datos_nu[,2] , labels = arboles_nu, cex = 0.7, pos = 2)

#######################################
# Support Vector Machines
set.seed(111)
grid = expand.grid(C=seq(1,5,1), sigma=seq(0.02, 0.04, 0.005))
svm_fit = train(x = datos[indices_train,], y = as.factor(etiquetas[indices_train]),method = "svmRadial", trControl = control, prob.model = TRUE, preProcess = c("YeoJohnson","center","scale"), tuneGrid = grid)
svm_pred = predict(svm_fit, datos[-indices_train,])
error_svm = porcentaje_error(as.numeric(svm_pred), etiquetas[-indices_train])

c_sigma = svm_fit$result[which(svm_fit$results[,2] == 0.02)[which.max(svm_fit$results[svm_fit$results[,2] == 0.02,3])],1]
c_sigma = c(c_sigma,svm_fit$result[which(svm_fit$results[,2] == 0.025)[which.max(svm_fit$results[svm_fit$results[,2] == 0.025,3])],1])
c_sigma = c(c_sigma,svm_fit$result[which(svm_fit$results[,2] == 0.03)[which.max(svm_fit$results[svm_fit$results[,2] == 0.03,3])],1])
c_sigma = c(c_sigma,svm_fit$result[which(svm_fit$results[,2] == 0.035)[which.max(svm_fit$results[svm_fit$results[,2] == 0.035,3])],1])
c_sigma = c(c_sigma,svm_fit$result[which(svm_fit$results[,2] == 0.04)[which.max(svm_fit$results[svm_fit$results[,2] == 0.04,3])],1])
datos_sigma = c(0.02, max(svm_fit$results[svm_fit$results[,2] == 0.02,3]))
datos_sigma = rbind(datos_sigma,c(0.025, max(svm_fit$results[svm_fit$results[,2] == 0.025,3])))
datos_sigma = rbind(datos_sigma,c(0.03, max(svm_fit$results[svm_fit$results[,2] == 0.03,3])))
datos_sigma = rbind(datos_sigma,c(0.035, max(svm_fit$results[svm_fit$results[,2] == 0.035,3])))
datos_sigma = rbind(datos_sigma,c(0.04, max(svm_fit$results[svm_fit$results[,2] == 0.04,3])))

plot(datos_sigma, pch = 20, ylab = "Precisión", xlab = "Sigma", xlim = c(0.018,0.042),col = "blue", type = "o")
text(x = datos_sigma[,1], y = datos_sigma[,2] , labels = c_sigma, cex = 0.7, pos = 2)

#######################################
# Neural Networks

# layers = c(3,3)
# datos_nn = cbind(datos,etiquetas)
# colnames(datos_nn)[ncol(datos_nn)] = "etiquetas"
# formula_nn = paste("etiquetas~",paste(colnames(datos),collapse = "+"))
# net = neuralnet(formula_nn,data=datos_nn[indices_train,],hidden=layers,linear.output=FALSE)
# prediccion = compute(net,datos[-indices_train,])
# porcentaje_error(categorizar(prediccion$net.result),etiquetas[-indices_train])
# 
# set.seed(111)
# grid = expand.grid(layer1=c(seq(1,50,3),50), layer2=0, layer3=0)
# nn_fit_una = train(x = datos[indices_train,], y = etiquetas[indices_train],method = "neuralnet", linear.output=FALSE, trControl = control, preProcess = c("YeoJohnson","center","scale"), tuneGrid = grid)
# grid = expand.grid(layer1=c(seq(1,50,5),50), layer2=seq(0,50,5), layer3=0)
# nn_fit_dos = train(x = datos[indices_train,], y = etiquetas[indices_train],method = "neuralnet", linear.output=FALSE, trControl = control, preProcess = c("YeoJohnson","center","scale"), tuneGrid = grid)
# grid = expand.grid(layer1=c(seq(1,50,10),50), layer2=seq(0,50,10), layer3=seq(0,50,10))
# nn_fit_tres = train(x = datos[indices_train,], y = etiquetas[indices_train],method = "neuralnet", linear.output=FALSE, trControl = control, preProcess = c("YeoJohnson","center","scale"), tuneGrid = grid)

###############################################################################
# Curvas ROC

calcula_curva_roc = function(pred,truth){
    predob = prediction(pred,truth)
    area = performance(predob,"auc")
    curva = performance(predob,"tpr","fpr")
    list(curva=curva,area=area)
}

# Curva ROC regresión logística
modelo_glm = ajustes_glm_sin_pca[3,glm_sin_pca_min_error_index]
eval_glm = unlist(evaluar_modelo(modelo_glm,datos_procesados_sin_pca[-indices_train,]))
roc_glm = calcula_curva_roc(eval_glm,etiquetas[-indices_train])
plot(roc_glm$curva)
area_roc_glm = roc_glm$area@y.values
# Curva ROC Random Forest
modelo_rf = ajustes_rf_cv[3,rf_cv_min_error_index]
eval_rf = predict(modelo_rf,datos[-indices_train,],type="prob")$rf[,2]
roc_rf = calcula_curva_roc(eval_rf,etiquetas[-indices_train])
plot(roc_rf$curva)
area_roc_rf = roc_rf$area@y.values
# Curva ROC AdaBoost
modelo_ada = ada_fit
eval_ada = predict(modelo_ada,datos[-indices_train,],type="prob")[,2]
roc_ada = calcula_curva_roc(eval_ada,etiquetas[-indices_train])
plot(roc_ada$curva)
area_roc_ada = roc_ada$area@y.values
# Curva ROC Support Vector Machines
modelo_svm = svm_fit
eval_svm = predict(modelo_svm,datos[-indices_train,],type="prob")[,2]
roc_svm = calcula_curva_roc(eval_svm,etiquetas[-indices_train])
plot(roc_svm$curva)
area_roc_svm = roc_svm$area@y.values