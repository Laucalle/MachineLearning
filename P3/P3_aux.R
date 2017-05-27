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
o_procesados = preprocesar_datos(ozono$datos,o_indexes_train,c("YeoJohnson","center","scale","pca"),0.9)
o_procesados_sin_pca = preprocesar_datos(ozono$datos,o_indexes_train,c("YeoJohnson","center","scale"),0.9)
#o_procesados = as.data.frame(poly(x=as.matrix(o_procesados[,-ncol(o_procesados)]),degree=2))
#o_procesados_sin_pca = as.data.frame(poly(x=as.matrix(o_procesados_sin_pca[,-ncol(o_procesados_sin_pca)]),degree=2))

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

# Gráficas de errores medios cuadráticos
con_pca_min_error_index = which.min(unlist(ajustes[2,]))
sin_pca_min_error_index = which.min(unlist(ajustes_sin_pca[2,]))
plot(x=1:ncol(ajustes),y=ajustes[2,],pch=20,ylim=c(17,28),type="o",col="blue",xlab="Tamaño del conjunto",ylab="Error cuadrático", main = "Regresión lineal con PCA")
points(x=con_pca_min_error_index, y=ajustes[2,con_pca_min_error_index], pch=10, col="red")
plot(x=1:ncol(ajustes_sin_pca),y=ajustes_sin_pca[2,],pch=20,ylim=c(17,28),type="o",col="blue",xlab="Tamaño del conjunto",ylab="Error cuadrático", main = "Regresión lineal sin PCA")
points(x=sin_pca_min_error_index, y=ajustes_sin_pca[2,sin_pca_min_error_index], pch=10, col="red")

plot(ajustes[3,6]$reg,which=c(1),pch=20,col="blue")
plot(ajustes_sin_pca[3,6]$reg,which=c(1),pch=20,col="blue")

#   Aplicamos transformaciones no lineales a los mejores modelos
# Ya que son pocas características, vamos a extraerlas manualmente
mejor_formula_pca = ajustes[1,6]
mejor_formula_sin_pca = ajustes_sin_pca[1,6]
print(mejor_formula_pca) # PC1 + PC2 + PC3 + PC4 + PC5 + PC6
print(mejor_formula_sin_pca) # humidity + temp + ibh + ibt + vis + doy

# Aplicamos polinomios de grado 2
o_procesados_poly = poly(as.matrix(o_procesados[,-ncol(o_procesados)]),degree=2)
o_procesados_poly = cbind(as.data.frame(o_procesados_poly),o_labels)
colnames(o_procesados_poly)[ncol(o_procesados_poly)] = "etiquetas"
o_procesados_sin_pca_poly = poly(as.matrix(o_procesados_sin_pca[,c("humidity","temp","ibh","ibt","vis","doy")]),degree=2)
o_procesados_sin_pca_poly = cbind(as.data.frame(o_procesados_sin_pca_poly),o_labels)
colnames(o_procesados_sin_pca_poly)[ncol(o_procesados_sin_pca_poly)] = "etiquetas"

# Calculamos los nuevos conjuntos de fórmulas a partir de las nuevas características
max_car_poly = ncol(o_procesados_poly)-1
max_car_sin_pca_poly = ncol(o_procesados_sin_pca_poly)-1
formulas_poly = subconjuntos_formulas(o_procesados_poly,max_car_poly)$formulas
formulas_sin_pca_poly = subconjuntos_formulas(o_procesados_sin_pca_poly,max_car_sin_pca_poly)$formulas

# Ajustamos todos los modelos
ajustes_poly = mapply(evalua_lm,formulas_poly, MoreArgs = list(datos = o_procesados_poly, subconjunto = o_indexes_train))
ajustes_sin_pca_poly = mapply(evalua_lm,formulas_sin_pca_poly, MoreArgs = list(datos = o_procesados_sin_pca_poly, subconjunto = o_indexes_train))

# Gráficas de errores medios cuadráticos
con_pca_poly_min_error_index = which.min(unlist(ajustes_poly[2,]))
sin_pca_poly_min_error_index = which.min(unlist(ajustes_sin_pca_poly[2,]))
plot(x=1:ncol(ajustes_poly),y=ajustes_poly[2,],pch=20,ylim=c(14,26),type="o",col="blue",xlab="Tamaño del conjunto",ylab="Error cuadrático", main = "R. Lin. con PCA y poly")
points(x=con_pca_poly_min_error_index, y=ajustes_poly[2,con_pca_poly_min_error_index], pch=10, col="red")
plot(x=1:ncol(ajustes_sin_pca_poly),y=ajustes_sin_pca_poly[2,],pch=20,ylim=c(14,26),type="o",col="blue",xlab="Tamaño del conjunto",ylab="Error cuadrático", main = "R. Lin. sin PCA y con poly")
points(x=sin_pca_poly_min_error_index, y=ajustes_sin_pca_poly[2,sin_pca_poly_min_error_index], pch=10, col="red")

# Comparativa de errores cuadráticos previos y actuales
layout(matrix(c(1,2,3,3),ncol=2,byrow=T))

par(mai=rep(0.5,4))
plot(x=1:ncol(ajustes_poly),y=ajustes_poly[2,],pch=20,xlim=c(1,27),ylim=c(14,26),type="o",col="blue",xlab="Tamaño del conjunto",ylab="Error cuadrático", main = "R. Lin. con PCA")
points(x=con_pca_poly_min_error_index, y=ajustes_poly[2,con_pca_poly_min_error_index], pch=10, col="green")
points(x=1:ncol(ajustes),y=ajustes[2,],type="o",pch=20,col="red")
plot(x=1:ncol(ajustes_sin_pca_poly),y=ajustes_sin_pca_poly[2,],pch=20,ylim=c(14,26),type="o",col="blue",xlab="Tamaño del conjunto",ylab="Error cuadrático", main = "R. Lin. sin PCA")
points(x=sin_pca_poly_min_error_index, y=ajustes_sin_pca_poly[2,sin_pca_poly_min_error_index], pch=10, col="green")
points(x=1:ncol(ajustes_sin_pca),y=ajustes_sin_pca[2,],type="o",pch=20,col="red")
par(mai=c(0,0,0,0))
plot.new()
legend(x="top",ncol=2,26,c("Con transformaciones","Sin transformaciones"),lty=c(1,1),lwd=c(2.5,2.5),col=c("blue","red"))

# Gráficas de errores residuales
par(dev.off())
layout(matrix(c(1,1)))
plot(ajustes_poly[3,con_pca_poly_min_error_index]$reg,which=c(1),pch=20,col="blue")
plot(ajustes_sin_pca_poly[3,sin_pca_poly_min_error_index]$reg,which=c(1),pch=20,col="blue")

#   Aplicamos raíz cuadrada a las etiquetas para solucionar la heterocedasticidad y recalculamos el error
o_procesados_poly[,ncol(o_procesados_poly)] = sqrt(o_procesados_poly[,ncol(o_procesados_poly)])
o_procesados_sin_pca_poly[,ncol(o_procesados_sin_pca_poly)] = sqrt(o_procesados_sin_pca_poly[,ncol(o_procesados_sin_pca_poly)])
mejor_formula_pca_poly = ajustes_poly[1,17]
mejor_formula_sin_pca_poly = ajustes_sin_pca_poly[1,16]

# Obtenemos los dos mejores modelos pero con etiquetas de raíz cuadrada
sqrt_mejor_modelo_poly = evalua_lm(mejor_formula_pca_poly,o_procesados_poly,o_indexes_train)
sqrt_mejor_modelo_sin_pca_poly = evalua_lm(mejor_formula_sin_pca_poly,o_procesados_sin_pca_poly,o_indexes_train)
etiquetas_sqrt_mejor_modelo_poly = evaluar_regresion(sqrt_mejor_modelo_poly$reg,o_procesados_poly[-o_indexes_train,-ncol(o_procesados_poly)])
etiquetas_sqrt_mejor_modelo_sin_pca_poly = evaluar_regresion(sqrt_mejor_modelo_sin_pca_poly$reg,o_procesados_sin_pca_poly[-o_indexes_train,-ncol(o_procesados_sin_pca_poly)])
# Calculamos MSE
error_sqrt_mejor_modelo_poly = error_cuadratico_medio(etiquetas_sqrt_mejor_modelo_poly^2,o_labels[-o_indexes_train])
error_sqrt_mejor_modelo_sin_pca_poly = error_cuadratico_medio(etiquetas_sqrt_mejor_modelo_sin_pca_poly^2,o_labels[-o_indexes_train])
# Representamos los errores residuales
plot(sqrt_mejor_modelo_poly$reg,which=1,pch=20,col="blue")
plot(sqrt_mejor_modelo_sin_pca_poly$reg,which=1,pch=20,col="blue")
