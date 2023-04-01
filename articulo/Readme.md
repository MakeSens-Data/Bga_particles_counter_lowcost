# Articulo: estaciones de bajo costo

Se hace un estudio de diferentes modelos para la calibración de sensores bajo costo, donde partimos de los metodos de regresión liean habituales, además de un estudio de componentes principales para estudiar la influencia de otras variables, para el uso en regresión multilian. Finalmente hemos explorado el uso de estimación no supervisada con la ayuda de algotismos de aprendizaje automatico, especificamente Random Forest.

### Data
La data de referencia corresponde a la red de monitoreo del area metropolitana de Bucaramanga y la de los sensores bajo costo, a los proporcionados por la red ciudadana de monitoreo (RACIMO), poryecto del grupo halley UIS. Los datos empleados corresponde al 2019 entre los meses de Abril y Agosto.

### Directorios y archivos
Este repositorio contiene un directorio llamado **Data**, que contiene la data descrita anteriormente. Adicional contiene los siguientes archivos:
- **rervice.py:** Cuenta con todas las funciones que son utilizadas para el procesamiento de los datos y la implementación de los diferentes modelos de calibración
- **analisis.ipynb:** contiene todo el análisis necesario para el articulo
