# MEFE

Este repositorio está hecho para subir notebooks de jupyter con las resoluciones de algunos ejercicios y TPs de la materia [Métodos Estadísticos en Física Experimental (MEFE)](http://materias.df.uba.ar/meefea2023c1/). Además de los notebooks, algunos ejercicios están publicados en formato web en [esta página](https://tdinapoli.github.io/).

El archivo ```conversor.py``` es un (muy precario) script que se encarga de convertir los notebooks de jupyter a archivos de markdown (que luego se pueden convertir fácilmente a latex con la ayuda de otro programa). Para usarlo hay que cambiar la variable ```json_path``` por el path al notebook que se quiera convertir, y ```md_path``` por el path donde quieran que se guarde el archivo de markdown.

Para convertirlo a latex tienen que tener instalado [pandoc](https://pandoc.org/) y por lo tanto algún compilador de latex como pdflatex. Con eso listo, el comando

```bash
pandoc archivo-markdown.md -o archivo-latex.pdf --highlight-style=tango -V colorlinks=true
```

convierte el archivo ```archivo-markdown.md``` a ```archivo-latex.pdf```.
