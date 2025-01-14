# Proyecto de IA de reconocimiento de Marca de Vehiculo
Hecho por: Byron Serrano


###  Requisitos
- Python 3.9
- pip (instalado con Python por defecto)
- Dataset con marcas de carros [url_para_dataset](https://www.kaggle.com/datasets/kshitij192/cars-image-dataset?resource=download)
- Video de prueba [url_para_video_de_prueba](https://pixabay.com/es/videos/carros-autopista-velocidad-1900/)


###  Instalación
1. Clona el repositorio en tu máquina:
```bash
git clone https://github.com/ByronSerrano/exam-2p-ia-vehicle.git
```

2. Crea y dirigite a la rama "dev" y trae los cambios de la rama "dev"::
```bash
git checkout -b dev
```

```bash
git pull origin dev
```

3. Crea un enviroment y activalo:
```bash
  python -m venv env
  ```

  - Windows:
   ```bash
   env\Scripts\activate
   ```
   - Linux:
   ```bash
   source env/bin/activate
   ```

### Uso del Aplicativo
1. Ve al dataset con los datos de la marcas del carro, y ponlo dentro del directorio de:
  ```bash
   src/data
   ```
2. Cambia el nombre del dataset a:
  ```bash
   cars_dataset
   ```