import os

# Definir rutas a los datasets
lab_root = os.environ["SIDERUST_LAB_ROOT"]
astropy_path = os.path.join(lab_root, "astropy/dataset/")
libnova_path = os.path.join(lab_root, "libnova/dataset/")
siderust_path = os.path.join(lab_root, "siderust/dataset/")
planets = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]
