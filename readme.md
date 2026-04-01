# Face Classification API 

Cette API permet de classifier des images de visages à l’aide d’un modèle CNN. Elle offre un endpoint simple pour envoyer une image et recevoir une prédiction avec un niveau de confiance.

## Installation

#### Cloner le projet :

```bash
git clone https://github.com/jeanluckawel/FastAPIProject
cd FastAPIProject 
```
#### Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```
### install FastApi
bash
```bash 
pip install fastapi uvicorn["standard"]
```
#### Installer les dépendances :
```bash
pip install -r requirements.txt
or
pip install -r requirements-dev.txt
```
#### Lancer le serveur
```bash
uvicorn app.main:app --reload
```
#### Accès 
```bash
API : http://127.0.0.1:8000
Swagger : http://127.0.0.1:8000/docs
ReDoc : http://127.0.0.1:8000/redoc
```