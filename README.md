Run Server
1. Python
```
env\Scripts\activate
(env) python server/app.py
```
2. Conda
```
conda activate modelenv
python server/app.py
```

Run Client
```
$ npm run serve  
```
3. Docker (server)
```
docker build --tag maquina-backend .
docker run --name maquina-backend -p 5000:5000 maquina-backend 
```