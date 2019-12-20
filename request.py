import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'year':2017})

print(r.json())