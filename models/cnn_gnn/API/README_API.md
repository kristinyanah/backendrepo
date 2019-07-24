# Python API  

to run the server, clone the repo, and then install the requirements 

conda install -c anaconda gunicorn

to run the API for debugging purpose

```
python search_endpoint.py
```

if you have to deploy the server for website

```
bash /path/to/source/restartWrapper.sh /path/to/source
```

The server runs on port 5344, which can be changed in the search_endpoint.py
Logs for the API can be found in log.txt and other/older logs can be accessed from log/ directory.
