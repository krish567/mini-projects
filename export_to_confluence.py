import numpy as np
import os, shutil
import json
js = {"type":"page","title":"Testing REST API","space":{"key":"~krish"}, "body":{"storage":{"value":"<p>This is the information that is written automatically to this page</p>","representation":"storage"}}}
os.system("curl -u krish:55747462 -X POST -H 'Content-Type: application/json' -d'"+ json.dumps(js) +"' http://192.168.2.148/confluence/rest/api/content/ | python -mjson.tool")

# curl -u krish:55747462 -X POST -H 'Content-Type: application/json' -d'{"type":"page","title":"Testing REST API","space":{"key":"~krish"}, "body":{"storage":{"value":"<p>This is the information that is written automatically to this page</p>","representation":"storage"}}}' http://192.168.2.148/confluence/rest/api/content/ | python -mjson.tool
