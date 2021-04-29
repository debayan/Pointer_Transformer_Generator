#!/usr/bin/python
import sys,json

print(len(json.loads(open(sys.argv[1]).read())))
