import json

"""
correlate steps and min active
"""

with open('./exist0/steps.json') as json_file:
    steps_loaded = json.load(json_file)
    for entry in steps_loaded:
        print(entry['value'])
        print(entry['date'])
