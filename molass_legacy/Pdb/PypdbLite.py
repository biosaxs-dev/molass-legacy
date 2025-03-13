# coding: utf-8
"""
    Pdb.PypdbLite.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import requests

class Query:
    """
    See https://data.rcsb.org/#gql-usage-guidelines
    """
    url = "https://data.rcsb.org/graphql"

    def __init__(self, pdbid):
        self.pdbid = pdbid

    def search(self):
        query_text="""
        {
          entry(entry_id:"%s") {
            exptl {
              method
            }
          }
        }
        """
        response = requests.get(
            self.url, {"query": query_text % self.pdbid}
        )

        # print("response.status_code=", response.status_code)
        if response.status_code == 200:
            text = eval(response.text)
            # print("response.text=", text)
            return text["data"]["entry"]["exptl"]
        else:
            return None

def get_pdb_file(pdbid, filetype='pdb'):
    url = 'https://files.rcsb.org/download/'

    response = requests.get(url + pdbid + "." + filetype)
    print("response.status_code=", response.status_code)
    return response.content.decode()
