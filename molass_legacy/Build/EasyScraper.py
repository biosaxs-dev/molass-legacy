# coding: utf-8
"""
    EasyScraper.py

    Copyright (c) 2021, Masatsuyo Takahashi, KEK-PF
"""
import requests
from bs4 import BeautifulSoup

class EasyScraper:
    def __init__(self, url):
        self.parser = self.get_page_parser(url)

    def get_page_parser(self, url):
        response = requests.get(url)
        print('response=', response)
        page_html = response.text
        soup = BeautifulSoup(page_html, "html.parser")
        return soup

    def get_element(self, tag, key_text):
        ret_text = None
        for k, element in enumerate(self.parser.findAll(tag)):
            if element.text.find(key_text) >= 0:
                ret_text = element.text
                break
        return ret_text
