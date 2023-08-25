import requests 
import re 
import asyncio 
from pprint import pprint 
import pandas as pd 
from bs4 import BeautifulSoup 
import xml.etree.ElementTree as ET
import pickle 


def extract_xml_files_from_sitemap():
    
    sitemap_url = "https://docs.netapp.com/cloud/sitemap.xml"
    
    # Download the sitemap file
    response = requests.get(sitemap_url)
    sitemap_content = response.content

    # Parse the XML content
    root = ET.fromstring(sitemap_content)

    # Find all the URLs in the sitemap
    urls = root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")

    # Extract the XML file URLs
    xml_files = [url.text for url in urls if url.text.endswith(".xml")]

    return xml_files

def extract_html_files_from_xml(xml_url):
    # Download the sitemap file
    response = requests.get(xml_url)
    sitemap_content = response.content

    # Parse the XML content
    root = ET.fromstring(sitemap_content)

    # Find all the URLs in the sitemap
    urls = root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")

    # Extract the HTML file URLs
    html_files = [url.text for url in urls if url.text.endswith(".html")]

    return html_files

def getAllHtmlFiles(connect=False): 
    
    if connect == True: 
        out = []
        xml_files = extract_xml_files_from_sitemap()
        
        for xml_url in xml_files: 
            try: 
                out += extract_html_files_from_xml(xml_url)
            except: 
                pass 
        
        with open("data/all_html.pkl", "w") as f: 
            pickle.dump(out, f)
    
    else: 
        with open("data/all_html.pkl", "rb") as f: 
            out = pickle.load(f)
            
    return out
        
def convert_html_to_text(url, output_file):
    # Fetch the HTML content from the URL
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    res = soup.find("article", id="main")
    text = res.get_text().strip()
    text = re.sub(r'\r\n|\r', ' ', text)
    # Write the text to a text file
    with open(output_file, "w+", encoding="utf8") as file:
        file.write(text)
    
    return text

def get_doc(path: str): 
    with open(path, encoding="utf8") as f: 
        return f.read()