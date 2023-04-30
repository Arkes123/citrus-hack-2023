from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
# import os
# os.system("python model.py")
from model import call_model

def main():
    getURL = input("Enter URL: ")
    print("\n")

    #page = urlopen(getURL)
    req = Request(
        url=getURL,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    page = urlopen(req)

    html = page.read().decode('utf-8')

    soup = BeautifulSoup(html, "html.parser")

    pageContent = []

    title = soup.find("h1")

    pageContent.append(title.get_text())

    for pTag in soup.find_all("p"):
        pageContent.append(pTag.get_text())

    pageContent = " ".join(pageContent)
    
    per = call_model(pageContent)
    if 0.95 < per:
        print("Likely to fake news: ", per)
    else:
        print("Likely to be real news: ", per)

    
    
if __name__ == '__main__':
    main()

