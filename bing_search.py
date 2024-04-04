import requests
from bs4 import BeautifulSoup

def rechercher_bing(query):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    url = f"https://www.bing.com/search?q={query}"
    # url = f"https://www.google.fr/search?q={query}"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    # print(soup)
    results = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        # Optionnel : Filtrer les URLs indésirables ou spécifiques
        if "http" in href:  # Un exemple basique de filtrage
            results.append(href)
    
    return results[:15]

# Exemple d'utilisation
query = "qui sont les bene gesserit"
urls = rechercher_bing(query)
print("urls = [")
for url in urls[:-1]:
    print(f'    "{url}",')
print(f'    "{urls[-1]}"')  # Pour le dernier élément, on n'ajoute pas de virgule
print("]")
