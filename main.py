from bs4 import BeautifulSoup

from dataloader import load_data
from preprocessing import Preprocessor

# constants
PATH="data/realestate_data_london_2024_nov.csv"


def main():
    data = load_data(PATH)
    
    p = Preprocessor(data)

    cleaned_data = p.cleaning('descriptionHtml', [p.remove_html_tags, p.remove_double_whitespace])
    
    print(cleaned_data['descriptionHtml'][1])


if __name__ == "__main__":
    main()