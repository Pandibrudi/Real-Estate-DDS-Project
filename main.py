from bs4 import BeautifulSoup

from dataloader import load_data
from preprocessing import remove_html_tags

# constants
PATH="data/realestate_data_london_2024_nov.csv"


def main():
    data = load_data(PATH)
    
    print(remove_html_tags(data))


if __name__ == "__main__":
    main()