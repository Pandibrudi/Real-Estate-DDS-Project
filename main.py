from dataloader import load_data
from bs4 import BeautifulSoup

# constants
PATH="data/realestate_data_london_2024_nov.csv"


def main():
    data = load_data(PATH)
    print(data['descriptionHtml'].tolist())


if __name__ == "__main__":
    main()