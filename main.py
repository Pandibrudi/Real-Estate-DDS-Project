from bs4 import BeautifulSoup

from dataloader import load_data
from preprocessing import Preprocessor

# constants
PATH="data/realestate_data_london_2024_nov.csv"


def main():
    data = load_data(PATH)

    print(len(data))
    
    p = Preprocessor(data)

    cleaned_data = p.multi_cleaning('descriptionHtml', [p.remove_html_tags, p.remove_double_whitespace])
    
    print(cleaned_data['descriptionHtml'][1])

    p.remove_pattern_from_column('addedOn', r"[A-Za-z]+\s")

    print(cleaned_data['addedOn'][1:1019])

    # still need smth for addedOn "updated today"


if __name__ == "__main__":
    main()