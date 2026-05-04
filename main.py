from dataloader import load_data
from preprocessing import Preprocessor

# constants
PATH="data/realestate_data_london_2024_nov.csv"


def main():
    data = load_data(PATH)
    
    p = Preprocessor(data)

    # COLUMNS
    # descriptionHtml
    p.apply_method_to_column("descriptionHtml", p.remove_html_tags)

    # addedON

    p.remove_pattern_from_column("addedOn", r"[A-Za-z]*\W*([A-Z]|[a-z])") # needs to be done after filling missing entries

    print(p.data[1000:])




if __name__ == "__main__":
    main()