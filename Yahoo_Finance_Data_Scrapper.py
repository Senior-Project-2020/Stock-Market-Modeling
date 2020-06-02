import time
import requests
import pandas as pd

def main():
    #Import companies from S+P 500
    companies = pd.read_json("s-and-p-500-companies/data/constituents_json.json")
    
    for stock in companies.loc[:, 'Symbol']:
        stock_url= f'https://query1.finance.yahoo.com/v7/finance/download/{stock}?period1=86400&period2=1590364800&interval=1d&events=history'
        stock_page = requests.get(stock_url, allow_redirects=True)

        print(f"Writing {stock} to file ...")
        open(f'Yahoo_Data/{stock}.csv', 'wb').write(stock_page.content)
        print("Complete")

        #Sleep to prevent the website from timeing out our requests
        time.sleep(0.5)

if __name__ == '__main__':
    main()