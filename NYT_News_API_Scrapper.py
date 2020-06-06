import os
import time
import json
import requests
import pandas as pd
from datetime import date, timedelta

def get_news_data(company, start_date, end_date):
    #TODO: hide this key in environment variable
    key = 'fBNxpCDhmaMKMBqkS0xrEzmdPvyyYtFv'
    fields = f'headline="{company}"&begin_date={start_date}&end_date={end_date}&news-desk="Business"&type_of_materials="News"'
    return requests.get(f'https://api.nytimes.com/svc/search/v2/articlesearch.json?q={fields}&api-key={key}') 

def main():
    """
    NOTE

    This way of getting data from the NYT api was scrapped because of the amount of time we have to gather data. With some back
    of the envelope calculation it is easy to tell that if I wished to get the passed 30 years of data for all 500 stocks at an
    interval of 1 week at a time, with 4000 requests a day it would still take me 195 days to retrieve all of that data. This way
    of gathering data will still work however it just isn't quick enough for the time frame we are working in to gather any
    meaningful amount of data. Because of this we will no longer be pursing the idea of using news data in our model.
    """
    news_dir = 'News_Data'
    companies = pd.read_json("s-and-p-500-companies/data/constituents_json.json")

    latest_date = date.fromisoformat('2020-05-22')
    next_monday = latest_date

    while next_monday > earliest_date:
        for index, company in companies.iterrows():
            if not os.path.exists(f'{news_dir}/{company["Name"]}'):
                os.mkdir(f'{news_dir}/{company["Name"]}')

            market_data = pd.read_csv(f"Yahoo_Data/{company['Symbol']}.csv")

            earliest_date = date.fromisoformat(market_data.loc[0, 'Date'])
            latest_date = date.fromisoformat(market_data.loc[len(market_data)-1, 'Date'])

            last_monday = earliest_date - timedelta(days=earliest_date.weekday())
            next_monday = last_monday + timedelta(weeks=1)
            data = get_news_data(company['Name'], last_monday.isoformat(), next_monday.isoformat())

            if len(data.json()['response']['docs']) > 0:
                with open(f'{news_dir}/{company["Name"]}/Week_of_{last_monday}.json', 'w') as outfile:
                    json.dump(data.json(), outfile)

            last_monday = next_monday
            next_monday += timedelta(weeks=1)
            time.sleep(6)

if __name__=='__main__':
    main()

