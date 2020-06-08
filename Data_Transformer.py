import pandas as pd
from datetime import date

inflation_rates = pd.read_fwf('CPI-U.txt', infer_nrows=1300).loc[:, 'year':]

mappings = {'M01': 1, 'M02': 2, 'M03': 3, 'M04': 4, 'M05': 5, 'M06': 6, 'M07': 7, 'M08': 8, 'M09': 9, 'M10': 10, 'M11': 11, 'M12': 12}

inflation_rates = inflation_rates[inflation_rates['period'] != 'M13']
inflation_rates['period'] = inflation_rates['period'].map(mappings)
inflation_rates = inflation_rates.set_index(['year', 'period'])

def inflation_adjustment(row):
    stock_date =  date.fromisoformat(row['Date'])
    row.loc['Open'] = (row.loc['Open'] / inflation_rates.loc[stock_date.year].loc[stock_date.month]['value']) * inflation_rates.loc[2020].loc[5]['value']
    row.loc['Low'] = (row.loc['Low'] / inflation_rates.loc[stock_date.year].loc[stock_date.month]['value']) * inflation_rates.loc[2020].loc[5]['value']
    row.loc['High'] = (row.loc['High'] / inflation_rates.loc[stock_date.year].loc[stock_date.month]['value']) * inflation_rates.loc[2020].loc[5]['value']
    row.loc['Close'] = (row.loc['Close'] / inflation_rates.loc[stock_date.year].loc[stock_date.month]['value']) * inflation_rates.loc[2020].loc[5]['value']
    row.loc['Adj Close'] = (row.loc['Adj Close'] / inflation_rates.loc[stock_date.year].loc[stock_date.month]['value']) * inflation_rates.loc[2020].loc[5]['value']
    return row

def main():
    companies = pd.read_json("s-and-p-500-companies/data/constituents_json.json")
    
    for stock in companies.loc[78:, 'Symbol']:
        stock_data = pd.read_csv(f'Yahoo_Data/{stock}.csv')

        try:
            stock_data['Next Close'] = stock_data['Close'].shift(-1)

            stock_data = stock_data.apply(inflation_adjustment, axis=1)

            print(f"Writing {stock} to file ...")
            stock_data.to_csv(f'Inflation_Adjusted/{stock}.csv')
            print("Complete")
        except KeyError:
            print (f"Stock {stock} does not have data")


if __name__ == '__main__':
    main()