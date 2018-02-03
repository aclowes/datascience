import csv
import time
import urllib.parse

import requests

# api documentation: https://projects.propublica.org/nonprofits/api
base_url = 'https://projects.propublica.org/nonprofits/api/v2'

# field definitions: http://www.irs.gov/file_source/pub/irs-soi/12eofinextractdoc.xls
field_map = {
    'EIN': 'ein',
    'Name': 'name',
    # 'Secondary Name': 'sub_name',
    'Tax Year': 'tax_prd_yr',
    '990 PDF': 'pdf_url',
    'Revenue': 'totrevenue',
    'Expenses': 'totfuncexpns',
    'Assets': 'totassetsend',
    'Liabilities': 'totliabend',
    'Officer Expenses': 'compnsatncurrofcr',
    'Salary Expenses': 'othrsalwages',
    'Contribution Revenues': 'totcntrbgfts',
    'Program Revenues': 'totprgmrevnue',
}


def scrape():
    reader = csv.DictReader(open('inputs.csv'))
    fields = reader.fieldnames + list(field_map.keys())
    writer = csv.DictWriter(open('output.csv', 'w+'), fields)
    writer.writeheader()

    for row in reader:
        result = dict(row)
        escaped_name = urllib.parse.quote(row['Organization Name'], safe='')
        search_url = '{}/search.json?q={}'.format(base_url, escaped_name)
        response = requests.get(search_url)
        if not response.status_code == 200:
            print('Failed to search {}, response was: {}'.format(
                row['Organization Name'], response.content.decode()[:200]))
            continue
        data = response.json()

        if data['organizations']:
            organization = data['organizations'][0]
            ein = organization['ein']
            result.update({
                key: organization[value] for key, value
                in field_map.items() if value in organization
            })
            filing_url = '{}/organizations/{}.json'.format(base_url, ein)
            response = requests.get(filing_url)
            if not response.status_code == 200:
                print('Failed to load {}, response was: {}'.format(
                    row['Organization Name'], response.content.decode()[:200]))
                continue
            data = response.json()

            if data['filings_with_data']:
                filing = data['filings_with_data'][0]
                result.update({
                    key: filing[value] for key, value
                    in field_map.items() if value in filing
                })

        writer.writerow(result)
        time.sleep(5)  # so we don't get throttled

if __name__ == '__main__':
    scrape()
