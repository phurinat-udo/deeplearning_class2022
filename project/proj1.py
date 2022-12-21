# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %%
raw = pd.read_csv('Global_Mobility_Report.csv')
raw
# %%
rawGDP = pd.read_csv('WEO_Data.csv')
# %%
GDPout = rawGDP.drop(['ISO', 'Subject Descriptor', 'Subject Notes', 'Units', 'Scale', 'Country/Series-specific Notes', 'Estimates Start After', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2022', '2023', '2024', '2025', '2026', '2027'], axis=1).copy()
GDPout = GDPout.drop([196, 197], axis=0).copy()
# %%
mobilIn = raw.drop(['country_region_code', 'sub_region_1', 'sub_region_2', 'metro_area', 'iso_3166_2_code', 'census_fips_code', 'place_id'], axis=1).copy()
mobilIn
# %%
for reg in mobilIn.country_region.unique():
    print(reg)
# %%
