import numpy as np
import pandas as pd

dict_block = np.load('../data/block_features/block_f.npy').item()
#read the crawled .npy file as dataframe
x = pd.DataFrame.from_dict(dict_block).T
#read textual block features processed by LDA model
y = pd.read_csv('../data/block_features/blockfeatures_onehot.csv')
y = y.rename(columns={'Unnamed: 0':'blockid'})
#initialise an empty dataframe 
df = pd.DataFrame()
df['Cottages'] = x['Accommodation Type'].apply(lambda x : str(x).split('Cottages\', ')[1].split(')')[0] if ('Cottage' in str(x))
                                             else 0)
df['Hotels'] = x['Accommodation Type'].apply(lambda x : str(x).split('Hotels\', ')[1].split(')')[0] if ('Hotels' in str(x))
                                             else 0)

df['England'] = x['Country'].apply(lambda x : str(x).split('England\', ')[1].split(')')[0] if ('England' in str(x))
                                             else 0)
df['France'] = x['Country'].apply(lambda x : str(x).split('France\', ')[1].split(')')[0] if ('France' in str(x))
                                             else 0)
df['Ireland'] = x['Country'].apply(lambda x : str(x).split('Ireland\', ')[1].split(')')[0] if ('Ireland' in str(x))
                                             else 0)
df['Scotland'] = x['Country'].apply(lambda x : str(x).split('Scotland\', ')[1].split(')')[0] if ('Scotland' in str(x))
                                             else 0)
df['Wales'] = x['Country'].apply(lambda x : str(x).split('Wales\', ')[1].split(')')[0] if ('Wales' in str(x))
                                             else 0)

df['Barbecue'] = x['Features'].apply(lambda x : str(x).split('Barbecue\', ')[1].split(')')[0] if ('Barbecue' in str(x))
                                             else 0)
df['CarParking'] = x['Features'].apply(lambda x : str(x).split('Car Parking\', ')[1].split(')')[0] if ('Car Parking' in str(x))
                                             else 0)
df['City'] = x['Features'].apply(lambda x : str(x).split('City\', ')[1].split(')')[0] if ('City' in str(x))
                                             else 0)
df['Coastal'] = x['Features'].apply(lambda x : str(x).split('Coastal\', ')[1].split(')')[0] if ('Coastal' in str(x))
                                             else 0)
df['Cot Available'] = x['Features'].apply(lambda x : str(x).split('Cot Available\', ')[1].split(')')[0] if ('Cot Available' in str(x))
                                             else 0)
df['DineWithPet'] = x['Features'].apply(lambda x : str(x).split('Dine With Pet\', ')[1].split(')')[0] if ('Dine With Pet' in str(x))
                                             else 0)
df['EnclosedGarden/Patio'] = x['Features'].apply(lambda x : str(x).split('Enclosed Garden/Patio\', ')[1].split(')')[0] if ('Enclosed Garden/Patio' in str(x))
                                             else 0)
df['DogsStayFree'] = x['Features'].apply(lambda x : str(x).split('Dogs Stay Free\', ')[1].split(')')[0] if ('Dogs Stay Free' in str(x))
                                             else 0)
df['FamilyRooms'] = x['Features'].apply(lambda x : str(x).split('Family Rooms\', ')[1].split(')')[0] if ('Family Rooms' in str(x))
                                             else 0)
df['FitnessCentre'] = x['Features'].apply(lambda x : str(x).split('Fitness Centre\', ')[1].split(')')[0] if ('Fitness Centre' in str(x))
                                             else 0)
df['GamesRoom'] = x['Features'].apply(lambda x : str(x).split('Games Room\', ')[1].split(')')[0] if ('Games Room' in str(x))
                                             else 0)
df['Garden/Patio'] = x['Features'].apply(lambda x : str(x).split('Garden Or Patio\', ')[1].split(')')[0] if ('Garden Or Patio' in str(x))
                                             else 0)
df['GroundFloorFacilities'] = x['Features'].apply(lambda x : str(x).split('Ground Floor Facilities\', ')[1].split(')')[0] if ('Ground Floor Facilities' in str(x))
                                             else 0)
df['HotTub'] = x['Features'].apply(lambda x : str(x).split('Hot Tub\', ')[1].split(')')[0] if ('Hot Tub' in str(x))
                                             else 0)
df['PetAmenitiesOnsite'] = x['Features'].apply(lambda x : str(x).split('Pet Amenities Onsite\', ')[1].split(')')[0] if ('Pet Amenities Onsite' in str(x))
                                             else 0)
df['PubWithin1Mile'] = x['Features'].apply(lambda x : str(x).split('Pub Within 1 Mile\', ')[1].split(')')[0] if ('Pub Within 1 Mile' in str(x))
                                             else 0)
df['Restaurant'] = x['Features'].apply(lambda x : str(x).split('Restaurant\', ')[1].split(')')[0] if ('Restaurant' in str(x))
                                             else 0)
df['Rural'] = x['Features'].apply(lambda x : str(x).split('Rural\', ')[1].split(')')[0] if ('Rural' in str(x))
                                             else 0)
df['SatelliteTelevision'] = x['Features'].apply(lambda x : str(x).split('Satellite Television\', ')[1].split(')')[0] if ('Satellite Television' in str(x))
                                             else 0)
df['WiFi'] = x['Features'].apply(lambda x : str(x).split('WiFi\', ')[1].split(')')[0] if ('WiFi' in str(x))
                                             else 0)
df['SpaBreaks'] = x['Features'].apply(lambda x : str(x).split('Spa Breaks\', ')[1].split(')')[0] if ('Spa Breaks' in str(x))
                                             else 0)
df['SwimmingPool'] = x['Features'].apply(lambda x : str(x).split('Swimming Pool\', ')[1].split(')')[0] if ('Swimming Pool' in str(x))
                                             else 0)

df['CottagesforSunseekers'] = x['Inspiration'].apply(lambda x : str(x).split('Cottages for Sunseekers\', ')[1].split(')')[0] if ('Cottages for Sunseekers' in str(x))
                                             else 0)
df['Off-PeakBreaks'] = x['Inspiration'].apply(lambda x : str(x).split('Dog-friendly Off-Peak Breaks\', ')[1].split(')')[0] if ('Dog-friendly Off-Peak Breaks' in str(x))
                                             else 0)
df['BankHolidayBreaks'] = x['Inspiration'].apply(lambda x : str(x).split('Bank Holiday Breaks\', ')[1].split(')')[0] if ('Bank Holiday Breaks' in str(x))
                                             else 0)
df['CottagesHotTubs'] = x['Inspiration'].apply(lambda x : str(x).split('Cottages with Hot Tubs\', ')[1].split(')')[0] if ('Cottages with Hot Tubs' in str(x))
                                             else 0)
df['CountryCottages'] = x['Inspiration'].apply(lambda x : str(x).split('Country Cottages\', ')[1].split(')')[0] if ('Country Cottages' in str(x))
                                             else 0)
df['EscapetoCountry'] = x['Inspiration'].apply(lambda x : str(x).split('Escape to the Country\', ')[1].split(')')[0] if ('Escape to the Country' in str(x))
                                             else 0)
df['Exclusive Offers'] = x['Inspiration'].apply(lambda x : str(x).split('Exclusive Offers\', ')[1].split(')')[0] if ('Exclusive Offers' in str(x))
                                             else 0)
df['OctoberHT'] = x['Inspiration'].apply(lambda x : str(x).split('October Half Term Availability\', ')[1].split(')')[0] if ('October Half Term Availability' in str(x))
                                             else 0)
df['TheWinners2016'] = x['Inspiration'].apply(lambda x : str(x).split('The Winners - Travel Awards 2016\', ')[1].split(')')[0] if ('The Winners - Travel Awards 2016' in str(x))
                                             else 0)

df['Dogs1'] = x['Number of Dogs'].apply(lambda x : str(x).split('(\'1\', ')[1].split(')')[0] if ('(\'1' in str(x))
                                             else 0)
df['Dogs2'] = x['Number of Dogs'].apply(lambda x : str(x).split('(\'2\', ')[1].split(')')[0] if ('(\'2' in str(x))
                                             else 0)
df['Dogs3'] = x['Number of Dogs'].apply(lambda x : str(x).split('(\'3\', ')[1].split(')')[0] if ('(\'3' in str(x))
                                             else 0)
df['Dogs4+'] = x['Number of Dogs'].apply(lambda x : str(x).split('(\'4+\', ')[1].split(')')[0] if ('(\'4+' in str(x))
                                             else 0)

df['Prople1'] = x['Number of People'].apply(lambda x : str(x).split('(\'1\', ')[1].split(')')[0] if ('(\'1\', ' in str(x))
                                             else 0)
df['Prople2'] = x['Number of People'].apply(lambda x : str(x).split('(\'2\', ')[1].split(')')[0] if ('(\'2\', ' in str(x))
                                             else 0)
df['Prople3'] = x['Number of People'].apply(lambda x : str(x).split('(\'3\', ')[1].split(')')[0] if ('(\'3\', ' in str(x))
                                             else 0)
df['Prople4'] = x['Number of People'].apply(lambda x : str(x).split('(\'4\', ')[1].split(')')[0] if ('(\'4\', ' in str(x))
                                             else 0)
df['Prople5'] = x['Number of People'].apply(lambda x : str(x).split('(\'5\', ')[1].split(')')[0] if ('(\'5\', ' in str(x))
                                             else 0)
df['Prople6'] = x['Number of People'].apply(lambda x : str(x).split('(\'6\', ')[1].split(')')[0] if ('(\'6\', ' in str(x))
                                             else 0)
df['Prople7'] = x['Number of People'].apply(lambda x : str(x).split('(\'7\', ')[1].split(')')[0] if ('(\'7\', ' in str(x))
                                             else 0)
df['Prople8'] = x['Number of People'].apply(lambda x : str(x).split('(\'8\', ')[1].split(')')[0] if ('(\'8\', ' in str(x))
                                             else 0)
df['Prople9'] = x['Number of People'].apply(lambda x : str(x).split('(\'9\', ')[1].split(')')[0] if ('(\'9\', ' in str(x))
                                             else 0)
df['Prople10+'] = x['Number of People'].apply(lambda x : str(x).split('(\'10+\', ')[1].split(')')[0] if ('(\'10+\', ' in str(x))
                                             else 0)

df['blockid'] = list(pd.read_csv("../data/block_features/urls.csv",encoding = "utf-8")['URL'].values)
#merge two parts of block features
block_features = pd.merge(df,y,on=['blockid'])
block_features = pd.concat([y,df],axis=1).fillna(0)
x = block_features.drop(columns=['blockid'])
block_features = pd.concat([y[['blockid']],x],axis=1)
block_features.to_csv("../data/block_features/block_features.csv",index=False,sep=',')

'''
from sklearn import preprocessing
#block_f = pd.read_csv('../data/block_features.csv')
a = pd.DataFrame(preprocessing.normalize(block_features.iloc[:,1:].values, norm='l2'))
block_f_norm = pd.concat([block_features[['blockid']],a],axis = 1)
block_f_norm.to_csv("../data/block_features_norm.csv",index=False,sep=',')
'''