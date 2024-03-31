import pandas as pd

# i) Create a pandas dataframe for calories_data
calories_data = {'day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                 'calories_consumed': [2000, 2200, 2100, 2300, 2400],
                 'calories_burnt': [500, 600, 550, 700, 650]}
df = pd.DataFrame(calories_data)
print("DataFrame for calories_data:")
print(df)
print()

# ii) Add an additional column calories_remaining and calculate it
df['calories_remaining'] = df['calories_consumed'] - df['calories_burnt']
print("DataFrame with additional column calories_remaining:")
print(df)
print()

# iii) Display calories_consumed and calories_burnt daywise with days as index
df.set_index('day', inplace=True)
print("DataFrame with days as index:")
print(df[['calories_consumed', 'calories_burnt']])
print()

# iv) Store the dataframe to a CSV file
df.to_csv('calories_data.csv')
print("DataFrame stored to 'calories_data.csv'")

# v) Display the pandas version
print("Pandas version:", pd.__version__)
