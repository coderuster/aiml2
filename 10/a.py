import pandas as pd

# i) Create a pandas dataframe for the string "He is a good person"
string_data = "He is a good person"
df_string = pd.DataFrame(list(string_data.split()), columns=['Word'])
print("DataFrame for string 'He is a good person':")
print(df_string)
print()

# ii) Create a pandas dataframe for car_data
car_data = {'Car Name': ['Toyota', 'Honda', 'Ford', 'BMW'],
            'Price': [25000, 27000, 23000, 40000]}
df_car = pd.DataFrame(car_data)
print("DataFrame for car_data:")
print(df_car)
print()

# Display list of car names
print("List of car names:")
print(df_car['Car Name'])
print()

# Display the details of the second car using index
print("Details of the second car:")
print(df_car.iloc[1])
print()

# iii) Append the new car_data to the existing dataframe
new_car_data = {'Car Name': ['Mercedes', 'Audi'],
                'Price': [55000, 50000]}
df_car = df_car.append(pd.DataFrame(new_car_data), ignore_index=True)
print("Updated DataFrame with new car_data:")
print(df_car)
print()

# iv) Update any data in the dataframe
# Updating the price of Toyota to 26000
df_car.loc[df_car['Car Name'] == 'Toyota', 'Price'] = 26000
print("Updated DataFrame with modified price of Toyota:")
print(df_car)
print()

# v) Store the dataframe to a CSV file
df_car.to_csv('car_data.csv', index=False)
print("DataFrame stored to 'car_data.csv'")

# vi) Reading a CSV file and storing it in another dataframe
df_read_csv = pd.read_csv('car_data.csv')
print("\nDataframe from 'car_data.csv':")
print(df_read_csv)
