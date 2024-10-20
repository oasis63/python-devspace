import random
import mysql.connector
from faker import Faker

# Establishing connection to MySQL
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="oasis_db"
)

cursor = connection.cursor()

# Using Faker library to generate random sample data
fake = Faker()

genders = ['Male', 'Female', 'Other']
marital_statuses = ['Single', 'Married', 'Divorced', 'Widowed']
education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
hobbies = ['Cycling', 'Reading', 'Photography', 'Running', 'Gaming', 'Painting', 'Hiking', 'Gardening', 'Traveling', 'Yoga']

# Generate and insert 100 rows of data
for _ in range(100):
    first_name = fake.first_name()
    last_name = fake.last_name()
    date_of_birth = fake.date_of_birth(minimum_age=18, maximum_age=80)
    gender = random.choice(genders)
    email = fake.email()
    phone_number = fake.phone_number()
    address = fake.street_address()
    city = fake.city()
    state = fake.state()
    country = 'USA'
    postal_code = fake.zipcode()
    occupation = fake.job()
    company = fake.company()
    annual_income = round(random.uniform(30000, 150000), 2)
    marital_status = random.choice(marital_statuses)
    number_of_children = random.randint(0, 5)
    education_level = random.choice(education_levels)
    hobby = random.choice(hobbies)
    
    query = """
    INSERT INTO person (first_name, last_name, date_of_birth, gender, email, phone_number, address, city, state, country, postal_code, occupation, company, annual_income, marital_status, number_of_children, education_level, hobby)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    cursor.execute(query, (first_name, last_name, date_of_birth, gender, email, phone_number, address, city, state, country, postal_code, occupation, company, annual_income, marital_status, number_of_children, education_level, hobby))

# Committing the transaction
connection.commit()

# Closing the connection
cursor.close()
connection.close()
