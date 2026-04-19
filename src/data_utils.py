"""
data_utils.py — Synthetic data generators for Aadhaar card fields.

Provides:
  - Indian name generation (male / female)
  - DOB generation (age-consistent or deliberately mismatched)
  - 12-digit Aadhaar number generation (valid format, Verhoeff checksum)
  - Utility: detect gender from filename heuristics (fallback)
"""

import random
import string
from datetime import date, timedelta

# ─── Embedded Indian Name Lists ───────────────────────────────────────────────

MALE_FIRST_NAMES = [
    "Aarav", "Arjun", "Vikram", "Rohan", "Karan", "Rahul", "Aditya", "Siddharth",
    "Mihir", "Kunal", "Nikhil", "Suresh", "Rajesh", "Amit", "Ajay", "Vijay",
    "Manish", "Pradeep", "Deepak", "Santosh", "Harish", "Ramesh", "Dinesh",
    "Sunil", "Ankit", "Varun", "Gaurav", "Shubham", "Pranav", "Yash", "Dev",
    "Sachin", "Rohit", "Virat", "Hardik", "Ishaan", "Kabir", "Rishi", "Akshat",
    "Naveen", "Pavan", "Kiran", "Sanjay", "Vijayendra", "Ravi", "Mohan",
    "Bhaskar", "Arvind", "Girish", "Manoj", "Himanshu", "Aniket", "Tushar",
    "Lokesh", "Parth", "Nishant", "Sarang", "Chetan", "Mohit", "Lakshya",
    "Tanmay", "Abhishek", "Vinit", "Shoaib", "Faisal", "Irfan", "Adnan",
    "Rajan", "Vivek", "Aakash", "Sumit", "Nitin", "Lalit", "Hemant",
    "Satish", "Vinod", "Rakesh", "Ashok", "Mahesh", "Umesh", "Suresh",
    "Dilip", "Prakash", "Ganesh", "Naresh", "Ramakrishna", "Venkat",
    "Balaji", "Srikanth", "Murali", "Raghav", "Sriram", "Anand", "Arun",
    "Chandramohan", "Prashant", "Sourabh", "Aviral", "Shreyas", "Dhruv",
    "Vignesh", "Karthik", "Aravind", "Sudhir", "Rajiv", "Sandeep",
]

FEMALE_FIRST_NAMES = [
    "Priya", "Ananya", "Deepika", "Shreya", "Pooja", "Kavya", "Neha", "Aarti",
    "Sunita", "Rekha", "Geeta", "Meena", "Lata", "Seema", "Ritu", "Sonia",
    "Nisha", "Divya", "Swati", "Jyoti", "Alka", "Vandana", "Shalini", "Radha",
    "Sudha", "Usha", "Savita", "Anita", "Smita", "Vinita", "Amita", "Kavitha",
    "Manjula", "Ratna", "Parvathi", "Lakshmi", "Saraswathi", "Meenakshi",
    "Bhavana", "Rashmi", "Ankita", "Tanvi", "Richa", "Isha", "Nidhi",
    "Pallavi", "Shweta", "Kritika", "Mansi", "Megha", "Aishwarya", "Sakshi",
    "Aditi", "Ruhi", "Zara", "Farah", "Asha", "Kiran", "Varsha", "Madhuri",
    "Sadhana", "Hemali", "Jalpa", "Komal", "Monal", "Foram", "Riddhi",
    "Siddhi", "Krupa", "Ruchita", "Niyati", "Drashti", "Heena", "Bhoomi",
    "Poonam", "Harsha", "Jayshree", "Renuka", "Uma", "Anuradha", "Indira",
    "Sushma", "Chameli", "Nalini", "Saroja", "Gomathi", "Padmini", "Vimala",
    "Chitra", "Hema", "Revathi", "Amrutha", "Keerthi", "Archana", "Swapna",
    "Lavanya", "Yamini", "Rohini", "Shakunthala", "Pramila", "Vijaya",
    "Sumitra", "Kalyani", "Malathi", "Bharathi", "Saranya", "Kiruthika",
]

SURNAMES = [
    "Sharma", "Patel", "Iyer", "Verma", "Singh", "Kumar", "Gupta", "Joshi",
    "Mehta", "Shah", "Nair", "Pillai", "Reddy", "Rao", "Naidu", "Menon",
    "Krishnan", "Subramaniam", "Rajan", "Sundar", "Bhat", "Hegde",
    "Shetty", "D'Souza", "Fernandez", "Pereira", "Thomas", "Joseph",
    "Jacob", "Mathew", "George", "Philip", "Abraham", "John", "Paul",
    "Agarwal", "Agrawal", "Banerjee", "Chatterjee", "Mukherjee", "Das",
    "Ghosh", "Bose", "Sen", "Roy", "Dey", "Mondal", "Chakraborty",
    "Mishra", "Pandey", "Tiwari", "Dubey", "Yadav", "Chauhan", "Rawat",
    "Thakur", "Saxena", "Srivastava", "Shukla", "Tripathi", "Dube",
    "Patil", "Desai", "Jain", "Kothari", "Modi", "Parekh", "Thakkar",
    "Chaudhary", "Malik", "Kapoor", "Khanna", "Bhatia", "Arora", "Walia",
    "Gill", "Dhillon", "Wahan", "Sandhu", "Grewal", "Sidhu", "Bajwa",
    "Pillai", "Kutty", "Unni", "Warrier", "Panikar", "Thampi", "Namboothiri",
    "Gowda", "Hebbar", "Urs", "Swamy", "Murthy", "Prasad", "Narayana",
]

# ─── Verhoeff Algorithm (Aadhaar uses Verhoeff checksum) ─────────────────────

_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def _verhoeff_checksum(number: str) -> int:
    """Compute Verhoeff check digit for a numeric string."""
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][int(ch)]]
    return _VERHOEFF_INV[c]


def generate_aadhaar_number(rng: random.Random = None) -> str:
    """
    Generate a valid 12-digit Aadhaar number with Verhoeff checksum.
    Returns formatted string: 'XXXX XXXX XXXX'
    """
    if rng is None:
        rng = random
    # First digit must be 2–9 (Aadhaar spec)
    first = str(rng.randint(2, 9))
    middle = "".join([str(rng.randint(0, 9)) for _ in range(10)])
    base = first + middle          # 11 digits
    check = _verhoeff_checksum(base)
    full = base + str(check)       # 12 digits
    return f"{full[:4]} {full[4:8]} {full[8:]}"


# ─── Name Generators ─────────────────────────────────────────────────────────

def generate_indian_name(gender: str, rng: random.Random = None) -> str:
    """
    Generate a random Indian full name matching the given gender.

    Args:
        gender: 'male' | 'female'  (case-insensitive)
        rng:    Optional seeded Random instance for reproducibility.

    Returns:
        'FirstName Surname'
    """
    if rng is None:
        rng = random
    gender = gender.lower().strip()
    pool = MALE_FIRST_NAMES if gender in ("male", "man", "m") else FEMALE_FIRST_NAMES
    first   = rng.choice(pool)
    surname = rng.choice(SURNAMES)
    return f"{first} {surname}"


def get_mismatched_name(actual_gender: str, rng: random.Random = None) -> str:
    """Return a name of the OPPOSITE gender (intentional mismatch for fakes)."""
    if rng is None:
        rng = random
    actual_gender = actual_gender.lower().strip()
    opposite = "female" if actual_gender in ("male", "man", "m") else "male"
    return generate_indian_name(opposite, rng)


# ─── DOB Generators ──────────────────────────────────────────────────────────

def _dob_from_age(target_age: int, variance: int, rng: random.Random) -> str:
    """Internal: build a DOB string (DD/MM/YYYY) for a person of approximately target_age."""
    actual_age = target_age + rng.randint(-variance, variance)
    actual_age = max(1, min(actual_age, 120))
    today = date.today()
    birth_year = today.year - actual_age
    # Random day/month within the year
    birth_date = date(birth_year, rng.randint(1, 12), 1)
    max_day = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][birth_date.month - 1]
    # Leap year feb correction
    if birth_date.month == 2 and (birth_year % 4 == 0 and (birth_year % 100 != 0 or birth_year % 400 == 0)):
        max_day = 29
    birth_date = birth_date.replace(day=rng.randint(1, max_day))
    return birth_date.strftime("%d/%m/%Y")


def generate_dob(target_age: int, variance: int = 3, rng: random.Random = None) -> str:
    """
    Generate a DOB consistent with the given face age (±variance years).
    Format: 'DD/MM/YYYY'
    """
    if rng is None:
        rng = random
    return _dob_from_age(target_age, variance, rng)


def get_mismatched_dob(actual_age: int, offset_years: int = 25, rng: random.Random = None) -> str:
    """
    Generate a deliberately mismatched DOB (shifted by offset_years in a random direction).
    Keeps the resulting age plausible (5–90 years).
    """
    if rng is None:
        rng = random
    direction = rng.choice([-1, 1])
    offset = offset_years + rng.randint(0, 10)   # 25–35 year mismatch
    mismatched_age = actual_age + direction * offset
    mismatched_age = max(5, min(mismatched_age, 90))
    return _dob_from_age(mismatched_age, variance=0, rng=rng)


# ─── Quick Sanity Check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = random.Random(42)
    print("Male name:   ", generate_indian_name("male",   rng))
    print("Female name: ", generate_indian_name("female", rng))
    print("Mismatch:    ", get_mismatched_name("male",    rng))
    print("DOB (age 28):", generate_dob(28, rng=rng))
    print("Mismatch DOB:", get_mismatched_dob(28, rng=rng))
    for _ in range(5):
        print("Aadhaar:     ", generate_aadhaar_number(rng))
