# Please make a copy of this file in this directory named login.py and fill in the login details for your database(s).
# Dictionary to store login details
LOGIN_DETAILS = {
    "postgresql": {
        "user": "your_username",
        "pass": "your_password",
        "host": "localhost", # At EPA, use the static IP address of the PostgreSQL server
        "port": "5432"
    },
    "oracle": {
        "user": "your_username",
        "pass": "your_password",
        "host": "localhost", # At EPA, use the static IP address of the Oracle server
        "port": "1521"
    }
}