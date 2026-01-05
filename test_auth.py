import json
import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

SA_JSON = "noza-481210-2dce90c6dbfd.json"  # mets le vrai nom exact
SPREADSHEET_ID = "1lSqjJb-6HZQVfsDVrjeN2wULtFW7luK4d-HmBQOlx58"

with open(SA_JSON, "r", encoding="utf-8-sig") as f:
    info = json.load(f)

creds = Credentials.from_service_account_info(info, scopes=SCOPES)
gc = gspread.authorize(creds)

sh = gc.open_by_key(SPREADSHEET_ID)
print("OK ->", sh.title)